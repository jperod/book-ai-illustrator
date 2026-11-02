"""Book Illustrator: paste a page of text, get an illustration from a local ComfyUI.

Standard library only. ComfyUI must be running separately (default
http://127.0.0.1:8188) and workflow_api.json must exist next to this file.
See README.md for the one-time setup.
"""
import base64
import json
import os
import random
import re
import socket
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
WORKFLOW_PATH = os.path.join(BASE_DIR, "workflow_api.json")
BOOKS_DIR = os.path.join(BASE_DIR, "books")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

CONFIG = {
    "comfy_url": "http://127.0.0.1:8188",
    "port": 8000,
    "width": 1024,
    "height": 1024,
    "timeout_seconds": 600,
    "open_browser": True,
    "steps": 4,
}

# One generation at a time: the GPU can only do one well.
GENERATE_LOCK = threading.Lock()


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            CONFIG.update(json.load(f))


# ---------- visual identity (style box) ----------

# ---------- books: each book file holds its own visual style and characters ----------
# books/<id>.json looks like:
#   {"id", "title", "author", "series", "book_number", "visual_style", "characters": [{"name", "description"}]}

BOOK_ID_RE = re.compile(r"^[a-z0-9-]+$")


def book_file(book_id):
    if not BOOK_ID_RE.match(book_id):
        raise ValueError(f"bad book id: {book_id!r}")
    return os.path.join(BOOKS_DIR, book_id + ".json")


def load_book(book_id):
    path = book_file(book_id)
    if not os.path.exists(path):
        raise FileNotFoundError(book_id)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_book(book):
    os.makedirs(BOOKS_DIR, exist_ok=True)
    with open(book_file(book["id"]), "w", encoding="utf-8") as f:
        json.dump(book, f, indent=2, ensure_ascii=False)
        f.write("\n")


def list_books():
    books = []
    if os.path.isdir(BOOKS_DIR):
        for fn in os.listdir(BOOKS_DIR):
            if fn.endswith(".json"):
                with open(os.path.join(BOOKS_DIR, fn), encoding="utf-8") as f:
                    b = json.load(f)
                books.append({
                    "id": b["id"],
                    "title": b.get("title", b["id"]),
                    "author": b.get("author", ""),
                    "series": b.get("series", ""),
                    "book_number": b.get("book_number", 0),
                })
    return sorted(books, key=lambda b: (b["series"], b["book_number"], b["title"]))


def characters_to_text(characters):
    return "\n".join(f"{c['name']}: {c['description']}" for c in characters)


def text_to_characters(text):
    # Lines without a 'Name: description' shape are dropped on save.
    return [{"name": n, "description": d} for n, d in parse_characters(text)]


def book_view(book):
    return {
        "id": book["id"],
        "title": book.get("title", book["id"]),
        "author": book.get("author", ""),
        "series": book.get("series", ""),
        "book_number": book.get("book_number", 0),
        "visual_style": book.get("visual_style", ""),
        "characters_text": characters_to_text(book.get("characters", [])),
    }


def update_book(book_id, fields):
    """Applies any of visual_style / characters_text that are present, saves, and returns the book."""
    book = load_book(book_id)
    if "visual_style" in fields:
        book["visual_style"] = str(fields["visual_style"])
    if "characters_text" in fields:
        book["characters"] = text_to_characters(str(fields["characters_text"]))
    save_book(book)
    return book


SINGLE_IMAGE_NOTE = "A single painting of one continuous scene filling the whole canvas, edge to edge."


def parse_characters(characters_text):
    """One character per line, 'Name: description'. Returns a list of (name, description)."""
    chars = []
    for line in characters_text.splitlines():
        if ":" not in line:
            continue
        name, desc = (part.strip() for part in line.split(":", 1))
        if name and desc:
            chars.append((name, desc))
    return chars


def characters_in_scene(characters_text, scene_text):
    """Only the characters whose name appears in the pasted text are added to the prompt."""
    return [
        (name, desc)
        for name, desc in parse_characters(characters_text)
        # Case-sensitive on purpose: "patience", "regal" or "fool" in lowercase are ordinary words.
        if re.search(r"\b" + re.escape(name) + r"\b", scene_text)
    ]


def build_prompt(style, text, characters_text=""):
    # Scene first, then characters, then style: putting the long style text first
    # was pulling compositions indoors.
    text = text.strip()
    parts = [SINGLE_IMAGE_NOTE, "", "Scene from the book:", text]
    chars = characters_in_scene(characters_text, text)
    if chars:
        parts += ["", "Characters in this scene (keep their appearance consistent):"]
        parts += [f"- {name}: {desc}" for name, desc in chars]
    if style.strip():
        parts += ["", f"Visual style: {style.strip()}"]
    return "\n".join(parts)


# ---------- workflow handling ----------

def is_text_prompt_node(node):
    class_type = str(node.get("class_type", ""))
    inputs = node.get("inputs", {})
    return ("TextEncode" in class_type or "CLIPText" in class_type) and isinstance(inputs.get("text"), str)


def make_workflow(prompt_text):
    """Load the exported ComfyUI workflow (API format) and fill in prompt, seed and size."""
    if not os.path.exists(WORKFLOW_PATH):
        raise RuntimeError(
            "workflow_api.json not found. Export your ComfyUI workflow in API format "
            "and save it in this folder (see README.md)."
        )
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        workflow = json.load(f)

    prompt_nodes = 0
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})

        if is_text_prompt_node(node):
            inputs["text"] = prompt_text
            prompt_nodes += 1

        for key in ("seed", "noise_seed"):
            if isinstance(inputs.get(key), int):
                inputs[key] = random.randint(0, 2**32 - 1)

        # Size lives on the latent node and on the scheduler (Flux2Scheduler uses it for sigmas).
        if isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int):
            inputs["width"] = int(CONFIG["width"])
            inputs["height"] = int(CONFIG["height"])
        if "Scheduler" in str(node.get("class_type", "")) and isinstance(inputs.get("steps"), int):
            inputs["steps"] = int(CONFIG["steps"])

    if prompt_nodes == 0:
        raise RuntimeError(
            "No text prompt node found in workflow_api.json (looked for CLIPTextEncode / TextEncode "
            "nodes with a 'text' input)."
        )
    return workflow


# ---------- ComfyUI HTTP API ----------

def comfy_url(path):
    return CONFIG["comfy_url"].rstrip("/") + path


def comfy_get(path, timeout=30):
    with urllib.request.urlopen(comfy_url(path), timeout=timeout) as r:
        return r.read()


def comfy_post_json(path, payload):
    req = urllib.request.Request(
        comfy_url(path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected the workflow (HTTP {e.code}): {detail[:1500]}")


def comfy_available():
    try:
        comfy_get("/system_stats", timeout=5)
        return True
    except Exception:
        return False


# ---------- live progress (ComfyUI websocket) ----------

PROGRESS_LOCK = threading.Lock()
PROGRESS = {"active": False, "label": "", "value": 0, "max": 0}

# Friendly name for each ComfyUI node type, shown while that node runs.
STAGE_LABELS = {
    "UNETLoader": "Loading diffusion model",
    "CLIPLoader": "Loading text encoder",
    "VAELoader": "Loading VAE",
    "CLIPTextEncode": "Encoding prompt",
    "ConditioningZeroOut": "Encoding prompt",
    "CFGGuider": "Preparing sampler",
    "Flux2Scheduler": "Preparing sampler",
    "KSamplerSelect": "Preparing sampler",
    "RandomNoise": "Preparing sampler",
    "EmptyFlux2LatentImage": "Preparing sampler",
    "SamplerCustomAdvanced": "Sampling",
    "VAEDecode": "Decoding image",
    "SaveImage": "Saving",
}


def set_progress(**fields):
    with PROGRESS_LOCK:
        PROGRESS.update(fields)


def get_progress():
    with PROGRESS_LOCK:
        return dict(PROGRESS)


class ProgressListener(threading.Thread):
    """Minimal WebSocket client for ComfyUI's /ws. Reads progress events for one client_id."""

    def __init__(self, client_id, node_types):
        super().__init__(daemon=True)
        self.client_id = client_id
        self.node_types = node_types
        self.stop_event = threading.Event()
        self.sock = None
        self.buffer = b""

    def connect(self):
        parsed = urllib.parse.urlparse(CONFIG["comfy_url"])
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.sendall((
            f"GET /ws?clientId={self.client_id} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode())
        head = self._read_until(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError("ComfyUI websocket handshake failed")
        self.sock.settimeout(1.0)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=3)

    def run(self):
        try:
            while not self.stop_event.is_set():
                opcode, payload = self._read_frame()
                if opcode == 0x1:
                    try:
                        self._handle(json.loads(payload))
                    except Exception:
                        pass
                elif opcode == 0x9:
                    self._send_frame(0xA, payload)
                elif opcode == 0x8:
                    break
                # binary frames (image previews) are ignored
        except (OSError, ConnectionError):
            pass
        finally:
            self.close()

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass

    def _handle(self, msg):
        kind = msg.get("type")
        data = msg.get("data") or {}
        if kind == "progress":
            set_progress(value=int(data.get("value", 0)), max=int(data.get("max", 0)))
        elif kind == "executing" and data.get("node") is not None:
            ctype = self.node_types.get(str(data["node"]), "")
            set_progress(label=STAGE_LABELS.get(ctype, "Working"), value=0, max=0)

    def _read_until(self, marker):
        while marker not in self.buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket closed during handshake")
            self.buffer += chunk
        head, self.buffer = self.buffer.split(marker, 1)
        return head

    def _recv_exact(self, n):
        while len(self.buffer) < n:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                if self.stop_event.is_set():
                    raise ConnectionError("listener stopped")
                continue
            if not chunk:
                raise ConnectionError("websocket closed")
            self.buffer += chunk
        data, self.buffer = self.buffer[:n], self.buffer[n:]
        return data

    def _read_frame(self):
        b1, b2 = self._recv_exact(2)
        opcode = b1 & 0x0F
        masked = b2 & 0x80
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else None
        payload = self._recv_exact(length)
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    def _send_frame(self, opcode, payload):
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            header = bytes([0x80 | opcode, 0x80 | n])
        elif n < 65536:
            header = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack("!H", n)
        else:
            header = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack("!Q", n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + mask + masked)


def start_progress_listener(client_id, node_types):
    """Returns a running listener, or None if ComfyUI's websocket is unavailable (generation still works)."""
    listener = ProgressListener(client_id, node_types)
    try:
        listener.connect()
    except Exception:
        listener.close()
        return None
    listener.start()
    return listener


def generate_image(text, style, characters_text=""):
    """Run the workflow on ComfyUI, wait for it, save the PNG. Returns (filename, prompt)."""
    prompt_text = build_prompt(style, text, characters_text)
    workflow = make_workflow(prompt_text)
    client_id = uuid.uuid4().hex
    node_types = {nid: str(node.get("class_type", "")) for nid, node in workflow.items() if isinstance(node, dict)}

    set_progress(active=True, label="Starting", value=0, max=0)
    listener = start_progress_listener(client_id, node_types)
    try:
        queued = comfy_post_json("/prompt", {"prompt": workflow, "client_id": client_id})
        return _wait_for_result(queued["prompt_id"], prompt_text)
    finally:
        if listener:
            listener.stop()
        set_progress(active=False)


def _wait_for_result(prompt_id, prompt_text):
    deadline = time.time() + CONFIG["timeout_seconds"]
    while time.time() < deadline:
        history = json.loads(comfy_get(f"/history/{prompt_id}"))
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI reported an error: " + json.dumps(status.get("messages", []))[:1500])

            for node_output in entry.get("outputs", {}).values():
                for image in node_output.get("images", []):
                    if image.get("type") != "output":
                        continue
                    data = comfy_get("/view?" + urllib.parse.urlencode({
                        "filename": image["filename"],
                        "subfolder": image.get("subfolder", ""),
                        "type": "output",
                    }))
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    ext = os.path.splitext(image["filename"])[1] or ".png"
                    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6] + ext
                    with open(os.path.join(OUTPUT_DIR, name), "wb") as f:
                        f.write(data)
                    return name, prompt_text

            if status.get("completed"):
                raise RuntimeError("ComfyUI finished but produced no image. Make sure the workflow ends in a Save Image node.")
        time.sleep(1)
    raise RuntimeError("Timed out waiting for ComfyUI.")


# ---------- web server ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s %s" % (self.command, self.path))

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            with open(INDEX_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(200, {
                "comfy_ok": comfy_available(),
                "workflow_ok": os.path.exists(WORKFLOW_PATH),
                "comfy_url": CONFIG["comfy_url"],
            })
        elif path == "/api/books":
            self._json(200, list_books())
        elif path.startswith("/api/books/"):
            try:
                self._json(200, book_view(load_book(path[len("/api/books/"):])))
            except (FileNotFoundError, ValueError):
                self._json(404, {"error": "Unknown book."})
        elif path == "/api/progress":
            self._json(200, get_progress())
        elif path.startswith("/outputs/"):
            full = os.path.join(OUTPUT_DIR, os.path.basename(path[len("/outputs/"):]))
            if os.path.isfile(full):
                with open(full, "rb") as f:
                    self._send(200, f.read(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "Invalid JSON."})
            return

        if path.startswith("/api/books/"):
            try:
                update_book(path[len("/api/books/"):], body)
                self._json(200, {"ok": True})
            except (FileNotFoundError, ValueError):
                self._json(404, {"error": "Unknown book."})

        elif path == "/api/generate":
            text = str(body.get("text", "")).strip()
            if not text:
                self._json(400, {"error": "Paste some text first."})
                return
            try:
                # Save whatever the page shows, then generate from the saved book.
                book = update_book(str(body.get("book", "")), body)
            except (FileNotFoundError, ValueError):
                self._json(400, {"error": "Pick a book first."})
                return
            style = book.get("visual_style", "")
            characters_text = characters_to_text(book.get("characters", []))
            with GENERATE_LOCK:
                try:
                    name, prompt = generate_image(text, style, characters_text)
                except urllib.error.HTTPError as e:
                    self._json(500, {"error": f"ComfyUI returned HTTP {e.code} for {e.url}"})
                    return
                except urllib.error.URLError:
                    self._json(503, {"error": f"Can't reach ComfyUI at {CONFIG['comfy_url']}. Start ComfyUI first."})
                    return
                except Exception as e:
                    self._json(500, {"error": str(e)})
                    return
            self._json(200, {"image_url": f"/outputs/{name}", "prompt": prompt})

        else:
            self._send(404, b"not found", "text/plain")


def main():
    load_config()
    url = f"http://127.0.0.1:{CONFIG['port']}"
    server = ThreadingHTTPServer(("127.0.0.1", CONFIG["port"]), Handler)
    print(f"Book Illustrator running at {url}")
    print(f"Expecting ComfyUI at {CONFIG['comfy_url']}")
    if CONFIG.get("open_browser"):
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
