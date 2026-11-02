"""Book Illustrator: paste a page of text, get an illustration from a local ComfyUI.

Standard library only. ComfyUI must be running separately (default
http://127.0.0.1:8188) and workflow_api.json must exist next to this file.
See README.md for the one-time setup.
"""
import base64
import collections
import difflib
import itertools
import json
import os
import random
import re
import socket
import struct
import threading
import time
import traceback
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
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")  # per-machine settings, e.g. the default book
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

CONFIG = {
    "comfy_url": "http://127.0.0.1:8188",
    "port": 8000,
    "width": 1024,
    "height": 1024,
    "timeout_seconds": 600,
    "open_browser": True,
    "steps": 8,
    # Scene planner: a local LLM served by Ollama. Off unless enabled in config.json.
    "llm_enabled": False,
    "llm_url": "http://127.0.0.1:11434",
    "llm_model": "qwen3:8b",
    "llm_timeout": 60,
    "llm_num_gpu": 99,
    "llm_keep_alive": "2m",
}

MAX_BODY_BYTES = 1_000_000  # largest request body the server accepts

# ---------- logging: console, logs/server.log, and a live buffer shown in the page ----------

LOG_PATH = os.path.join(BASE_DIR, "logs", "server.log")
LOG_BUFFER = collections.deque(maxlen=1000)
LOG_LOCK = threading.Lock()
LOG_SEQ = itertools.count(1)


def log(level, message, **fields):
    """Records one event. Never raises: logging must not break a request."""
    try:
        entry = {"seq": next(LOG_SEQ), "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                 "level": level, "msg": message, "fields": fields}
        line = f"{entry['time']} {level:<5} {message}"
        if fields:
            line += " " + json.dumps(fields, ensure_ascii=False, default=str)
        with LOG_LOCK:
            LOG_BUFFER.append(entry)
            try:
                os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        try:
            print(line, flush=True)
        except UnicodeEncodeError:  # some consoles cannot show every character
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    except Exception:
        pass


def log_entries_after(seq):
    with LOG_LOCK:
        return [e for e in LOG_BUFFER if e["seq"] > seq]


def add_stage(timeline, stage, seconds, kind="app", model=""):
    """Adds one timed step to a generation's timeline (None means do not record).
    kind is "llm", "image" or "app"; model names the model or file that ran the step."""
    if timeline is not None:
        timeline.append({"stage": stage, "seconds": round(max(seconds, 0.0), 2), "kind": kind, "model": model})


IMAGE_STEPS = {
    "Loading diffusion model": ("load diffusion model", "diffusion"),
    "Sampling": ("sampling", "diffusion"),
    "Loading text encoder": ("load text encoder", "text_encoder"),
    "Encoding prompt": ("encode prompt", "text_encoder"),
    "Loading VAE": ("load VAE", "vae"),
    "Decoding image": ("decode image", "vae"),
    "Preparing sampler": ("prepare sampler", None),
    "Saving": ("save PNG", None),
}


def image_step(label, models):
    """Readable stage name and the model file that ran it, for one ComfyUI step."""
    name, role = IMAGE_STEPS.get(label, (label.lower(), None))
    return name, (models.get(role, "") if role else "")


def workflow_models(workflow):
    """The image model files this workflow uses, by role."""
    models = {}
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        ctype = node.get("class_type")
        if ctype == "UNETLoader":
            models["diffusion"] = str(inputs.get("unet_name", ""))
        elif ctype == "CLIPLoader":
            models["text_encoder"] = str(inputs.get("clip_name", ""))
        elif ctype == "VAELoader":
            models["vae"] = str(inputs.get("vae_name", ""))
    return models

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
    if not BOOK_ID_RE.fullmatch(book_id):
        raise ValueError(f"bad book id: {book_id!r}")
    return os.path.join(BOOKS_DIR, book_id + ".json")


class BookFileError(Exception):
    """A book file exists but cannot be read as a book."""


def load_book(book_id):
    path = book_file(book_id)
    if not os.path.exists(path):
        raise FileNotFoundError(book_id)
    try:
        with open(path, encoding="utf-8") as f:
            book = json.load(f)
    except (OSError, ValueError) as e:
        raise BookFileError(f"{book_id}.json could not be read: {e}") from e
    if not isinstance(book, dict):
        raise BookFileError(f"{book_id}.json does not contain a JSON object")
    # The file name is the book's id, so a copied file cannot take over another book.
    book["id"] = book_id
    return book


BOOK_WRITE_LOCK = threading.Lock()
SETTINGS_LOCK = threading.Lock()


def read_settings():
    """Settings saved on this machine. A missing or unreadable file means the defaults."""
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"Ignoring unreadable settings file: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def write_settings(settings):
    tmp = SETTINGS_PATH + ".tmp"
    with SETTINGS_LOCK:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        os.replace(tmp, SETTINGS_PATH)


def save_book(book):
    """Writes books/<id>.json atomically (temp file, then replace)."""
    target = book_file(book["id"])
    os.makedirs(BOOKS_DIR, exist_ok=True)
    tmp = target + ".tmp"
    with BOOK_WRITE_LOCK:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(book, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, target)


def list_books():
    books = []
    if os.path.isdir(BOOKS_DIR):
        for fn in os.listdir(BOOKS_DIR):
            stem = fn[: -len(".json")] if fn.endswith(".json") else ""
            if not BOOK_ID_RE.fullmatch(stem):
                continue
            try:
                with open(os.path.join(BOOKS_DIR, fn), encoding="utf-8") as f:
                    b = json.load(f)
                books.append({
                    "id": stem,
                    "title": str(b.get("title", stem)),
                    "author": str(b.get("author", "")),
                    "series": str(b.get("series", "")),
                    "book_number": int(b.get("book_number", 0) or 0),
                })
            except (OSError, ValueError, TypeError, AttributeError) as e:
                print(f"Skipping unreadable book file {fn}: {e}")
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
        "visual_style_short": book.get("visual_style_short", ""),
        "narrator": book.get("narrator", ""),
        "characters_text": characters_to_text(book.get("characters", [])),
    }


def update_book(book_id, fields):
    """Applies any of visual_style / characters_text that are present, saves, and returns the book."""
    book = load_book(book_id)
    if "visual_style" in fields:
        book["visual_style"] = str(fields["visual_style"])
    if "visual_style_short" in fields:
        book["visual_style_short"] = str(fields["visual_style_short"])
    if "characters_text" in fields:
        book["characters"] = text_to_characters(str(fields["characters_text"]))
    if "narrator" in fields:
        book["narrator"] = str(fields["narrator"]).strip()
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


def name_in_text(name, text):
    # Case-sensitive on purpose: "patience", "regal" or "fool" in lowercase are ordinary words.
    return re.search(r"\b" + re.escape(name) + r"\b", text) is not None


# First-person narration: "I" (capital only, so the letter does not match inside words),
# and me/my/mine/myself in any case. Quoted dialogue is removed before this is checked.
FIRST_PERSON_RE = re.compile(r"\bI\b|\b(?i:me|my|mine|myself)\b")
QUOTE_RE = re.compile(r'"[^"]*"|“[^”]*”')


def has_first_person(text):
    """True if the narration (dialogue removed) is written in the first person."""
    return FIRST_PERSON_RE.search(QUOTE_RE.sub(" ", text)) is not None


def characters_in_scene(characters_text, scene_text, narrator="", present=None):
    """With present=None: characters named in the text, plus the narrator when the text is first
    person. With present (from the planner): exactly those characters. The narrator is listed first."""
    chars = parse_characters(characters_text)
    if present is not None:
        wanted = {p.lower() for p in present}
        chosen = [c for c in chars if c[0].lower() in wanted]
        return sorted(chosen, key=lambda c: c[0] != narrator)
    narrator_entry = [c for c in chars if narrator and c[0] == narrator]
    others = [c for c in chars if not (narrator and c[0] == narrator) and name_in_text(c[0], scene_text)]
    narrator_present = bool(narrator_entry) and (has_first_person(scene_text) or name_in_text(narrator, scene_text))
    return (narrator_entry if narrator_present else []) + others


def build_prompt(style, text, characters_text="", narrator="", scene="", present=None, glossary_notes=None):
    # Subject first: who is in the picture, then the scene, then the style.
    # `text` is the pasted page and is used for name matching. `scene`, when given,
    # replaces the page text in the prompt itself. `present` (from the planner) fixes who is shown.
    matching_text = text.strip() + "\n" + scene.strip()
    scene_text = scene.strip() or text.strip()
    parts = [SINGLE_IMAGE_NOTE]
    chars = characters_in_scene(characters_text, matching_text, narrator, present)
    if chars:
        parts += ["", "Characters in this scene (keep their appearance consistent):"]
        for name, desc in chars:
            tag = " (the narrator, the viewpoint character)" if narrator and name == narrator else ""
            parts.append(f"- {name}{tag}: {desc}")
    if glossary_notes:
        parts += ["", "Visual notes:"] + [f"- {term}: {visual}" for term, visual in glossary_notes]
    parts += ["", "Scene from the book:", scene_text]
    if style.strip():
        parts += ["", f"Visual style: {style.strip()}"]
    return "\n".join(parts)


# ---------- scene planner (local LLM through Ollama) ----------
# The planner decides which candidate characters are visible in one moment and writes a short
# visual scene. It runs only when llm_enabled is true. Any problem gives None, and the app falls
# back to name matching and the pasted page as before.

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")

PLANNER_SYSTEM = (
    "You plan one illustration for a page of a novel. Choose ONE moment from the page: the most visual one. "
    'For EVERY allowed name, write a "checks" entry with "visible_in_moment": true only if that person is '
    "physically in the picture in that moment. Someone who is only remembered, mentioned, avoided, absent, "
    'dead or spoken about is false. Then write "scene": at most 50 words of concrete visible detail '
    "(poses, setting, which hand holds each prop). No thoughts or backstory. Refer to people by name, never "
    'as "I". Only the visible people appear in the scene. Copy every name exactly as written in the allowed '
    "list. A person who is elsewhere, avoids the narrator, left, or is only spoken about is "
    "visible_in_moment false, even if the page mentions them in the same passage."
)
# One synthetic example, with invented names, shown before the real request.
PLANNER_FEW_SHOT_USER = (
    'Allowed names: Mara (narrator, "I"), Orrin, Bel, Corran\n'
    "Page:\nI waited at the gate while Orrin argued with the guard. Bel had left us two days ago, "
    "and I still missed her voice. Corran had ridden off to the coast at dawn. "
    "Orrin finally stepped aside, laughing at something the guard had said."
)
PLANNER_FEW_SHOT_ASSISTANT = json.dumps({
    "checks": [
        {"name": "Mara", "visible_in_moment": True},
        {"name": "Orrin", "visible_in_moment": True},
        {"name": "Bel", "visible_in_moment": False},
        {"name": "Corran", "visible_in_moment": False},
    ],
    "scene": "Mara waits by a stone gate. Orrin, arms folded, argues with a guard in a grey tabard, "
             "then steps aside, laughing.",
})
PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "visible_in_moment": {"type": "boolean"}},
            "required": ["name", "visible_in_moment"],
        }},
        "scene": {"type": "string"},
    },
    "required": ["checks", "scene"],
}


def candidate_names(text, chars, narrator):
    """Names the planner may choose from: names found in the page, plus the narrator when the
    narration is first person."""
    names = [n for n, _ in chars if name_in_text(n, text)]
    known = [n for n, _ in chars]
    if narrator and narrator in known and narrator not in names and has_first_person(text):
        names.append(narrator)
    return names


def ollama_post(path, payload, timeout):
    req = urllib.request.Request(
        CONFIG["llm_url"].rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def llm_release():
    """Unload the planner model so its VRAM is free for ComfyUI."""
    if not CONFIG.get("llm_enabled"):
        return
    try:
        ollama_post("/api/generate", {"model": CONFIG["llm_model"], "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def free_comfy_gpu():
    """Ask ComfyUI to unload its models, so the planner has room on the GPU."""
    req = urllib.request.Request(
        comfy_url("/free"),
        data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except OSError:
        pass


def plan_scene(text, characters_text, narrator, glossary):
    """Returns {"present", "scene", "seconds", "terms"}, or None. Never raises."""
    if not CONFIG.get("llm_enabled"):
        return None
    try:
        chars = parse_characters(characters_text)
        names = candidate_names(text, chars, narrator)
        terms = [g for g in glossary if g.get("term") and name_in_text(g["term"], text)]
        if not names:
            return None  # nothing to decide, so no planner call
        system = PLANNER_SYSTEM
        if terms:
            system += "\n\nGlossary (use these meanings, do not invent others):\n" + "\n".join(
                f"- {g['term']}: {g.get('meaning', '')}" for g in terms)

        def tag(name):
            return f'{name} (narrator, "I")' if narrator and name == narrator else name

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": PLANNER_FEW_SHOT_USER},
            {"role": "assistant", "content": PLANNER_FEW_SHOT_ASSISTANT},
            {"role": "user", "content": "Allowed names: " + ", ".join(tag(n) for n in names) + "\nPage:\n" + text},
        ]
        t_free = time.perf_counter()
        if CONFIG["llm_num_gpu"] > 0:
            free_comfy_gpu()
        free_s = time.perf_counter() - t_free
        t0 = time.perf_counter()
        data = ollama_post("/api/chat", {
            "model": CONFIG["llm_model"], "stream": False, "think": False,
            "keep_alive": CONFIG["llm_keep_alive"], "messages": messages, "format": PLANNER_SCHEMA,
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 260,
                        "num_gpu": CONFIG["llm_num_gpu"]},
        }, timeout=CONFIG["llm_timeout"])
        seconds = round(time.perf_counter() - t0, 2)
        obj = json.loads(FENCE_RE.sub("", data["message"]["content"].strip()))
        checks, scene = obj.get("checks"), obj.get("scene")
        if not isinstance(checks, list) or not isinstance(scene, str) or not scene.strip():
            log("WARN", "planner reply rejected: wrong shape", reply=str(data["message"]["content"])[:300])
            return None
        known = {n.lower(): n for n in names}
        present = []
        for c in checks:
            if not (isinstance(c, dict) and isinstance(c.get("name"), str)
                    and isinstance(c.get("visible_in_moment"), bool)):
                return None
            key = c["name"].strip().lower()
            if key not in known:  # a close spelling counts, e.g. "Patiance" for "Patience"
                close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.75)
                key = close[0] if close else key
            if key in known and c["visible_in_moment"] and known[key] not in present:
                present.append(known[key])
        timings = {
            "free_comfy": round(free_s, 2),
            "load": round(data.get("load_duration", 0) / 1e9, 2),
            "prompt": round(data.get("prompt_eval_duration", 0) / 1e9, 2),
            "generate": round(data.get("eval_duration", 0) / 1e9, 2),
            "wall": seconds,
        }
        log("INFO", "planner done", present=present, terms=[g["term"] for g in terms], **timings)
        return {"present": present, "scene": " ".join(scene.split()[:60]), "seconds": seconds,
                "timings": timings, "terms": [g["term"] for g in terms]}
    except Exception as e:
        log("ERROR", "planner failed; using name matching", error=str(e), traceback=traceback.format_exc())
        return None


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
        # Time spent in each ComfyUI step, measured from the 'executing' events.
        self.stage_lock = threading.Lock()
        self.stage_label = None
        self.stage_started = 0.0
        self.stage_totals = {}

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
        self.finish_stages()

    def finish_stages(self):
        """Closes the open ComfyUI step and returns the seconds spent in each step, by label."""
        with self.stage_lock:
            self._end_stage(time.perf_counter())
            return dict(self.stage_totals)

    def _end_stage(self, now):
        if self.stage_label is not None:
            total = self.stage_totals.get(self.stage_label, 0.0)
            self.stage_totals[self.stage_label] = total + (now - self.stage_started)
            self.stage_label = None

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
        elif kind == "executing":
            now = time.perf_counter()
            with self.stage_lock:
                self._end_stage(now)  # the previous step has finished
                if data.get("node") is not None:
                    ctype = self.node_types.get(str(data["node"]), "")
                    label = STAGE_LABELS.get(ctype, "Working")
                    self.stage_label, self.stage_started = label, now
                    set_progress(label=label, value=0, max=0)

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


def generate_image(text, style, characters_text="", narrator="", scene="", present=None, glossary_notes=None,
                   timeline=None):
    """Run the workflow on ComfyUI, wait for it, save the PNG. Returns (filename, prompt, seed).
    If a timeline list is given, the time spent in each step is appended to it."""
    t = time.perf_counter()
    prompt_text = build_prompt(style, text, characters_text, narrator, scene, present, glossary_notes)
    workflow = make_workflow(prompt_text)
    add_stage(timeline, "Build workflow", time.perf_counter() - t)
    models = workflow_models(workflow)
    seed = next(
        (n["inputs"]["noise_seed"] for n in workflow.values()
         if isinstance(n, dict) and isinstance(n.get("inputs", {}).get("noise_seed"), int)),
        None,
    )
    client_id = uuid.uuid4().hex
    node_types = {nid: str(node.get("class_type", "")) for nid, node in workflow.items() if isinstance(node, dict)}

    set_progress(active=True, label="Starting", value=0, max=0)
    listener = start_progress_listener(client_id, node_types)
    try:
        t = time.perf_counter()
        queued = comfy_post_json("/prompt", {"prompt": workflow, "client_id": client_id})
        add_stage(timeline, "queue request", time.perf_counter() - t, kind="image", model="ComfyUI")
        log("INFO", "comfy queued", prompt_id=queued.get("prompt_id"), live_progress=listener is not None)
        t = time.perf_counter()
        name, prompt = _wait_for_result(queued["prompt_id"], prompt_text)
        waited = time.perf_counter() - t
        steps = listener.finish_stages() if listener else {}
        for label, secs in steps.items():
            stage, model = image_step(label, models)
            add_stage(timeline, stage, secs, kind="image", model=model)
        add_stage(timeline, "queue wait and PNG save", waited - sum(steps.values()), kind="image", model="ComfyUI")
        log("INFO", "image saved", file=name, waited=round(waited, 2),
            steps={k: round(v, 2) for k, v in steps.items()})
        return name, prompt, seed
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
    def parse_request(self):
        self._t0 = time.perf_counter()
        return super().parse_request()

    def log_request(self, code="-", size="-"):
        # The page polls these two every second; keep them out of the log.
        if self.path.startswith(("/api/progress", "/api/log")):
            return
        ms = (time.perf_counter() - getattr(self, "_t0", time.perf_counter())) * 1000
        log("INFO", f"{self.command} {self.path} {code}", ms=round(ms))

    def log_message(self, fmt, *args):
        log("WARN", fmt % args)

    def _guard(self, handler):
        """Turns any unexpected error into a logged 500, so the page never gets a dropped connection."""
        try:
            handler()
        except Exception as e:
            log("ERROR", f"{self.command} {self.path} failed", error=str(e), traceback=traceback.format_exc())
            try:
                self._json(500, {"error": f"Server error: {e}. See the debug log."})
            except Exception:
                pass

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _host_ok(self):
        # Blocks DNS rebinding: only requests addressed to this server's own port are served.
        port = self.server.server_address[1]
        return (self.headers.get("Host") or "").lower() in (f"127.0.0.1:{port}", f"localhost:{port}")

    def do_GET(self):
        self._guard(self._get)

    def _get(self):
        path = urllib.parse.urlparse(self.path).path
        if not self._host_ok():
            self._send(403, b"forbidden", "text/plain")
            return
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
            except BookFileError as e:
                self._json(500, {"error": str(e)})
        elif path == "/api/settings":
            self._json(200, read_settings())
        elif path == "/api/progress":
            self._json(200, get_progress())
        elif path == "/api/log":
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            after = int((query.get("after") or ["0"])[0] or 0)
            entries = log_entries_after(after)[-200:]
            self._json(200, {"entries": entries, "last": entries[-1]["seq"] if entries else after})
        elif path.startswith("/outputs/"):
            full = os.path.join(OUTPUT_DIR, os.path.basename(path[len("/outputs/"):]))
            if os.path.isfile(full):
                with open(full, "rb") as f:
                    self._send(200, f.read(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def _read_json_body(self):
        """Returns the request's JSON object, or None after sending an error response."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "Bad Content-Length."})
            return None
        if length < 0 or length > MAX_BODY_BYTES:
            self._json(413, {"error": "Request too large."})
            return None
        raw = self.rfile.read(length) if length else b""  # read the body before replying
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            self._json(415, {"error": "Send JSON with Content-Type: application/json."})
            return None
        try:
            body = json.loads(raw or b"{}")
        except ValueError:  # JSON syntax errors and bad UTF-8 are both ValueErrors
            self._json(400, {"error": "Invalid JSON."})
            return None
        if not isinstance(body, dict):
            self._json(400, {"error": "Expected a JSON object."})
            return None
        return body

    def do_POST(self):
        self._guard(self._post)

    def _post(self):
        path = urllib.parse.urlparse(self.path).path
        if not self._host_ok():
            self._send(403, b"forbidden", "text/plain")
            return
        body = self._read_json_body()
        if body is None:
            return

        if path.startswith("/api/books/"):
            try:
                update_book(path[len("/api/books/"):], body)
                self._json(200, {"ok": True})
            except (FileNotFoundError, ValueError):
                self._json(404, {"error": "Unknown book."})
            except BookFileError as e:
                self._json(500, {"error": str(e)})

        elif path == "/api/settings":
            book_id = str(body.get("default_book", "")).strip()
            if book_id:
                try:
                    load_book(book_id)  # the default must be a book that exists
                except (FileNotFoundError, ValueError):
                    self._json(404, {"error": "Unknown book."})
                    return
                except BookFileError as e:
                    self._json(500, {"error": str(e)})
                    return
            settings = read_settings()
            settings["default_book"] = book_id
            write_settings(settings)
            self._json(200, {"default_book": book_id})

        elif path == "/api/generate":
            started = time.perf_counter()
            text = str(body.get("text", "")).strip()
            scene = str(body.get("scene", "")).strip()
            if not text:
                self._json(400, {"error": "Paste some text first."})
                return
            timeline = []
            t = time.perf_counter()
            try:
                # Save whatever the page shows, then generate from the saved book.
                book = update_book(str(body.get("book", "")), body)
            except (FileNotFoundError, ValueError):
                self._json(400, {"error": "Pick a book first."})
                return
            except BookFileError as e:
                self._json(500, {"error": str(e)})
                return
            add_stage(timeline, "Save book", time.perf_counter() - t)
            style = book.get("visual_style_short") or book.get("visual_style", "")
            characters_text = characters_to_text(book.get("characters", []))
            narrator = book.get("narrator", "")
            glossary = book.get("glossary", [])
            log("INFO", "generate start", book=book["id"], typed_scene=bool(scene),
                characters=len(book.get("characters", [])))
            with GENERATE_LOCK:
                set_progress(active=True, label="Planning scene", value=0, max=0)
                try:
                    # A scene the user typed is used as it is; otherwise the planner may write one.
                    planned = None
                    if not scene:
                        t = time.perf_counter()
                        planned = plan_scene(text, characters_text, narrator, glossary)
                        plan_wall = time.perf_counter() - t
                        where = "GPU" if CONFIG["llm_num_gpu"] > 0 else "CPU"
                        llm = f"{CONFIG['llm_model']} ({where})"
                        if planned:
                            tm = planned["timings"]
                            add_stage(timeline, "unload ComfyUI models for the LLM", tm["free_comfy"],
                                      kind="image", model="ComfyUI")
                            add_stage(timeline, "load model", tm["load"], kind="llm", model=llm)
                            add_stage(timeline, "read page and write scene", tm["prompt"] + tm["generate"],
                                      kind="llm", model=llm)
                            add_stage(timeline, "HTTP and parsing", plan_wall - tm["wall"], kind="llm", model=llm)
                        else:
                            add_stage(timeline, "no result (name matching used)", plan_wall, kind="llm", model=llm)
                    present = None
                    if planned:
                        scene, present = planned["scene"], planned["present"]
                    scene_source = "scene" if scene and not planned else ("planner" if planned else "page")
                    planner_info = {
                        "used": bool(planned),
                        "present": planned["present"] if planned else [],
                        "seconds": planned["seconds"] if planned else None,
                        "terms": planned["terms"] if planned else [],
                    }
                    notes = [(g["term"], g.get("visual", "")) for g in glossary
                             if g.get("term") and g.get("visual") and name_in_text(g["term"], text)]
                    if CONFIG.get("llm_enabled"):
                        t = time.perf_counter()
                        llm_release()  # give the GPU back to ComfyUI before the image is made
                        add_stage(timeline, "unload model", time.perf_counter() - t,
                                  kind="llm", model=CONFIG["llm_model"])
                    name, prompt, seed = generate_image(
                        text, style, characters_text, narrator, scene, present, notes, timeline)
                except urllib.error.HTTPError as e:
                    log("ERROR", "ComfyUI returned an HTTP error", code=e.code, url=e.url)
                    self._json(500, {"error": f"ComfyUI returned HTTP {e.code} for {e.url}"})
                    return
                except urllib.error.URLError as e:
                    log("ERROR", "cannot reach ComfyUI", error=str(e.reason), url=CONFIG["comfy_url"])
                    self._json(503, {"error": f"Can't reach ComfyUI at {CONFIG['comfy_url']}. Start ComfyUI first."})
                    return
                except Exception as e:
                    log("ERROR", "generation failed", error=str(e), traceback=traceback.format_exc())
                    self._json(500, {"error": str(e)})
                    return
                finally:
                    set_progress(active=False)
            total = time.perf_counter() - started
            log("INFO", "generate done", image=name, total=round(total, 2), stages=len(timeline))
            self._json(200, {
                "image_url": f"/outputs/{name}",
                "prompt": prompt,
                "seed": seed,
                "scene_source": scene_source,
                "planner": planner_info,
                "timeline": timeline,
                "total_seconds": round(total, 2),
            })

        else:
            self._send(404, b"not found", "text/plain")


class LoggingServer(ThreadingHTTPServer):
    # On Windows, address reuse would let a second server share the port without any error.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request, client_address):
        log("ERROR", "connection error", client=str(client_address), traceback=traceback.format_exc())


def main():
    load_config()
    url = f"http://127.0.0.1:{CONFIG['port']}"
    server = LoggingServer(("127.0.0.1", CONFIG["port"]), Handler)
    log("INFO", "server started", url=url, comfy=CONFIG["comfy_url"], planner=bool(CONFIG.get("llm_enabled")))
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
