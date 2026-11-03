"""ComfyUI integration: workflow preparation, WebSocket progress listening, and image generation."""

import base64
import contextlib
import json
import os
import random
import socket
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime

import config
from config import add_stage, log

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


def is_text_prompt_node(node):
    class_type = str(node.get("class_type", ""))
    inputs = node.get("inputs", {})
    return ("TextEncode" in class_type or "CLIPText" in class_type) and isinstance(inputs.get("text"), str)


def make_workflow(prompt_text, seed=None):
    """Load the exported ComfyUI workflow (API format) and fill in prompt, seed and size."""
    if not os.path.exists(config.WORKFLOW_PATH):
        raise RuntimeError(
            "workflow_api.json not found. Export your ComfyUI workflow in API format "
            "and save it in this folder (see README.md)."
        )
    with open(config.WORKFLOW_PATH, encoding="utf-8") as f:
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
                inputs[key] = seed if seed is not None else random.randint(0, 2**32 - 1)

        # Size lives on the latent node and on the scheduler (Flux2Scheduler uses it for sigmas).
        if isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int):
            inputs["width"] = int(config.CONFIG["width"])
            inputs["height"] = int(config.CONFIG["height"])
        if "Scheduler" in str(node.get("class_type", "")) and isinstance(inputs.get("steps"), int):
            inputs["steps"] = int(config.CONFIG["steps"])

    if prompt_nodes == 0:
        raise RuntimeError(
            "No text prompt node found in workflow_api.json (looked for CLIPTextEncode / TextEncode "
            "nodes with a 'text' input)."
        )
    return workflow


# ---------- ComfyUI HTTP API ----------


def comfy_url(path):
    return config.CONFIG["comfy_url"].rstrip("/") + path


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
        raise RuntimeError(f"ComfyUI rejected the workflow (HTTP {e.code}): {detail[:1500]}") from e


def comfy_available():
    try:
        comfy_get("/system_stats", timeout=5)
        return True
    except Exception:
        return False


def free_comfy():
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
        parsed = urllib.parse.urlparse(config.CONFIG["comfy_url"])
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.sendall(
            (
                f"GET /ws?clientId={self.client_id} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
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
                    with contextlib.suppress(Exception):
                        self._handle(json.loads(payload))
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
            with contextlib.suppress(OSError):
                self.sock.close()

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
            except TimeoutError:
                if self.stop_event.is_set():
                    raise ConnectionError("listener stopped") from None
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


def _wait_for_result(prompt_id, prompt_text):
    deadline = time.time() + config.CONFIG["timeout_seconds"]
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
                    data = comfy_get(
                        "/view?"
                        + urllib.parse.urlencode(
                            {
                                "filename": image["filename"],
                                "subfolder": image.get("subfolder", ""),
                                "type": "output",
                            }
                        )
                    )
                    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                    ext = os.path.splitext(image["filename"])[1] or ".png"
                    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6] + ext
                    with open(os.path.join(config.OUTPUT_DIR, name), "wb") as f:
                        f.write(data)
                    return name, prompt_text

            if status.get("completed"):
                raise RuntimeError(
                    "ComfyUI finished but produced no image. Make sure the workflow ends in a Save Image node."
                )
        time.sleep(1)
    raise RuntimeError("Timed out waiting for ComfyUI.")


def generate_image(prompt_text, timeline=None, seed=None):
    """Run the workflow on ComfyUI with this prompt, wait for it, save the PNG. Returns (filename, prompt, seed).
    If a timeline list is given, the time spent in each step is appended to it. A fixed seed makes a run
    repeatable."""
    t = time.perf_counter()
    workflow = make_workflow(prompt_text, seed)
    add_stage(timeline, "Build workflow", time.perf_counter() - t)
    models = workflow_models(workflow)
    seed = next(
        (
            n["inputs"]["noise_seed"]
            for n in workflow.values()
            if isinstance(n, dict) and isinstance(n.get("inputs", {}).get("noise_seed"), int)
        ),
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
        log("INFO", "image saved", file=name, waited=round(waited, 2), steps={k: round(v, 2) for k, v in steps.items()})
        return name, prompt, seed
    finally:
        if listener:
            listener.stop()
        set_progress(active=False)
