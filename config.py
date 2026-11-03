"""Configuration, paths, logging, and system metrics for Book Illustrator."""

import collections
import itertools
import json
import os
import subprocess
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
WORKFLOW_PATH = os.path.join(BASE_DIR, "workflow_api.json")
BOOKS_DIR = os.path.join(BASE_DIR, "books")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")  # per-machine settings, e.g. the default book
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
LOG_PATH = os.path.join(BASE_DIR, "logs", "server.log")

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
    "llm_timeout": 180,  # seconds; a long page with a big cast can take a while, and calls queue in Ollama
    "llm_num_gpu": 99,
    "llm_keep_alive": "2m",
    "planner_validate": True,  # a second LLM call reviews and fixes the planner's scene (see review_scene)
}

MAX_BODY_BYTES = 1_000_000  # largest request body the server accepts

# ---------- logging: console, logs/server.log, and a live buffer shown in the page ----------

LOG_BUFFER = collections.deque(maxlen=1000)
LOG_LOCK = threading.Lock()
LOG_SEQ = itertools.count(1)


def log(level, message, **fields):
    """Records one event. Never raises: logging must not break a request."""
    try:
        entry = {
            "seq": next(LOG_SEQ),
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "level": level,
            "msg": message,
            "fields": fields,
        }
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


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            CONFIG.update(json.load(f))
    # Overrides for test instances (e.g. the golden-set runs) that must not touch the main server.
    if os.environ.get("BOOK_ILLUSTRATOR_PORT"):
        CONFIG["port"] = int(os.environ["BOOK_ILLUSTRATOR_PORT"])
    if os.environ.get("BOOK_ILLUSTRATOR_NO_BROWSER"):
        CONFIG["open_browser"] = False


def _number(text):
    try:
        return float(text)
    except ValueError:
        return None


def gpu_snapshot():
    """GPU clock, temperature, memory, power and load from nvidia-smi, or None when it is not available.
    Used to see whether a slow image came from the GPU running slow (heat, power, other apps)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,memory.used,memory.total,"
                "power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    values = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    keys = ("clock_mhz", "clock_max_mhz", "temp_c", "vram_used_mb", "vram_total_mb", "power_w", "gpu_util_pct")
    return {key: _number(value) for key, value in zip(keys, values, strict=False)}
