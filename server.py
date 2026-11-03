"""Book Illustrator: paste a page of text, get an illustration from a local ComfyUI.

Standard library only. ComfyUI must be running separately (default
http://127.0.0.1:8188) and workflow_api.json must exist next to this file.
See README.md for the one-time setup.
"""

import base64
import collections
import contextlib
import hashlib
import itertools
import json
import os
import random
import re
import socket
import struct
import subprocess
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

from bible import facts, lint_book, normalise_book
from prompting import PROMPT_VERSION, build_prompt

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
    "llm_timeout": 180,  # seconds; a long page with a big cast can take a while, and calls queue in Ollama
    "llm_num_gpu": 99,
    "llm_keep_alive": "2m",
    "planner_validate": True,  # a second LLM call reviews and fixes the planner's scene (see review_scene)
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
    # Overrides for test instances (e.g. the golden-set runs) that must not touch the main server.
    if os.environ.get("BOOK_ILLUSTRATOR_PORT"):
        CONFIG["port"] = int(os.environ["BOOK_ILLUSTRATOR_PORT"])
    if os.environ.get("BOOK_ILLUSTRATOR_NO_BROWSER"):
        CONFIG["open_browser"] = False


# ---------- visual identity (style box) ----------

# ---------- books: each book file holds its own visual style and characters ----------
# books/<id>.json looks like (schema 2; see bible.py for every character field):
#   {"id", "schema_version": 2, "title", "author", "series", "book_number", "visual_style", "narrator",
#    "characters": [{"name", "aliases", "kind", "sex", "age", "species", "look"}], "glossary": [...]}
# visual_style is the style the picture prompt uses; any other keys (notes, longer drafts) are kept as they are.

BOOK_ID_RE = re.compile(r"^[a-z0-9-]+$")


def book_file(book_id):
    if not BOOK_ID_RE.fullmatch(book_id):
        raise ValueError(f"bad book id: {book_id!r}")
    return os.path.join(BOOKS_DIR, book_id + ".json")


class BookFileError(Exception):
    """A book file exists but cannot be read as a book."""


def load_book(book_id):
    """The book, normalised to the current shape (see bible.normalise_book). Everything downstream relies on it."""
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
    return normalise_book(book)


BOOK_WRITE_LOCK = threading.Lock()
SETTINGS_LOCK = threading.Lock()


def write_json_atomic(path, obj, lock=None):
    """Writes JSON to a temp file, then replaces the target, so a crash never leaves half a file."""
    tmp = path + ".tmp"
    with lock or threading.Lock():
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)


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
    write_json_atomic(SETTINGS_PATH, settings, SETTINGS_LOCK)


def save_book(book):
    """Writes books/<id>.json atomically."""
    os.makedirs(BOOKS_DIR, exist_ok=True)
    write_json_atomic(book_file(book["id"]), book, BOOK_WRITE_LOCK)


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
                books.append(
                    {
                        "id": stem,
                        "title": str(b.get("title", stem)),
                        "author": str(b.get("author", "")),
                        "series": str(b.get("series", "")),
                        "book_number": int(b.get("book_number", 0) or 0),
                    }
                )
            except (OSError, ValueError, TypeError, AttributeError) as e:
                print(f"Skipping unreadable book file {fn}: {e}")
    return sorted(books, key=lambda b: (b["series"], b["book_number"], b["title"]))


def book_view(book):
    """What the page needs: the book's fields, characters, glossary and its structural problems."""
    return {
        "id": book["id"],
        "title": book.get("title", book["id"]),
        "author": book.get("author", ""),
        "series": book.get("series", ""),
        "book_number": book.get("book_number", 0),
        "visual_style": book.get("visual_style", ""),
        "narrator": book.get("narrator", ""),
        "characters": book["characters"],
        "glossary": book["glossary"],
        "problems": lint_book(book),
    }


def update_book(book_id, fields):
    """Applies whichever of visual_style, narrator, characters (a list) and glossary (a list) are present,
    saves, and returns the book."""
    book = load_book(book_id)
    if "visual_style" in fields:
        book["visual_style"] = str(fields["visual_style"])
    if isinstance(fields.get("characters"), list):
        book["characters"] = fields["characters"]
    if isinstance(fields.get("glossary"), list):
        book["glossary"] = fields["glossary"]
    if "narrator" in fields:
        book["narrator"] = str(fields["narrator"]).strip()
    book = normalise_book(book)
    save_book(book)
    return book


# ---------- scene planner (local LLM through Ollama) ----------
# The planner reads the page with the book's whole cast and glossary, and decides who is visible in one
# moment, which glossary terms it shows, where it happens, and writes a short visual scene. A second short
# call (the reviewer) checks that scene and fixes it. All content judgements are made by the LLM: there are
# no word lists or text patterns in this code. Every example uses invented names. It runs only when
# llm_enabled is true; any failure gives None, and the app uses the pasted page with no character list.

PLANNER_SYSTEM = (
    "You plan one illustration for a page of a novel. Choose ONE moment from the page: the most visual one. "
    "You get the book's cast. For every cast member the page refers to in any way (by name, by another name, "
    'by "he" or "she", or as the narrator "I"), write a "checks" entry with "visible_in_moment": true only if '
    "that person or animal is physically in the picture in that moment. Someone who is only remembered, "
    "mentioned, avoided, absent, dead or spoken about is false. Leave out cast members the page does not refer "
    'to. Copy every name exactly as written in the cast. "terms": the glossary terms whose thing is visible in '
    'the moment. "setting": "place" (a short phrase the page gives, or "unknown"), "indoors" ("yes", "no", or '
    '"unknown" when the page does not say) and "time" (for example "dusk", or "unknown"); do not guess a place '
    'the page does not give. "scene": at most 50 words of concrete visible detail: poses, setting, which hand '
    'holds each prop. Refer to people by name, never as "I". Only the visible cast appears in the scene and no '
    "one else. When two people are visible, write one sentence per person and name who does what to whom; do "
    "not use a pronoun that could mean either of them. No thoughts or backstory."
)

# Two synthetic examples with invented names, shown before the real request.
PLANNER_FEW_SHOT = [
    (
        'Cast:\n- Lira (narrator, "I"; female, young adult)\n- Orrin (male, adult)\n- Bel (female, adult)\n'
        "- Corran (male, adult)\n- Tess (female, child)\nGlossary terms: none\n"
        "Page:\nI waited at the gate while Orrin argued with the guard. Bel had left us two days ago, "
        "and I still missed her voice. Corran had ridden off to the coast at dawn. "
        "Orrin finally stepped aside, laughing at something the guard had said.",
        {
            "checks": [
                {"name": "Lira", "visible_in_moment": True},
                {"name": "Orrin", "visible_in_moment": True},
                {"name": "Bel", "visible_in_moment": False},
                {"name": "Corran", "visible_in_moment": False},
            ],
            "terms": [],
            "setting": {"place": "a stone gate by a road", "indoors": "no", "time": "unknown"},
            "scene": "Lira waits by a stone gate. Orrin, arms folded, stands beside Lira, laughing.",
        },
    ),
    (
        'Cast:\n- Ren (narrator, "I"; male, teen)\n- Ama (female, adult)\n- Koll (wolf)\nGlossary terms: none\n'
        "Page:\nI lay by the fire, too weak to sit up. Ama knelt beside me and held the water skin to my lips.",
        {
            "checks": [
                {"name": "Ren", "visible_in_moment": True},
                {"name": "Ama", "visible_in_moment": True},
            ],
            "terms": [],
            "setting": {"place": "beside a campfire", "indoors": "unknown", "time": "unknown"},
            "scene": "Ren lies on a blanket beside a fire, head on a rolled cloak. Ama kneels beside Ren and "
            "holds a water skin to Ren's lips.",
        },
    ),
]

SETTING_SCHEMA = {
    "type": "object",
    "properties": {
        "place": {"type": "string"},
        "indoors": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "time": {"type": "string"},
    },
    "required": ["place", "indoors", "time"],
}

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "visible_in_moment": {"type": "boolean"}},
                "required": ["name", "visible_in_moment"],
            },
        },
        "terms": {"type": "array", "items": {"type": "string"}},
        "setting": SETTING_SCHEMA,
        "scene": {"type": "string"},
    },
    "required": ["checks", "terms", "setting", "scene"],
}

REVIEW_SYSTEM = (
    "You check a short scene description that will be drawn as one picture. Rules: (1) every visible person "
    "is named and does something; (2) no other person, figure or crowd appears; (3) no pronoun could mean "
    "either of two visible people: write the name instead; (4) nothing contradicts the setting (no indoor "
    "objects outdoors, no sky or landscape indoors); (5) each person has one pose; (6) at most 60 words, "
    'complete sentences. Answer "ok": true if all rules hold. Otherwise list the "problems" and give a fixed '
    '"scene" that keeps the same moment and follows every rule. If the scene is already fine, return an '
    'empty "scene".'
)
REVIEW_FEW_SHOT = [
    (
        "Visible: Ren, Ama\nSetting: beside a campfire\n"
        "Scene: Ren lies by the fire. Ama kneels and holds the water skin to his lips while a stranger watches.",
        {
            "ok": False,
            "problems": ["'his' could mean Ren or Ama", "a stranger is not visible in this moment"],
            "scene": "Ren lies by the fire. Ama kneels and holds the water skin to Ren's lips.",
        },
    ),
]
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
    },
    "required": ["ok", "problems", "scene"],
}

BOOK_REVIEW_SYSTEM = (
    "You review the setup of a book for an illustration app. The JSON gives the visual style, the cast and the "
    "glossary. Find problems that would make pictures wrong: a look that contradicts the character's sex, age "
    "or kind; a character described only by reference to another character; a style text that names places, "
    "objects, light sources or scene content (it pulls every picture toward them) or words that invite "
    "lettering such as cover, poster or title; a glossary visual note that describes a named character; "
    "missing details the pictures need. Report only real problems, each with the field it is about."
)
BOOK_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["error", "warning"]},
                    "field": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["level", "field", "message"],
            },
        }
    },
    "required": ["problems"],
}


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
    with contextlib.suppress(Exception):
        ollama_post("/api/generate", {"model": CONFIG["llm_model"], "keep_alive": 0}, timeout=30)


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


def setting_line(setting):
    """'open road, outdoors, daytime' from the planner's setting object. Unknown parts are left out."""
    if not isinstance(setting, dict):
        return ""
    parts = [str(setting.get("place", "")).strip()]
    if setting.get("indoors") == "yes":
        parts.append("indoors")
    elif setting.get("indoors") == "no":
        parts.append("outdoors")
    parts.append(str(setting.get("time", "")).strip())
    return ", ".join(p for p in parts if p and p.lower() != "unknown")


class PlannerError(Exception):
    """The planner LLM gave no usable answer."""


def context_size(prompt_chars, num_predict):
    """A context window that holds the request (about 3 characters per token) and the answer."""
    need = prompt_chars // 3 + num_predict + 256
    return next((size for size in (4096, 8192, 16384) if need <= size), 32768)


def llm_json(system, examples, user, schema, num_predict):
    """One chat call that must answer with JSON matching `schema`. Examples are (user, answer) pairs.
    The context window is sized from the request. An answer cut off by the token budget (unparseable JSON)
    is retried once with double the budget. Returns (raw Ollama reply, parsed object, seconds)."""
    messages = [{"role": "system", "content": system}]
    for example_user, example_answer in examples:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": json.dumps(example_answer)})
    messages.append({"role": "user", "content": user})
    prompt_chars = sum(len(m["content"]) for m in messages) + len(json.dumps(schema))
    t0 = time.perf_counter()
    for attempt in (1, 2):
        data = ollama_post(
            "/api/chat",
            {
                "model": CONFIG["llm_model"],
                "stream": False,
                "think": False,
                "keep_alive": CONFIG["llm_keep_alive"],
                "messages": messages,
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_ctx": context_size(prompt_chars, num_predict),
                    "num_predict": num_predict,
                    "num_gpu": CONFIG["llm_num_gpu"],
                },
            },
            timeout=CONFIG["llm_timeout"],
        )
        try:
            obj = json.loads(data["message"]["content"])
            break
        except ValueError as e:
            if attempt == 2:
                raise PlannerError(
                    f"the model's answer was not complete JSON even with {num_predict} tokens "
                    f"(done_reason: {data.get('done_reason')}): {e}"
                ) from e
            log(
                "WARN",
                "LLM answer cut off; retrying with a larger budget",
                num_predict=num_predict,
                done_reason=data.get("done_reason"),
                error=str(e),
            )
            num_predict *= 2
    seconds = round(time.perf_counter() - t0, 2)
    return data, obj, seconds


def planner_schema(names, terms):
    """PLANNER_SCHEMA with the book's own names and glossary terms as the only allowed values, so the
    model cannot invent or misspell a name."""
    schema = json.loads(json.dumps(PLANNER_SCHEMA))
    if names:
        schema["properties"]["checks"]["items"]["properties"]["name"]["enum"] = names
    if terms:
        schema["properties"]["terms"]["items"]["enum"] = terms
    return schema


def cast_text(characters, narrator, glossary):
    """The cast and glossary terms as the planner reads them, from the book's own fields (bible.facts)."""
    lines = ["Cast:"]
    for c in characters:
        tags = ['narrator, "I"'] if narrator and c["name"] == narrator else []
        tags += [t for t in (facts(c), "also called " + ", ".join(c["aliases"]) if c["aliases"] else "") if t]
        lines.append(f"- {c['name']} ({'; '.join(tags)})" if tags else f"- {c['name']}")
    if len(lines) == 1:
        lines.append("- none")
    terms = [g["term"] for g in glossary]
    lines.append("Glossary terms: " + (", ".join(terms) if terms else "none"))
    return "\n".join(lines)


def review_scene(scene, present, setting):
    """The reviewer call. Returns (ok, problems, fixed scene or "", seconds)."""
    user = (
        f"Visible: {', '.join(present) if present else 'nobody'}\n"
        f"Setting: {setting_line(setting) or 'unknown'}\nScene: {scene}"
    )
    _, obj, seconds = llm_json(REVIEW_SYSTEM, REVIEW_FEW_SHOT, user, REVIEW_SCHEMA, num_predict=600)
    problems = [str(p) for p in obj.get("problems", []) if str(p).strip()]
    return bool(obj.get("ok")), problems, str(obj.get("scene", "")).strip(), seconds


def plan_scene(text, characters, narrator, glossary):
    """Returns {"present", "scene", "setting", "terms", "seconds", "timings", "review"}, or None. Never raises.
    `characters` and `glossary` come from load_book (already normalised). With planner_validate on, the
    reviewer checks the scene and its fixed scene is used; with no usable fix the planner's scene is kept.
    Returns None when the planner is off. Raises PlannerError when the LLM gives no usable answer: the raw
    page is never drawn in its place."""
    if not CONFIG.get("llm_enabled"):
        return None
    try:
        alias_of = {}
        for c in characters:
            for label in [c["name"]] + c["aliases"]:
                alias_of[label] = c["name"]
        term_names = [g["term"] for g in glossary]
        system = PLANNER_SYSTEM
        meanings = [g for g in glossary if g.get("meaning")]
        if meanings:
            system += "\n\nGlossary (use these meanings, do not invent others):\n" + "\n".join(
                f"- {g['term']}: {g['meaning']}" for g in meanings
            )
        t_free = time.perf_counter()
        if CONFIG["llm_num_gpu"] > 0:
            free_comfy_gpu()
        free_s = time.perf_counter() - t_free
        data, obj, seconds = llm_json(
            system,
            PLANNER_FEW_SHOT,
            cast_text(characters, narrator, glossary) + "\nPage:\n" + text,
            planner_schema(list(alias_of), term_names),
            num_predict=400 + 30 * len(characters),  # a check per cast member, plus setting and scene
        )
        checks, scene = obj.get("checks"), obj.get("scene")
        if (
            not isinstance(checks, list)
            or not isinstance(scene, str)
            or not scene.strip()
            or not all(isinstance(c, dict) and isinstance(c.get("visible_in_moment"), bool) for c in checks)
        ):
            raise PlannerError("the planner's answer had the wrong shape")
        present = []
        for c in checks:
            name = alias_of.get(c.get("name"))
            if c["visible_in_moment"] and name and name not in present:
                present.append(name)
        terms = [t for t in term_names if t in (obj.get("terms") or [])]
        setting = obj.get("setting") if isinstance(obj.get("setting"), dict) else {}
        scene = scene.strip()
        review = {"used": False}
        review_s = 0.0
        if CONFIG.get("planner_validate"):
            t_review = time.perf_counter()
            try:
                ok, problems, fixed, review_s = review_scene(scene, present, setting)
            except Exception as e:  # a failed review keeps the planner's scene rather than losing the plan
                review_s = round(time.perf_counter() - t_review, 2)
                log("WARN", "scene review failed; keeping the planner's scene", error=str(e))
                ok, problems, fixed = True, [], ""
            review = {"used": True, "ok": ok, "problems": problems}
            if not ok and fixed:
                scene = fixed  # with no usable fix, the planner's own scene is kept
            log("INFO", "scene review", ok=ok, problems=problems, fixed=bool(fixed), seconds=review_s)
        timings = {
            "free_comfy": round(free_s, 2),
            "load": round(data.get("load_duration", 0) / 1e9, 2),
            "prompt": round(data.get("prompt_eval_duration", 0) / 1e9, 2),
            "generate": round(data.get("eval_duration", 0) / 1e9, 2),
            "wall": seconds,
            "review": review_s,
        }
        log("INFO", "planner done", present=present, terms=terms, setting=setting_line(setting), **timings)
        return {
            "present": present,
            "scene": " ".join(scene.split()[:60]),
            "setting": setting_line(setting),
            "terms": terms,
            "seconds": round(seconds + review_s, 2),
            "timings": timings,
            "review": review,
        }
    except PlannerError as e:
        log("ERROR", "planner failed", error=str(e))
        raise
    except Exception as e:
        log("ERROR", "planner failed", error=str(e), traceback=traceback.format_exc())
        raise PlannerError(str(e)) from e


def review_book(book):
    """Asks the LLM to read the book's setup and list content problems. Returns [{level, message}]."""
    payload = {
        "visual_style": book.get("visual_style", ""),
        "narrator": book.get("narrator", ""),
        "characters": book["characters"],
        "glossary": book["glossary"],
    }
    _, obj, _ = llm_json(
        BOOK_REVIEW_SYSTEM, [], json.dumps(payload, ensure_ascii=False), BOOK_REVIEW_SCHEMA, num_predict=900
    )
    out = []
    for p in obj.get("problems", []):
        if isinstance(p, dict) and str(p.get("message", "")).strip():
            level = p.get("level") if p.get("level") in ("error", "warning") else "warning"
            field = str(p.get("field", "")).strip()
            out.append(
                {
                    "level": level,
                    "message": (f"{field}: " if field else "") + str(p["message"]).strip(),
                    "source": "review",
                }
            )
    return out


# ---------- workflow handling ----------


def is_text_prompt_node(node):
    class_type = str(node.get("class_type", ""))
    inputs = node.get("inputs", {})
    return ("TextEncode" in class_type or "CLIPText" in class_type) and isinstance(inputs.get("text"), str)


def make_workflow(prompt_text, seed=None):
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
                inputs[key] = seed if seed is not None else random.randint(0, 2**32 - 1)

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
        raise RuntimeError(f"ComfyUI rejected the workflow (HTTP {e.code}): {detail[:1500]}") from e


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


EMPTY_PLAN = {"scene": "", "present": [], "setting": "", "terms": []}


def plan_from_request(plan):
    """A plan in the shape plan_scene returns ({scene, present, setting, terms}), checked. None if not a dict.
    Used for eval variants and for a re-roll's saved plan. An empty scene means the page text is drawn."""
    if not isinstance(plan, dict):
        return None
    present = plan.get("present")
    terms = plan.get("terms")
    return {
        "scene": str(plan.get("scene") or "").strip(),
        "present": [str(p) for p in present] if isinstance(present, list) else [],
        "setting": str(plan.get("setting") or ""),
        "terms": [str(t) for t in terms] if isinstance(terms, list) else [],
    }


def sidecar_path(image_name):
    """outputs/<image>.json for an image this app saved, or None for any other name."""
    name = os.path.basename(str(image_name or ""))
    if name != image_name or not name.endswith(".png"):
        return None
    return os.path.join(OUTPUT_DIR, name[: -len(".png")] + ".json")


def read_saved_plan(image_name):
    """The plan saved with an earlier image (its sidecar), so a re-roll draws the same moment."""
    path = sidecar_path(image_name)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return plan_from_request(json.load(f).get("plan"))
    except (OSError, ValueError, AttributeError):
        return None


def write_sidecar(image_name, record):
    """Writes outputs/<image>.json next to the picture: what made it, so any image can be traced or re-made."""
    write_json_atomic(sidecar_path(image_name), record)


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


class NeedsSceneError(Exception):
    """The planner is off and no scene was typed, so there is nothing sensible to draw."""


def decide_scene(body, text, book, characters, timeline):
    """What to draw: (plan, source, planner info). Sources, in order: an eval variant, the saved plan of an
    earlier image (a re-roll), or the planner; a scene typed by the user replaces only the planner's scene.
    The raw page is never drawn: with no typed scene, a planner failure raises PlannerError and a planner that
    is off raises NeedsSceneError."""
    variant = body.get("variant")
    if isinstance(variant, dict) and ("scene" in variant or "present" in variant):
        return plan_from_request(variant), "variant", {"used": False}
    if body.get("from_image"):
        saved = read_saved_plan(body["from_image"])
        if saved:
            return saved, "saved", {"used": False}
    typed = str(body.get("scene", "")).strip()
    llm = f"{CONFIG['llm_model']} ({'GPU' if CONFIG['llm_num_gpu'] > 0 else 'CPU'})"
    t = time.perf_counter()
    try:
        planned = plan_scene(text, characters, book.get("narrator", ""), book["glossary"])
    except PlannerError:
        add_stage(timeline, "planner failed", time.perf_counter() - t, kind="llm", model=llm)
        if not typed:
            raise
        planned = None  # the typed scene is drawn as typed, with no character list
    wall = time.perf_counter() - t
    if planned:
        tm = planned["timings"]
        add_stage(timeline, "unload ComfyUI models for the LLM", tm["free_comfy"], kind="image", model="ComfyUI")
        add_stage(timeline, "load model", tm["load"], kind="llm", model=llm)
        add_stage(timeline, "read page and write scene", tm["prompt"] + tm["generate"], kind="llm", model=llm)
        if tm["review"]:
            add_stage(timeline, "review scene", tm["review"], kind="llm", model=llm)
        add_stage(timeline, "HTTP and parsing", wall - tm["wall"] - tm["review"], kind="llm", model=llm)
    elif not typed:
        raise NeedsSceneError("The scene planner is off (llm_enabled in config.json), so type a scene to draw.")
    info = {
        "used": bool(planned),
        "called": CONFIG.get("llm_enabled", False),
        "seconds": planned["seconds"] if planned else None,
        "review": planned["review"] if planned else {"used": False},
    }
    plan = plan_from_request(planned) if planned else dict(EMPTY_PLAN)
    if typed:
        plan["scene"] = typed
        return plan, "typed", info
    return plan, "planner", info


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
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    ext = os.path.splitext(image["filename"])[1] or ".png"
                    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6] + ext
                    with open(os.path.join(OUTPUT_DIR, name), "wb") as f:
                        f.write(data)
                    return name, prompt_text

            if status.get("completed"):
                raise RuntimeError(
                    "ComfyUI finished but produced no image. Make sure the workflow ends in a Save Image node."
                )
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
            with contextlib.suppress(Exception):
                self._json(500, {"error": f"Server error: {e}. See the debug log."})

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _book_or_reply(self, book_id, missing=(404, "Unknown book.")):
        """The book, or None after replying with the error (missing or bad id, or an unreadable file)."""
        try:
            return load_book(book_id)
        except (FileNotFoundError, ValueError):
            self._json(missing[0], {"error": missing[1]})
        except BookFileError as e:
            self._json(500, {"error": str(e)})
        return None

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
            self._json(
                200,
                {
                    "comfy_ok": comfy_available(),
                    "workflow_ok": os.path.exists(WORKFLOW_PATH),
                    "comfy_url": CONFIG["comfy_url"],
                },
            )
        elif path == "/api/books":
            self._json(200, list_books())
        elif path.startswith("/api/books/"):
            book = self._book_or_reply(path[len("/api/books/") :])
            if book is not None:
                self._json(200, book_view(book))
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
            full = os.path.join(OUTPUT_DIR, os.path.basename(path[len("/outputs/") :]))
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

        if path.startswith("/api/books/") and path.endswith("/review"):
            book = self._book_or_reply(path[len("/api/books/") : -len("/review")])
            if book is None:
                return
            if not CONFIG.get("llm_enabled"):
                self._json(400, {"error": "The book review needs the planner LLM (llm_enabled in config.json)."})
                return
            try:
                problems = lint_book(book) + review_book(book)
            except Exception as e:
                log("ERROR", "book review failed", error=str(e), traceback=traceback.format_exc())
                self._json(502, {"error": f"The book review failed: {e}"})
                return
            self._json(200, {"problems": problems})

        elif path.startswith("/api/books/"):
            if self._book_or_reply(path[len("/api/books/") :]) is not None:
                book = update_book(path[len("/api/books/") :], body)
                self._json(200, {"ok": True, "problems": lint_book(book)})

        elif path == "/api/settings":
            book_id = str(body.get("default_book", "")).strip()
            if book_id and self._book_or_reply(book_id) is None:  # the default must be a book that exists
                return
            settings = read_settings()
            settings["default_book"] = book_id
            write_settings(settings)
            self._json(200, {"default_book": book_id})

        elif path == "/api/generate":
            started = time.perf_counter()
            text = str(body.get("text", "")).strip()
            # `seed` fixes the picture's noise, so a run is repeatable. `from_image` re-draws an earlier
            # image's moment (its saved plan) with a new seed. `variant` is for evals: it supplies the plan
            # itself and may replace the cast for this one request; nothing in it is saved to the book.
            seed_req = body.get("seed")
            seed_req = seed_req if isinstance(seed_req, int) and not isinstance(seed_req, bool) else None
            variant = body.get("variant") if isinstance(body.get("variant"), dict) else None
            if not text:
                self._json(400, {"error": "Paste some text first."})
                return
            book = self._book_or_reply(str(body.get("book", "")), missing=(400, "Pick a book first."))
            if book is None:
                return
            characters = book["characters"]
            if variant and isinstance(variant.get("characters"), list):
                characters = normalise_book({"characters": variant["characters"]})["characters"]
            narrator = book.get("narrator", "")
            timeline = []
            log(
                "INFO",
                "generate start",
                book=book["id"],
                typed_scene=bool(body.get("scene")),
                rerun=bool(body.get("from_image")),
                characters=len(characters),
            )
            with GENERATE_LOCK:
                set_progress(active=True, label="Planning scene", value=0, max=0)
                try:
                    plan, source, planner_info = decide_scene(body, text, book, characters, timeline)
                    notes = [
                        (g["term"], g["visual"]) for g in book["glossary"] if g["term"] in plan["terms"] and g["visual"]
                    ]
                    if planner_info.get("called"):
                        t = time.perf_counter()
                        llm_release()  # give the GPU back to ComfyUI before the image is made
                        add_stage(
                            timeline, "unload model", time.perf_counter() - t, kind="llm", model=CONFIG["llm_model"]
                        )
                    prompt_text = build_prompt(
                        book.get("visual_style", ""),
                        text,
                        characters,
                        narrator,
                        plan["scene"],
                        plan["present"],
                        notes,
                        plan["setting"],
                    )
                    gpu_before = gpu_snapshot()
                    name, prompt, seed = generate_image(prompt_text, timeline, seed_req)
                    gpu_after = gpu_snapshot()
                except NeedsSceneError as e:
                    self._json(400, {"error": str(e)})
                    return
                except PlannerError as e:
                    self._json(502, {"error": f"The scene planner failed ({e}). Try again, or type a scene."})
                    return
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
            planner_info.pop("called", None)
            record = {
                "image": name,
                "created": datetime.now().isoformat(timespec="seconds"),
                "book": book["id"],
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "prompt_version": PROMPT_VERSION,
                "seed": seed,
                "prompt": prompt,
                "plan": dict(plan, source=source),
                "planner": planner_info,
                "timeline": timeline,
                "gpu": {"before": gpu_before, "after": gpu_after},
                "total_seconds": round(total, 2),
                "variant": variant is not None,
            }
            try:
                write_sidecar(name, record)
            except OSError as e:
                log("WARN", "could not write the image's sidecar file", error=str(e))
            log("INFO", "generate done", image=name, total=round(total, 2), stages=len(timeline))
            self._json(200, dict(record, image_url=f"/outputs/{name}"))

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
