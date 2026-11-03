"""Book Illustrator: paste a page of text, get an illustration from a local ComfyUI.

Standard library only. ComfyUI must be running separately (default
http://127.0.0.1:8188) and workflow_api.json must exist next to this file.
See README.md for the one-time setup.
"""

import contextlib
import hashlib
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import comfy
import config
from bible import facts, lint_book, normalise_book
from comfy import (
    GENERATE_LOCK,
    IMAGE_STEPS,
    STAGE_LABELS,
    ProgressListener,
    comfy_available,
    comfy_get,
    comfy_post_json,
    comfy_url,
    free_comfy,
    generate_image,
    get_progress,
    image_step,
    is_text_prompt_node,
    make_workflow,
    set_progress,
    start_progress_listener,
    workflow_models,
)
from config import (
    BASE_DIR,
    BOOKS_DIR,
    CONFIG,
    CONFIG_PATH,
    INDEX_PATH,
    LOG_BUFFER,
    LOG_LOCK,
    LOG_PATH,
    LOG_SEQ,
    MAX_BODY_BYTES,
    OUTPUT_DIR,
    SETTINGS_PATH,
    WORKFLOW_PATH,
    add_stage,
    gpu_snapshot,
    load_config,
    log,
    log_entries_after,
)
from planner import (
    EMPTY_PLAN,
    NeedsSceneError,
    PlannerError,
    cast_text,
    decide_scene,
    free_comfy_gpu,
    llm_json,
    llm_release,
    ollama_post,
    plan_from_request,
    plan_scene,
    planner_schema,
    review_book,
    review_scene,
    setting_line,
)
from planner_prompts import (
    BOOK_REVIEW_SCHEMA,
    BOOK_REVIEW_SYSTEM,
    PLANNER_FEW_SHOT,
    PLANNER_SCHEMA,
    PLANNER_SYSTEM,
    REVIEW_FEW_SHOT,
    REVIEW_SCHEMA,
    REVIEW_SYSTEM,
    SETTING_SCHEMA,
)
from prompting import PROMPT_VERSION, build_prompt
from storage import (
    BOOK_ID_RE,
    BOOK_WRITE_LOCK,
    SETTINGS_LOCK,
    BookFileError,
    book_file,
    book_view,
    list_books,
    load_book,
    read_saved_plan,
    read_settings,
    save_book,
    sidecar_path,
    update_book,
    write_json_atomic,
    write_settings,
    write_sidecar,
)

__all__ = [
    "BASE_DIR",
    "BOOKS_DIR",
    "BOOK_ID_RE",
    "BOOK_REVIEW_SCHEMA",
    "BOOK_REVIEW_SYSTEM",
    "BOOK_WRITE_LOCK",
    "BookFileError",
    "CONFIG",
    "CONFIG_PATH",
    "EMPTY_PLAN",
    "GENERATE_LOCK",
    "Handler",
    "IMAGE_STEPS",
    "INDEX_PATH",
    "LOG_BUFFER",
    "LOG_LOCK",
    "LOG_PATH",
    "LOG_SEQ",
    "LoggingServer",
    "MAX_BODY_BYTES",
    "NeedsSceneError",
    "OUTPUT_DIR",
    "PLANNER_FEW_SHOT",
    "PLANNER_SCHEMA",
    "PLANNER_SYSTEM",
    "PROMPT_VERSION",
    "PlannerError",
    "ProgressListener",
    "REVIEW_FEW_SHOT",
    "REVIEW_SCHEMA",
    "REVIEW_SYSTEM",
    "SETTINGS_LOCK",
    "SETTINGS_PATH",
    "SETTING_SCHEMA",
    "STAGE_LABELS",
    "WORKFLOW_PATH",
    "add_stage",
    "book_file",
    "book_view",
    "build_prompt",
    "cast_text",
    "comfy_available",
    "comfy_get",
    "comfy_post_json",
    "comfy_url",
    "decide_scene",
    "facts",
    "free_comfy",
    "free_comfy_gpu",
    "generate_image",
    "get_progress",
    "gpu_snapshot",
    "image_step",
    "is_text_prompt_node",
    "lint_book",
    "list_books",
    "llm_json",
    "llm_release",
    "load_book",
    "load_config",
    "log",
    "log_entries_after",
    "main",
    "make_workflow",
    "normalise_book",
    "ollama_post",
    "plan_from_request",
    "plan_scene",
    "planner_schema",
    "read_saved_plan",
    "read_settings",
    "review_book",
    "review_scene",
    "save_book",
    "set_progress",
    "setting_line",
    "sidecar_path",
    "start_progress_listener",
    "update_book",
    "workflow_models",
    "write_json_atomic",
    "write_settings",
    "write_sidecar",
]

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
            with open(config.INDEX_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(
                200,
                {
                    "comfy_ok": comfy_available(),
                    "workflow_ok": os.path.exists(config.WORKFLOW_PATH),
                    "comfy_url": config.CONFIG["comfy_url"],
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
            full = os.path.join(config.OUTPUT_DIR, os.path.basename(path[len("/outputs/") :]))
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
        if length < 0 or length > config.MAX_BODY_BYTES:
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
            if not config.CONFIG.get("llm_enabled"):
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
                            timeline,
                            "unload model",
                            time.perf_counter() - t,
                            kind="llm",
                            model=config.CONFIG["llm_model"],
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
                    log("ERROR", "cannot reach ComfyUI", error=str(e.reason), url=config.CONFIG["comfy_url"])
                    self._json(
                        503, {"error": f"Can't reach ComfyUI at {config.CONFIG['comfy_url']}. Start ComfyUI first."}
                    )
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
    url = f"http://127.0.0.1:{config.CONFIG['port']}"
    server = LoggingServer(("127.0.0.1", config.CONFIG["port"]), Handler)
    log(
        "INFO",
        "server started",
        url=url,
        comfy=config.CONFIG["comfy_url"],
        planner=bool(config.CONFIG.get("llm_enabled")),
    )
    print(f"Book Illustrator running at {url}")
    print(f"Expecting ComfyUI at {config.CONFIG['comfy_url']}")
    if config.CONFIG.get("open_browser"):
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


class _ServerModule(sys.modules[__name__].__class__):
    """Synchronizes monkeypatched module attributes with shared config and comfy state."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(config, name):
            setattr(config, name, value)
        if name in ("get_progress", "set_progress"):
            setattr(comfy, name, value)


sys.modules[__name__].__class__ = _ServerModule

if __name__ == "__main__":
    main()
