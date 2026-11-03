"""Tests for the Book Illustrator server.

Run from the project root. No GPU or ComfyUI is needed: a small fake ComfyUI
stands in for it, and books and outputs go to temporary folders.

    python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import server  # noqa: E402

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-bytes"

SAMPLE_BOOK = {
    "id": "test-book",
    "title": "Test Book",
    "author": "Test Author",
    "series": "Test Series",
    "book_number": 2,
    "visual_style": "Ink and watercolour",
    "characters": [{"name": "Hero", "description": "tall woman, red cloak"}],
}


def cast(text):
    """Test fixtures written as "Name: look" lines, turned into the structured cast the app uses."""
    return [
        {"name": n.strip(), "look": look.strip()}
        for n, look in (line.split(":", 1) for line in text.splitlines() if ":" in line)
    ]


class FakeComfy(BaseHTTPRequestHandler):
    """The ComfyUI endpoints one generation needs. There is no /ws, so the app
    must still generate without live progress."""

    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        state = self.server.state
        if self.path.startswith("/system_stats"):
            self._send(200, b"{}")
        elif self.path.startswith("/history/"):
            state["polls"] += 1
            if state["polls"] < 3:
                self._send(200, b"{}")
                return
            entry = {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"13": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}]}},
            }
            self._send(200, json.dumps({state["prompt_id"]: entry}).encode())
        elif self.path.startswith("/view"):
            self._send(200, PNG_BYTES, "image/png")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        state = self.server.state
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if self.path.startswith("/free"):  # ComfyUI frees its models before the planner runs
            self._send(200, b"{}")
            return
        state["received"].append(body)
        self._send(200, json.dumps({"prompt_id": state["prompt_id"], "number": 1}).encode())


class TempDirsMixin:
    """Points the server at temporary books and outputs folders."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bi-test-")
        self._saved = (server.BOOKS_DIR, server.OUTPUT_DIR, server.SETTINGS_PATH, server.LOG_PATH, dict(server.CONFIG))
        server.BOOKS_DIR = os.path.join(self.tmp, "books")
        server.OUTPUT_DIR = os.path.join(self.tmp, "outputs")
        server.SETTINGS_PATH = os.path.join(self.tmp, "settings.json")
        server.LOG_PATH = os.path.join(self.tmp, "server.log")

    def tearDown(self):
        server.BOOKS_DIR, server.OUTPUT_DIR, server.SETTINGS_PATH, server.LOG_PATH, config = self._saved
        server.CONFIG.clear()
        server.CONFIG.update(config)
        shutil.rmtree(self.tmp, ignore_errors=True)


class PromptTests(unittest.TestCase):
    def test_subject_then_scene_then_style(self):
        prompt = server.build_prompt("Oil paint", "The Hero waits.", cast("Hero: tall woman"), present=["Hero"])
        self.assertLess(prompt.index("Characters in this scene"), prompt.index("\nScene:\n"))
        self.assertLess(prompt.index("\nScene:\n"), prompt.index("Visual style"))

    def test_scene_replaces_page_text(self):
        prompt = server.build_prompt(
            "",
            "Hero rode out at dawn.",
            cast("Hero: rider"),
            scene="A rider leads a horse out at dawn.",
            present=["Hero"],
        )
        self.assertIn("Scene:\nA rider leads", prompt)
        self.assertNotIn("rode out", prompt)

    def test_optional_sections_are_left_out_when_empty(self):
        prompt = server.build_prompt("", "A quiet forest.", cast("Hero: tall woman"))
        self.assertNotIn("Characters in this scene", prompt)
        self.assertNotIn("Visual style", prompt)


class BookTests(TempDirsMixin, unittest.TestCase):
    def test_round_trip(self):
        server.save_book(dict(SAMPLE_BOOK))
        self.assertEqual([b["id"] for b in server.list_books()], ["test-book"])
        view = server.book_view(server.load_book("test-book"))
        self.assertEqual(view["characters"][0]["look"], "tall woman, red cloak")  # schema 1 description

        server.update_book("test-book", {"characters": cast("Hero: tall\nGuard: old man") + [{"look": "no name"}]})
        self.assertEqual(
            server.load_book("test-book")["characters"],
            [
                {"name": "Hero", "aliases": [], "kind": "person", "sex": "", "age": "", "species": "", "look": "tall"},
                {
                    "name": "Guard",
                    "aliases": [],
                    "kind": "person",
                    "sex": "",
                    "age": "",
                    "species": "",
                    "look": "old man",
                },
            ],
        )
        server.update_book("test-book", {"visual_style": "Oil"})
        self.assertEqual(server.load_book("test-book")["visual_style"], "Oil")

    def test_bad_ids_are_rejected(self):
        for bad in ["../secrets", "Bad Id", ""]:
            with self.assertRaises(ValueError):
                server.load_book(bad)

    def test_missing_book(self):
        with self.assertRaises(FileNotFoundError):
            server.load_book("no-such-book")

    def test_structured_characters_and_glossary_are_saved(self):
        server.save_book(dict(SAMPLE_BOOK))
        server.update_book(
            "test-book",
            {
                "characters": [
                    {"name": "Ren", "kind": "person", "sex": "male", "age": "teen", "look": "slim"},
                    {"name": "Kestrel", "kind": "animal", "species": "wolf", "look": "grey"},
                ],
                "glossary": [{"term": "Wending", "meaning": "a bond", "visual": "a shared gaze", "extra": "dropped"}],
            },
        )
        view = server.book_view(server.load_book("test-book"))
        self.assertEqual(view["characters"][0]["sex"], "male")
        self.assertEqual(view["characters"][1]["species"], "wolf")
        self.assertEqual(view["glossary"], [{"term": "Wending", "meaning": "a bond", "visual": "a shared gaze"}])
        self.assertEqual(view["problems"], [])

    def test_book_view_lists_problems(self):
        server.save_book(dict(SAMPLE_BOOK))  # the sample hero has no sex or age yet
        problems = server.book_view(server.load_book("test-book"))["problems"]
        self.assertTrue(any(p["level"] == "error" and "set sex" in p["message"] for p in problems))


class GenerationTests(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        server.CONFIG.update(width=768, height=640, steps=6, timeout_seconds=30)
        self.fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
        self.fake.state = {"polls": 0, "prompt_id": "pid-1", "received": []}
        threading.Thread(target=self.fake.serve_forever, daemon=True).start()
        server.CONFIG["comfy_url"] = f"http://127.0.0.1:{self.fake.server_address[1]}"

    def tearDown(self):
        self.fake.shutdown()
        self.fake.server_close()
        super().tearDown()

    def test_prompt_and_settings_reach_comfy_and_image_is_saved(self):
        chars = cast("Hero: tall woman, red cloak\nGuard: old man with a spear\nOther: not in the scene")
        page = "The Hero crossed the bridge while the Guard watched."
        prompt = server.build_prompt("Watercolour, muted colours.", page, chars, present=["Hero", "Guard"])
        name, prompt, seed = server.generate_image(prompt)

        sent = self.fake.state["received"][-1]["prompt"]
        self.assertEqual(seed, sent["7"]["inputs"]["noise_seed"])
        self.assertEqual(sent["1"]["inputs"]["text"], prompt)
        self.assertIn("- Hero: tall woman, red cloak", prompt)
        self.assertIn("- Guard: old man with a spear", prompt)
        self.assertNotIn("- Other:", prompt)
        self.assertEqual(sent["5"]["inputs"]["steps"], 6)
        self.assertEqual((sent["6"]["inputs"]["width"], sent["6"]["inputs"]["height"]), (768, 640))
        with open(os.path.join(server.OUTPUT_DIR, name), "rb") as f:
            self.assertEqual(f.read(), PNG_BYTES)

    def test_generation_works_without_websocket(self):
        # The fake has no /ws, so no progress listener starts.
        self.assertIsNone(server.start_progress_listener("cid", {}))
        name, _, _ = server.generate_image("A quiet forest.")
        self.assertTrue(name.endswith(".png"))

    def test_seed_changes_between_runs(self):
        server.generate_image("A quiet forest.")
        server.generate_image("A quiet forest.")
        seeds = [r["prompt"]["7"]["inputs"]["noise_seed"] for r in self.fake.state["received"]]
        self.assertNotEqual(seeds[0], seeds[1])

    def test_unreachable_comfy_is_reported(self):
        server.CONFIG["comfy_url"] = "http://127.0.0.1:1"
        self.assertIsNone(server.start_progress_listener("cid", {}))
        with self.assertRaises(urllib.error.URLError):
            server.generate_image("A quiet forest.")


class ProgressTests(unittest.TestCase):
    def setUp(self):
        server.set_progress(active=True, label="", value=0, max=0)

    def tearDown(self):
        server.set_progress(active=False, label="", value=0, max=0)

    @staticmethod
    def text_frame(payload):
        """A server-to-client WebSocket text frame (not masked)."""
        n = len(payload)
        header = bytes([129, n]) if n < 126 else bytes([129, 126]) + struct.pack("!H", n)
        return header + payload

    def test_progress_and_stage_messages_update_state(self):
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        listener = server.ProgressListener("cid", {"8": "SamplerCustomAdvanced"})
        listener.sock = a
        messages = [
            {"type": "executing", "data": {"node": "8"}},
            {"type": "progress", "data": {"value": 2, "max": 4, "node": "8", "pad": "x" * 300}},
            {"type": "progress", "data": {"value": 3, "max": 4, "node": "8"}},
        ]
        for m in messages:
            b.sendall(self.text_frame(json.dumps(m).encode()))
        for _ in messages:
            _, payload = listener._read_frame()
            listener._handle(json.loads(payload))

        p = server.get_progress()
        self.assertEqual((p["label"], p["value"], p["max"]), ("Sampling", 3, 4))


class HttpTests(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        server.save_book(dict(SAMPLE_BOOK))
        self.app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.app.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.app.server_address[1]}"

    def tearDown(self):
        self.app.shutdown()
        self.app.server_close()
        super().tearDown()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_index_page_is_served(self):
        status, body = self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b'id="books"', body)

    def test_book_endpoints(self):
        status, body = self.call("GET", "/api/books")
        self.assertEqual(status, 200)
        self.assertEqual([b["id"] for b in json.loads(body)], ["test-book"])

        status, _ = self.call("POST", "/api/books/test-book", {"visual_style": "Oil"})
        self.assertEqual(status, 200)
        status, body = self.call("GET", "/api/books/test-book")
        self.assertEqual(json.loads(body)["visual_style"], "Oil")
        self.assertEqual(self.call("GET", "/api/books/missing")[0], 404)

    def test_book_save_returns_the_problems_list(self):
        status, body = self.call(
            "POST",
            "/api/books/test-book",
            {
                "characters": [{"name": "Hero", "look": "tall"}],
            },
        )
        self.assertEqual(status, 200)
        problems = json.loads(body)["problems"]
        self.assertTrue(any(p["level"] == "error" for p in problems))

    def test_generate_refuses_an_unknown_book(self):
        status, body = self.call("POST", "/api/generate", {"text": "A quiet forest.", "book": "missing"})
        self.assertEqual(status, 400)
        self.assertIn(b"Pick a book", body)


class BookSafetyTests(TempDirsMixin, unittest.TestCase):
    def test_copied_file_cannot_take_over_another_book(self):
        server.save_book(dict(SAMPLE_BOOK))
        # A copy keeps the original "id" inside but has its own file name.
        with open(os.path.join(server.BOOKS_DIR, "mine.json"), "w", encoding="utf-8") as f:
            json.dump(dict(SAMPLE_BOOK), f)
        server.update_book("mine", {"visual_style": "Oil"})
        self.assertEqual(server.load_book("test-book")["visual_style"], SAMPLE_BOOK["visual_style"])
        self.assertEqual(server.load_book("mine")["visual_style"], "Oil")
        self.assertIn("mine", [b["id"] for b in server.list_books()])

    def test_corrupt_file_is_skipped_and_reported(self):
        server.save_book(dict(SAMPLE_BOOK))
        with open(os.path.join(server.BOOKS_DIR, "broken.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual([b["id"] for b in server.list_books()], ["test-book"])
        with self.assertRaises(server.BookFileError):
            server.load_book("broken")

    def test_atomic_save_leaves_no_temp_file(self):
        server.save_book(dict(SAMPLE_BOOK))
        self.assertFalse([f for f in os.listdir(server.BOOKS_DIR) if f.endswith(".tmp")])

    def test_narrator_round_trips(self):
        server.save_book(dict(SAMPLE_BOOK))
        server.update_book("test-book", {"narrator": " Hero "})
        self.assertEqual(server.book_view(server.load_book("test-book"))["narrator"], "Hero")


class RequestSafetyTests(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        server.save_book(dict(SAMPLE_BOOK))
        self.app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.app.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.app.server_address[1]}"

    def tearDown(self):
        self.app.shutdown()
        self.app.server_close()
        super().tearDown()

    def request(self, method, path, data=None, headers=None):
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def test_text_plain_post_is_refused(self):
        status = self.request("POST", "/api/books/test-book", b'{"visual_style":"x"}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_non_object_json_is_refused(self):
        status = self.request("POST", "/api/books/test-book", b"[1, 2]", {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_undecodable_body_is_refused(self):
        status = self.request("POST", "/api/books/test-book", b"\xff\xfe\xfd", {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_foreign_host_is_refused(self):
        self.assertEqual(self.request("GET", "/api/books", headers={"Host": "evil.example"}), 403)

    def test_own_host_is_served(self):
        self.assertEqual(self.request("GET", "/api/books"), 200)

    def test_corrupt_book_by_id_is_a_server_error(self):
        with open(os.path.join(server.BOOKS_DIR, "broken.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.request("GET", "/api/books/broken"), 500)


class SettingsTests(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        server.save_book(dict(SAMPLE_BOOK))
        self.app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.app.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.app.server_address[1]}"

    def tearDown(self):
        self.app.shutdown()
        self.app.server_close()
        super().tearDown()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_no_default_until_one_is_pinned(self):
        self.assertEqual(self.call("GET", "/api/settings"), (200, {}))

    def test_pin_is_saved_and_read_back(self):
        self.assertEqual(self.call("POST", "/api/settings", {"default_book": "test-book"})[0], 200)
        self.assertEqual(self.call("GET", "/api/settings")[1]["default_book"], "test-book")
        self.assertTrue(os.path.exists(server.SETTINGS_PATH))

    def test_unknown_book_cannot_be_pinned(self):
        status, _ = self.call("POST", "/api/settings", {"default_book": "missing"})
        self.assertEqual(status, 404)
        self.assertNotIn("default_book", self.call("GET", "/api/settings")[1])

    def test_empty_value_clears_the_default(self):
        self.call("POST", "/api/settings", {"default_book": "test-book"})
        self.call("POST", "/api/settings", {"default_book": ""})
        self.assertEqual(self.call("GET", "/api/settings")[1]["default_book"], "")

    def test_unreadable_settings_file_falls_back_to_defaults(self):
        with open(server.SETTINGS_PATH, "w", encoding="utf-8") as f:
            f.write("{broken")
        self.assertEqual(self.call("GET", "/api/settings"), (200, {}))


class FakeLLM(BaseHTTPRequestHandler):
    """Stands in for Ollama: /api/chat returns a canned reply; anything else (unload) returns OK."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        state = self.server.state
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if self.path == "/api/chat":
            state["received"].append(body)
            reply = state["replies"].pop(0) if state.get("replies") else state["reply"]
            out = json.dumps({"message": {"role": "assistant", "content": reply}}).encode()
        else:
            out = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


PLANNER_CHARS = server.normalise_book(
    {"characters": cast("Ren: teenage boy, dark hair\nAma: calm woman\nKestrel: grey wolf")}
)["characters"]
PLANNER_GLOSSARY = [{"term": "Wending", "meaning": "people whose minds were taken", "visual": "dull-eyed, stiff"}]
ROAD = {"place": "a road", "indoors": "no", "time": "dusk"}


def planner_reply(checks, scene="Ren walks a road at dusk.", terms=None, setting=None):
    return json.dumps({"checks": checks, "terms": terms or [], "setting": setting or ROAD, "scene": scene})


def review_reply(ok=True, problems=(), scene=""):
    return json.dumps({"ok": ok, "problems": list(problems), "scene": scene})


def visible(*names, hidden=()):
    return [{"name": n, "visible_in_moment": True} for n in names] + [
        {"name": n, "visible_in_moment": False} for n in hidden
    ]


class FakePlannerMixin(TempDirsMixin):
    validate = False

    def setUp(self):
        super().setUp()
        self.llm = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        self.llm.state = {"reply": "", "replies": [], "received": []}
        threading.Thread(target=self.llm.serve_forever, daemon=True).start()
        server.CONFIG.update(
            llm_enabled=True,
            planner_validate=self.validate,
            llm_url=f"http://127.0.0.1:{self.llm.server_address[1]}",
            llm_model="fake-planner",
            llm_num_gpu=0,  # CPU here: no ComfyUI /free call in the tests
            llm_timeout=5,
            llm_keep_alive="1m",
        )

    def tearDown(self):
        self.llm.shutdown()
        self.llm.server_close()
        super().tearDown()

    def request(self, n=0):
        return self.llm.state["received"][n]["messages"]


class PlannerTests(FakePlannerMixin, unittest.TestCase):
    def test_planner_picks_visible_characters(self):
        self.llm.state["reply"] = planner_reply(visible("Ren", "Ama", hidden=["Kestrel"]))
        page = "I met Ama at the gate. Kestrel was gone, and I thought of him."
        plan = server.plan_scene(page, PLANNER_CHARS, "Ren", [])
        self.assertEqual(plan["present"], ["Ren", "Ama"])
        prompt = server.build_prompt("", page, PLANNER_CHARS, "Ren", plan["scene"], plan["present"])
        self.assertIn("- Ren (narrator): teenage boy", prompt)
        self.assertIn("- Ama", prompt)
        self.assertNotIn("- Kestrel", prompt)

    def test_whole_cast_and_glossary_are_sent(self):
        self.llm.state["reply"] = planner_reply(visible("Ren"))
        server.plan_scene(
            "The road was long.",
            PLANNER_CHARS,
            "Ren",
            server.normalise_book({"glossary": PLANNER_GLOSSARY})["glossary"],
        )
        user = self.request()[-1]["content"]
        for line in (
            '- Ren (narrator, "I")',
            "- Ama\n",
            "- Kestrel\n",
            "Glossary terms: Wending",
            "Page:\nThe road was long.",
        ):
            self.assertIn(line, user)
        self.assertIn("Wending: people whose minds were taken", self.request()[0]["content"])

    def test_page_without_names_still_gets_a_setting(self):
        self.llm.state["reply"] = planner_reply([], scene="An empty road at dusk.")
        plan = server.plan_scene("The road was long and empty.", PLANNER_CHARS, "Ren", [])
        self.assertEqual(plan["present"], [])
        self.assertEqual(plan["setting"], "a road, outdoors, dusk")

    def test_names_outside_the_cast_are_ignored(self):
        self.llm.state["reply"] = planner_reply(visible("Ren", "Bob"))
        self.assertEqual(server.plan_scene("I walked on.", PLANNER_CHARS, "Ren", [])["present"], ["Ren"])

    def test_names_must_come_from_the_cast(self):
        self.llm.state["reply"] = planner_reply(visible("Amma", "Ama"))
        plan = server.plan_scene("I saw Ama at the gate.", PLANNER_CHARS, "Ren", PLANNER_GLOSSARY)
        self.assertEqual(plan["present"], ["Ama"])  # a name the book does not have is dropped
        schema = self.llm.state["received"][0]["format"]
        self.assertEqual(
            schema["properties"]["checks"]["items"]["properties"]["name"]["enum"], ["Ren", "Ama", "Kestrel"]
        )
        self.assertEqual(schema["properties"]["terms"]["items"]["enum"], ["Wending"])

    def test_glossary_terms_come_from_the_planner(self):
        self.llm.state["reply"] = planner_reply(visible("Ren"), terms=["Wending", "Unknown term"])
        plan = server.plan_scene(
            "Wending ones on the road.",
            PLANNER_CHARS,
            "Ren",
            server.normalise_book({"glossary": PLANNER_GLOSSARY})["glossary"],
        )
        self.assertEqual(plan["terms"], ["Wending"])

    def test_bad_replies_and_dead_server_raise_planner_error(self):
        self.llm.state["reply"] = "not json"
        with self.assertRaises(server.PlannerError):
            server.plan_scene("I saw Ama.", PLANNER_CHARS, "Ren", [])
        self.llm.state["reply"] = json.dumps({"checks": "nope", "terms": [], "setting": ROAD, "scene": ""})
        with self.assertRaises(server.PlannerError):
            server.plan_scene("I saw Ama.", PLANNER_CHARS, "Ren", [])
        server.CONFIG["llm_url"] = "http://127.0.0.1:1"
        with self.assertRaises(server.PlannerError):
            server.plan_scene("I saw Ama.", PLANNER_CHARS, "Ren", [])

    def test_cut_off_answer_is_retried_with_a_bigger_budget(self):
        self.llm.state["replies"] = ['{"checks": [{"name": "Ren", "visi', planner_reply(visible("Ren"))]
        plan = server.plan_scene("I walked.", PLANNER_CHARS, "Ren", [])
        self.assertEqual(plan["present"], ["Ren"])
        first, second = (r["options"]["num_predict"] for r in self.llm.state["received"])
        self.assertEqual(second, first * 2)

    def test_answer_budget_grows_with_the_cast(self):
        self.llm.state["reply"] = planner_reply(visible("Ren"))
        server.plan_scene("I walked.", PLANNER_CHARS, "Ren", [])
        small = self.llm.state["received"][-1]["options"]["num_predict"]
        big_cast = (
            PLANNER_CHARS
            + server.normalise_book({"characters": [{"name": f"Extra{i}"} for i in range(20)]})["characters"]
        )
        server.plan_scene("I walked.", big_cast, "Ren", [])
        self.assertGreater(self.llm.state["received"][-1]["options"]["num_predict"], small)

    def test_disabled_planner_makes_no_calls(self):
        server.CONFIG["llm_enabled"] = False
        self.assertIsNone(server.plan_scene("I saw Ama.", PLANNER_CHARS, "Ren", []))
        self.assertEqual(self.llm.state["received"], [])

    def test_examples_use_invented_names_only(self):
        self.llm.state["reply"] = planner_reply(visible("Ren"))
        server.plan_scene("I walked on.", PLANNER_CHARS, "Ren", [])
        examples = " ".join(m["content"] for m in self.request()[1:-1])
        self.assertIn("Lira", examples)
        self.assertIn("holds a water skin to Ren's lips", examples)


class ReviewerTests(FakePlannerMixin, unittest.TestCase):
    """The second LLM call that checks the planner's scene and fixes it."""

    validate = True

    def test_reviewer_fix_is_used(self):
        self.llm.state["replies"] = [
            planner_reply(visible("Ren", "Ama"), scene="Ama holds the cup to his lips."),
            review_reply(False, ["'his' could mean Ren or Ama"], "Ama holds the cup to Ren's lips."),
        ]
        plan = server.plan_scene("Ama held the cup to my lips.", PLANNER_CHARS, "Ren", [])
        self.assertEqual(plan["scene"], "Ama holds the cup to Ren's lips.")
        self.assertEqual(plan["review"]["problems"], ["'his' could mean Ren or Ama"])

    def test_reviewer_sees_the_visible_people_and_setting(self):
        self.llm.state["replies"] = [planner_reply(visible("Ren", "Ama")), review_reply(True)]
        server.plan_scene("Ama walked beside me.", PLANNER_CHARS, "Ren", [])
        user = self.request(1)[-1]["content"]
        self.assertIn("Visible: Ren, Ama", user)
        self.assertIn("Setting: a road, outdoors, dusk", user)

    def test_ok_review_keeps_the_scene(self):
        self.llm.state["replies"] = [planner_reply(visible("Ren"), scene="Ren walks a road."), review_reply(True)]
        self.assertEqual(server.plan_scene("I walked.", PLANNER_CHARS, "Ren", [])["scene"], "Ren walks a road.")

    def test_no_fix_keeps_the_planners_scene(self):
        self.llm.state["replies"] = [
            planner_reply(visible("Ren"), scene="Ren walks a road."),
            review_reply(False, ["bad"], ""),
        ]
        plan = server.plan_scene("I walked.", PLANNER_CHARS, "Ren", [])
        self.assertEqual((plan["scene"], plan["present"]), ("Ren walks a road.", ["Ren"]))

    def test_review_off_makes_one_call(self):
        server.CONFIG["planner_validate"] = False
        self.llm.state["replies"] = [planner_reply(visible("Ren"))]
        server.plan_scene("I walked.", PLANNER_CHARS, "Ren", [])
        self.assertEqual(len(self.llm.state["received"]), 1)


class LoggingAndTimingTests(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        server.save_book(dict(SAMPLE_BOOK))
        self.app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.app.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.app.server_address[1]}"
        self.comfy = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
        self.comfy.state = {"polls": 0, "prompt_id": "pid-t", "received": []}
        threading.Thread(target=self.comfy.serve_forever, daemon=True).start()
        server.CONFIG.update(
            comfy_url=f"http://127.0.0.1:{self.comfy.server_address[1]}",
            timeout_seconds=30,
            steps=4,
            width=512,
            height=512,
        )

    def tearDown(self):
        for s in (self.app, self.comfy):
            s.shutdown()
            s.server_close()
        super().tearDown()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, path, body):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_log_entries_are_kept_and_served_after_a_sequence_number(self):
        server.log("INFO", "first test line")
        server.log("WARN", "second test line")
        status, body = self.get("/api/log?after=0")
        msgs = [e["msg"] for e in body["entries"]]
        self.assertEqual(status, 200)
        self.assertIn("first test line", msgs)
        self.assertIn("second test line", msgs)
        status, body = self.get(f"/api/log?after={body['last']}")
        self.assertEqual(body["entries"], [])
        with open(server.LOG_PATH, encoding="utf-8") as f:
            self.assertIn("second test line", f.read())

    def test_unexpected_error_is_logged_and_returned_as_json(self):
        original = server.get_progress

        def broken():
            raise RuntimeError("boom for the test")

        server.get_progress = broken
        try:
            status, body = self.get("/api/progress")
        finally:
            server.get_progress = original
        self.assertEqual(status, 500)
        self.assertIn("boom for the test", body["error"])
        errors = [e for e in server.log_entries_after(0) if e["level"] == "ERROR"]
        self.assertTrue(any("traceback" in e["fields"] for e in errors))

    def test_generation_timeline_names_each_step(self):
        timeline = []
        server.generate_image("I walked the road.", timeline)
        stages = [s["stage"] for s in timeline]
        self.assertIn("Build workflow", stages)
        self.assertIn("queue request", stages)
        self.assertIn("queue wait and PNG save", stages)
        image_steps = [s for s in timeline if s["kind"] == "image"]
        self.assertTrue(image_steps and all(s["model"] for s in image_steps))
        self.assertTrue(all(s["seconds"] >= 0 for s in timeline))

    def test_listener_times_each_comfy_step(self):
        listener = server.ProgressListener("cid", {"8": "SamplerCustomAdvanced", "9": "VAEDecode"})
        listener._handle({"type": "executing", "data": {"node": "8"}})
        time.sleep(0.05)
        listener._handle({"type": "executing", "data": {"node": "9"}})
        time.sleep(0.02)
        totals = listener.finish_stages()
        self.assertGreaterEqual(totals["Sampling"], 0.04)
        self.assertGreaterEqual(totals["Decoding image"], 0.01)

    def test_generate_endpoint_returns_timeline(self):
        status, body = self.post(
            "/api/generate",
            {
                "text": "I walked the road at dusk.",
                "book": "test-book",
                "scene": "A road at dusk.",
            },
        )
        self.assertEqual(status, 200)
        stages = [s["stage"] for s in body["timeline"]]
        self.assertIn("Build workflow", stages)
        self.assertIn("queue request", stages)
        self.assertGreater(body["total_seconds"], 0)


class EvalVariantTests(TempDirsMixin, unittest.TestCase):
    """Golden-set eval hooks: head count, negative line, fixed seed, and the variant that skips the planner."""

    def setUp(self):
        super().setUp()
        server.save_book(dict(SAMPLE_BOOK))
        self.app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.app.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.app.server_address[1]}"
        self.comfy = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
        self.comfy.state = {"polls": 0, "prompt_id": "pid-v", "received": []}
        threading.Thread(target=self.comfy.serve_forever, daemon=True).start()
        server.CONFIG.update(
            comfy_url=f"http://127.0.0.1:{self.comfy.server_address[1]}",
            timeout_seconds=30,
            steps=4,
            width=512,
            height=512,
            llm_enabled=True,
            llm_url="http://127.0.0.1:1",
            llm_timeout=2,
        )

    def tearDown(self):
        self.app.shutdown()
        self.app.server_close()
        self.comfy.shutdown()
        self.comfy.server_close()
        super().tearDown()

    def post(self, path, body):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_count_line_lists_the_planners_people(self):
        prompt = server.build_prompt(
            "", "Hero walks.", cast("Hero: tall woman\nGuard: old man"), "", "", ["Hero", "Guard"]
        )
        self.assertIn("Figures in the picture, exactly 2: Hero, Guard.", prompt)
        self.assertLess(prompt.index("Figures in the picture"), prompt.index("\nScene:\n"))

    def test_planner_failure_draws_nothing(self):
        # The fake planner address refuses connections, so the planner fails.
        status, body = self.post("/api/generate", {"text": "I walked the road at dusk.", "book": "test-book"})
        self.assertEqual(status, 502)
        self.assertIn("scene planner failed", body["error"])
        self.assertEqual(self.comfy.state["received"], [])

    def test_planner_off_asks_for_a_typed_scene(self):
        server.CONFIG["llm_enabled"] = False
        status, body = self.post("/api/generate", {"text": "I walked the road at dusk.", "book": "test-book"})
        self.assertEqual(status, 400)
        self.assertIn("type a scene", body["error"])
        status, _ = self.post("/api/generate", {"text": "I walked.", "book": "test-book", "scene": "A road."})
        self.assertEqual(status, 200)

    def test_fixed_seed_is_sent_to_comfy(self):
        server.generate_image("A quiet forest.", seed=12345)
        sent = self.comfy.state["received"][-1]["prompt"]
        self.assertEqual(sent["7"]["inputs"]["noise_seed"], 12345)

    def test_variant_skips_planner_and_uses_its_people(self):
        status, body = self.post(
            "/api/generate",
            {
                "text": "I walked the road at dusk.",
                "book": "test-book",
                "seed": 777,
                "variant": {
                    "present": ["Hero"],
                    "scene": "A tall woman walks a road at dusk.",
                    "characters": [{"name": "Hero", "look": "stocky man in a grey cloak"}],
                },
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["variant"])
        self.assertFalse(body["planner"]["used"])
        self.assertEqual((body["seed"], body["plan"]["source"]), (777, "variant"))
        self.assertFalse([s for s in body["timeline"] if s["kind"] == "llm"])  # no planner work at all
        self.assertIn("Figures in the picture, exactly 1: Hero.", body["prompt"])
        self.assertIn("- Hero: stocky man in a grey cloak", body["prompt"])
        self.assertIn("A tall woman walks a road at dusk.", body["prompt"])

    def test_generate_does_not_change_the_saved_book(self):
        self.post(
            "/api/generate",
            {
                "text": "I walked the road at dusk.",
                "book": "test-book",
                "visual_style": "ignored",
                "variant": {"present": ["Hero"], "scene": "A road.", "characters": [{"name": "Hero", "look": "other"}]},
            },
        )
        saved = server.load_book("test-book")
        self.assertEqual(saved["characters"][0]["look"], "tall woman, red cloak")
        self.assertEqual(saved["visual_style"], "Ink and watercolour")

    def test_rerun_draws_the_saved_plan_of_an_earlier_image(self):
        plan = {"scene": "A tall woman walks a road at dusk.", "present": ["Hero"], "setting": "road, outdoors"}
        _, first = self.post("/api/generate", {"text": "I walked the road.", "book": "test-book", "variant": plan})
        status, body = self.post(
            "/api/generate",
            {"text": "I walked the road.", "book": "test-book", "from_image": first["image"], "seed": 5},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["plan"]["source"], "saved")
        self.assertEqual(body["plan"]["scene"], plan["scene"])
        self.assertIn("Setting: road, outdoors", body["prompt"])
        self.assertFalse([s for s in body["timeline"] if s["kind"] == "llm"])

    def test_unknown_saved_image_is_ignored(self):
        self.assertIsNone(server.read_saved_plan("../books/test-book.json"))
        self.assertIsNone(server.read_saved_plan("missing.png"))

    def test_each_image_gets_a_sidecar_file(self):
        status, body = self.post(
            "/api/generate",
            {
                "text": "I walked the road at dusk.",
                "book": "test-book",
                "seed": 21,
                "variant": {"present": ["Hero"], "scene": "A road."},
            },
        )
        self.assertEqual(status, 200)
        with open(server.sidecar_path(body["image"]), encoding="utf-8") as f:
            record = json.load(f)
        self.assertEqual((record["seed"], record["image"]), (21, body["image"]))
        self.assertIn("prompt", record)
        self.assertIn("gpu", record)
        self.assertEqual(record["prompt_version"], server.PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
