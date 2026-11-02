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
                "outputs": {"13": {"images": [
                    {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
                ]}},
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
        self._saved = (server.BOOKS_DIR, server.OUTPUT_DIR, server.SETTINGS_PATH, server.LOG_PATH,
                       dict(server.CONFIG))
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
    def test_only_named_characters_are_added(self):
        chars = "Hero: tall woman\nGuard: old man\nPatience: calm woman"
        prompt = server.build_prompt("Oil paint", "The Hero waits. She shows patience.", chars)
        self.assertIn("- Hero: tall woman", prompt)
        self.assertNotIn("- Guard:", prompt)
        self.assertNotIn("- Patience:", prompt)  # lowercase "patience" is an ordinary word

    def test_subject_then_scene_then_style(self):
        prompt = server.build_prompt("Oil paint", "The Hero waits.", "Hero: tall woman")
        self.assertLess(prompt.index("Characters in this scene"), prompt.index("Scene from the book"))
        self.assertLess(prompt.index("Scene from the book"), prompt.index("Visual style"))

    def test_narrator_is_marked_and_listed_first(self):
        chars = "Burrich: stablemaster\nFitz: teenage boy"
        prompt = server.build_prompt("", "I ran past Burrich.", chars, narrator="Fitz")
        self.assertIn("- Fitz (the narrator, the viewpoint character): teenage boy", prompt)
        self.assertLess(prompt.index("- Fitz"), prompt.index("- Burrich"))

    def test_scene_replaces_page_text_but_matching_uses_both(self):
        chars = "Burrich: stablemaster\nGalen: duke"
        prompt = server.build_prompt(
            "", "Burrich rode out at dawn.", chars,
            scene="A grey-haired stablemaster leads a horse out at dawn.",
        )
        self.assertIn("Scene from the book:\nA grey-haired stablemaster", prompt)
        self.assertNotIn("rode out", prompt)
        self.assertIn("- Burrich: stablemaster", prompt)  # matched on the page text
        self.assertNotIn("- Galen:", prompt)

    def test_optional_sections_are_left_out_when_empty(self):
        prompt = server.build_prompt("", "A quiet forest.", "Hero: tall woman")
        self.assertNotIn("Characters in this scene", prompt)
        self.assertNotIn("Visual style", prompt)

    def test_lines_without_a_colon_are_ignored(self):
        self.assertEqual(server.parse_characters("just a note\nHero: tall"), [("Hero", "tall")])


class BookTests(TempDirsMixin, unittest.TestCase):
    def test_round_trip(self):
        server.save_book(dict(SAMPLE_BOOK))
        self.assertEqual([b["id"] for b in server.list_books()], ["test-book"])
        view = server.book_view(server.load_book("test-book"))
        self.assertEqual(view["characters_text"], "Hero: tall woman, red cloak")

        server.update_book("test-book", {"characters_text": "Hero: tall\nGuard: old man\nnote without colon"})
        self.assertEqual(
            server.load_book("test-book")["characters"],
            [{"name": "Hero", "description": "tall"}, {"name": "Guard", "description": "old man"}],
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
        chars = "Hero: tall woman, red cloak\nGuard: old man with a spear\nOther: not in the scene"
        page = "The Hero crossed the bridge while the Guard watched."
        name, prompt, seed = server.generate_image(page, "Watercolour, muted colours.", chars)

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
        name, _, _ = server.generate_image("A quiet forest.", "", "")
        self.assertTrue(name.endswith(".png"))

    def test_seed_changes_between_runs(self):
        server.generate_image("A quiet forest.", "", "")
        server.generate_image("A quiet forest.", "", "")
        seeds = [r["prompt"]["7"]["inputs"]["noise_seed"] for r in self.fake.state["received"]]
        self.assertNotEqual(seeds[0], seeds[1])

    def test_unreachable_comfy_is_reported(self):
        server.CONFIG["comfy_url"] = "http://127.0.0.1:1"
        self.assertIsNone(server.start_progress_listener("cid", {}))
        with self.assertRaises(urllib.error.URLError):
            server.generate_image("A quiet forest.", "", "")


class ProgressTests(unittest.TestCase):
    def setUp(self):
        server.set_progress(active=True, label="", value=0, max=0)

    def tearDown(self):
        server.set_progress(active=False, label="", value=0, max=0)

    @staticmethod
    def text_frame(payload):
        """A server-to-client WebSocket text frame (not masked)."""
        n = len(payload)
        if n < 126:
            header = bytes([0x81, n])
        else:
            header = bytes([0x81, 126]) + struct.pack("!H", n)
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
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
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

    def test_generate_refuses_an_unknown_book(self):
        status, body = self.call("POST", "/api/generate", {"text": "A quiet forest.", "book": "missing"})
        self.assertEqual(status, 400)
        self.assertIn(b"Pick a book", body)


class NarratorTests(unittest.TestCase):
    def test_first_person_detected_outside_dialogue(self):
        self.assertTrue(server.has_first_person("I walked on until the road ended."))
        self.assertFalse(server.has_first_person('"Are you hurt?" Burrich asked the boy.'))
        self.assertTrue(server.has_first_person('"Go," he said. My feet were already moving.'))

    def test_narrator_listed_first_on_first_person_page(self):
        chars = "Galen: duke\nFitz: teenage boy\nBurrich: stablemaster"
        matched = server.characters_in_scene(chars, "I ran past Burrich.", narrator="Fitz")
        self.assertEqual([n for n, _ in matched], ["Fitz", "Burrich"])

    def test_narrator_left_out_of_third_person_page(self):
        chars = "Fitz: teenage boy\nBurrich: stablemaster"
        matched = server.characters_in_scene(chars, "Burrich watched the road.", narrator="Fitz")
        self.assertEqual([n for n, _ in matched], ["Burrich"])

    def test_narrator_named_in_third_person_page_is_included(self):
        matched = server.characters_in_scene("Fitz: teenage boy", "Fitz watched the road.", narrator="Fitz")
        self.assertEqual([n for n, _ in matched], ["Fitz"])


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
        status = self.request("POST", "/api/books/test-book", b'{"visual_style":"x"}',
                              {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_non_object_json_is_refused(self):
        status = self.request("POST", "/api/books/test-book", b"[1, 2]",
                              {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_undecodable_body_is_refused(self):
        status = self.request("POST", "/api/books/test-book", b"\xff\xfe\xfd",
                              {"Content-Type": "application/json"})
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
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
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
            out = json.dumps({"message": {"role": "assistant", "content": state["reply"]}}).encode()
        else:
            out = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


PLANNER_CHARS = "Fitz: teenage boy, dark hair\nPatience: calm woman\nSmithy: black terrier"
PLANNER_GLOSSARY = [{"term": "Forged", "meaning": "mindless living humans", "visual": "dull-eyed, stiff"}]


class PlannerTests(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.llm = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        self.llm.state = {"reply": "", "received": []}
        threading.Thread(target=self.llm.serve_forever, daemon=True).start()
        server.CONFIG.update(
            llm_enabled=True,
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

    def reply(self, checks, scene="Fitz walks a road at dusk."):
        self.llm.state["reply"] = json.dumps({"checks": checks, "scene": scene})

    def test_planner_picks_visible_characters(self):
        self.reply([{"name": "Fitz", "visible_in_moment": True},
                    {"name": "Patience", "visible_in_moment": True},
                    {"name": "Smithy", "visible_in_moment": False}])
        page = "I met Patience at the gate. Smithy was dead, and I thought of him."
        plan = server.plan_scene(page, PLANNER_CHARS, "Fitz", [])
        self.assertEqual(plan["present"], ["Fitz", "Patience"])
        prompt = server.build_prompt("", page, PLANNER_CHARS, "Fitz", plan["scene"], plan["present"])
        self.assertIn("- Fitz (the narrator, the viewpoint character)", prompt)
        self.assertIn("- Patience", prompt)
        self.assertNotIn("- Smithy", prompt)

    def test_names_outside_the_candidates_are_ignored(self):
        self.reply([{"name": "Fitz", "visible_in_moment": True}, {"name": "Bob", "visible_in_moment": True}])
        plan = server.plan_scene("I walked on.", PLANNER_CHARS, "Fitz", [])
        self.assertEqual(plan["present"], ["Fitz"])

    def test_close_spelling_is_matched(self):
        self.reply([{"name": "Patiance", "visible_in_moment": True}, {"name": "Fitz", "visible_in_moment": False}])
        plan = server.plan_scene("I saw Patience at the gate.", PLANNER_CHARS, "Fitz", [])
        self.assertEqual(plan["present"], ["Patience"])

    def test_narrator_is_left_out_when_planner_says_so(self):
        self.reply([{"name": "Fitz", "visible_in_moment": False}, {"name": "Patience", "visible_in_moment": True}])
        plan = server.plan_scene("I heard Patience laugh.", PLANNER_CHARS, "Fitz", [])
        self.assertEqual(plan["present"], ["Patience"])

    def test_bad_replies_and_dead_server_fall_back_to_none(self):
        self.llm.state["reply"] = "not json at all"
        self.assertIsNone(server.plan_scene("I saw Patience.", PLANNER_CHARS, "Fitz", []))
        self.llm.state["reply"] = json.dumps({"checks": "nope", "scene": ""})
        self.assertIsNone(server.plan_scene("I saw Patience.", PLANNER_CHARS, "Fitz", []))
        server.CONFIG["llm_url"] = "http://127.0.0.1:1"
        self.assertIsNone(server.plan_scene("I saw Patience.", PLANNER_CHARS, "Fitz", []))

    def test_disabled_planner_makes_no_calls(self):
        server.CONFIG["llm_enabled"] = False
        self.reply([{"name": "Fitz", "visible_in_moment": True}])
        self.assertIsNone(server.plan_scene("I saw Patience.", PLANNER_CHARS, "Fitz", []))
        self.assertEqual(self.llm.state["received"], [])

    def test_no_names_means_no_planner_call(self):
        self.reply([{"name": "Fitz", "visible_in_moment": True}])
        self.assertIsNone(server.plan_scene("The road was long.", PLANNER_CHARS, "Fitz", []))
        self.assertEqual(self.llm.state["received"], [])

    def test_glossary_reaches_planner_and_image_notes(self):
        self.reply([{"name": "Fitz", "visible_in_moment": True}],
                   scene="Fitz faces a group of Forged ones in the road.")
        page = "I rode on, past the Forged ones in the road."
        plan = server.plan_scene(page, PLANNER_CHARS, "Fitz", PLANNER_GLOSSARY)
        system = self.llm.state["received"][-1]["messages"][0]["content"]
        self.assertIn("- Forged: mindless living humans", system)
        self.assertEqual(plan["terms"], ["Forged"])
        notes = [(g["term"], g["visual"]) for g in PLANNER_GLOSSARY]
        prompt = server.build_prompt("", page, PLANNER_CHARS, "Fitz", plan["scene"], plan["present"], notes)
        self.assertIn("Visual notes:\n- Forged: dull-eyed, stiff", prompt)

    def test_explicit_present_list_overrides_name_matching(self):
        prompt = server.build_prompt("", "Smithy barked at Patience.", PLANNER_CHARS, "", "", ["Patience"])
        self.assertIn("- Patience", prompt)
        self.assertNotIn("- Smithy", prompt)


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
        server.CONFIG.update(comfy_url=f"http://127.0.0.1:{self.comfy.server_address[1]}",
                             timeout_seconds=30, steps=4, width=512, height=512)

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
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
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
        server.generate_image("I walked the road.", "", "", "", "", None, None, timeline)
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
        status, body = self.post("/api/generate", {
            "text": "I walked the road at dusk.", "book": "test-book", "scene": "A road at dusk.",
        })
        self.assertEqual(status, 200)
        stages = [s["stage"] for s in body["timeline"]]
        self.assertIn("Save book", stages)
        self.assertIn("queue request", stages)
        self.assertGreater(body["total_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
