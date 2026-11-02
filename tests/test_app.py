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
        state["received"].append(json.loads(self.rfile.read(length)))
        self._send(200, json.dumps({"prompt_id": state["prompt_id"], "number": 1}).encode())


class TempDirsMixin:
    """Points the server at temporary books and outputs folders."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bi-test-")
        self._saved = (server.BOOKS_DIR, server.OUTPUT_DIR, dict(server.CONFIG))
        server.BOOKS_DIR = os.path.join(self.tmp, "books")
        server.OUTPUT_DIR = os.path.join(self.tmp, "outputs")

    def tearDown(self):
        server.BOOKS_DIR, server.OUTPUT_DIR, config = self._saved
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

    def test_scene_then_characters_then_style(self):
        prompt = server.build_prompt("Oil paint", "The Hero waits.", "Hero: tall woman")
        self.assertLess(prompt.index("Scene from the book"), prompt.index("Characters in this scene"))
        self.assertLess(prompt.index("Characters in this scene"), prompt.index("Visual style"))

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
        name, prompt = server.generate_image(page, "Watercolour, muted colours.", chars)

        sent = self.fake.state["received"][-1]["prompt"]
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
        name, _ = server.generate_image("A quiet forest.", "", "")
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
        self.assertIn(b'id="book"', body)

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


if __name__ == "__main__":
    unittest.main()
