"""Tests for pure helpers around the planner: the setting line, saved plans, and the picture style choice.
The planner and reviewer calls themselves are tested in test_app.py with a fake LLM."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import server  # noqa: E402


class SettingLineTests(unittest.TestCase):
    def test_outdoors_daytime(self):
        self.assertEqual(
            server.setting_line({"place": "open dirt road", "indoors": "no", "time": "daytime"}),
            "open dirt road, outdoors, daytime",
        )

    def test_unknown_parts_are_left_out(self):
        self.assertEqual(
            server.setting_line({"place": "stone kitchen", "indoors": "yes", "time": "unknown"}),
            "stone kitchen, indoors",
        )
        self.assertEqual(server.setting_line({"place": "unknown", "indoors": "unknown", "time": "unknown"}), "")

    def test_missing_setting_gives_nothing(self):
        self.assertEqual(server.setting_line(None), "")


class CastTextTests(unittest.TestCase):
    def test_whole_cast_and_terms_from_the_book_fields(self):
        chars = server.normalise_book(
            {
                "characters": [
                    {"name": "Ren", "kind": "person", "sex": "male", "age": "teen", "aliases": ["the lad"]},
                    {"name": "Kestrel", "kind": "animal", "species": "wolf"},
                ]
            }
        )["characters"]
        text = server.cast_text(chars, "Ren", [{"term": "Wending"}])
        self.assertIn('- Ren (narrator, "I"; male, teen; also called the lad)', text)
        self.assertIn("- Kestrel (wolf)", text)
        self.assertIn("Glossary terms: Wending", text)


class PlanRequestTests(unittest.TestCase):
    def test_saved_plan_is_checked(self):
        plan = server.plan_from_request({"scene": " Ren rides. ", "present": ["Ren"], "setting": "road"})
        self.assertEqual((plan["scene"], plan["present"]), ("Ren rides.", ["Ren"]))

    def test_plan_without_a_scene_keeps_people_and_setting(self):
        plan = server.plan_from_request({"present": ["Ren"], "setting": "road"})
        self.assertEqual((plan["scene"], plan["present"], plan["setting"]), ("", ["Ren"], "road"))

    def test_non_plan_is_refused(self):
        self.assertIsNone(server.plan_from_request("not a plan"))


if __name__ == "__main__":
    unittest.main()
