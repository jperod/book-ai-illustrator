"""Tests for the book bible (bible.py): structure, prompt lines from the book's own fields, and the
structural linter. All names are invented."""

import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bible  # noqa: E402
import prompting  # noqa: E402

REN = {"name": "Ren", "kind": "person", "sex": "male", "age": "teen", "look": "dark hair, plain wool shirt"}
AMA = {"name": "Ama", "kind": "person", "sex": "female", "age": "adult", "look": "grey cloak, a scar on the chin"}
KESTREL = {"name": "Kestrel", "kind": "animal", "species": "wolf", "look": "grey, dark muzzle"}


def problems_at(problems, level):
    return [p["message"] for p in problems if p["level"] == level]


class PromptLineTests(unittest.TestCase):
    def test_line_copies_the_book_fields(self):
        self.assertEqual(
            bible.character_line(bible.normalise_character(REN)), "- Ren (male, teen): dark hair, plain wool shirt"
        )

    def test_narrator_is_marked(self):
        self.assertEqual(
            bible.character_line(bible.normalise_character(REN), narrator="Ren"),
            "- Ren (narrator; male, teen): dark hair, plain wool shirt",
        )

    def test_animals_show_their_species(self):
        line = bible.character_line(bible.normalise_character(KESTREL))
        self.assertEqual(line, "- Kestrel (wolf): grey, dark muzzle")

    def test_old_characters_without_fields_show_only_the_look(self):
        chars = bible.as_characters([{"name": "Hero", "description": "tall woman, red cloak"}])
        self.assertEqual(bible.character_line(chars[0]), "- Hero: tall woman, red cloak")

    def test_count_line(self):
        self.assertEqual(bible.count_line([REN, KESTREL]), "Figures in the picture, exactly 2: Ren, Kestrel.")
        self.assertEqual(bible.count_line([]), "")


class CharacterTextTests(unittest.TestCase):
    def test_schema_one_books_load_with_description_as_look(self):
        book = bible.normalise_book({"characters": [{"name": "Hero", "description": "tall woman"}]})
        self.assertEqual(book["schema_version"], bible.SCHEMA_VERSION)
        self.assertEqual(book["characters"][0]["look"], "tall woman")


class StructuralLintTests(unittest.TestCase):
    def test_a_complete_book_has_no_errors(self):
        book = {"narrator": "Ren", "characters": [REN, AMA, KESTREL]}
        self.assertEqual(problems_at(bible.lint_book(book), "error"), [])

    def test_a_person_needs_sex_and_age(self):
        errors = problems_at(bible.lint_book({"characters": [{"name": "Hero", "look": "tall"}]}), "error")
        self.assertTrue(any("set sex" in e for e in errors))
        self.assertTrue(any("set age" in e for e in errors))

    def test_values_must_be_from_the_allowed_lists(self):
        errors = problems_at(bible.lint_book({"characters": [dict(REN, age="youngish")]}), "error")
        self.assertTrue(any("age must be one of" in e for e in errors))

    def test_name_or_alias_used_twice_is_an_error(self):
        errors = problems_at(bible.lint_book({"characters": [dict(REN, aliases=["Ama"]), AMA]}), "error")
        self.assertTrue(any("'Ama'" in e for e in errors))

    def test_narrator_must_be_a_character(self):
        errors = problems_at(bible.lint_book({"narrator": "Koll", "characters": [REN]}), "error")
        self.assertTrue(any("narrator" in e for e in errors))

    def test_glossary_needs_a_term(self):
        self.assertTrue(problems_at(bible.lint_book({"glossary": [{"meaning": "no term"}]}), "error"))


class GenericCodeTests(unittest.TestCase):
    """The code must stay generic: no character names from any book, and no text-pattern rules."""

    CODE = (
        "server.py",
        "prompting.py",
        "bible.py",
        "index.html",
        "config.py",
        "planner.py",
        "planner_prompts.py",
        "comfy.py",
        "storage.py",
    )

    def read(self, name):
        with open(os.path.join(ROOT, name), encoding="utf-8") as f:
            return f.read()

    def test_no_book_character_names_in_code(self):
        names = set()
        for fn in os.listdir(os.path.join(ROOT, "books")):
            if fn.endswith(".json"):
                with open(os.path.join(ROOT, "books", fn), encoding="utf-8") as f:
                    for c in json.load(f).get("characters", []):
                        names.update([c["name"]] + c.get("aliases", []))
        for code in self.CODE:
            text = self.read(code)
            found = sorted(n for n in names if re.search(r"\b" + re.escape(n) + r"\b", text))
            self.assertEqual(found, [], f"{code} mentions book characters: {found}")

    def test_only_the_book_id_check_is_a_pattern(self):
        for code in (
            "server.py",
            "prompting.py",
            "bible.py",
            "config.py",
            "planner.py",
            "planner_prompts.py",
            "comfy.py",
            "storage.py",
        ):
            compiled = re.findall(r"^(\w+)\s*=\s*re\.compile", self.read(code), re.MULTILINE)
            self.assertLessEqual(set(compiled), {"BOOK_ID_RE"}, f"{code} defines text patterns: {compiled}")


class ShippedBooksTests(unittest.TestCase):
    """The books in the repo must pass the structural checks."""

    def test_shipped_books_lint_clean(self):
        for name in ("example-book.json", "assassins-apprentice.json"):
            with self.subTest(book=name):
                with open(os.path.join(ROOT, "books", name), encoding="utf-8") as f:
                    book = json.load(f)
                self.assertEqual(problems_at(bible.lint_book(book), "error"), [])


class PromptLayoutTests(unittest.TestCase):
    def test_only_the_planners_people_are_listed(self):
        prompt = prompting.build_prompt("", "Ren and Ama talk.", [REN, AMA, KESTREL], present=["Ama"])
        self.assertIn("- Ama (female, adult)", prompt)
        self.assertNotIn("- Ren", prompt)
        self.assertNotIn("- Kestrel", prompt)

    def test_no_planner_answer_means_no_character_list(self):
        prompt = prompting.build_prompt("", "Ren and Ama talk.", [REN, AMA])
        self.assertNotIn("Characters in this scene", prompt)
        self.assertNotIn("Figures in the picture", prompt)

    def test_narrator_first_and_count_line(self):
        prompt = prompting.build_prompt("", "page", [AMA, REN], narrator="Ren", present=["Ama", "Ren"])
        self.assertLess(prompt.index("- Ren (narrator"), prompt.index("- Ama"))
        self.assertIn("Figures in the picture, exactly 2: Ren, Ama.", prompt)

    def test_order_people_setting_scene_style(self):
        prompt = prompting.build_prompt(
            "Ink", "page", [REN], present=["Ren"], scene="Ren walks.", setting="open road, outdoors"
        )
        order = [prompt.index(s) for s in ("- Ren", "Setting: open road", "Scene:\nRen walks.", "Visual style: Ink")]
        self.assertEqual(order, sorted(order))

    def test_scene_wording_has_no_title_words(self):
        prompt = prompting.build_prompt("", "page", [])
        self.assertIn("\nScene:\npage", prompt)
        self.assertNotIn("book", prompt.lower())


if __name__ == "__main__":
    unittest.main()
