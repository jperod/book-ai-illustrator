"""Offline prompt checks for Book AI Illustrator. No GPU, no ComfyUI, no network.

    python eval/harness.py

Reads every case in eval/local/cases/*.json. That folder is git-ignored because
cases hold real book text. Case format:

    {
      "book_id": "example-book",
      "page_text": "...",
      "scene": "",                       optional: replaces the page in the prompt
      "present": ["Ren", "Ama"],         who is visible (the planner's answer); without it no one is listed
      "expect_narrator": "Ren",          optional: must be the first character listed
      "must_contain": ["- Ama"],         optional: substrings the prompt must include
      "must_not_contain": ["- Koll"], optional: substrings it must not include
      "max_scene_words": 120             optional: cap on the scene section
    }
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import server  # noqa: E402

CASES_DIR = os.path.join(ROOT, "eval", "local", "cases")
SCENE_HEADER = "Scene:"


def build_for_case(case):
    book = server.load_book(case["book_id"])
    return server.build_prompt(
        book.get("visual_style", ""),
        case["page_text"],
        book.get("characters", []),
        book.get("narrator", ""),
        case.get("scene", ""),
        case.get("present"),
    )


def scene_section(prompt):
    start = prompt.index(SCENE_HEADER) + len(SCENE_HEADER)
    end = prompt.find("\n\nVisual style:", start)
    return prompt[start:] if end < 0 else prompt[start:end]


def check(case, prompt):
    failures = []
    if "expect_narrator" in case:
        want = case["expect_narrator"]
        listed = [line for line in prompt.splitlines() if line.startswith("- ")]
        if not listed or not listed[0].startswith(f"- {want} "):
            first = listed[0] if listed else "none"
            failures.append(f"narrator {want!r} is not listed first (first listed: {first})")
    for s in case.get("must_contain", []):
        if s not in prompt:
            failures.append(f"missing: {s!r}")
    for s in case.get("must_not_contain", []):
        if s in prompt:
            failures.append(f"should not appear: {s!r}")
    if "max_scene_words" in case:
        words = len(scene_section(prompt).split())
        if words > case["max_scene_words"]:
            failures.append(f"scene section is {words} words, limit {case['max_scene_words']}")
    return failures


def main():
    paths = sorted(glob.glob(os.path.join(CASES_DIR, "*.json")))
    if not paths:
        print(f"No cases found in {CASES_DIR}.")
        return 0
    failed = 0
    for path in paths:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
        try:
            prompt = build_for_case(case)
        except (OSError, ValueError) as e:
            print(f"FAIL {name}: could not build the prompt: {e}")
            failed += 1
            continue
        problems = check(case, prompt)
        if problems:
            failed += 1
            print(f"FAIL {name}")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(paths) - failed}/{len(paths)} cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
