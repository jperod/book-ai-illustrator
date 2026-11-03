"""Shared helpers for the eval tools: loading case files and reading a run folder. Standard library only."""

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cases(*patterns):
    """Every JSON case matching the patterns (relative to the project folder), sorted, each tagged with
    "_id" (its file name without .json) and "_file"."""
    cases = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            with open(path, encoding="utf-8") as f:
                case = json.load(f)
            case["_id"] = os.path.splitext(os.path.basename(path))[0]
            case["_file"] = os.path.relpath(path, ROOT).replace("\\", "/")
            cases.append(case)
    return cases


def load_rows(run_dir):
    """The rows of a run's results.jsonl (written by run_golden.py)."""
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_plan(row):
    """The plan a row was drawn with. Rows written before the plan field kept it under "planner"."""
    return row.get("plan") or row.get("planner") or {}
