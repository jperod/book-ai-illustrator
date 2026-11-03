"""Planner benchmark: runs the app's own planner (server.plan_scene, the exact prompt and schema that ship)
on the planner cases and scores who it says is visible. Standard library only; needs Ollama running.

    uv run python eval/bench_planner.py --model qwen3:8b
    uv run python eval/bench_planner.py --model qwen3:8b --review --runs 2 --num-gpu 99

Cases: eval/cases/planner/*.json (synthetic, committed) and eval/local/cases/planner/*.json (git-ignored).
A case is {"names": [...], "narrator": "...", "text": "...", "must_be_present": [...],
"must_not_be_present": [...]}; or {"book": "<id>", "text": ..., "must_be_present": ..., ...}, which takes the
cast, narrator and glossary from books/<id>.json. Results go to eval/local/out/.

Recall = share of must_be_present names the planner listed as visible. Precision = share of listed names
that are not in must_not_be_present. The first run is cold (model load); later runs show steady speed.
"""

import argparse
import json
import os
import statistics
import sys

from common import load_cases as load_json_cases

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import server  # noqa: E402


def load_cases():
    cases = load_json_cases("eval/cases/planner/*.json", "eval/local/cases/planner/*.json")
    for case in cases:
        if case.get("book"):
            book = server.load_book(case["book"])
        else:
            book = server.normalise_book(
                {"characters": [{"name": n} for n in case["names"]], "narrator": case.get("narrator", "")}
            )
        case["characters"], case["glossary"], case["narrator"] = book["characters"], book["glossary"], book["narrator"]
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-gpu", type=int, default=99, help="layers on the GPU; 0 = CPU only")
    ap.add_argument("--runs", type=int, default=2, help="passes over all cases; the first is cold")
    ap.add_argument("--review", action="store_true", help="also run the scene reviewer (a second LLM call)")
    args = ap.parse_args()

    server.load_config()
    server.CONFIG.update(
        llm_enabled=True,
        llm_model=args.model,
        llm_num_gpu=args.num_gpu,
        llm_keep_alive="10m",
        planner_validate=args.review,
    )
    cases = load_cases()
    rows = []
    for run in range(args.runs):
        for case in cases:
            try:
                plan = server.plan_scene(case["text"], case["characters"], case["narrator"], case["glossary"])
            except server.PlannerError:
                plan = None
            present = plan["present"] if plan else []
            must, must_not = set(case.get("must_be_present", [])), set(case.get("must_not_be_present", []))
            row = {
                "run": run,
                "case": case["_id"],
                "ok": plan is not None,
                "present": present,
                "missing": sorted(must - set(present)),
                "extra": sorted(must_not & set(present)),
                "seconds": plan["seconds"] if plan else None,
                "setting": plan["setting"] if plan else "",
                "scene": plan["scene"] if plan else "",
                "review": plan["review"] if plan else None,
            }
            rows.append(row)
            print(
                f"run {run} {case['_id']:<26} present={present} missing={row['missing']} extra={row['extra']} "
                f"{row['seconds']}s setting={row['setting']!r}",
                flush=True,
            )
    server.llm_release()

    must_total = sum(len(c.get("must_be_present", [])) for c in cases) * args.runs
    missing = sum(len(r["missing"]) for r in rows)
    listed = sum(len(r["present"]) for r in rows)
    extra = sum(len(r["extra"]) for r in rows)
    warm = [r["seconds"] for r in rows if r["run"] > 0 and r["seconds"] is not None] or [
        r["seconds"] for r in rows if r["seconds"] is not None
    ]
    summary = {
        "model": args.model,
        "review": args.review,
        "cases": len(cases),
        "runs": args.runs,
        "recall": round(1 - missing / must_total, 3) if must_total else None,
        "precision": round(1 - extra / listed, 3) if listed else None,
        "failed_replies": sum(1 for r in rows if not r["ok"]),
        "median_seconds_warm": statistics.median(warm) if warm else None,
    }
    print(json.dumps(summary, indent=2))
    out_dir = os.path.join(ROOT, "eval", "local", "out")
    os.makedirs(out_dir, exist_ok=True)
    tag = "".join(ch if ch.isalnum() else "-" for ch in args.model).strip("-")
    path = os.path.join(out_dir, f"bench-{tag}-shipped{'-review' if args.review else ''}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"written to {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
