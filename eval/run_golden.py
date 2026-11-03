"""Golden-set runner. Generates each golden case with fixed seeds through the running app, and saves prompts,
plans, timings and images for scoring. Standard library only.

    uv run python eval/run_golden.py --arm live --app http://127.0.0.1:8001
    uv run python eval/run_golden.py --arm reuse --from eval/runs/<run-dir> --app http://127.0.0.1:8001
    uv run python eval/run_golden.py --arm reuse --from <run-dir> --wording F1 --only syn-argue --seeds 1,2,3

Arms:
  live    the app as it is configured: the planner (and reviewer) decide who is in the picture and write the scene.
  reuse   the plans (people, scene, setting, glossary terms) recorded in an earlier run (--from, first seed per
          case), so a change to the prompt or the book is compared on the same moments without planner noise.
With --wording KEY, the cast of each case that has arms.KEY.characters (a list of character objects, in
the case file) replaces the book's cast for that request only.

The app must be running. Cases come from eval/golden/*.json (synthetic, committed) and eval/local/golden/*.json
(real book pages, git-ignored). Each run is written to eval/runs/ (git-ignored) with results.jsonl and one PNG
per case and seed. Prompt wording is part of the code (PROMPT_VERSION), so compare runs across versions.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import urllib.request

from common import ROOT, load_cases, load_rows, row_plan

APP = "http://127.0.0.1:8000"  # set from --app in main()


def git_hash():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def call(path, body=None, timeout=900):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        APP + path,
        data=data,
        method="POST" if body is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def recorded_plans(run_dir):
    """Per case: the plan of the first seed in an earlier run."""
    plans = {}
    for row in load_rows(run_dir):
        plans.setdefault(row["case"], row_plan(row))
    return plans


def main():
    global APP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["live", "reuse"], required=True)
    ap.add_argument("--from", dest="from_run", default=None, help="earlier run whose plans the reuse arm draws")
    ap.add_argument("--wording", default=None, help="use the cast in each case's arms.<KEY>.characters")
    ap.add_argument("--only", default=None, help="run only these case ids, comma separated")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, to replace the case's seeds (a sweep)")
    ap.add_argument("--app", default="http://127.0.0.1:8000", help="address of the running app")
    args = ap.parse_args()
    APP = args.app.rstrip("/")
    seeds_override = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    if args.arm == "reuse" and not args.from_run:
        print("the reuse arm needs --from <run directory>")
        return 2
    plans = recorded_plans(os.path.abspath(args.from_run)) if args.arm == "reuse" else {}

    try:
        call("/api/status", timeout=10)
    except OSError:
        print(f"The app is not running at {APP}. Start it with: uv run python server.py")
        return 2

    only = set(args.only.split(",")) if args.only else None
    cases = [c for c in load_cases("eval/golden/*.json", "eval/local/golden/*.json") if only is None or c["id"] in only]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{git_hash()}_{args.arm}" + (f"-{args.wording}" if args.wording else "")
    run_dir = os.path.join(ROOT, "eval", "runs", name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as out:
        for case in cases:
            for seed in seeds_override or case["seeds"]:
                body = {"text": case["page_text"], "book": case["book"], "seed": seed}
                if args.arm == "reuse":
                    body["variant"] = dict(plans.get(case["id"], {}))
                if args.wording:
                    cast = case.get("arms", {}).get(args.wording, {}).get("characters")
                    if isinstance(cast, list):
                        body.setdefault("variant", {}).update(characters=cast)
                result = json.loads(call("/api/generate", body))
                image_name = f"{case['id']}_s{seed}.png"
                with open(os.path.join(run_dir, image_name), "wb") as f:
                    f.write(call(result["image_url"], timeout=60))
                row = {
                    "case": case["id"],
                    "tags": case.get("tags", []),
                    "seed": seed,
                    "arm": args.arm,
                    "wording": args.wording,
                    "image": image_name,
                    "expect": case.get("expect", {}),
                }
                row.update(
                    {k: result.get(k) for k in ("prompt", "plan", "planner", "gpu", "total_seconds", "prompt_version")}
                )
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                plan = result["plan"]
                print(
                    f"{case['id']:<22} seed {seed:<3} present={plan['present']} setting={plan['setting']!r} "
                    f"total={result['total_seconds']}s -> {image_name}",
                    flush=True,
                )

    print(f"\nrun written to {os.path.relpath(run_dir, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
