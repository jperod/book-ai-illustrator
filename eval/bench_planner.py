"""Benchmark a local planner model through Ollama. Standard library only.

    uv run python eval/bench_planner.py --model qwen3:4b-instruct-2507-q4_K_M --mode checks --num-gpu 0 --threads 8 --candidates

Modes:
  full        {"scene", "present": [names]}                      (first design)
  present     {"present": [names]}                               (fastest)
  checks      {"checks": [{"name", "visible_in_moment"}], "scene"}  (decide each name, then the scene)
  checks_only {"checks": [{"name", "visible_in_moment"}]}

--candidates offers only names that appear in the page, plus the narrator when the narration is
first person. This mirrors the app, where name matching runs before the planner.

Cases: eval/cases/planner/*.json (synthetic, committed) and eval/local/cases/planner/*.json
(real book pages, git-ignored). A case is {"names", "narrator", "text", "must_be_present",
"must_not_be_present"}, or {"book": "<id>", "text", ...} to take names and narrator from books/.
Each run writes one JSON report to eval/local/out/ (git-ignored).
"""
import argparse
import difflib
import glob
import json
import os
import re
import statistics
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA = "http://127.0.0.1:11434"
MODES = ("full", "present", "checks", "checks_only", "checks2")

QUOTE_RE = re.compile(r'"[^"]*"|“[^”]*”')
FIRST_PERSON_RE = re.compile(r"\bI\b|\b(?i:me|my|mine|myself)\b")
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")

SYSTEM = {
    "full": (
        "You plan one illustration for a page of a novel. Choose ONE moment from the page: the most visual one. "
        'Write "scene": at most 50 words of concrete visible detail (poses, setting, which hand holds each prop). '
        "No thoughts, backstory or feelings. Refer to people by name, never as \"I\". "
        'Write "present": the names of characters who are physically visible in that moment. '
        "A character who is only remembered, mentioned, avoided, absent or dead is NOT present. "
        "Use only names from the allowed list."
    ),
    "present": (
        "You plan one illustration for a page of a novel. Choose ONE moment from the page: the most visual one. "
        'Write "present": the names of characters who are physically visible in that moment. '
        "A character who is only remembered, mentioned, avoided, absent or dead is NOT present. "
        "Use only names from the allowed list."
    ),
    "checks": (
        "You plan one illustration for a page of a novel. Choose ONE moment from the page: the most visual one. "
        'For EVERY allowed name, write a "checks" entry with "visible_in_moment": true only if that person is '
        "physically in the picture in that moment. Someone who is only remembered, mentioned, avoided, absent, "
        "dead or spoken about is false. Then write \"scene\": at most 50 words of concrete visible detail "
        "(poses, setting, which hand holds each prop). No thoughts or backstory. Refer to people by name, never "
        'as "I". Only the visible people appear in the scene.'
    ),
    "checks_only": (
        "You plan one illustration for a page of a novel. Choose ONE moment from the page: the most visual one. "
        'For EVERY allowed name, write a "checks" entry with "visible_in_moment": true only if that person is '
        "physically in the picture in that moment. Someone who is only remembered, mentioned, avoided, absent, "
        "dead or spoken about is false."
    ),
}
SYSTEM["checks2"] = SYSTEM["checks"] + (
    " Copy every name exactly as written in the allowed list. A person who is elsewhere, avoids the narrator, "
    "left, or is only spoken about is visible_in_moment false, even if the page mentions them in the same passage."
)

# A second synthetic example for checks2: one person is remembered, one has gone elsewhere.
FEW_SHOT_USER_V2 = (
    'Allowed names: Mara (narrator, "I"), Orrin, Bel, Corran\n'
    "Page:\nI waited at the gate while Orrin argued with the guard. Bel had left us two days ago, "
    "and I still missed her voice. Corran had ridden off to the coast at dawn. "
    "Orrin finally stepped aside, laughing at something the guard had said."
)

# One synthetic example with invented names, shown before the real request.
FEW_SHOT_USER = (
    'Allowed names: Mara (narrator, "I"), Orrin, Bel\n'
    "Page:\nI waited at the gate while Orrin argued with the guard. Bel had left us two days ago, "
    "and I still missed her voice. Orrin finally stepped aside, laughing at something the guard had said."
)
FEW_SHOT = {
    "full": json.dumps({
        "scene": "Mara waits by a stone gate. Orrin, arms folded, argues with a guard in a grey tabard, "
                 "then steps aside, laughing.",
        "present": ["Mara", "Orrin"],
    }),
    "present": json.dumps({"present": ["Mara", "Orrin"]}),
    "checks": json.dumps({
        "checks": [
            {"name": "Mara", "visible_in_moment": True},
            {"name": "Orrin", "visible_in_moment": True},
            {"name": "Bel", "visible_in_moment": False},
        ],
        "scene": "Mara waits by a stone gate. Orrin, arms folded, argues with a guard in a grey tabard, "
                 "then steps aside, laughing.",
    }),
    "checks_only": json.dumps({"checks": [
        {"name": "Mara", "visible_in_moment": True},
        {"name": "Orrin", "visible_in_moment": True},
        {"name": "Bel", "visible_in_moment": False},
    ]}),
    "checks2": json.dumps({
        "checks": [
            {"name": "Mara", "visible_in_moment": True},
            {"name": "Orrin", "visible_in_moment": True},
            {"name": "Bel", "visible_in_moment": False},
            {"name": "Corran", "visible_in_moment": False},
        ],
        "scene": "Mara waits by a stone gate. Orrin, arms folded, argues with a guard in a grey tabard, "
                 "then steps aside, laughing.",
    }),
}

CHECKS_SCHEMA = {"type": "array", "items": {
    "type": "object",
    "properties": {"name": {"type": "string"}, "visible_in_moment": {"type": "boolean"}},
    "required": ["name", "visible_in_moment"],
}}


def schema_for(mode):
    present = {"type": "array", "items": {"type": "string"}}
    if mode == "full":
        return {"type": "object", "properties": {"scene": {"type": "string"}, "present": present},
                "required": ["scene", "present"]}
    if mode == "present":
        return {"type": "object", "properties": {"present": present}, "required": ["present"]}
    if mode in ("checks", "checks2"):
        return {"type": "object", "properties": {"checks": CHECKS_SCHEMA, "scene": {"type": "string"}},
                "required": ["checks", "scene"]}
    return {"type": "object", "properties": {"checks": CHECKS_SCHEMA}, "required": ["checks"]}


def candidates_for(case):
    """Names the planner may choose from: names found in the page, plus the narrator when the
    narration is first person. Mirrors the app, where name matching runs before the planner."""
    text = case["text"]
    found = [n for n in case["names"] if re.search(r"\b" + re.escape(n) + r"\b", text)]
    narrator = case.get("narrator", "")
    if narrator and narrator not in found and FIRST_PERSON_RE.search(QUOTE_RE.sub(" ", text)):
        found.append(narrator)
    return found


def build_messages(mode, case):
    narrator = case.get("narrator", "")

    def tag(name):
        return f'{name} (narrator, "I")' if narrator and name == narrator else name

    user = "Allowed names: " + ", ".join(tag(n) for n in case["names"]) + "\nPage:\n" + case["text"]
    return [
        {"role": "system", "content": SYSTEM[mode]},
        {"role": "user", "content": FEW_SHOT_USER_V2 if mode == "checks2" else FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT[mode]},
        {"role": "user", "content": user},
    ]


def load_cases():
    cases = []
    for pattern in ("eval/cases/planner/*.json", "eval/local/cases/planner/*.json"):
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            with open(path, encoding="utf-8") as f:
                case = json.load(f)
            if "book" in case:
                with open(os.path.join(ROOT, "books", case["book"] + ".json"), encoding="utf-8") as f:
                    book = json.load(f)
                case.setdefault("names", [c["name"] for c in book.get("characters", [])])
                case.setdefault("narrator", book.get("narrator", ""))
            case["_file"] = os.path.relpath(path, ROOT).replace("\\", "/")
            cases.append(case)
    return cases


def post(path, payload, timeout):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def unload(model):
    post("/api/generate", {"model": model, "keep_alive": 0}, timeout=120)


def chat(model, messages, schema, options, keep_alive="30m", timeout=900, think=None):
    payload = {"model": model, "stream": False, "keep_alive": keep_alive,
               "messages": messages, "format": schema, "options": options}
    if think is not None:
        payload["think"] = think  # Qwen3.5 thinks by default; off keeps replies short and fast
    t0 = time.perf_counter()
    data = post("/api/chat", payload, timeout)
    return data, time.perf_counter() - t0


def residency():
    with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=10) as r:
        return json.loads(r.read())


def score(case, reply, mode):
    invalid = {"valid": False, "picked": [], "unknown": 0, "precision": 0.0, "recall": 0.0,
               "forbidden": [], "scene_words": 0, "uses_I": False}
    try:
        obj = json.loads(FENCE_RE.sub("", reply.strip()))
    except ValueError:
        return invalid
    if not isinstance(obj, dict):
        return invalid

    # Turn the reply into (name, visible) pairs. Old modes only list names, so every listed name is visible.
    if mode.startswith("checks"):
        checks = obj.get("checks")
        if not isinstance(checks, list) or not all(
            isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("visible_in_moment"), bool)
            for c in checks
        ):
            return invalid
        pairs = [(c["name"], c["visible_in_moment"]) for c in checks]
    else:
        raw = obj.get("present")
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            return invalid
        pairs = [(x, True) for x in raw]

    scene = obj.get("scene", "")
    if mode in ("full", "checks", "checks2"):
        if not isinstance(scene, str):
            return invalid
    else:
        scene = ""

    known = {n.lower(): n for n in case["names"]}
    picked, unknown = set(), 0
    for name, visible in pairs:
        key = name.strip().lower()
        if key not in known:
            # Close spellings ("Orr-irin") count as the allowed name, as the app will do.
            close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.75)
            key = close[0] if close else key
        if key not in known:
            unknown += 1
        elif visible:
            picked.add(known[key])
    must = set(case["must_be_present"])
    forbid = set(case["must_not_be_present"])
    tp = len(picked & must)
    fp = len(picked - must)
    fn = len(must - picked)
    return {
        "valid": True,
        "picked": sorted(picked),
        "unknown": unknown,
        "precision": round(tp / (tp + fp), 2) if tp + fp else 1.0,
        "recall": round(tp / (tp + fn), 2) if tp + fn else 1.0,
        "forbidden": sorted(picked & forbid),
        "scene_words": len(scene.split()),
        "uses_I": bool(re.search(r"\bI\b", scene)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=MODES, default="full")
    ap.add_argument("--num-gpu", type=int, default=0, help="layers on the GPU; 0 = CPU only")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads; 0 = Ollama default")
    ap.add_argument("--runs", type=int, default=2, help="passes over all cases; the first is cold")
    ap.add_argument("--num-predict", type=int, default=260)
    ap.add_argument("--think", choices=["on", "off"], default=None, help="omit to leave the model's default")
    ap.add_argument("--candidates", action="store_true",
                    help="offer only names found in the page (plus the narrator if first person)")
    args = ap.parse_args()
    think = None if args.think is None else (args.think == "on")

    cases = load_cases()
    if not cases:
        print("No cases found.")
        return 1

    options = {"temperature": 0, "num_ctx": 2048, "num_predict": args.num_predict, "num_gpu": args.num_gpu}
    if args.threads:
        options["num_thread"] = args.threads
    schema = schema_for(args.mode)

    # Clear every resident model first, so the target starts cold and RAM is not shared.
    for m in residency().get("models", []):
        unload(m["name"])
    rows = []
    placement = None
    for run in range(args.runs):
        for i, case in enumerate(cases):
            if args.candidates:
                case = dict(case, names=candidates_for(case))
            data, wall = chat(args.model, build_messages(args.mode, case), schema, options, think=think)
            reply = data["message"]["content"]
            if placement is None:
                placement = residency().get("models", [])
            row = score(case, reply, args.mode)
            row["offered"] = case["names"]
            row.update({
                "case": case["_file"], "run": run, "cold": run == 0 and i == 0,
                "wall_s": round(wall, 2),
                "load_s": round(data.get("load_duration", 0) / 1e9, 2),
                "prompt_s": round(data.get("prompt_eval_duration", 0) / 1e9, 2),
                "gen_s": round(data.get("eval_duration", 0) / 1e9, 2),
                "prompt_tokens": data.get("prompt_eval_count"),
                "out_tokens": data.get("eval_count"),
                "reply": reply,
            })
            rows.append(row)
            ok = row["valid"] and row["recall"] == 1.0 and row["precision"] == 1.0 and not row["forbidden"]
            print(f"{'PASS' if ok else 'miss'} {case['_file']:<55} run{run} "
                  f"valid={row['valid']} P={row['precision']} R={row['recall']} "
                  f"forbidden={row['forbidden']} wall={row['wall_s']}s out={row['out_tokens']}")

    warm = [r for r in rows if not r["cold"]]
    summary = {
        "valid_rate": round(sum(r["valid"] for r in rows) / len(rows), 2),
        "pass_rate": round(sum(
            1 for r in rows if r["valid"] and r["recall"] == 1.0 and r["precision"] == 1.0 and not r["forbidden"]
        ) / len(rows), 2),
        "mean_precision": round(statistics.mean(r["precision"] for r in rows), 2),
        "mean_recall": round(statistics.mean(r["recall"] for r in rows), 2),
        "forbidden_total": sum(len(r["forbidden"]) for r in rows),
        "cold_wall_s": rows[0]["wall_s"],
        "warm_median_wall_s": round(statistics.median(r["wall_s"] for r in warm), 2) if warm else None,
        "warm_median_out_tokens": statistics.median(r["out_tokens"] for r in warm) if warm else None,
        "scene_words_median": statistics.median(r["scene_words"] for r in rows)
        if args.mode in ("full", "checks", "checks2") else None,
    }
    report = {"model": args.model, "mode": args.mode, "candidates": args.candidates, "options": options,
              "think": think, "placement": placement, "summary": summary, "rows": rows}

    out_dir = os.path.join(ROOT, "eval", "local", "out")
    os.makedirs(out_dir, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
    think_tag = "" if think is None else ("-think" if think else "-nothink")
    cand_tag = "-cand" if args.candidates else ""
    name = f"bench-{tag}-{args.mode}-gpu{args.num_gpu}-t{args.threads or 'auto'}{think_tag}{cand_tag}.json"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nSUMMARY", json.dumps(summary))
    for m in placement or []:
        print(f"placement: size={m.get('size', 0) / 1e9:.2f} GB  vram={m.get('size_vram', 0) / 1e9:.2f} GB")
    print(f"written: eval/local/out/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
