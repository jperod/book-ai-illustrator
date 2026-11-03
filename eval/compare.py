"""Compare golden runs: pass rates per criterion, hard pass, clean pass and full pass, per tag, and a paired diff.

    uv run python eval/compare.py eval/runs/<run-a> [eval/runs/<run-b> ...]
    uv run python eval/compare.py eval/runs/<run-a> eval/runs/<run-b> --paired
    uv run python eval/compare.py eval/runs/<sweep-run> --by-seed

Each run folder holds results.jsonl (written by run_golden.py) and scores.json, which maps
"<case>|<seed>" to {"C": 0|1, "S": 0|1, "A": 0|1, "X": 0|1, "St": 0|1, "L": 0|1, "E": 0|1, "T": 0|1}.
Criteria: C count of people correct, S sex and age band correct, A action and pose correct,
X no absent character drawn, St style matches, L setting matches the page, E no unlisted person drawn,
T no lettering or caption in the picture. L, E and T are optional: older runs without them still compare.
Hard pass = C and S and X. Clean pass = hard pass and E and T. Full pass = all eight that are scored.

--paired compares two runs image by image (same case and seed): how many images improved, regressed or
stayed the same, for the hard pass and for each criterion. Use it for any change: a paired diff is
much more sensitive than two pass rates, because the seed and the page are held fixed.

--by-seed lists, for each seed, how many images were scored and how many showed lettering. Use it on a
seed sweep to see whether lettering follows the seed.
"""

import json
import os
import sys

from common import load_rows

BASE = ("C", "S", "A", "X", "St")
OPTIONAL = ("L", "E", "T")


def load_run(run_dir):
    rows = load_rows(run_dir)
    scores = {}
    path = os.path.join(run_dir, "scores.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            scores = json.load(f)
    return rows, scores


def key(row):
    return f"{row['case']}|{row['seed']}"


def is_hard(s):
    return s.get("C") == 1 and s.get("S") == 1 and s.get("X") == 1


def is_clean(s):
    return is_hard(s) and s.get("E") == 1 and s.get("T") == 1


def summarize(rows, scores):
    scored = [(r, scores[key(r)]) for r in rows if key(r) in scores]
    if not scored:
        return None
    n = len(scored)

    def rate(check, group=scored):
        return sum(1 for _, s in group if check(s)) / len(group)

    summary = {"n": n, "unscored": len(rows) - n}
    criteria = list(BASE) + [c for c in OPTIONAL if any(c in s for _, s in scored)]
    summary["criteria"] = criteria
    for c in criteria:
        summary[c] = rate(lambda s, c=c: s.get(c) == 1)
    summary["hard"] = rate(lambda s: is_hard(s))
    with_clean = [(r, s) for r, s in scored if "E" in s and "T" in s]
    summary["clean"] = rate(lambda s: is_clean(s), with_clean) if with_clean else None
    summary["full"] = rate(lambda s: all(s.get(c) == 1 for c in criteria))
    by_tag = {}
    for r, s in scored:
        for tag in r.get("tags", []):
            by_tag.setdefault(tag, []).append(s)
    summary["tags"] = {tag: rate(is_hard, [(None, s) for s in group]) for tag, group in by_tag.items()}
    return summary


def print_table(runs):
    columns = list(BASE) + list(OPTIONAL)
    present = [c for c in columns if any(s and c in s["criteria"] for s in runs.values() if s)]
    head = f"{'run':<44}{'n':>4}" + "".join(f"{c:>6}" for c in present) + f"{'hard':>7}{'clean':>7}{'full':>7}"
    print(head)
    for name, s in runs.items():
        if s is None:
            print(f"{name:<44}  not scored yet")
            continue
        line = f"{name:<44}{s['n']:>4}" + "".join(
            f"{s[c]:>6.0%}" if c in s["criteria"] else f"{'-':>6}" for c in present
        )
        clean = f"{s['clean']:>7.0%}" if s["clean"] is not None else f"{'-':>7}"
        print(line + f"{s['hard']:>7.0%}" + clean + f"{s['full']:>7.0%}")
        if s["unscored"]:
            print(f"    {s['unscored']} image(s) not scored yet")
        for tag, rate in sorted(s["tags"].items()):
            print(f"    tag {tag:<14} hard pass {rate:.0%}")


def paired(rows_a, scores_a, rows_b, scores_b):
    """Image by image, same case and seed. Prints improved / regressed / same per criterion."""
    b_by_key = {key(r): r for r in rows_b}
    pairs = [
        (scores_a[key(r)], scores_b[key(r)])
        for r in rows_a
        if key(r) in scores_a and key(r) in scores_b and key(r) in b_by_key
    ]
    if not pairs:
        print("no images scored in both runs")
        return
    print(f"paired images: {len(pairs)}")
    checks = [("hard", is_hard)] + [
        (c, lambda s, c=c: s.get(c) == 1) for c in BASE + OPTIONAL if any(c in a and c in b for a, b in pairs)
    ]
    for name, check in checks:
        better = sum(1 for a, b in pairs if not check(a) and check(b))
        worse = sum(1 for a, b in pairs if check(a) and not check(b))
        same = len(pairs) - better - worse
        print(f"  {name:<6} improved {better:>3}   regressed {worse:>3}   same {same:>3}")


def by_seed(rows, scores):
    seeds = {}
    for r in rows:
        if key(r) not in scores:
            continue
        s = scores[key(r)]
        bucket = seeds.setdefault(r["seed"], {"n": 0, "text": 0, "hard": 0})
        bucket["n"] += 1
        bucket["text"] += 1 if s.get("T") == 0 else 0
        bucket["hard"] += 1 if is_hard(s) else 0
    print(f"{'seed':>6}{'images':>8}{'lettering':>11}{'hard pass':>11}")
    for seed in sorted(seeds):
        b = seeds[seed]
        print(f"{seed:>6}{b['n']:>8}{b['text']:>11}{b['hard']:>11}")
    total = sum(b["n"] for b in seeds.values())
    text = sum(b["text"] for b in seeds.values())
    if total:
        print(f"{'all':>6}{total:>8}{text:>11}  lettering rate {text / total:.1%}")


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__)
        return 2
    if "--by-seed" in flags:
        rows, scores = load_run(paths[0])
        by_seed(rows, scores)
        return 0
    if "--paired" in flags:
        if len(paths) != 2:
            print("--paired needs exactly two run folders")
            return 2
        rows_a, scores_a = load_run(paths[0])
        rows_b, scores_b = load_run(paths[1])
        print(f"A: {os.path.basename(os.path.normpath(paths[0]))}")
        print(f"B: {os.path.basename(os.path.normpath(paths[1]))}")
        paired(rows_a, scores_a, rows_b, scores_b)
        return 0
    runs = {}
    for run_dir in paths:
        rows, scores = load_run(run_dir)
        runs[os.path.basename(os.path.normpath(run_dir))] = summarize(rows, scores)
    print_table(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
