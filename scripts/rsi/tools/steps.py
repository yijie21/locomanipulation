"""TODO 32.6: the rounds of a directory as a table -- per round: ok, the goal's why or the planner's refusal, and every
step's name, time span and scalar fields -- or, with --vs, the rounds whose outcome differs between two directories
(an incumbent's and a candidate's) side by side. What 41 of 44 Reviewers wrote by hand (`summ.py`, `steps.py`).

    python3 scripts/rsi/tools/steps.py <rounds dir> [--vs <rounds dir>] [--task pick_Mug_2] [--labels a,b]
        [--failed] [--step walk,pre] [--brief]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SKIP = {"step", "ok", "why", "t0", "t1"}


def rounds(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if not p.name.startswith("_")}


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, list) and len(v) <= 4 and all(isinstance(x, (int, float)) for x in v):
        return "[" + ",".join(fmt(float(x)) for x in v) + "]"
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= 40 else s[:37] + "..."


def why(r: dict) -> str:
    return r.get("plan_why") or r.get("why") or r.get("error") or ""


def step_lines(r: dict, only: set | None) -> list[str]:
    out = []
    for s in r.get("steps") or []:
        if only and s["step"] not in only:
            continue
        mark = "" if s.get("ok", True) else " FAILED: " + str(s.get("why", ""))[:80]
        fields = " ".join(f"{k}={fmt(v)}" for k, v in s.items() if k not in SKIP)
        out.append(f"    {s['step']:<10s} {s.get('t0', 0):6.2f}-{s.get('t1', 0):6.2f}s {fields}{mark}")
    return out


def head(r: dict) -> str:
    flags = "".join(f" {k}" for k in ("fell", "dropped") if r.get(k))
    t = r.get("t_end")
    return f"{'OK  ' if r.get('ok') else 'FAIL'} t_end={t if t is not None else '-'}{flags}  {why(r)[:110]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--vs", type=Path, help="a second rounds directory: show the rounds whose outcome differs")
    ap.add_argument("--task", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--failed", action="store_true", help="only the failed rounds (of the first directory)")
    ap.add_argument("--step", default="", help="only these steps (comma-separated names)")
    ap.add_argument("--brief", action="store_true", help="one line per round, no steps")
    a = ap.parse_args(argv)
    A = rounds(a.dir)
    B = rounds(a.vs) if a.vs else None
    only = set(a.step.split(",")) if a.step else None
    labels = [x for x in a.labels.split(",") if x] or sorted(A)
    labels = [x for x in labels if x in A and (not a.task or x.rsplit("_s", 1)[0] == a.task)]
    if B is not None:
        labels = [x for x in labels if x in B and bool(A[x].get("ok")) != bool(B[x].get("ok"))]
        print(f"{len(labels)} rounds differ: {sum(bool(B[x].get('ok')) for x in labels)} only in --vs, "
              f"{sum(bool(A[x].get('ok')) for x in labels)} only in {a.dir.name}")  # fmt: skip
    if a.failed:
        labels = [x for x in labels if not A[x].get("ok")]
    n_ok = sum(bool(A[x].get("ok")) for x in labels)
    print(f"{a.dir}: {n_ok}/{len(labels)} ok")
    for x in labels:
        print(f"{x}  {head(A[x])}")
        if not a.brief:
            print("\n".join(step_lines(A[x], only)))
        if B is not None:
            print(f"  --vs  {head(B[x])}")
            if not a.brief:
                print("\n".join(step_lines(B[x], only)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
