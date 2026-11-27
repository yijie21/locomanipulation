"""selfmodel_simpledata_20261126, evaluation 3: one label's real episodes (sd_real.py --out DIR, the four tasks) merged into
DIR/data/all.npz with DIR/run.json, the layout xfer's sx_eval.py reads (--xfer DIR), and checked; read-only on the rows.

    $PY sd_real_merge.py DIR --label main|task [--a-rows <bench_rgbd_20261124/scalebfm/rows>]

Merge: the tasks' data/<task>.npz concatenated key by key in TASKS order (data.load's own concatenation). Checks, printed:
  - every episode 0-29 of every task has a row and a log line, reset ids unique;
  - per episode two segments (the pregrasp, chain 0, kind 1; the entry, chain 1, kind 2) unless it fell or had no grasp;
  - task label: the episode's simple_bench row equals the A arm's (--a-rows/A_<task>.jsonl, same episode) in palm_cm and
    in every stage's err_cm and ik_cm (the recorder does not change the episode);
  - the counts: fallen, settled at the end label, waits (main label), label rows on the 0.1 s grid from the command.
Prints "REAL MERGE <label> PASS" or "... FAIL" (exit 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
A_ROWS = Path("/workspace/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm/rows")
TICKS = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--label", required=True, choices=["main", "task"])
    ap.add_argument("--a-rows", type=Path, default=A_ROWS)
    ap.add_argument("--episodes", default="0-29")
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.episodes.split("-"))
    want = set(range(lo, hi + 1))
    ok = True
    zs, n_ep, n_fell, n_nograsp, n_settled, n_seg, waits, offgrid, same_a, diff_a = [], 0, 0, 0, 0, 0, [], 0, 0, []
    for t in a.tasks.split(","):
        rows = [json.loads(s) for s in (a.dir / "rows" / f"{t}.jsonl").read_text().splitlines() if s.strip()]
        logs = [json.loads(s) for s in (a.dir / "log" / f"{t}.jsonl").read_text().splitlines() if s.strip()]
        eps = {int(r["ep"]) for r in rows}
        if eps != want or len(rows) != len(want) or len(logs) != len(want):
            print(f"{t}: rows for episodes {sorted(eps)[:3]}.. ({len(rows)} rows, {len(logs)} log lines), want {len(want)}")
            ok = False
        z = dict(np.load(a.dir / "data" / f"{t}.npz"))
        zs.append(z)
        n_seg += len(z["n_steps"])
        for r, lg in zip(rows, logs):  # both written once per episode, in the same order
            n_ep += 1
            if (str(r["level"]), int(r["index"])) != (str(lg["level"]), int(lg["index"])):
                print(f"{t}: row L{r['level']} e{r['index']} beside log L{lg['level']} e{lg['index']}")
                ok = False
            fell = lg["fell"] is not None
            n_fell += fell
            n_nograsp += r.get("why") == "no grasp"
            rc = lg["reaches"]
            if not fell and r.get("why") != "no grasp" and ([x["kind"] for x in rc] != ["pre", "entry"] or lg["n_segs"] != 2):
                print(f"{t} e{r['ep']}: reaches {[x['kind'] for x in rc]}, segments {lg['n_segs']}")
                ok = False
            for x in rc:
                n_settled += bool(x["settled_end"])
                if x["f_label"] is not None and (x["f_label"] - x["f_cmd"]) % TICKS:
                    offgrid += 1
            waits += [w["waited_s"] for w in lg["waits"]]
        if a.label == "task":
            A = {int(r["ep"]): r for r in (json.loads(s) for s in (a.a_rows / f"A_{t}.jsonl").read_text().splitlines() if s.strip())}
            for r in rows:
                ar = A.get(int(r["ep"]))
                key = lambda x: (x.get("palm_cm"), [(s.get("stage"), s.get("err_cm"), s.get("ik_cm")) for s in x.get("stages", [])])  # noqa: E731
                if ar is not None and key(ar) == key(r):
                    same_a += 1
                else:
                    diff_a.append((t, r["ep"], key(r)[0], None if ar is None else key(ar)[0]))
    keys = set.intersection(*(set(z) for z in zs))
    allz = {k: np.concatenate([z[k] for z in zs]) for k in zs[0] if k in keys}
    dup = len(set(zip(allz["reset"].tolist(), allz["chain"].tolist()))) != len(allz["reset"])
    ok &= not dup and offgrid == 0
    (a.dir / "data").mkdir(exist_ok=True)
    tmp = a.dir / "data" / ".all.tmp.npz"
    np.savez_compressed(tmp, **allz)
    tmp.replace(a.dir / "data" / "all.npz")
    (a.dir / "run.json").write_text(json.dumps(dict(controller="scalebfm", planner="simple_bench (Motor.reach)",
                                                    stance=f"SIMPLE A arm episodes, {a.label} label", delay_f=0,
                                                    label=a.label, tasks=a.tasks.split(","), episodes=a.episodes), indent=1))  # fmt: skip
    print(f"{a.label}: {n_ep} episodes ({n_fell} fell, {n_nograsp} no grasp), {n_seg} segments, {n_settled} settled at the end "
          f"label, duplicate (reset, chain) {dup}, label rows off the 0.1 s grid from the command {offgrid}")  # fmt: skip
    if waits:
        w = np.array(waits)
        print(f"  waits (main label): {len(w)}, median {np.median(w):.2f} s, p90 {np.percentile(w, 90):.2f} s, max {w.max():.2f} s")
    if a.label == "task":
        print(f"  task label rows equal to the A arm's (palm_cm, stages err_cm / ik_cm): {same_a} / {n_ep}; differ: {diff_a[:8]}")
        ok &= same_a == n_ep
    print(f"  merged {len(allz['n_steps'])} segments -> {a.dir / 'data' / 'all.npz'}")
    print(f"REAL MERGE {a.label}", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
