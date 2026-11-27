"""selfmodel_simplecmp2_20261126 V1: the --joint-dur file that gives another arm's durations; read-only on its inputs.

    python3 scmp2_jdur_make.py <the other arm's via dir> <the other arm's rows jsonl (a --joint arm with --correct)> <out json>

Per episode of the rows, {"<task>_L<level>_e<index>": {"plan": the other arm's plan npz (via_<name>.npz's src /
plan_<name>.npz), "speed": the via's speed (null: reach_speed), "dur_s": the duration recorded in the row (correct.dur_s,
4 decimals; scmp2_jdur.py recomputes it live and records both)}}.
"""

import json
import sys
from pathlib import Path

import numpy as np

via_dir, src, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
D = {}
for line in open(src):
    r = json.loads(line)
    name = f"{r['task']}_L{r['level']}_e{r['index']}"
    s = [x for x in r["stages"] if x["stage"] == "pregrasp"][0]
    c = s.get("correct") or {}
    assert c.get("rule") == "joint" and c.get("dur_s") is not None, (name, c.get("rule"))
    assert abs(float(c["dur_s"]) - float(s["joint_dur_s"])) <= 0.006, (name, c["dur_s"], s["joint_dur_s"])
    via = np.load(via_dir / f"via_{name}.npz", allow_pickle=True)
    plan = Path(str(via["src"])) / f"plan_{name}.npz"
    assert plan.exists() and str(via["src"]) == s["joint_src"], (name, plan, s["joint_src"])
    D[name] = dict(plan=str(plan), speed=float(via["speed"]) if "speed" in via.files else None, dur_s=float(c["dur_s"]))
json.dump(D, open(out, "w"), indent=1, sort_keys=True)
d = [x["dur_s"] for x in D.values()]
print(f"{out}: {len(D)} episodes, recorded durations {min(d):.3f}-{max(d):.3f} s")
