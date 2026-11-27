"""simple_relook_20261126: with --relook off, does simple_bench.py still give the rows and films it gave? New rows and
films against an old run's, episode by episode (simple_sonic_vrref_20261126's repro_cmp.py, any arm and run).

    python3 -I idcmp.py <new dir> <old dir> <arm> <episodes, e.g. 3,24> [filmref]
A row is the same when every key and value is equal after dropping wall_s (the wall clock) and the record's option key
bench.options.sonic_vr_frame (added 2026-11-26; absent from older rows, "robot" in new ones). A film is the same when every
array is bitwise equal. "filmref": the new run had --film-ref and the old did not (each stage's S3, t_cmd, W, J and the
film's ref dropped from the new side).
"""

import json
import sys
from pathlib import Path

import numpy as np

NEW, OLD, ARM, EPS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], [int(e) for e in sys.argv[4].split(",")]
FILMREF = len(sys.argv) > 5 and sys.argv[5] == "filmref"
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")


def rows(d: Path, task: str) -> dict:
    f = d / "rows" / f"{ARM}_{task}.jsonl"
    return {r["ep"]: r for r in map(json.loads, open(f))} if f.exists() else {}


def diffs(a, b, path="") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in a else 'old'}")
            else:
                out += diffs(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diffs(x, y, f"{path}[{i}]")
        return out
    return [] if (a == b and type(a) is type(b)) else [f"{path}: {a!r} vs {b!r}"[:200]]


same_rows = same_films = n = 0
for t in TASKS:
    new, old = rows(NEW, t), rows(OLD, t)
    for e in EPS:
        if e not in new or e not in old:
            continue
        n += 1
        a, b = json.loads(json.dumps(new[e])), json.loads(json.dumps(old[e]))
        for r in (a, b):
            r.pop("wall_s", None)
            r.get("bench", {}).get("options", {}).pop("sonic_vr_frame", None)
        if FILMREF:
            for st in a.get("stages", []):
                for k in ("S3", "t_cmd", "W", "J"):
                    st.pop(k, None)
        d = diffs(a, b)
        name = f"{t}_L{e // 10}_e{e % 10}"
        za = np.load(NEW / f"films_{ARM}" / f"{name}_perception_direct.npz", allow_pickle=False)
        zb = np.load(OLD / f"films_{ARM}" / f"{name}_perception_direct.npz", allow_pickle=False)
        fd = [k for k in sorted((set(za.files) - ({"ref"} if FILMREF else set())) | set(zb.files))
              if k not in za.files or k not in zb.files or za[k].shape != zb[k].shape
              or za[k].tobytes() != zb[k].tobytes()]  # fmt: skip
        same_rows += not d
        same_films += not fd
        print(f"{ARM} {name}: row {'same' if not d else 'DIFFERS'} (success_45 {a.get('success_45')}/{b.get('success_45')}, "
              f"t_success {a.get('t_success')}/{b.get('t_success')}), film {'same' if not fd else 'DIFFERS ' + ','.join(fd)} "
              f"({za['qpos'].shape[0]} frames)")  # fmt: skip
        for x in d[:8]:
            print("   ", x)
print(f"{ARM}: rows the same {same_rows}/{n}, films the same {same_films}/{n}")
