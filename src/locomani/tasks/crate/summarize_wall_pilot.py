"""Per-round summary of a wall pilot from its manifest + state traces.

    $VENV_SIM_PY src/locomani/tasks/crate/summarize_wall_pilot.py <pilot_dir>   # needs numpy: not the system python3

Columns: ok, why, stand attempts, close trigger, crate first move (s) vs
episode length, static-frame fraction (50 Hz frames where every joint moved
< 1e-3 rad AND the pelvis < 0.5 mm AND the crate < 0.5 mm -- the frames a
single-frame policy cannot tell apart, i.e. the absorbing-state budget).
"""

import glob
import json
import os
import re
import sys

import numpy as np

d = sys.argv[1]
# the collect stage batches into manifest_<tag>.jsonl (never a single
# manifest.jsonl) so that two pilots writing the same directory can't merge
# into one file; read every batch the same way it does before printing totals.
manifests = sorted(glob.glob(os.path.join(d, "manifest_*.jsonl")))
if not manifests:
    raise FileNotFoundError(f"no manifest_*.jsonl in {d}")
rows = [json.loads(l) for m in manifests for l in open(m) if l.strip()]


def _key(r):
    # The trace is named after the run's --label (wall_s<seed>), which the
    # manifest carries in the video path; wall_s0 is the pre-label default.
    label = os.path.basename(str(r.get("video", ""))).rsplit("_", 1)[0] or "wall_s0"
    return label, r["round"]


# Re-running the same seed into the same directory appends another round-N row
# but OVERWRITES wall_s<seed>_NN.{mp4,npz}: only the LAST row of a (label, round)
# still owns the files on disk. Earlier duplicates get '*' and no trace stats,
# otherwise six stale round-0 rows all read the newest trace (2026-09-11).
last_owner = {_key(r): i for i, r in enumerate(rows)}
dups = sorted({k for k, i in last_owner.items()} & {_key(r) for i, r in enumerate(rows) if last_owner[_key(r)] != i})
print(f"{'rd':>2} {'ok':>3} {'att':>3} {'trig':>8} {'move_s':>6} {'len_s':>5} {'static':>6} {'so':>5} {'yaw':>5}  why")
stat = []
for i, r in enumerate(rows):
    label, rd = _key(r)
    stale = last_owner[(label, rd)] != i
    f = os.path.join(d, "trace", f"{label}_{rd:02d}_state_trace.npz")
    move = length = static = float("nan")
    if os.path.exists(f) and not stale:
        q = np.load(f)["qpos"]
        c = q[:, -7:]
        w, x, y, z = c[:, 3], c[:, 4], c[:, 5], c[:, 6]
        cy = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        cy -= cy[0]
        dxy = np.linalg.norm(c[:, :2] - c[0, :2], axis=1)
        mv = np.where((dxy > 0.005) | (np.abs(cy) > 1.0))[0]
        move = mv[0] / 50 if len(mv) else float("nan")
        length = len(q) / 50
        dq = np.abs(np.diff(q[:, 7:-7], axis=0)).max(1)
        dp = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
        dc = np.linalg.norm(np.diff(c[:, :3], axis=0), axis=1)
        static = float(np.mean((dq < 1e-3) & (dp < 5e-4) & (dc < 5e-4)))
        stat.append(static)
    att = len(r.get("stand_attempts", [])) or 1
    so = r.get("stand_off_meas", float("nan"))
    yaw = r.get("yaw_err_deg", float("nan"))
    print(
        f"{rd:2d}{'*' if stale else ' '}{'OK' if r['ok'] else '--':>3} {att:3d} {str(r.get('close_trigger')):>8} {move:6.1f} {length:5.1f} {static:6.2f} {so:5.2f} {yaw:5.1f}  {r.get('why', '')[:60]}"
    )
print(
    f"\nOK {sum(r['ok'] for r in rows)}/{len(rows)}; static-frame fraction median {np.median(stat):.2f} max {max(stat):.2f}"
    if stat
    else ""
)
if dups:
    print(
        f"WARNING: {len(dups)} (label, round) repeated in the manifest {sorted(f'{l}_{n:02d}' for l, n in dups)}: "
        "the same seed was run again into this directory, so the video/trace of the earlier rows "
        "(marked *) were overwritten and their trace columns are blank. Use a fresh directory or seed per run."
    )
