"""TODO 33.1.4 (plan section 1.5, D22; user 2026-10-23 "继续做 33.1.4，先做预测器"): the reach samples a body model
learns from -- one row per reach step of every recorded round that has a 50 Hz trace. The command is what the skill
asked the controller for (the wrist reference at the end of the reach, in the heading frame of the pelvis where the
reach started); the outcome is what the skill measured when the step ended (`jaw_err_cm`, `attitude_err_deg`, the IK
residual `ik_err_cm`) and what the trace shows (the wrist against its reference, the pelvis drift during the reach).
The 2026-10-23 failure analysis traced 12-13 of the incumbent's 19 failures to "the reference is reachable, the hand
landed 3-19 cm away": this table is where that gap becomes predictable.

    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/body_data.py [--roots GLOB ...] [--out FILE] [--procs N]

Rows are written to one .npz (`outputs/scene_rsi_20261016/body/reach_samples.npz`: a structured array) with a JSON
summary beside it. Records without a trace or a reach step are skipped and counted. Frames: `robot` in a trace is the
pelvis pose (xyz + wxyz), `ref_wrists` / `wrists` are world positions (failure_class.py reads them the same way); the
heading frame keeps the pelvis yaw only, x forward, y left, z up, origin at the pelvis at the reach's first tick. The
left hand's y and the target's lateral offset are mirrored (`y_mirror`) so both hands share one model."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOTS = ["/workspace/code/rsi-scene/*/outputs/scene_rsi_20261016/grasp_atomic/diag/*"]
OUT = REPO / "outputs" / "scene_rsi_20261016" / "body" / "reach_samples.npz"
REACH_STEPS = ("pre", "pregrasp", "entry", "insert", "descend", "hand_entry", "raise")
FIRST = ("pre", "pregrasp")  # the step that carries the hand to the pre-grasp: the predictor's main target
FIELDS = [
    ("sandbox", "U40"), ("label", "U48"), ("task", "U32"), ("skills", "U16"), ("ok", "?"), ("failure_class", "U16"),
    ("step", "U12"), ("first", "?"), ("t0", "f4"), ("dur", "f4"), ("hand", "i1"), ("pinch", "?"),
    ("pelvis_z0", "f4"), ("tx", "f4"), ("ty", "f4"), ("tz", "f4"), ("dist", "f4"),
    ("wx0", "f4"), ("wy0", "f4"), ("wz0", "f4"), ("drift_xy", "f4"), ("drift_z", "f4"), ("mode", "f4"),
    ("jaw_err_cm", "f4"), ("attitude_err_deg", "f4"), ("ik_err_cm", "f4"), ("wrist_err_cm", "f4"),
    ("shoulder_to_obj", "f4"), ("jaw_z_minus_top", "f4"), ("approach_deg", "f4"),
    ("gx", "f4"), ("gy", "f4"), ("gz", "f4"),  # the grasp point the skill chose (world -> heading frame), nan if unrecorded
    ("clear", "f4"), ("corridor_cm", "f4"), ("half_height", "f4"),  # the stance step's clearance to the furniture
]  # fmt: skip


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def heading(p: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """A world point in the heading frame at `origin` (xyz) turned by `yaw`."""
    d = p - origin
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


def first_step(steps: list, names) -> dict | None:
    return next((s for s in steps if isinstance(s, dict) and s.get("step") in names), None)


def hand_and_pinch(r: dict, steps: list) -> tuple[int | None, bool | None]:
    """0 left / 1 right and whether the grasp is a top pinch: the new records' events, the old records' `choose`."""
    ev = r.get("events") or {}
    ch = first_step(steps, ("choose",)) or {}
    side = ev.get("grip_side") or ch.get("side")
    hand = {"left": 0, "right": 1}.get(side) if side else None
    if ev.get("pinch") is not None:
        pinch = bool(ev["pinch"])
    elif ch.get("kind"):
        pinch = ch["kind"] == "top"
    else:
        pinch = any(s.get("step") == "descend" for s in steps if isinstance(s, dict))
    return hand, pinch


def samples_of(path: str) -> tuple[list[tuple], str]:
    """The reach rows of one round, and why none when there are none."""
    p = Path(path)
    try:
        r = json.loads(p.read_text())
    except (OSError, ValueError):
        return [], "unreadable"
    steps = r.get("steps") or []
    if not steps or "ok" not in r:
        return [], "no_steps"
    npz = p.with_suffix(".npz")
    if not npz.exists():
        return [], "no_trace"
    reaches = [s for s in steps if isinstance(s, dict) and s.get("step") in REACH_STEPS and s.get("jaw_err_cm") is not None]  # fmt: skip
    if not reaches:
        return [], "no_reach_step"
    try:
        d = np.load(npz, allow_pickle=True)
        t, robot, refs, wrists = d["t"], d["robot"], d["ref_wrists"], d["wrists"]
        prim = d["primitive"]
        names = [str(x) for x in d["primitives"]]
        mode = d["mode"] if "mode" in d.files else np.full(len(t), np.nan)
    except (OSError, ValueError, KeyError):
        return [], "bad_trace"
    if len(t) < 5:
        return [], "short_trace"
    reach_id = names.index("reach") if "reach" in names else -1
    hand, pinch = hand_and_pinch(r, steps)
    sb = path.split("/rsi-scene/")[1].split("/")[0] if "/rsi-scene/" in path else p.parents[4].name
    geom = first_step(steps, ("geom",)) or first_step(steps, ("stance", "choose")) or {}
    # the grasp point the skill chose before reaching: new records' pregrasp `grasp`, old records' choose `g`
    gpt = (first_step(steps, ("pregrasp",)) or {}).get("grasp") or (first_step(steps, ("choose",)) or {}).get("g")
    stance = first_step(steps, ("stance",)) or {}
    rows, seen_first = [], False
    for s in reaches:
        i0 = int(np.clip(np.searchsorted(t, float(s.get("t0", 0))), 0, len(t) - 1))
        i1 = int(np.clip(np.searchsorted(t, float(s.get("t1", 0))) - 1, i0, len(t) - 1))
        if i1 - i0 < 2:
            continue
        win = np.arange(i0, i1 + 1)
        # the hand: the record's, else the wrist whose reference moved most in the window
        h = hand
        if h is None:
            moved = [np.linalg.norm(refs[i1, k] - refs[i0, k]) for k in (0, 1)]
            h = int(np.argmax(moved))
        # the command: the reference at the last reach tick in the window (the motor layer then holds it)
        idx = win[prim[win] == reach_id] if reach_id >= 0 else win
        ic = int(idx[-1]) if len(idx) else i1
        yaw = yaw_of(robot[i0, 3:7])
        origin = robot[i0, :3].copy()
        tgt = heading(refs[ic, h], origin, yaw)
        w0 = heading(wrists[i0, h], origin, yaw)
        sign = -1.0 if h == 0 else 1.0  # mirror the left hand onto the right
        drift = robot[i1, :3] - robot[i0, :3]
        first = s.get("step") in FIRST and not seen_first
        seen_first = seen_first or first
        g = heading(np.asarray(gpt, dtype=float), origin, yaw) if gpt and len(gpt) == 3 else np.full(3, np.nan)
        rows.append((
            sb, r.get("label", p.stem), r.get("task", ""), str(r.get("skills", "")), bool(r["ok"]),
            str(r.get("failure_class") or ""), s["step"], first, float(s.get("t0", 0)),
            float(s.get("t1", 0)) - float(s.get("t0", 0)), h, bool(pinch), float(origin[2]),
            float(tgt[0]), float(sign * tgt[1]), float(tgt[2]), float(np.linalg.norm(tgt)),
            float(w0[0]), float(sign * w0[1]), float(w0[2]), float(np.hypot(drift[0], drift[1])), float(drift[2]),
            float(mode[ic]), float(s["jaw_err_cm"]), float(s.get("attitude_err_deg") or np.nan),
            float(s.get("ik_err_cm") if s.get("ik_err_cm") is not None else np.nan),
            float(100 * np.linalg.norm(refs[ic, h] - wrists[ic, h])),
            float(geom.get("shoulder_to_obj") if geom.get("shoulder_to_obj") is not None else geom.get("D", np.nan)),
            float(geom.get("jaw_z_minus_top", np.nan)), float(geom.get("approach_deg", np.nan)),
            float(g[0]), float(sign * g[1]), float(g[2]),
            float(stance.get("clear", np.nan)), float(stance.get("corridor_cm", np.nan)),
            float(stance.get("half_height", np.nan)),
        ))  # fmt: skip
    return rows, "ok" if rows else "no_window"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=ROOTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(argv)
    files = sorted({f for g in a.roots for f in glob.glob(os.path.join(g, "pick_*.json"))})
    print(f"{len(files)} round files", flush=True)
    why = collections.Counter()
    rows = []
    with Pool(a.procs) as pool:
        for k, (rs, w) in enumerate(pool.imap_unordered(samples_of, files, chunksize=16)):
            why[w] += 1
            rows += rs
            if k % 1000 == 999:
                print(f"  {k + 1} files, {len(rows)} rows", flush=True)
    arr = np.array(rows, dtype=FIELDS)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, samples=arr)
    first = arr[arr["first"]]
    summary = dict(files=len(files), rows=len(arr), first_reaches=int(len(first)), why=dict(why),
                   sandboxes=len(set(arr["sandbox"])), skills=len(set(arr["skills"])),
                   miss3_first=float(np.mean(first["jaw_err_cm"] >= 3)) if len(first) else None,
                   per_step=dict(collections.Counter(arr["step"].tolist())))  # fmt: skip
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
