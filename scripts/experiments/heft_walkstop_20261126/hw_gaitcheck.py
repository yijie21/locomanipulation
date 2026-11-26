"""heft_walkstop_20261126 revision 3: the gait check. HA's HEFT task episodes walk on gait_heft.HEFT through hadata's
hr_real_gait.py --heft-gait (heft_walk_frames, md5 7dfaa816 in hadata's gait smoke); the free-space P0 / P2 / P4 walk on
hw_run.walk with gait p0. `install` wraps hw_run.walk so that the first HEFT walk on gait_heft.HEFT is also built by
hadata's heft_walk_frames -- its source (with heft_gait_modules and HEFT_GAIT_DIR) taken from the given copy of
hr_real_gait.py and run against this copy's modules (HERE = this copy's selfmodel_hacollect_20261126, so HEFT_GAIT_DIR is
this copy's byte-identical heft_gait/; nothing is imported from the shared tree) -- on the same inputs, after ours, and
compared bit for bit; the result goes to <out>/gait_check.json and the log. No Isaac import: usable from LML too."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def install(src_path: Path, out: Path, H, L, HW, tag: str = "hw") -> None:
    src = Path(src_path).read_text()
    md5 = hashlib.md5(src.encode()).hexdigest()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("heft_gait_modules", "heft_walk_frames"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "HEFT_GAIT_DIR" for t in n.targets))]  # fmt: skip
    assert len(keep) == 3, [getattr(n, "name", "assign") for n in keep]
    ns = dict(np=np, sys=sys, Path=Path, HERE=HERE.parent / "selfmodel_hacollect_20261126")
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(src_path), "exec"), ns)
    orig, g0 = H.walk, H.gaits()["p0"]
    state = dict(done=False)

    def walk(dr, backend, g, x, ref_frame, goal_xy, goal_yaw, pose, info=None):
        wf = orig(dr, backend, g, x, ref_frame, goal_xy, goal_yaw, pose, info)
        if not state["done"] and backend == "heft" and g is g0:
            state["done"] = True
            theirs = ns["heft_walk_frames"](dr, np.array(ref_frame), np.asarray(goal_xy, float), float(goal_yaw), pose, {}, L, HW)
            same = bool(wf.shape == theirs.shape and np.array_equal(wf, theirs))
            diff = float(np.max(np.abs(wf.astype(float) - theirs.astype(float)))) if wf.shape == theirs.shape else None
            res = dict(same=same, shape=list(wf.shape), max_abs_diff=diff, hr_real_gait_md5=md5, src=str(src_path),
                       gait_is_heft=bool(g is ns["heft_gait_modules"]()[1].HEFT), heft_gait_dir=str(ns["HEFT_GAIT_DIR"]),
                       goal_xy=np.asarray(goal_xy, float).tolist(), pose=[np.asarray(pose[0], float).tolist(), float(pose[1])])  # fmt: skip
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "gait_check.json").write_text(json.dumps(res, indent=1))
            print(f"{tag} gait check: {json.dumps(res)}", flush=True)
        return wf

    H.walk = walk
