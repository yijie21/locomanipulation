"""Where the jaw truly was against the object when the grip started (selfwm item 12): exact FK (fk.py) of the recorded
joints and pelvis at the round's grip time, against the object's pose at that moment (AT_GRIP; False: at the start of the
round), in the wrist frame (x approach, y closing, z third). Per start, the change from a reference set of rounds
(truth, L0) removes each object's own offset of its centre from the grasp point.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY adt_grip_error.py <reference rounds dir> <rounds dir> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import fk as F  # noqa: E402

AT_GRIP = True  # the object where it is when the grip starts (a push during the entry moves the target with it)


def make_fk():
    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    cfg = ScaleBfmBackendConfig(scene_dir=str(REPO / "outputs" / "scene" / "walk_mars"), object_name="")
    sim = R.Sim(json.load(open(R.META)), backend=cfg)
    fk = F.FK(sim.m, sim.qadr)
    # the trace's joints are Motor.joints(): MuJoCo joint-name order; FK takes ScaleBFM's (sim.qadr)
    kp = [int(sim.m.jnt_qposadr[sim.m.joint(n).id]) for n in MUJOCO_JOINT_NAMES]
    fk.perm = [kp.index(int(a)) for a in sim.qadr]
    return fk


def grip_frames(fk, d: Path) -> dict:
    """label -> (jaw - object centre) in the wrist frame at the grip (cm), for rounds that gripped."""
    out = {}
    for f in sorted(d.glob("pick_*.json")):
        r = json.loads(f.read_text())
        t = (r.get("events") or {}).get("grip_t")
        z = f.with_suffix(".npz")
        if t is None or not z.exists():
            continue
        z = np.load(z, allow_pickle=True)
        i = int(np.searchsorted(z["t"], t))
        ob = [k for k in z.files if k.startswith("obj_")][0]
        root = z["robot"][i].astype(float)
        jaw, rot = fk(torch.tensor(z["joints"][i : i + 1, fk.perm], dtype=torch.float32))["jaw_right"]
        Rr = F.quat_to_mat(root[3:7])
        jaw_w = root[:3] + Rr @ jaw[0].numpy()
        Rw = Rr @ rot[0].numpy()
        out[r["label"]] = dict(ok=bool(r["ok"]), e=100 * Rw.T @ (jaw_w - z[ob][i if AT_GRIP else 0, :3].astype(float)))
    return out


def main(argv) -> None:
    fk = make_fk()
    ref = grip_frames(fk, Path(argv[0]))
    for d in argv[1:]:
        cur = grip_frames(fk, Path(d))
        keys = sorted(set(ref) & set(cur))
        de = np.array([cur[k]["e"] - ref[k]["e"] for k in keys])
        a = np.abs(de)
        print(f"{d}: {len(keys)} rounds gripped in both; |change from reference| cm  "
              f"approach med {np.median(a[:, 0]):.2f} p90 {np.percentile(a[:, 0], 90):.2f} | "
              f"closing med {np.median(a[:, 1]):.2f} p90 {np.percentile(a[:, 1], 90):.2f} | "
              f"third med {np.median(a[:, 2]):.2f} p90 {np.percentile(a[:, 2], 90):.2f}")  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:])
