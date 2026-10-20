"""TODO 28.3: how steady the body is in a scene round, from its motor trace (scene_round --npz), per controller.

Per round:
  torso_rate_p95   deg/s, 95th percentile of the torso_link angular speed (the head camera rides on it; the
                   controller-compare guard used the ego camera's)
  torso_tilt_max   deg, largest torso_link tilt from vertical
  stand_drift      cm/s, median pelvis speed in the horizontal plane while a stationary primitive runs (reach,
                   grip, release, hold, posture): how much the feet / body shuffle when they should not
  hand_jitter      mm, RMS of the commanded-minus-measured wrist distance's tick-to-tick change while gripping or
                   holding (high-frequency hand shake, not the slow tracking error)
  carry_slip       cm, how far a tracked object moves relative to the closest wrist during walk_to with a closed jaw
Writes outputs/amo_compare_20261020/stability.json and prints per-task medians.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/amo_compare_20261020/stability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts/rsi"), str(REPO / "src")]

import tasks as T  # noqa: E402

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

ROBOT = REPO / "outputs/scene_discovery/adt_apartment_mars_physics/robot.xml"
STILL = {"reach", "grip", "release", "hold", "posture"}
DT = 0.02


class Torso:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROBOT))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.b = self.m.body("torso_link").id

    def __call__(self, robot, joints) -> np.ndarray:
        out = np.zeros((len(robot), 3, 3))
        for i in range(len(robot)):
            self.d.qpos[:7], self.d.qpos[self.qadr] = robot[i], joints[i]
            mujoco.mj_kinematics(self.m, self.d)
            out[i] = self.d.xmat[self.b].reshape(3, 3)
        return out


def angle(Ra, Rb) -> np.ndarray:
    tr = np.einsum("nij,nij->n", Ra, Rb)
    return np.arccos(np.clip((tr - 1) / 2, -1, 1))


def round_numbers(z, torso: Torso) -> dict:
    t, robot, joints = z["t"], z["robot"], z["joints"]
    if len(t) < 10:
        return {}
    Rt = torso(robot, joints)
    rate = np.degrees(angle(Rt[1:], Rt[:-1])) / DT
    tilt = np.degrees(np.arccos(np.clip(Rt[:, 2, 2], -1, 1)))
    names = list(z["primitives"])
    prim = np.array([names[i] for i in z["primitive"]])
    still = np.isin(prim[1:], list(STILL)) & np.isin(prim[:-1], list(STILL))
    v = np.linalg.norm(np.diff(robot[:, :2], axis=0), axis=1) / DT
    out = dict(torso_rate_p95=float(np.percentile(rate, 95)), torso_tilt_max=float(tilt.max()),
               stand_drift=float(100 * np.median(v[still])) if still.any() else None)  # fmt: skip
    grip = np.isin(prim, ["grip", "hold"])
    err = np.linalg.norm(z["ref_wrists"] - z["wrists"], axis=2).max(axis=1)
    g = np.flatnonzero(grip[1:] & grip[:-1])
    out["hand_jitter"] = float(1000 * np.sqrt(np.mean(np.diff(err)[g] ** 2))) if len(g) else None
    objs = [k for k in z.files if k.startswith("obj_")]
    walk = (prim == "walk_to") & (z["jaw"].min(axis=1) < 0.0)  # a closed jaw: carrying
    if objs and walk.sum() > 10:
        o = z[objs[0]][:, :3]
        rel = np.min(np.linalg.norm(o[:, None] - z["wrists"], axis=2), axis=1)
        idx = np.flatnonzero(walk)
        out["carry_slip"] = float(100 * (rel[idx].max() - rel[idx].min()))
    else:
        out["carry_slip"] = None
    return out


def main() -> int:
    torso = Torso()
    res = {}
    for ctrl in ("amo", "scalebfm"):
        for f, task in T.TASKS:
            for lab in T.labels_of(f, task):
                p = T.ROUNDS / ctrl / f / f"{lab}.npz"
                if p.exists():
                    with np.load(p) as z:
                        res.setdefault(f"{f}/{task}", {}).setdefault(ctrl, {})[lab] = round_numbers(z, torso)
    (T.OUT / "stability.json").write_text(json.dumps(res, indent=1))
    keys = ("torso_rate_p95", "torso_tilt_max", "stand_drift", "hand_jitter", "carry_slip")
    for task, by in res.items():
        line = [f"{task:44s}"]
        for ctrl in ("amo", "scalebfm"):
            rows = list(by.get(ctrl, {}).values())
            med = {
                k: np.median([r[k] for r in rows if r.get(k) is not None])
                if any(r.get(k) is not None for r in rows)
                else None
                for k in keys
            }
            line.append(
                ctrl
                + " "
                + " ".join(f"{k.split('_')[0][:5]}={'-' if v is None else f'{v:.1f}'}" for k, v in med.items())
            )
        print(" | ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
