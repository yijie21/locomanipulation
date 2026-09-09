"""Can the arm be at the staging point AND the grip point, from anywhere?

The grasp has been stuck between two requirements that pull in opposite
directions:

    stage far enough not to clip the object   >= 0.21 m back from the grip
        (fingertips lead the grip centre by 0.11, reach() lands within 0.08,
         object radius 0.02)
    stage near enough for the arm to get there  <~ 0.15 m
        (measured: 5 of 8 episodes stalled reaching a 0.24 m stand-off)

Tuning the number cannot satisfy both. So before another round of it, settle
whether *any* standing pose satisfies both — and if none does, the horizontal
straight-in plan is finished and the fix has to change shape, not size.

Two envelopes, because they answer different questions:

  kinematic  where the arm *can* put the grip centre, by inverse kinematics
             over the arm's joint limits. An upper bound: outside this, no
             controller helps.
  effective  what the whole-body controller actually delivers. Much smaller,
             and it is the one that has been failing. Bounded from the live
             logs rather than assumed.

Purely offline — no sim, no deploy, no controller.

    python src/teleop_free/diag_reach_map.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grasping  # noqa: E402
import paths
from primitives import GRASP_CENTRE_OFFSET  # noqa: E402

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"

# What the controller was measured to deliver, in the pelvis frame, across the
# reach traces in outputs/logs: the palm never got past x = 0.36, and never
# below z = -0.09, however hard the command was pushed. The arm's own limits
# are far wider — this is the whole-body controller's working envelope, and it
# is what the stalls ran into.
EFFECTIVE_X_MAX = 0.36
EFFECTIVE_Z_MIN = -0.09

ARM_LINKS = ("shoulder", "elbow", "wrist")  # not hip / knee / ankle
IK_STEPS = 120
IK_RESTARTS = 5
IK_DAMPING = 0.05
IK_POS_TOL = 0.015  # metres
IK_AXIS_TOL = np.deg2rad(30.0)
AXIS_WEIGHT = 0.25  # palm-axis task vs position task


def arm_joints(model, side: str) -> list[int]:
    """The arm's own joints.

    Prefix matching alone catches the leg too — `right_knee_joint` starts with
    `right_` — and three spurious degrees of freedom turn a workspace query
    into a much harder search for nothing.
    """
    out = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.startswith(f"{side}_") and any(k in name for k in ARM_LINKS):
            out.append(j)

    return out


def solve_ik(model, data, palm, target_world, axis_world, qadr, dofadr, lo, hi, rng):
    """Damped least squares onto a point, palm +x roughly along `axis_world`.

    Random sampling was the first attempt and it does not work: a
    nearest-neighbour test needs the cloud dense at the scale of the tolerance,
    and over seven joints it never is — the first map came back almost entirely
    "unreachable" for points the arm plainly reaches. Solving each query is
    both exact and cheaper.

    The angular task is `cross(x_now, x_target)`, which rotates the palm's +x
    onto the approach and leaves the spin about it free; the jaw does not care
    how the hand is rolled.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    best = (False, 1e9, np.pi)

    for attempt in range(IK_RESTARTS):
        q = np.zeros(len(qadr)) if attempt == 0 else rng.uniform(lo, hi)
        for _ in range(IK_STEPS):
            data.qpos[qadr] = q
            mujoco.mj_kinematics(model, data)
            rot = data.xmat[palm].reshape(3, 3)
            grip = data.xpos[palm] + rot @ GRASP_CENTRE_OFFSET
            x_axis = rot @ np.array([1.0, 0.0, 0.0])

            e_pos = target_world - grip
            e_ax = np.cross(x_axis, axis_world)
            pos_err = float(np.linalg.norm(e_pos))
            ax_err = float(np.arccos(np.clip(float(x_axis @ axis_world), -1, 1)))
            if pos_err < IK_POS_TOL and ax_err < IK_AXIS_TOL:
                return True, pos_err, ax_err
            if pos_err < best[1]:
                best = (False, pos_err, ax_err)

            mujoco.mj_jac(model, data, jacp, jacr, grip, palm)
            j = np.vstack([jacp[:, dofadr], AXIS_WEIGHT * jacr[:, dofadr]])
            e = np.concatenate([e_pos, AXIS_WEIGHT * e_ax])
            jjt = j @ j.T + IK_DAMPING**2 * np.eye(6)
            q = np.clip(q + j.T @ np.linalg.solve(jjt, e), lo, hi)

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    model = mujoco.MjModel.from_xml_path(meta["scene_xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(args.seed)

    joints = arm_joints(model, args.side)
    qadr = np.array([model.jnt_qposadr[j] for j in joints])
    dofadr = np.array([model.jnt_dofadr[j] for j in joints])
    lo = np.array([model.jnt_range[j][0] for j in joints])
    hi = np.array([model.jnt_range[j][1] for j in joints])
    print(
        f"{args.side} arm: {len(joints)} joints — "
        + ", ".join(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).replace(f"{args.side}_", "").replace("_joint", "")
            for j in joints
        )
    )

    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{args.side}_wrist_yaw_link")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    p_pos = data.xpos[pelvis].copy()
    p_rot = data.xmat[pelvis].reshape(3, 3).copy()

    def query(local_pt, local_axis, effective: bool) -> bool:
        """Ask the arm for a point expressed in the pelvis frame."""
        if effective and (local_pt[0] > EFFECTIVE_X_MAX or local_pt[2] < EFFECTIVE_Z_MIN):
            return False
        ok, _, _ = solve_ik(model, data, palm, p_pos + p_rot @ local_pt, p_rot @ local_axis, qadr, dofadr, lo, hi, rng)

        return bool(ok)

    obj = meta["objects"][meta["target_object"]]
    surf = meta["surfaces"][obj["surface"]]
    pelvis_z = 0.78
    gz = surf["top"] + grasping.FINGER_DROP + grasping.CLEARANCE_MARGIN + grasping.GRIP_LIFT
    print(
        f"\ntarget {meta['target_object']} on a {surf['top']:.2f} m surface; "
        f"grip point {gz:.3f} m ({(gz - pelvis_z) * 100:+.0f} cm relative to "
        f"a {pelvis_z:.2f} m pelvis), pregrasp "
        f"{grasping.PREGRASP_GAP * 100:.0f} cm back"
    )

    dists = np.arange(0.32, 0.62, 0.04)
    bearings = np.deg2rad(np.arange(-40, 41, 10))
    for label, effective in (
        ("kinematic (arm limits only)", False),
        ("effective (what the controller delivers)", True),
    ):
        print(f"\n--- {label} ---")
        print(
            "        " + "".join(f"{np.rad2deg(b):>6.0f}" for b in bearings) + "    <- object bearing off the nose, deg"
        )
        any_ok = False
        for d in dists:
            row = []
            for b in bearings:
                # Object at distance d and bearing b, worked directly in the
                # pelvis frame — which is what face_for_grasp actually sets, so
                # GRASP_BEARING becomes one column of this table rather than a
                # fixed guess.
                grip = np.array([d * np.cos(b), d * np.sin(b), gz - pelvis_z])
                ah = np.array([np.cos(b), np.sin(b), 0.0])
                pre = grip - ah * (obj.get("radius", 0.02) + grasping.PREGRASP_GAP)
                g_ok = query(grip, ah, effective)
                p_ok = query(pre, ah, effective)
                row.append("  OK  " if (g_ok and p_ok) else "  g   " if g_ok else "   p  " if p_ok else "  --  ")
                any_ok |= g_ok and p_ok
            print(f"  {d:.2f}m" + "".join(row))
        print("        OK = both reachable    g = grip only    p = pregrasp only    -- = neither")
        if not any_ok:
            print(
                "        >>> NO standing pose satisfies both. Horizontal "
                "straight-in cannot work as specified; change the shape of "
                "the approach, not the numbers."
            )


if __name__ == "__main__":
    main()
