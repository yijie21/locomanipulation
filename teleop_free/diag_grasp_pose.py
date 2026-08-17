"""Put the hand exactly where the plan says, close it, and look.

The grasp has never been observed succeeding, so every failure so far has been
ambiguous between "the pose is wrong" and "the arm could not get to the pose".
This separates them: solve the arm's IK straight onto the planned grip point,
close the fingers, and report what touches what. No controller, no walking, no
servo — if the grasp is unsound here, nothing downstream can rescue it.

Two outputs:

  numbers   contact count, which links, penetration depth, and whether the
            object is held between the thumb and the fingers rather than
            merely resting against one of them
  pictures  the same pose rendered from three angles, so the geometry can be
            eyeballed rather than inferred

    python teleop_free/diag_grasp_pose.py [--out outputs/renders/grasp_check]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grasping  # noqa: E402
from diag_reach_map import arm_joints, solve_ik  # noqa: E402
from primitives import (  # noqa: E402
    GRASP_BEARING,
    GRASP_CENTRE_OFFSET,
    HAND_CLOSED_LEFT,
    HAND_CLOSED_RIGHT,
    STAND_OFF,
)
from privileged_pub import HAND_JOINTS  # noqa: E402

SCENE_META = ("/workspace/code/locomanipulation/outputs/scene/sim/"
              "scene_meta.json")
PELVIS_Z = 0.78
WIDTH, HEIGHT = 900, 700

# Views chosen to show the three things that can be wrong: whether the jaw
# straddles the object, whether the fingers clear the table, and whether the
# approach is square to it.
VIEWS = [("side", 180.0, -10.0, 0.55),
         ("above", 210.0, -55.0, 0.60),
         ("along_approach", 90.0, -8.0, 0.50)]


def stand_pose(meta, obj_key: str):
    """Where the robot would be standing, per the approach the FSM runs."""
    obj = meta["objects"][obj_key]
    surf = meta["surfaces"][obj["surface"]]
    can = np.array(obj["init_pos"][:2], dtype=float)
    out = can - np.array(surf["xy"], dtype=float)
    out /= max(float(np.linalg.norm(out)), 1e-6)
    stand = can + out * STAND_OFF
    # face_for_grasp turns GRASP_BEARING past the object, which leaves it in
    # front of the working hand instead of on the body's centreline.
    heading = float(np.arctan2(*(can - stand)[::-1]))

    return stand, heading + GRASP_BEARING


def place_robot(model, data, stand_xy, yaw) -> None:
    data.qpos[0:3] = [stand_xy[0], stand_xy[1], PELVIS_Z]
    data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def hand_qadr(model, side: str):
    out = []
    for j in HAND_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{side}_hand_{j}_joint")
        out.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)

    return out


def report_contacts(model, data, obj_body: int, side: str) -> dict:
    """Everything touching the object, with penetration depth."""
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if obj_body not in (b1, b2):
            continue
        other = b2 if b1 == obj_body else b1
        name = model.body(other).name
        rows.append((name, float(c.dist), np.array(c.pos)))

    thumb = [r for r in rows if "thumb" in r[0]]
    fingers = [r for r in rows if ("index" in r[0] or "middle" in r[0])]

    return {"rows": rows, "thumb": thumb, "fingers": fingers}


def render(model, data, out_dir, tag, lookat) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as r:
        for name, azim, elev, dist in VIEWS:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.azimuth, cam.elevation, cam.distance = azim, elev, dist
            r.update_scene(data, camera=cam)
            path = os.path.join(out_dir, f"{tag}_{name}.png")
            # cv2 rather than imageio: it is already a dependency (sim_video)
            # and expects BGR, while MuJoCo renders RGB.
            cv2.imwrite(path, r.render()[:, :, ::-1])
            paths.append(path)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--out", default="/workspace/code/locomanipulation/outputs/"
                                     "renders/grasp_check")
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    model = mujoco.MjModel.from_xml_path(meta["scene_xml"])
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)

    obj_key = meta["target_object"]
    obj = meta["objects"][obj_key]
    surf = meta["surfaces"][obj["surface"]]
    obj_body = obj["body_id"]

    stand, yaw = stand_pose(meta, obj_key)
    place_robot(model, data, stand, yaw)
    mujoco.mj_forward(model, data)

    plan = grasping.grasp_pose(
        obj_pos=np.array(obj["init_pos"]), obj_quat=obj["init_quat"],
        box=obj["box"], surface_top=surf["top"], pelvis_xy=stand,
        pelvis_yaw=yaw)
    print(f"standing at {np.round(stand, 3)} yaw {np.rad2deg(yaw):+.0f}deg, "
          f"{np.linalg.norm(np.array(obj['init_pos'][:2]) - stand):.3f} m from "
          f"the {obj_key}")
    print(grasping.describe(plan))
    if not plan.ok:
        sys.exit(1)

    joints = arm_joints(model, args.side)
    qadr = np.array([model.jnt_qposadr[j] for j in joints])
    dofadr = np.array([model.jnt_dofadr[j] for j in joints])
    lo = np.array([model.jnt_range[j][0] for j in joints])
    hi = np.array([model.jnt_range[j][1] for j in joints])
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             f"{args.side}_wrist_yaw_link")

    ok, pos_err, ax_err = solve_ik(model, data, palm, plan.grip, plan.approach,
                                   qadr, dofadr, lo, hi, rng)
    print(f"\nIK onto the grip point: {'solved' if ok else 'DID NOT SOLVE'} "
          f"(position error {pos_err * 1000:.0f} mm, palm axis off by "
          f"{np.rad2deg(ax_err):.0f} deg)")
    if not ok:
        print("  the pose is unreachable, so nothing below is meaningful")

    # Where the grip centre actually ended up, and where the object is in the
    # palm's frame — the same number servo_grasp servoes on, so the two are
    # directly comparable.
    mujoco.mj_forward(model, data)
    rot = data.xmat[palm].reshape(3, 3)
    grip = data.xpos[palm] + rot @ GRASP_CENTRE_OFFSET
    q_palm = rot.T @ (data.xpos[obj_body] - data.xpos[palm])
    print(f"  grip centre at {np.round(grip, 3)}, target {np.round(plan.grip, 3)}")
    print(f"  object in the palm frame {np.round(q_palm, 3)}, "
          f"want {np.round(GRASP_CENTRE_OFFSET, 3)}")

    lookat = grip.copy()
    open_paths = render(model, data, args.out, "1_open", lookat)

    closed = HAND_CLOSED_RIGHT if args.side == "right" else HAND_CLOSED_LEFT
    for a, v in zip(hand_qadr(model, args.side), closed):
        if a >= 0:
            data.qpos[a] = float(v)
    mujoco.mj_forward(model, data)
    con = report_contacts(model, data, obj_body, args.side)

    print(f"\nfingers closed — {len(con['rows'])} contacts on the {obj_key}:")
    for name, dist, pos in con["rows"]:
        print(f"    {name:28s} dist {dist * 1000:+7.1f} mm  at "
              f"{np.round(pos, 3)}")
    thumb, fingers = len(con["thumb"]), len(con["fingers"])
    print(f"\n  thumb side {thumb} contact(s), finger side {fingers}")
    if thumb and fingers:
        print("  VERDICT: opposed — the object is between the thumb and the "
              "fingers, which is what a grasp is")
    elif thumb or fingers:
        print("  VERDICT: one-sided — closing will push the object away, not "
              "hold it")
    else:
        print("  VERDICT: no contact at all — the jaw closed on air")

    # --- calibrate the grip centre ---------------------------------------
    # GRASP_CENTRE_OFFSET was measured as the closest approach between the
    # thumb's surface and the fingers' surface with the hand OPEN. That point
    # is at the knuckles, and it is not where a closing hand ends up holding
    # something — the fingers curl as they close, so the gap moves.
    #
    # Calibrate it against the only thing that matters instead: put the object
    # at each candidate palm-frame position, close, and see whether the thumb
    # and the fingers end up on opposite sides of it. Moving the object rather
    # than the arm keeps this to one forward-kinematics call per candidate.
    print("\n--- where does a closed hand actually hold something? ---")
    adr = obj["qpos_adr"]
    keep = data.qpos[adr:adr + 7].copy()
    rot = data.xmat[palm].reshape(3, 3)
    origin = data.xpos[palm].copy()
    xs = np.arange(0.04, 0.20, 0.01)
    ys = np.arange(-0.06, 0.09, 0.01)
    zs = np.arange(-0.05, 0.06, 0.01)
    hits = []
    for px in xs:
        for py in ys:
            for pz in zs:
                data.qpos[adr:adr + 3] = origin + rot @ np.array([px, py, pz])
                data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
                mujoco.mj_forward(model, data)
                c = report_contacts(model, data, obj_body, args.side)
                if c["thumb"] and c["fingers"]:
                    hits.append((px, py, pz))
    if not hits:
        print("  NO position gives opposed contact — this hand cannot pinch "
              "this object at all")
    else:
        h = np.array(hits)
        centre = h.mean(0)
        span = h.max(0) - h.min(0)
        print(f"  {len(hits)} of {len(xs) * len(ys) * len(zs)} positions give "
              f"a real pinch (thumb and fingers on opposite sides)")
        print(f"  region centre {np.round(centre, 3)}   "
              f"span {np.round(span, 3)} m")
        print(f"  half-widths   x +-{span[0] / 2 * 1000:.0f} mm   "
              f"y +-{span[1] / 2 * 1000:.0f} mm   "
              f"z +-{span[2] / 2 * 1000:.0f} mm   "
              f"<- the accuracy the servo has to hit")
        print(f"  GRASP_CENTRE_OFFSET is currently "
              f"{np.round(GRASP_CENTRE_OFFSET, 3)}; "
              f"{'inside' if np.all(np.abs(GRASP_CENTRE_OFFSET - centre) <= span / 2) else 'OUTSIDE'} "
              f"the region")
        # A slice through the region at its own centre height, for eyeballing.
        zc = zs[np.argmin(np.abs(zs - centre[2]))]
        print(f"\n  slice at z = {zc:+.2f}:")
        for px in xs[::2]:
            row = []
            for py in ys[::2]:
                row.append("##" if any(abs(a - px) < 1e-6 and abs(b - py) < 1e-6
                                       and abs(c2 - zc) < 1e-6
                                       for a, b, c2 in hits) else " .")
            print(f"    x={px:.2f}  " + " ".join(row))
        print("           y = " + " ".join(f"{v:+.2f}" for v in ys[::2]))

    data.qpos[adr:adr + 7] = keep
    mujoco.mj_forward(model, data)

    # --- is there a way IN? -----------------------------------------------
    # The pinch region says where a closed hand holds the object. It says
    # nothing about whether the hand can GET there. The servo traces show the
    # object being pushed from the moment it reaches palm x ~ 0.21, which is
    # exactly where the fingertips are — so slide the object along the approach
    # with the hand open and find the first contact.
    print("\n--- approaching with the hand open: where does it first touch? ---")
    for a2 in hand_qadr(model, args.side):
        if a2 >= 0:
            data.qpos[a2] = 0.0                  # HAND_OPEN
    mujoco.mj_forward(model, data)
    rot = data.xmat[palm].reshape(3, 3)
    origin = data.xpos[palm].copy()
    adr = obj["qpos_adr"]
    keep = data.qpos[adr:adr + 7].copy()
    first = None
    for px in np.arange(0.34, 0.09, -0.005):
        data.qpos[adr:adr + 3] = (origin + rot @ np.array(
            [px, GRASP_CENTRE_OFFSET[1], GRASP_CENTRE_OFFSET[2]]))
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        c = report_contacts(model, data, obj_body, args.side)
        if c["rows"] and first is None:
            first = px
            names = sorted({r[0].replace("right_hand_", "").replace("_link", "")
                            for r in c["rows"]})
            print(f"  first contact at palm x = {px:.3f}, on {', '.join(names)}")
    if first is None:
        print("  never touches — the approach is clear all the way in")
    else:
        print(f"  the grip centre needs palm x = {GRASP_CENTRE_OFFSET[0]:.3f}, "
              f"so the object is in contact for the last "
              f"{(first - GRASP_CENTRE_OFFSET[0]) * 100:.1f} cm of the "
              f"approach — it is pushed, not enclosed")
    data.qpos[adr:adr + 7] = keep
    mujoco.mj_forward(model, data)

    # --- how far should the hand close? -----------------------------------
    # HAND_CLOSED is a single full-close pose that ignores the object. Against
    # a 4 cm can it drives the fingers 11-25 mm *into* it, and a
    # position-controlled hand really does drive there — the can is squeezed
    # out and ends up on the floor, which the episode then reports as "no
    # contact after closing". Find the smallest closure that still opposes.
    print("\n--- how far to close ---")
    data.qpos[adr:adr + 7] = keep
    best_a = None
    for alpha in np.arange(0.3, 1.05, 0.05):
        for a2, v in zip(hand_qadr(model, args.side), closed):
            if a2 >= 0:
                data.qpos[a2] = float(v) * alpha
        mujoco.mj_forward(model, data)
        c = report_contacts(model, data, obj_body, args.side)
        t, f = len(c["thumb"]), len(c["fingers"])
        deep = min([r[1] for r in c["rows"]], default=0.0)
        tag = "opposed" if (t and f) else "one-sided" if (t or f) else "no contact"
        print(f"  alpha {alpha:.2f}  thumb {t}  fingers {f}  "
              f"deepest {deep * 1000:+6.1f} mm   {tag}")
        if t and f and best_a is None:
            best_a = alpha
    if best_a is not None:
        print(f"\n  first closure that opposes: alpha {best_a:.2f} — use that "
              f"rather than 1.0, which buries the fingers in the object")
    for a2, v in zip(hand_qadr(model, args.side), closed):
        if a2 >= 0:
            data.qpos[a2] = float(v) * (best_a or 1.0)
    mujoco.mj_forward(model, data)

    closed_paths = render(model, data, args.out, "2_closed", lookat)
    print("\nwrote:")
    for p in open_paths + closed_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
