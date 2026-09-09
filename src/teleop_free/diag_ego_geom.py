"""Where is the can visible from? Pure geometry, no sim needed.

The student only ever sees `observation.images.ego_view`, so every action
the expert takes has to be explainable by what is in that frame. The head
camera is pitched steeply down (fovy 70, quat in scene_locomani.xml), which
puts a hard FAR limit on where a tabletop object can be seen at all: past
some stand-off the can leaves the top of the frame.

This sweeps the pelvis-frame ground plane at can height and reports, for a
few waist-pitch values, the visible band -- both the full frustum and the
central-90% box the servo's visibility gate actually uses. Leaning forward
pitches the camera further down, so the lean that buys reach also costs
far-field visibility; the point is to see how much.

    python src/teleop_free/diag_ego_geom.py
    python src/teleop_free/diag_ego_geom.py --height-sweep

`--height-sweep` answers the question roadmap step 5 blocks on: as the table
height varies, does the target stay inside the ego gates? Two gates, two
regimes, and they pull in opposite directions --

  * `[view] object outside ego view at stand` fires while the robot is
    STANDING at the stand, before the squat. A higher object rises in the
    frame, so this one gets easier as the table gets taller.
  * the servo's `ego visibility >= 70%` runs while the robot is SQUATTING,
    and the squat is planned per table height, so the pelvis moves with it.

The sweep reports the object's normalised frame position for both, at the
planned stand depth, so a candidate height can be accepted or rejected
before any episode is run.
"""

from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

SCENE = f"{paths.REPO}/outputs/scene/sim/scene_locomani.xml"
META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"
CENTRAL = 0.90  # the servo's gate: central 90% of the frame
WAIST_PITCH = [0.0, 0.20, 0.35, 0.52]


def frame_pos(m, d, cam, pel, z_obj, x_rel, y_rel=0.0):
    """Where the object lands in the ego frame: (u_x, u_y) normalised so the
    frame edge is 1.0. The gym's gate is |u_x| <= 0.8, |u_y| <= 0.9."""
    C = np.array(d.cam_xpos[cam])
    R = np.array(d.cam_xmat[cam]).reshape(3, 3)
    P = np.array(d.xpos[pel])
    Rp = np.array(d.xmat[pel]).reshape(3, 3)
    w = P + Rp @ np.array([x_rel, y_rel, z_obj - P[2]])
    v = R.T @ (w - C)
    if -v[2] <= 0.05:
        return None
    half = np.tan(np.radians(float(m.cam_fovy[cam]) / 2.0))
    return float(v[0] / -v[2] / half), float(v[1] / -v[2] / half)


def height_sweep() -> None:
    """Ego-frame position of the target versus table height."""
    import grasping
    import reach_map
    from primitives import STANDING_PELVIS_Z

    meta = json.load(open(META))
    can = meta["objects"][meta["target_object"]]
    obj_h = float(can["height"])
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rm = reach_map.ReachMap()

    print(
        f"[ego] object height {obj_h * 100:.0f} cm, fovy "
        f"{float(m.cam_fovy[cam]):.0f} deg. Gate at the stand: "
        f"|u_x| <= 0.80, |u_y| <= 0.90."
    )
    print(f"{'table':>6} {'obj top':>8} {'squat':>7}   worst stand/squat position in frame")
    for top in np.arange(0.50, 0.95001, 0.02):
        obj_top = top + obj_h
        z_obj = top + obj_h / 2.0
        grip_z = obj_top - grasping.GRIP_BELOW_RIM
        plan = rm.plan_core_stand(float(grip_z - STANDING_PELVIS_Z))
        if plan is None:
            print(f"{top:6.2f} {obj_top:8.3f}      --         --         --  no stand plan")
            continue
        rel_cands, drop, _ = plan
        # EVERY stand candidate, at its own (x, y). The first version of this
        # sweep put the object dead ahead (y = 0) and passed every height --
        # then a 0.80 m table failed live on cam_x 0.994. The stand plan aims
        # the arm's core, which sits offset to the right, so the object is
        # never centred: rel_y runs -0.06 to -0.16. A taller table also
        # shortens the range at which the pitched-down camera sees it, which
        # widens the lateral angle it subtends. Both push u_x out, and u_x is
        # the tighter gate (0.80 against 0.90).
        worst = None
        for tag, dz in (("stand", 0.0), ("squat", -drop)):
            mujoco.mj_resetData(m, d)
            d.qpos[2] += dz
            mujoco.mj_forward(m, d)
            for c in rel_cands:
                p_ = frame_pos(m, d, cam, pel, z_obj, float(c[0]), float(c[1]))
                if p_ is None:
                    continue
                score = max(abs(p_[0]) / 0.80, abs(p_[1]) / 0.90)
                if worst is None or score > worst[0]:
                    worst = (score, tag, p_)
        if worst is None:
            print(f"{top:6.2f} {obj_top:8.3f} {drop * 100:6.1f}cm     behind the camera")
            continue
        score, tag, (ux, uy) = worst
        print(
            f"{top:6.2f} {obj_top:8.3f} {drop * 100:6.1f}cm "
            f"  u_x {ux:+.2f}  u_y {uy:+.2f}  ({tag}, worst of "
            f"{len(rel_cands)} stands)  "
            f"{'ok' if score <= 1.0 else 'OUT OF FRAME'}"
        )


def main() -> None:
    if "--height-sweep" in sys.argv:
        return height_sweep()
    meta = json.load(open(META))
    can = meta["objects"][meta["target_object"]]
    top = meta["surfaces"]["A"]["top"]
    z_can = top + can["height"] / 2.0
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    if cam < 0:
        sys.exit("no head_camera in the scene")
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
    adr = m.jnt_qposadr[jid]
    fovy = float(m.cam_fovy[cam])
    print(
        f"[ego] head_camera fovy {fovy:.0f} deg, can centre z {z_can:.3f} m (table {top:.3f} + {can['height'] / 2:.3f})"
    )

    # ground grid in the pelvis frame
    xs = np.arange(0.15, 2.01, 0.01)
    ys = np.arange(-0.80, 0.81, 0.02)
    for wp in WAIST_PITCH:
        mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else mujoco.mj_resetData(m, d)
        d.qpos[adr] = wp
        mujoco.mj_forward(m, d)
        C = np.array(d.cam_xpos[cam])
        # MuJoCo camera frame: -z is the view direction, +x right, +y up
        R = np.array(d.cam_xmat[cam]).reshape(3, 3)
        P = np.array(d.xpos[pel])
        Rp = np.array(d.xmat[pel]).reshape(3, 3)
        pel_yaw_R = Rp
        half = np.tan(np.radians(fovy / 2.0))
        # square-ish gate: use fovy for both axes (the exporter's 480x640
        # frame is wider than tall, so vertical is the binding constraint)
        band_x, cols = [], []
        for x in xs:
            seen_any = False
            for y in ys:
                w = P + pel_yaw_R @ np.array([x, y, z_can - P[2]])
                v = R.T @ (w - C)  # camera frame
                if -v[2] <= 0.05:
                    continue
                u_x, u_y = v[0] / -v[2], v[1] / -v[2]
                if abs(u_x) <= half * CENTRAL and abs(u_y) <= half * CENTRAL:
                    seen_any = True
                    cols.append((x, y))
            band_x.append(seen_any)
        vis_x = xs[np.array(band_x)]
        if len(vis_x) == 0:
            print(f"[ego] waist_pitch {wp:+.2f}: nothing visible at can height")
            continue
        near, far = float(vis_x.min()), float(vis_x.max())

        # lateral half-width at the far edge and at 0.6 m
        def halfw(xq):
            yy = [y for (x, y) in cols if abs(x - xq) < 0.006]
            return (max(yy) - min(yy)) / 2 if yy else 0.0

        print(
            f"[ego] waist_pitch {wp:+.2f} rad ({np.degrees(wp):4.0f} deg): "
            f"can visible x {near:.2f}..{far:.2f} m   "
            f"half-width {halfw(0.60) * 100:4.0f} cm @0.60 m, "
            f"{halfw(round(far, 2)) * 100:4.0f} cm @far  "
            f"(cam z {C[2]:.2f})"
        )


if __name__ == "__main__":
    main()
