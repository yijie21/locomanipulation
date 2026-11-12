"""Grasp poses from the head camera, route 1 (segment, depth to a point cloud, a grasp network), in place of the mesh grasp cache.

see    per placement: the robot at the episode's start pose (room_grasp.episode's draw), one head-camera depth + segmentation
       frame, the target's points in the world. The mask is MuJoCo's segmentation render (privileged; a detector + SAM later).
       -> $GRASP_ROOM_OUT/vision/see_<pose>_L<k>_s<j>.npz (points, depth, rgb, true mask), see_<pose>_<layouts>.json
seg    the mask from SAM 3 instead (scripts/rsi/sam3_server.py on --port, prompted with the object's phrase).

    MUJOCO_GL=egl $PY scripts/experiments/reachwm_20261030/vision_grasp.py see --layouts 0,1,2,3,4,5,6
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import room_grasp as RG  # noqa: E402
import vision_depth as VD  # noqa: E402
from vision_phrases import phrase_of  # noqa: E402

ROOM = RG.ROOM
OUT = ROOM.OUT / "vision"
W, H = 640, 480
DEPTH_MAX = 3.0


def start_pose(layout: dict, k: int, j: int):
    """room_grasp.episode's stance and start draw."""
    slot = layout["slots"][j]
    rng = np.random.default_rng(RG.SEED + 100 * k + j)
    st, c, a, ax, perp = RG.station_frame(slot["station"])
    lat = float(perp @ (np.array(slot["pos"])[:2] - c))
    sd = 1.0 if lat < 0 else -1.0
    back = (
        RG.STANCE_BACK_TRAY
        if st["name"] == "tray"
        else RG.STANCE_BACK_LOW
        if slot["surf"] < RG.STANCE_LOW_Z
        else RG.STANCE_BACK
    )
    stance = c - ax * back + perp * (lat + sd * RG.HAND_Y)
    p_start = stance - ax * float(rng.uniform(0.45, 0.90)) + perp * float(rng.uniform(-0.10, 0.10))
    yaw = a + math.radians(float(rng.uniform(-8.0, 8.0)))
    if RG.LOOK > 0:
        p_start = stance - ax * max(0.0, RG.LOOK - back - float(ax @ (np.array(slot["pos"])[:2] - c)))
    return p_start, yaw, stance, a


def frame(env, rd, rs, cam: str = "head_camera"):
    """Depth (m), segmentation (geom id) and the camera's pose: points = cam_p + cam_R @ [x, y, z] with z forward.
    frame.right: the depth of an imager vision_depth.BASELINE to the right (the stereo pair's other eye)."""
    m, d = env.sim.m, env.sim.d
    cid_ = m.camera(cam).id
    pos0 = m.cam_pos[cid_].copy()
    mujoco.mj_forward(m, d)
    m.cam_pos[cid_] = pos0 + d.xmat[m.cam_bodyid[cid_]].reshape(3, 3).T @ (
        d.cam_xmat[cid_].reshape(3, 3)[:, 0] * VD.BASELINE
    )
    mujoco.mj_forward(m, d)
    rd.update_scene(d, camera=cam)
    frame.right = rd.render().copy()
    m.cam_pos[cid_] = pos0
    mujoco.mj_forward(m, d)
    rd.update_scene(d, camera=cam)
    depth = rd.render().copy()
    rs.update_scene(d, camera=cam)
    seg = rs.render()[..., 0].copy()
    cid = m.camera(cam).id
    Rc = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # x right, y down, z forward
    return depth, seg, d.cam_xpos[cid].copy(), Rc, float(m.cam_fovy[cid])


def cloud(depth, mask, p, R, fovy):
    f = 0.5 * H / math.tan(math.radians(fovy) / 2)
    v, u = np.nonzero(mask & (depth < DEPTH_MAX))
    z = depth[v, u]
    pts = np.c_[(u - W / 2 + 0.5) * z / f, (v - H / 2 + 0.5) * z / f, z]
    return p + pts @ R.T


CALIB = int(os.environ.get("VISION_CALIB", 0))  # 1: the saved depth_rs and cam_R carry calibration errors
CALIB_SCALE, CALIB_DEG = 0.01, 0.5  # depth scale (1 sigma), mounting angle (deg, 1 sigma)


def mujoco_rot(w) -> np.ndarray:
    q, R = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(q, w / max(1e-12, np.linalg.norm(w)), float(np.linalg.norm(w)))
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


GLASS: dict = {}  # object name -> its surface defeats active stereo (glass, clear plastic)


def see(layouts: list[int], pose: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for o in json.load(open(os.path.join(os.environ["GRASP_ROOM_OBJECTS"], "objects.json")))["objects"]:
        GLASS[o["name"]] = any(w in o["type"] for w in ("glass", "bottledWater", "Bottle"))
    rows = []
    for k in layouts:
        env = ROOM.make_env(k)
        m, sm = env.m, env.sim.m
        layout = json.load(open(ROOM.OUT / f"layout_{k}" / "layout.json"))
        rd, rs, rc = mujoco.Renderer(sm, H, W), mujoco.Renderer(sm, H, W), mujoco.Renderer(sm, H, W)
        rd.enable_depth_rendering()
        rs.enable_segmentation_rendering()
        import leg_odo as LO

        jf = LO.JawFeed(env)
        m.reset((0.0, 0.0), 0.0)
        m.hold(1.0)
        for j, slot in enumerate(layout["slots"]):
            p_start, yaw, stance, a = start_pose(layout, k, j)
            at, y = (stance, a) if pose == "stance" else (p_start, yaw)
            m.reset((float(at[0]), float(at[1])), y)
            for s in layout["slots"]:
                m.place(s["object"], np.array(s["pos"]), np.array(s["quat"]))
            m.hold(1.0)
            n_views = (
                2 if pose.startswith("wrist2") else 1
            )  # wrist2…: a second look with the other hand, saved as <pose>v
            for view in range(n_views):
                name = pose + "v" * view
                cam = "head_camera"
                if pose.startswith(
                    "wrist"
                ):  # the hand as the eye: the grasping side's wrist camera, aimed at the station's surface
                    from locomani.backends.scalebfm import motor as MM

                    if view == 0:
                        pA, RA = env.pelvis()
                        jf.anchor(pA, RG.E.yaw_of(RA), MM.WRIST, MM.R.OFF)
                    else:
                        import gen_reach as GR

                        GR.home(m)
                    cam, look = RG.hand_look(env, slot, view=view) if m.fell_at is None else (None, None)
                    if look is None or m.fell_at is not None:
                        rows.append(dict(layout=k, slot=j, station=slot["station"], surf=slot["surf"], depth=slot.get("depth"),
                                         object=slot["object"], n_pix=0, dist=-1.0, centre_px=[-1, -1], centre_in=False, fell=True))  # fmt: skip
                        break
                depth, seg, pc, Rc, fovy = frame(env, rd, rs, cam)
                pc_true, est = pc.copy(), None
                if pose.startswith(
                    "wrist"
                ):  # the camera's pose from the joints and the leg odometry, not the simulator
                    side_ = cam.split("_")[0]
                    pj, Rw = jf.jaw(side_)
                    jf.on = view + 1 < n_views
                    cid = sm.camera(cam).id
                    Rl = np.zeros(9)
                    mujoco.mju_quat2Mat(Rl, sm.cam_quat[cid])
                    pc_e = pj - Rw @ MM.R.OFF + Rw @ sm.cam_pos[cid]
                    Rc_e = Rw @ Rl.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
                    est = [100 * float(np.linalg.norm(pc_e - pc)),
                           math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rc_e @ Rc.T) - 1) / 2))))]  # fmt: skip
                    Rc_sim, pc, Rc = Rc, pc_e, Rc_e
                b = sm.body(slot["object"]).id
                ids = np.array([g for g in range(sm.ngeom) if sm.body_rootid[sm.geom_bodyid[g]] == sm.body_rootid[b]])
                mask = np.isin(seg, ids)
                root = sm.body_rootid[sm.body("pelvis").id]
                robot = np.isin(
                    seg, np.array([g for g in range(sm.ngeom) if sm.body_rootid[sm.geom_bodyid[g]] == root])
                )
                glass = np.isin(
                    seg,
                    np.array(
                        [g for g in range(sm.ngeom) if GLASS.get(sm.body(int(sm.body_rootid[sm.geom_bodyid[g]])).name)]
                        or [-9]
                    ),
                )
                crng = np.random.default_rng(7000 + 100 * k + j)
                depth_rs = VD.realsense(depth, frame.right, fovy, crng, glass)
                Rc_true = Rc_sim if est is not None else Rc.copy()
                if CALIB:  # what a calibrated camera still carries: the depth's scale and the mounting angle
                    depth_rs = (depth_rs * (1.0 + crng.normal(0.0, CALIB_SCALE))).astype(np.float32)
                    Rc = mujoco_rot(crng.normal(0.0, math.radians(CALIB_DEG) / math.sqrt(3), 3)) @ Rc
                pts = cloud(depth, mask, pc_true, Rc_true, fovy)
                # how much of it the frame holds: the same object alone, nothing in front, against what is seen
                p_o, R_o = env.sim.d.xpos[b].copy(), RG.quat_mat(env.sim.d.xquat[b])
                cam_o = Rc_true.T @ (p_o - pc_true)
                f = 0.5 * H / math.tan(math.radians(fovy) / 2)
                uv = [W / 2 + f * cam_o[0] / cam_o[2], H / 2 + f * cam_o[1] / cam_o[2]] if cam_o[2] > 0 else [-1, -1]
                inside = bool(0 <= uv[0] < W and 0 <= uv[1] < H)
                rows.append(dict(layout=k, slot=j, station=slot["station"], surf=slot["surf"], depth=slot.get("depth"),
                                 object=slot["object"], n_pix=int(mask.sum()), dist=float(np.linalg.norm(p_o - pc_true)), cam_est=est,
                                 centre_px=[float(uv[0]), float(uv[1])], centre_in=inside, cam=cam, view=view))  # fmt: skip
                rc.update_scene(env.sim.d, camera=cam)
                img = rc.render().copy()
                np.savez_compressed(OUT / f"see_{name}_L{k}_s{j}.npz", pts=pts.astype(np.float32), p_o=p_o, R_o=R_o, cam_p=pc,
                                    cam_R=Rc, cam_R_true=Rc_true, cam_p_true=pc_true, fovy=fovy, depth=depth.astype(np.float32), mask=mask, rgb=img, robot=robot,
                                    depth_rs=depth_rs)  # fmt: skip
                if j % 6 == 0:
                    from PIL import Image

                    img[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
                    Image.fromarray(img).save(OUT / f"see_{name}_L{k}_s{j}.png")
                print(k, j, slot["station"], slot["surf"], rows[-1]["n_pix"], inside, flush=True)
    json.dump(rows, open(OUT / f"see_{pose}_{'_'.join(map(str, layouts))}.json", "w"))


def seg(port: int, pose: str, min_score: float) -> None:
    """SAM 3 on each saved head frame, prompted with the target's phrase: the best-scored instance is the target (nothing else
    says which one). -> see_sam_L<k>_s<j>.npz with that instance's points, seg_<pose>.json with its overlap with the true mask."""
    from locomani.tasks.scene.services import Client, unpack_mask

    kinds = {
        o["name"]: o["type"]
        for o in json.load(open(os.path.join(os.environ["GRASP_ROOM_OBJECTS"], "objects.json")))["objects"]
    }
    sam, rows = Client(("127.0.0.1", port)), []
    for f in sorted(OUT.glob(f"see_{pose}_L*_s*.npz")):
        z = dict(np.load(f))
        k, j = (int(x[1:]) for x in f.stem.split("_")[2:])
        slot = json.load(open(ROOM.OUT / f"layout_{k}" / "layout.json"))["slots"][j]
        prompt = phrase_of(kinds[slot["object"]])
        found = sam.call(dict(image=z["rgb"], prompts=[prompt]))["results"][prompt]
        found = sorted(((s, unpack_mask(m)) for s, m in found if s >= min_score), key=lambda t: -t[0])
        row = dict(layout=k, slot=j, station=slot["station"], surf=slot["surf"], prompt=prompt, true_px=int(z["mask"].sum()),
                   n=len(found))  # fmt: skip
        mask = np.zeros_like(z["mask"])
        if found:
            score, mask = found[0]
            inter = int((mask & z["mask"]).sum())
            row.update(score=float(score), px=int(mask.sum()), iou=inter / max(1, int((mask | z["mask"]).sum())),
                       best_iou=max(int((m & z["mask"]).sum()) / max(1, int((m | z["mask"]).sum())) for _, m in found))  # fmt: skip
        pts = cloud(z["depth"], mask, z["cam_p"], z["cam_R"], float(z["fovy"]))
        np.savez_compressed(OUT / f"see_sam_L{k}_s{j}.npz", pts=pts.astype(np.float32), p_o=z["p_o"], R_o=z["R_o"])
        rows.append(row)
        print(row, flush=True)
    json.dump(rows, open(OUT / f"seg_{pose}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["see", "seg"])
    ap.add_argument("--port", type=int, default=18832)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--layouts", default="0")
    ap.add_argument(
        "--pose",
        default="start",
        help="stance: at the grasp stance; any other name: the episode start (ROOM_GRASP_LOOK sets it)",
    )
    a = ap.parse_args()
    if a.cmd == "seg":
        seg(a.port, a.pose, a.min_score)
    else:
        see([int(x) for x in a.layouts.split(",")], a.pose)
