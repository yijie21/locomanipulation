"""The one look of an episode (selfmodel_hardgrasp_20261125, LocoManiLab part): one RGB-D frame of the host's head
camera, SAM 3 on the object's name, the object and furniture clouds, GraspGen-X grasps for the right Dex3 hand, cached
so both groups of an episode read the same look.

Camera: the host's own `robot0_head_camera` (robosuite models/assets/robots/g1/robot.xml: fixed on torso_link, pos
0.08 0 0.45, xyaxes 0 -1 0 0.72 0 0.7, fovy 60; "G1's head is a geom on torso, no separate head body"), rendered with our
own mujoco.Renderer at 640 x 480 (MUJOCO_GL=egl; on 4090b libEGL.so.1 comes from look_dev/egllib, see EGL_HINT).

No simulator truth in the look: the camera's pose relative to the pelvis is FK of the joint readings on the host model
(a scratch MjData, the free joint at identity), and the pelvis pose is the caller's estimate (`pelvis`; default: the
host pelvis reading, which the caller must state -- leg odometry can replace it, the clouds are also stored in the
pelvis frame). The robot's own pixels are dropped by FK geometry (segmentation render of the robot's geoms at its read
joint angles: the same model, so this is self-knowledge, not object truth).

Grasp frame -> host wrist_yaw_link: GraspGen-X's unitree_g1 description puts its palm link at T_GP (gripper.urdf
world_joint: xyz 0.07 0 0.02, rpy 90 deg 0 180 deg). The host hand has no palm body: the fingers hang off
wrist_yaw_link, axes parallel to the description's palm (fingers +x, index +z, middle -z). Finger bases, description palm
frame: thumb (0.050, 0, 0), index (0.085, -0.004, 0.024), middle (0.085, -0.004, -0.024); host wrist_yaw_link frame:
thumb (0.067, -0.003, 0), index (0.1192, -0.0046, 0.0285), middle (0.1192, -0.0046, -0.0285) (host robot.xml FK,
2026-11-25). The two frames are aligned by the bases' centroids: wrist -> description palm = (0.0285, -0.0014, 0),
rotation identity. The host hand is ~1.2x the description's (base spread 5.7 vs 4.8 cm): a residual of up to ~1.5 cm.

    look(dr, "apple", out / "look_ep000")  -> dict (and out/look_ep000.npz + .json + _rgb.png + _mask.png)

The mask's source (selfmodel_lmlroute_20261126, the user's decision of 2026-11-26): HG_MASK=sam3 (default) is SAM 3 as
above; HG_MASK=gt (or look(..., mask_source="gt")) replaces only the SAM 3 call by the simulator's segmentation of the
task's target object -- the pixels of the segmentation render (the same render that drops the robot's own pixels) whose
geom belongs to a body of the target's body tree (HGDriver.obj_tree). Everything after the mask (erosion, the 15-point
rule, the clouds, GraspGen-X, the cache) is unchanged; the cache then states mask_source "sim_gt" and the mask's pixel
counts. The default leaves the look exactly as before (no new field).
"""

from __future__ import annotations

import json
import math
import os
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

AUTHKEY = b"locomani-vision-rsi"
SAM_ADDR = ("127.0.0.1", int(os.environ.get("HG_SAM3_PORT", "18852")))
MASK_SOURCE = os.environ.get("HG_MASK", "sam3")  # "sam3" (default) or "gt": the simulator's segmentation of the target
GGX_ADDR = ("127.0.0.1", int(os.environ.get("HG_GGX_PORT", "18853")))
EGL_HINT = "/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/hardgrasp_20261125/locomanilab/look_dev/egllib"
W_PX, H_PX = 640, 480
CAM = "robot0_head_camera"
NEAR, FAR = 0.1, 4.5  # depth kept (m); the approach tasks start 1.5-2.9 m away
VOX = 0.015  # furniture cloud voxel (m)
GGX_PTS = 2048  # object points sent to GraspGen-X
FURN_MAX, FLOOR_Z = 20000, 0.01  # furn_pts: at most this many points, the floor plane (z <= 1 cm) and below dropped
HAND_OFF = np.array([0.13, 0.0, 0.0])  # lml_selfmodel.HAND_OFF: the point the self model tracks
T_GP = np.eye(4)  # grasp frame -> description palm (gripper.urdf world_joint)
T_GP[:3, :3] = [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
T_GP[:3, 3] = [0.07, 0.0, 0.02]
T_WP = np.eye(4)  # host wrist_yaw_link -> description palm (finger-base centroids)
T_WP[:3, 3] = [0.0285, -0.0014, 0.0]
T_G_WRIST = T_GP @ np.linalg.inv(T_WP)  # grasp frame -> host wrist_yaw_link
TIP_G = np.array([0.0, 0.0, 0.07])  # the description's fingertip point (config.json), grasp frame
# the hand's volume in the wrist_yaw_link frame for the furniture check (palm + three fingers, open): x from the wrist
# joint to past the fingertips, y across the palm (+y: the side the palm faces, GraspGen's approach), z between the
# outer finger edges
HAND_BOX = (np.array([0.02, -0.035, -0.05]), np.array([0.20, 0.035, 0.05]))


def _call(addr, req: dict) -> dict:
    c = Client(addr, authkey=AUTHKEY)
    try:
        c.send(req)
        rep = c.recv()
    finally:
        c.close()
    if "error" in rep:
        raise RuntimeError(f"service {addr}: {rep['error']}\n{rep.get('trace', '')}")
    return rep


def sam3(rgb: np.ndarray, prompts: list[str]) -> dict:
    """{prompt: [(score, mask), ...]} from our SAM 3 (scripts/rsi/sam3_server.py protocol)."""
    rep = _call(SAM_ADDR, dict(image=rgb, prompts=list(prompts)))
    out = {}
    for p, inst in rep["results"].items():
        out[p] = [
            (float(s), np.unpackbits(pk["bits"], count=int(np.prod(pk["shape"]))).astype(bool).reshape(pk["shape"]))
            for s, pk in inst
        ]
    return out


class Head:
    """Our renderer on the host model plus the FK of the camera relative to the pelvis from joint readings."""

    def __init__(self, dr):
        import mujoco

        self.mj = mujoco
        m = dr.m
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W_PX)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), H_PX)
        self.r = mujoco.Renderer(m, H_PX, W_PX)
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
        assert self.cam >= 0, f"no {CAM} in the host model"
        fovy = float(m.cam_fovy[self.cam])
        self.f = 0.5 * H_PX / math.tan(math.radians(fovy) / 2)
        self.cx, self.cy = (W_PX - 1) / 2, (H_PX - 1) / 2
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        self.robot_body = np.array([n.startswith("robot0_") for n in names])
        self.scratch = mujoco.MjData(m)

    def cam_in_pelvis(self, dr) -> tuple[np.ndarray, np.ndarray]:
        """Camera pose in the host pelvis body frame, FK of the joint readings (free joint at the origin)."""
        mj, m, s = self.mj, dr.m, self.scratch
        s.qpos[:] = dr.d.qpos  # the scene's other joints are irrelevant to the robot's kinematic chain
        s.qpos[dr.rq : dr.rq + 3] = 0.0
        s.qpos[dr.rq + 3 : dr.rq + 7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_kinematics(m, s)
        mj.mj_camlight(m, s)  # cameras are placed here, not in mj_kinematics
        Rp, pp = s.xmat[dr.pelvis].reshape(3, 3), s.xpos[dr.pelvis]
        return Rp.T @ (s.cam_xpos[self.cam] - pp), Rp.T @ s.cam_xmat[self.cam].reshape(3, 3)

    def frame(self, dr) -> dict:
        mj, r, d = self.mj, self.r, dr.d
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        r.update_scene(d, camera=self.cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=self.cam)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        geom = (seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        own = np.zeros(geom.shape, bool)
        own[geom] = self.robot_body[dr.m.geom_bodyid[seg[..., 0][geom]]]
        valid = np.isfinite(depth) & (depth > NEAR) & (depth < FAR)
        v, u = np.mgrid[0:H_PX, 0:W_PX]
        pc = np.stack([(u - self.cx) / self.f * depth, -(v - self.cy) / self.f * depth, -depth], axis=-1)
        body = np.full(geom.shape, -1, np.int64)  # the body of each pixel's geom (-1: none); only the "gt" mask reads it
        body[geom] = dr.m.geom_bodyid[seg[..., 0][geom]]
        return dict(rgb=rgb, depth=depth, own=own, valid=valid, pc=pc, body=body)


def host_pelvis(dr) -> tuple[np.ndarray, np.ndarray]:
    """The host pelvis body's world pose as read from the simulator (the stated fallback for self-localisation)."""
    return dr.d.xpos[dr.pelvis].copy(), dr.d.xmat[dr.pelvis].reshape(3, 3).copy()


def voxel(P: np.ndarray, s: float) -> np.ndarray:
    if len(P) == 0:
        return P
    k = np.floor(P / s).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return P[np.sort(i)]


def wrist_of(G: np.ndarray) -> np.ndarray:
    """(n,4,4) GraspGen-X grasp-frame poses -> host right wrist_yaw_link poses (same frame)."""
    return G @ T_G_WRIST


def command_form(Tw: np.ndarray) -> dict:
    """Each wrist pose in the self model's command parametrisation (lml_tasks.command: wrist x along the approach,
    rot = rot_z(yaw) rot_y(pitch), no roll): approach yaw, pitch, and the roll the parametrisation cannot express
    (angle between the wrist's y axis and the y axis of rot_z(yaw) rot_y(pitch))."""
    x = Tw[:, :3, 0]
    yaw = np.arctan2(x[:, 1], x[:, 0])
    pitch = np.arcsin(np.clip(-x[:, 2], -1, 1))
    y0 = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)], axis=1)  # rot_z(yaw) rot_y(pitch) e_y
    y = Tw[:, :3, 1]
    roll = np.arctan2(np.einsum("ij,ij->i", np.cross(y0, y), x), np.einsum("ij,ij->i", y0, y))
    return dict(yaw=yaw, pitch=pitch, roll=roll)


def hand_box_hits(Tw: np.ndarray, P: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Number of cloud points P (world) inside each wrist pose's hand box (HAND_BOX grown by margin)."""
    if len(P) == 0:
        return np.zeros(len(Tw), int)
    lo, hi = HAND_BOX[0] - margin, HAND_BOX[1] + margin
    out = np.empty(len(Tw), int)
    for i, T in enumerate(Tw):
        q = (P - T[:3, 3]) @ T[:3, :3]
        out[i] = int(np.all((q > lo) & (q < hi), axis=1).sum())
    return out


def gt_mask(dr, fr: dict) -> np.ndarray:
    """The simulator's mask of the task's target: pixels whose geom's body is in the target's body tree."""
    return np.isin(fr["body"], np.array(sorted(dr.obj_tree), np.int64))


def look(dr, obj_name: str, out_path, pelvis=None, prompts: list[str] | None = None, num_grasps: int = 400,
         seed: int = 0, head: Head | None = None, mask_source: str | None = None) -> dict:  # fmt: skip
    """The look. `pelvis` = (p, R) of the host pelvis body in the world (the robot's own estimate); None -> the host
    reading (stated in the cache as pelvis_source). Returns the cache dict; status 'ok', 'not_recognized' (no SAM 3
    instance / too few object points) or 'no_grasps'. mask_source None -> MASK_SOURCE (HG_MASK)."""
    src_mask = mask_source or MASK_SOURCE
    assert src_mask in ("sam3", "gt"), f"mask source {src_mask!r}"
    from scipy import ndimage

    t0 = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    head = head or Head(dr)
    src = "caller" if pelvis is not None else "host_pelvis_reading"
    pp, Rp = pelvis if pelvis is not None else host_pelvis(dr)
    cp, cR = head.cam_in_pelvis(dr)
    Rc, pc_w = Rp @ cR, pp + Rp @ cp  # camera in the world
    fr = head.frame(dr)
    P_pel = fr["pc"] @ cR.T + cp  # (H, W, 3) pelvis frame
    P_w = P_pel @ Rp.T + pp
    t_render = time.time() - t0
    prompts = prompts or [obj_name]
    t1 = time.time()
    if src_mask == "gt":  # the simulator's mask in SAM 3's place (one instance, score 1); the rest is unchanged
        gm = gt_mask(dr, fr)
        res = {prompts[0]: [(1.0, gm)]}
    else:
        res = sam3(fr["rgb"], prompts)
    t_sam = time.time() - t1
    best = None
    cand = []
    for p in prompts:
        for s, m in res.get(p, []):
            n = int((m & fr["valid"] & ~fr["own"]).sum())
            cand.append(dict(prompt=p, score=s, pixels=int(m.sum()), pts=n))
            if n >= 15 and (best is None or s > best[1]):
                best = (p, s, m)
    meta = dict(obj_name=obj_name, prompts=prompts, sam_instances=cand, pelvis_source=src, pelvis_p=pp.tolist(),
                pelvis_R=Rp.tolist(), cam_p_w=pc_w.tolist(), cam_R_w=Rc.tolist(), cam_p_pel=cp.tolist(),
                cam_R_pel=cR.tolist(), f_px=head.f, size=[W_PX, H_PX], t_render_s=t_render, t_sam_s=t_sam,
                hand_off=HAND_OFF.tolist(), T_G_wrist=T_G_WRIST.tolist(), seed=seed)  # fmt: skip
    if src_mask == "gt":
        meta.update(mask_source="sim_gt", gt_mask_px=int(gm.sum()), gt_mask_pts=int((gm & fr["valid"] & ~fr["own"]).sum()),
                    gt_bodies=sorted(int(b) for b in dr.obj_tree))  # fmt: skip
    arr = dict(rgb=fr["rgb"], depth=fr["depth"].astype(np.float32), own=fr["own"])
    if best is None:
        meta.update(status="not_recognized", mask_score=None)
        mask = np.zeros(fr["valid"].shape, bool)
        P_obj = np.zeros((0, 3))
    else:
        meta.update(prompt_used=best[0], mask_score=best[1])
        mask = best[2]
        inner = ndimage.binary_erosion(mask, iterations=1)
        sel = (inner if (inner & fr["valid"]).sum() >= 15 else mask) & fr["valid"] & ~fr["own"]
        P_obj = P_w[sel]
        c = np.median(P_obj, axis=0)
        P_obj = P_obj[np.linalg.norm(P_obj - c, axis=1) < 0.3]
    grow = ndimage.binary_dilation(mask, iterations=3)
    P_furn = voxel(P_w[fr["valid"] & ~fr["own"] & ~grow], VOX)
    arr.update(mask=mask, obj_w=P_obj.astype(np.float32), furn_w=P_furn.astype(np.float32),
               obj_pel=((P_obj - pp) @ Rp).astype(np.float32), furn_pel=((P_furn - pp) @ Rp).astype(np.float32))  # fmt: skip
    meta.update(n_obj=len(P_obj), n_furn=len(P_furn), obj_center_w=P_obj.mean(0).tolist() if len(P_obj) else None)
    if best is not None and len(P_obj) >= 15:
        t2 = time.time()
        P_in = P_obj
        if len(P_in) > GGX_PTS:  # GraspGen-X's point budget; a fixed draw so the look is repeatable
            P_in = P_in[np.sort(np.random.default_rng(seed).choice(len(P_in), GGX_PTS, replace=False))]
        meta.update(obj_extent_cm=(100 * (P_obj.max(0) - P_obj.min(0))).round(1).tolist())
        try:
            rep = _call(GGX_ADDR, dict(points=P_in.astype(np.float32), num_grasps=num_grasps, seed=seed))
        except RuntimeError as e:  # recorded, the episode then counts as no grasps
            rep = dict(G=np.zeros((0, 4, 4)), conf=np.zeros(0), n_points=len(P_in), s=0.0)
            meta.update(ggx_error=str(e)[:300])
        meta.update(t_ggx_s=time.time() - t2, ggx_server_s=rep["s"], ggx_points=rep["n_points"])
        G = np.asarray(rep["G"], float).reshape(-1, 4, 4)
        Tw = wrist_of(G)
        cf = command_form(Tw) if len(Tw) else dict(yaw=np.zeros(0), pitch=np.zeros(0), roll=np.zeros(0))
        arr.update(grasp_G=G, grasp_score=np.asarray(rep["conf"], float), wrist_T=Tw,
                   hand_w=Tw[:, :3, 3] + Tw[:, :3, :3] @ HAND_OFF if len(Tw) else np.zeros((0, 3)),
                   ad_x=Tw[:, :3, 0] if len(Tw) else np.zeros((0, 3)),  # the wrist x axis: the command's approach
                   palm_n=G[:, :3, 2] if len(G) else np.zeros((0, 3)),  # GraspGen-X's approach (palm normal, wrist +y)
                   tip_w=(G[:, :3, :3] @ TIP_G + G[:, :3, 3]) if len(G) else np.zeros((0, 3)),
                   cmd_yaw=cf["yaw"], cmd_pitch=cf["pitch"], cmd_roll=cf["roll"],
                   furn_hits=hand_box_hits(Tw, P_furn) if len(Tw) else np.zeros(0, int))  # fmt: skip
        meta.update(status="ok" if len(G) else "no_grasps", n_grasps=len(G))
    elif best is not None:
        meta.update(status="not_recognized", n_grasps=0)
    meta["t_total_s"] = time.time() - t0
    # the main thread's cache keys (2026-11-25 message): one npz per episode
    fp = P_furn[P_furn[:, 2] > FLOOR_Z] if len(P_furn) else P_furn
    if len(fp) > FURN_MAX:
        fp = fp[np.sort(np.random.default_rng(0).choice(len(fp), FURN_MAX, replace=False))]
    cam_T = np.eye(4)
    cam_T[:3, :3], cam_T[:3, 3] = Rc, pc_w
    K = len(arr.get("wrist_T", []))
    arr.update(found=np.bool_(meta.get("status") == "ok"), mask_score=np.float32(meta.get("mask_score") or 0.0),
               obj_pts=P_obj.astype(np.float32), furn_pts=fp.astype(np.float32),
               grasp_T=arr.get("wrist_T", np.zeros((0, 4, 4))), grasp_hand=arr.get("hand_w", np.zeros((0, 3))),
               grasp_score=arr.get("grasp_score", np.zeros(0)), cam_T=cam_T, pelvis=np.asarray(dr.sample()["root"], float),
               t_sam=np.float32(t_sam), t_grasp=np.float32(meta.get("t_ggx_s", 0.0)), obj_name=np.str_(obj_name))  # fmt: skip
    meta.update(n_furn_pts=len(fp), n_grasps=K)
    np.savez_compressed(str(out_path) + ".npz", **arr)
    Path(str(out_path) + ".json").write_text(json.dumps(meta, indent=1))
    try:
        from PIL import Image

        Image.fromarray(fr["rgb"]).save(str(out_path) + "_rgb.png")
        ov = fr["rgb"].copy()
        ov[mask] = (0.5 * ov[mask] + [0, 127, 0]).astype(np.uint8)
        Image.fromarray(ov).save(str(out_path) + "_mask.png")
    except Exception:
        pass
    return dict(meta=meta, **arr)


def load(out_path) -> dict:
    """The cached look (both groups of an episode read this)."""
    z = np.load(str(out_path) + ".npz", allow_pickle=False)
    return dict(meta=json.loads(Path(str(out_path) + ".json").read_text()), **{k: z[k] for k in z.files})


def prompt_of(dr) -> str:
    """The SAM 3 prompt: the task's own language name of its target (IThor lift / approach classes: TARGET_LANG, e.g.
    "salt shaker"; else TARGET_NAME with underscores as spaces). Task metadata, not simulator state."""
    env = dr.env
    lang = getattr(env, "TARGET_LANG", None)
    return lang if isinstance(lang, str) and lang else str(getattr(env, "TARGET_NAME", "object")).replace("_", " ")


# -- self-check after the episode (simulator truth: scoring / audit only) ---------------------------------------------
def true_obj_pos(dr, obj_key: str | None = None) -> tuple[str, np.ndarray]:
    """World position of the task object's root body (simulator truth). obj_key: a key of env.objects; None -> the
    task's target object if the env names one, else the only object."""
    env = dr.env
    objs = getattr(env, "objects", {})
    key = obj_key
    if key is None:
        for a in ("TARGET_NAME", "obj_name", "target_obj_name", "target_object_name", "object_name"):
            v = getattr(env, a, None)
            if isinstance(v, str) and v in objs:
                key = v
                break
    if key is None and len(objs) == 1:
        key = next(iter(objs))
    assert key in objs, f"object key {key!r} not in env.objects {list(objs)}"
    bid = dr.m.body(objs[key].root_body).id
    return key, dr.d.xpos[bid].copy()


def check(cache: dict, dr, obj_key: str | None = None) -> dict:
    """Object-cloud centre vs the true object position (cm), and the closest cloud point to it."""
    key, p = true_obj_pos(dr, obj_key)
    P = np.asarray(cache["obj_w"], float)
    if len(P) == 0:
        return dict(obj_key=key, true_w=p.tolist(), center_err_cm=None)
    c = P.mean(0)
    out = dict(obj_key=key, true_w=p.tolist(), center_w=c.tolist(), center_err_cm=100 * float(np.linalg.norm(c - p)),
               center_err_xy_cm=100 * float(np.linalg.norm((c - p)[:2])), nearest_cm=100 * float(np.min(np.linalg.norm(P - p, axis=1))))  # fmt: skip
    if "hand_w" in cache and len(cache["hand_w"]):
        out["best_grasp_hand_to_true_cm"] = 100 * float(np.linalg.norm(cache["hand_w"][0] - p))
    return out
