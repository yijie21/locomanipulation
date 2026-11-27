# notebook: skip render for paper Figure 1, not an experiment
"""fig1_render.py: Figure 1 teaser frames from fig1_replay.py's close-time snapshots (MuJoCo offscreen EGL, mujoco.Renderer).

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 python fig1_render.py search <tag>
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 python fig1_render.py final <tag> <az_off> <el> <dist>

State: each arm's model (MJB, the simulator's own) and its MjData at the close: mj_setState(mjSTATE_INTEGRATION) +
mj_forward, then the body / geom poses overwritten with the ones the simulator held when lrt_run sampled the close
(after mj_step MjData's kinematics belong to the step's start, one physics step before qpos; the record's hand point and
close distances were taken from those), so the rendered hand point is the record's to the bit.

Look: geom groups 1-2 (the scene's visual geoms; group 0 = collision geoms hidden, robosuite's convention); the model's
5 lights kept, one soft directional key light added (castshadow); vis.quality.offsamples 16, shadowsize 4096; pixels
that see no geometry composited to light grey (none in the chosen view, reported).

search: free cameras around the close: lookat = midpoint of the pelvis and the hand / object centre, azimuth offset from
the robot's heading 0-345 step 15, elevation -15..-45, distance 1.6-3.0 m. A view is valid when, for both arms: no
non-robot geom between the camera and the object or the hand (mj_ray, visible groups), the object >= 50 % visible
(segmentation against an isolated render), <= 5 % of it hidden by furniture and <= 5 % by robot links other than the
hand, the hand >= 40 % visible with <= 10 % hidden by furniture, and the whole robot inside the frame.
"""

import copy
import hashlib
import json
import os
import sys

import mujoco
import numpy as np
from PIL import Image

F = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/fig1_teaser_20261126"
MODE, TAG = sys.argv[1], sys.argv[2]
S = f"{F}/state/{TAG}"
OUT = f"{F}/render/{TAG}"
os.makedirs(OUT, exist_ok=True)
ARMS = ("alone", "reachable")
GROUPS = (0, 1, 1, 0, 0, 0)
HIDE = 5
BG = np.array([236, 236, 236], np.uint8)
GHOST_RGBA = (0 / 255, 114 / 255, 178 / 255, 0.40)
KEY = dict(diffuse=0.35, specular=0.05, ambient=0.0)
KEY_TILT = 25.0


def load_arm(m: str):
    M = mujoco.MjModel.from_binary_path(f"{S}/{m}.mjb")
    z = np.load(f"{S}/{m}.npz")
    meta = json.load(open(f"{S}/{m}.json"))
    return M, z, meta


def set_state(M, z):
    D = mujoco.MjData(M)
    mujoco.mj_setState(M, D, z["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(M, D)
    D.xpos[:], D.xmat[:] = z["xpos"], z["xmat"]
    D.geom_xpos[:], D.geom_xmat[:] = z["geom_xpos"], z["geom_xmat"]
    return D


def file_md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


arms = {m: load_arm(m) for m in ARMS}
M0, z0, meta0 = arms["alone"]
same_model = file_md5(f"{S}/alone.mjb") == file_md5(f"{S}/reachable.mjb")
names = [mujoco.mj_id2name(M0, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(M0.nbody)]
ROOT = int(M0.body_rootid[meta0["wrist_body"]])
ROBOT_B = np.where(M0.body_rootid == ROOT)[0]
HAND_B = np.array([b for b in ROBOT_B if names[b].startswith(("robot0_right_hand", "robot0_right_wrist"))])
OBJ_B = np.array(meta0["obj_tree"])
gb = M0.geom_bodyid
VIS = np.isin(M0.geom_group, [g for g in range(6) if GROUPS[g]])
G_ROBOT = np.isin(gb, ROBOT_B) & VIS
G_HAND = np.isin(gb, HAND_B) & VIS
G_OBJ = np.isin(gb, OBJ_B) & VIS
G_SCENE = ~np.isin(gb, ROBOT_B) & ~np.isin(gb, OBJ_B) & VIS


def prep_model(M, keep=None, w=4000, h=4000, samples=16):
    """A render copy: offscreen buffer, AA, shadows; keep = geoms to keep visible (others to the hidden group)."""
    R = copy.deepcopy(M)
    R.vis.global_.offwidth, R.vis.global_.offheight = w, h
    R.vis.quality.offsamples = samples
    R.vis.quality.shadowsize = 4096
    if keep is not None:
        R.geom_group[~keep] = HIDE
    return R


def option():
    o = mujoco.MjvOption()
    for g in range(6):
        o.geomgroup[g] = GROUPS[g]
    return o


def free_cam(lookat, az, el, dist):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.azimuth, c.elevation, c.distance = az, el, dist
    return c


def cam_pos(lookat, az, el, dist):
    a, e = np.radians(az), np.radians(el)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return np.asarray(lookat) - dist * fwd


def add_key_light(scn, cpos, lookat):
    """A soft directional key light from above, tilted KEY_TILT deg towards the camera's side, castshadow (near-vertical,
    so the room's walls throw no large shadows across the floor); the model's lights stay."""
    if scn.nlight >= len(scn.lights):
        return
    lt = scn.lights[scn.nlight]
    hz = np.asarray(cpos, float) - np.asarray(lookat, float)
    hz[2] = 0.0
    hz /= np.linalg.norm(hz)
    t = np.radians(KEY_TILT)
    off = np.sin(t) * hz + np.cos(t) * np.array([0.0, 0.0, 1.0])
    lt.pos[:] = np.asarray(lookat) + 3.0 * off
    lt.dir[:] = -off
    lt.directional = 1
    lt.headlight = 0
    lt.castshadow = 1
    lt.diffuse[:] = KEY["diffuse"]
    lt.specular[:] = KEY["specular"]
    lt.ambient[:] = KEY["ambient"]
    lt.attenuation[:] = (1.0, 0.0, 0.0)
    lt.cutoff, lt.exponent = 180.0, 0.0
    scn.nlight += 1


def add_ghost(scn, Mg, Dg, opt):
    """R's robot as a translucent copy in A's scene (mjv_addGeoms on a model copy where only robot geoms are visible)."""
    n0 = scn.ngeom
    mujoco.mjv_addGeoms(Mg, Dg, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    for i in range(n0, scn.ngeom):
        g = scn.geoms[i]
        g.rgba[:] = GHOST_RGBA
        g.matid = -1
        g.emission = 0.25
        g.specular = 0.2
        g.reflectance = 0.0
    return scn.ngeom - n0


def render(rnd, D, cam, opt, key=True, ghost=None, seg=False, marker=None):
    rnd.update_scene(D, camera=cam, scene_option=opt)
    scn = rnd.scene
    if ghost is not None:
        add_ghost(scn, *ghost, opt)
    if marker is not None and scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0, 0]),
                            np.asarray(marker, float), np.eye(3).ravel(), np.array([1.0, 0.0, 1.0, 1.0], np.float32))
        scn.geoms[scn.ngeom].emission = 1.0
        scn.ngeom += 1
    if key:
        p = cam_pos(cam.lookat, cam.azimuth, cam.elevation, cam.distance)
        add_key_light(scn, p, cam.lookat)
    if seg:
        rnd.enable_segmentation_rendering()
        img = rnd.render().copy()
        rnd.disable_segmentation_rendering()
        return img
    return rnd.render().copy()


def seg_geom(img):
    """(H, W) geom id per pixel, -1 where none."""
    ids = img[..., 0].astype(int)
    ty = img[..., 1].astype(int)
    return np.where(ty == int(mujoco.mjtObj.mjOBJ_GEOM), ids, -1)


def ray_clear(M, D, p, cpos):
    """No visible non-robot geom on the segment p -> camera (robot geoms are stepped over)."""
    d = cpos - p
    L = np.linalg.norm(d)
    d = d / L
    start, travelled = p.copy(), 0.0
    gid = np.zeros(1, np.int32)
    grp = np.array(GROUPS, np.uint8)
    for _ in range(40):
        x = mujoco.mj_ray(M, D, start, d, grp, 1, -1, gid)
        if x < 0 or travelled + x >= L:
            return True
        g = int(gid[0])
        if not G_ROBOT[g] and not G_OBJ[g]:
            return False
        travelled += x + 1e-4
        start = p + travelled * d
    return True


def setup():
    st = {}
    for m in ARMS:
        M, z, meta = arms[m]
        D = set_state(M, z)
        wb = meta["wrist_body"]
        hp = D.xpos[wb] + D.xmat[wb].reshape(3, 3) @ np.asarray(meta["hand_off"])
        st[m] = dict(M=M, D=D, z=z, meta=meta, hp=hp, obj=D.xpos[meta["obj_bid"]].copy(),
                     hp_err=float(np.abs(hp - z["hp"]).max()), obj_err=float(np.abs(D.xpos[meta["obj_bid"]] - z["obj_pos"]).max()))
        print(f"{m}: hand point from the rendered state vs the snapshot max |d| {st[m]['hp_err']:.3g} m; object {st[m]['obj_err']:.3g} m; "
              f"close_dist_cm {100 * np.linalg.norm(hp - z['goal']):.9f} (snapshot {meta['close_dist_cm']:.9f})")
    return st


def base_view(st):
    D = st["alone"]["D"]
    pel = D.xpos[ROOT].copy()
    Rp = D.xmat[ROOT].reshape(3, 3)
    yaw = float(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0])))
    focus = np.mean([st[m]["hp"] for m in ARMS] + [st[m]["obj"] for m in ARMS], axis=0)
    return pel, yaw, focus


def visibility(rf, iso, st, cam, opt, w, h):
    """Per arm: object / hand pixel shares (visible, hidden by furniture, by robot links other than the hand, by the
    hand) against renders of the object alone / the hand alone; the whole robot (rendered alone) inside the frame."""
    out = {}
    for m in ARMS:
        D = st[m]["D"]
        full = seg_geom(render(rf, D, cam, opt, key=False, seg=True))
        r = {}
        for name, G in (("obj", G_OBJ), ("hand", G_HAND)):
            mask = seg_geom(render(iso[name], D, cam, opt, key=False, seg=True)) >= 0
            n = int(mask.sum())
            f = full[mask]
            fc = np.clip(f, 0, None)
            fv = f >= 0
            r[name] = dict(px=n, self=float((fv & G[fc]).sum()) / max(n, 1), furn=float((fv & G_SCENE[fc]).sum()) / max(n, 1),
                           hand=float((fv & G_HAND[fc] & ~G[fc]).sum()) / max(n, 1),
                           arm=float((fv & G_ROBOT[fc] & ~G_HAND[fc] & ~G[fc]).sum()) / max(n, 1))
        rob = seg_geom(render(iso["robot"], D, cam, opt, key=False, seg=True)) >= 0
        ys, xs = np.where(rob)
        r["robot_in_frame"] = bool(len(xs) and xs.min() > 0.02 * w and xs.max() < 0.98 * w and ys.min() > 0.02 * h and ys.max() < 0.98 * h)
        out[m] = r
    return out


def valid(v, rays):
    ok = rays
    for m in ARMS:
        o, hd = v[m]["obj"], v[m]["hand"]
        ok &= o["px"] > 0 and o["self"] >= 0.5 and o["furn"] <= 0.05 and o["arm"] <= 0.05
        ok &= hd["px"] > 0 and hd["self"] >= 0.4 and hd["furn"] <= 0.10
        ok &= v[m]["robot_in_frame"]
    return bool(ok)


def search():
    st = setup()
    pel, yaw, focus = base_view(st)
    lookat = 0.5 * (pel + focus)
    print(f"pelvis {pel.round(3).tolist()} heading {yaw:.1f} deg; hand/object focus {focus.round(3).tolist()}; lookat {lookat.round(3).tolist()}")
    w, h = 480, 320
    Mr = prep_model(M0, w=w, h=h, samples=4)
    rf = mujoco.Renderer(Mr, h, w)
    iso = {k: mujoco.Renderer(prep_model(M0, keep=G, w=w, h=h, samples=4), h, w) for k, G in (("obj", G_OBJ), ("hand", G_HAND), ("robot", G_ROBOT))}
    opt = option()
    rows = []
    for az_off in range(0, 360, 15):
        for el in (-15, -20, -25, -30, -35, -45):
            for dist in (1.6, 2.0, 2.4, 2.8, 3.2):
                az = yaw + az_off
                cam = free_cam(lookat, az, el, dist)
                cp = cam_pos(lookat, az, el, dist)
                rays = all(ray_clear(st[m]["M"], st[m]["D"], p, cp) for m in ARMS for p in (st[m]["obj"], st[m]["hp"]))
                v = visibility(rf, iso, st, cam, opt, w, h)
                ok = valid(v, rays)
                rows.append(dict(az_off=az_off, el=el, dist=dist, rays=rays, valid=ok, vis=v))
    good = [r for r in rows if r["valid"]]
    print(f"views {len(rows)}, valid {len(good)}")
    for r in good:
        a, b = r["vis"]["alone"], r["vis"]["reachable"]
        print(f"  az_off {r['az_off']:3d} el {r['el']:3d} dist {r['dist']:.1f} | obj vis A {a['obj']['self']:.2f} R {b['obj']['self']:.2f} "
              f"(px {a['obj']['px']}/{b['obj']['px']}; arm {a['obj']['arm']:.2f}/{b['obj']['arm']:.2f}, hand {a['obj']['hand']:.2f}/{b['obj']['hand']:.2f}) "
              f"| hand vis {a['hand']['self']:.2f}/{b['hand']['self']:.2f}")
    json.dump(dict(pelvis=pel.tolist(), heading_deg=yaw, focus=focus.tolist(), lookat=lookat.tolist(), rows=rows),
              open(f"{OUT}/search.json", "w"), indent=1)
    # contact sheet of valid views (A, RGB) at elevation -25 and -30, nearest valid distance per azimuth
    Mr2 = prep_model(M0, w=w, h=h, samples=8)
    r2 = mujoco.Renderer(Mr2, h, w)
    tiles = []
    for el in (-20, -25, -30, -35):
        row = []
        for az_off in range(0, 360, 30):
            cand = [r for r in good if r["el"] == el and r["az_off"] == az_off]
            if cand:
                c = min(cand, key=lambda r: r["dist"])
                img = render(r2, st["alone"]["D"], free_cam(lookat, yaw + az_off, el, c["dist"]), opt)
                img[:24, :160] = 255
            else:
                img = np.full((h, w, 3), 40, np.uint8)
            row.append(img)
        tiles.append(np.concatenate(row, 1))
    sheet = np.concatenate(tiles, 0)
    Image.fromarray(sheet).save(f"{OUT}/search_sheet.png")
    print("search sheet: rows el -20 -25 -30 -35; cols az_off 0..330 step 30 (nearest valid distance; dark = none valid)")



def models_for_render(w, h, samples):
    Mr = prep_model(M0, w=w, h=h, samples=samples)
    Mg = prep_model(M0, keep=G_ROBOT, w=w, h=h, samples=samples)  # the ghost's geoms: robot only
    return Mr, Mg


def composite_bg(img, sg):
    """Pixels that see no geometry -> light grey (returns the count)."""
    void = sg < 0
    img[void] = BG
    return int(void.sum())


def preview(cands):
    st = setup()
    pel, yaw, focus = base_view(st)
    lookat = 0.5 * (pel + focus)
    w, h = 900, 600
    Mr, Mg = models_for_render(w, h, 8)
    rnd = mujoco.Renderer(Mr, h, w)
    opt = option()
    rows = []
    for az_off, el, dist in cands:
        cam = free_cam(lookat, yaw + az_off, el, dist)
        a = render(rnd, st["alone"]["D"], cam, opt)
        r = render(rnd, st["reachable"]["D"], cam, opt)
        o = render(rnd, st["alone"]["D"], cam, opt, ghost=(Mg, st["reachable"]["D"]))
        rows.append(np.concatenate([a, r, o], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(f"{OUT}/preview.png")
    print("preview rows:", cands, "cols: A, R, A + R ghost")


def intrinsics(M, w, h):
    fovy = float(M.vis.global_.fovy)
    fy = 0.5 * h / np.tan(np.radians(fovy) / 2)
    return dict(fovy_deg=fovy, width=w, height=h, fx=fy, fy=fy, cx=w / 2, cy=h / 2,
                convention="pinhole, square pixels; u right, v down; (0, 0) = the image's top-left corner, pixel (i, j) "
                           "covers [i, i+1) x [j, j+1) (its centre is i + 0.5, j + 0.5); MuJoCo free camera, vis.global_.fovy")


def extrinsics(rnd):
    # mjv_updateScene writes two eye cameras (+-vis.global_.ipd / 2 along the right axis); a mono render uses their
    # average (mjv_averageCamera), so the extrinsics are the average's
    c = mujoco.mjv_averageCamera(rnd.scene.camera[0], rnd.scene.camera[1])
    pos, fwd, up = np.array(c.pos, float), np.array(c.forward, float), np.array(c.up, float)
    right = np.cross(fwd, up)
    R_wc = np.stack([right, -up, fwd])  # rows: camera x (right), y (down), z (forward) in world
    return dict(position_world=pos.tolist(), forward_world=fwd.tolist(), up_world=up.tolist(),
                R_world_to_cam=R_wc.tolist(), t_world_to_cam=(-R_wc @ pos).tolist(),
                frame="OpenCV camera frame: x right, y down, z forward; X_cam = R_world_to_cam @ X_world + t_world_to_cam",
                mjv_frustum=dict(near=float(c.frustum_near), far=float(c.frustum_far), top=float(c.frustum_top),
                                 bottom=float(c.frustum_bottom), width=float(c.frustum_width), center=float(c.frustum_center)))


def project(P, ex, K):
    R, t = np.asarray(ex["R_world_to_cam"]), np.asarray(ex["t_world_to_cam"])
    X = R @ np.asarray(P, float) + t
    return [K["fx"] * X[0] / X[2] + K["cx"], K["fy"] * X[1] / X[2] + K["cy"]], float(X[2])


def marker_check(rnd, D, cam, opt, P):
    """Render a small emissive sphere at P; the centroid of the changed pixels (+0.5) against the projection."""
    a = render(rnd, D, cam, opt, key=True).astype(int)
    b = render(rnd, D, cam, opt, key=True, marker=P).astype(int)
    m = np.abs(a - b).max(-1) > 40
    if not m.any():
        return None
    ys, xs = np.where(m)
    return [float(xs.mean() + 0.5), float(ys.mean() + 0.5), int(m.sum())]


def wpt(w, key="cmd_h"):
    p, y, g = np.asarray(w["p_cmd"], float), float(w["yaw_cmd"]), np.asarray(w[key], float)
    c, s_ = np.cos(y), np.sin(y)
    return np.array([p[0] + c * g[0] - s_ * g[1], p[1] + s_ * g[0] + c * g[1], g[2]])


def final(az_off, el, dist, close_dist):
    st = setup()
    pel, yaw, focus = base_view(st)
    lookat = 0.5 * (pel + focus)
    W, H = 3600, 2400
    Mr, Mg = models_for_render(W, H, 16)
    rnd = mujoco.Renderer(Mr, H, W)
    opt = option()
    cams = {"main": free_cam(lookat, yaw + az_off, el, dist), "closeup": free_cam(focus, yaw + az_off, el, close_dist)}
    task, ep = meta0["task"], meta0["ep"]
    rec = {m: json.load(open(f"/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126/scalebfm/episodes/{task}/{m}/ep{ep:+05d}.json")) for m in ARMS}
    kA = next(w for w in rec["alone"]["reaches"] if w["k"] == 1)
    kR = next(w for w in rec["reachable"]["reaches"] if w["k"] == 1)
    pts = {
        "p_star": dict(world=rec["reachable"]["plan"]["goal"], how="the grasp point: record plan.goal (world; the same in A and R; close_dist_cm is the hand's distance to it)"),
        "c_star": dict(world=wpt(kR).tolist(), how="R's entry command (reaches k = 1, reach_no 2): p_cmd + Rz(yaw_cmd) cmd_h[:2], z = cmd_h[2] -- the command target in the heading frame placed at the pelvis pose of the command row (lrt_eval's tgt_w form); = ref_end_w up to the IK error"),
        "c_star_ref_end_w": dict(world=kR["ref_end_w"], how="alternative: R's entry reference end, FK of the planned reference (record ref_end_w)"),
        "c_star_rel_goal": dict(world=(np.asarray(rec["reachable"]["plan"]["goal"]) + np.r_[(np.array([[np.cos(kR['yaw_cmd']), -np.sin(kR['yaw_cmd'])], [np.sin(kR['yaw_cmd']), np.cos(kR['yaw_cmd'])]]) @ (np.asarray(kR['cmd_h'][:2]) - np.asarray(kR['goal_h'][:2]))), kR['cmd_h'][2] - kR['goal_h'][2]]).tolist(),
                                how="alternative: p* + R's correction (cmd_h - goal_h, heading frame, rotated by yaw_cmd): the command relative to the grasp point, without the odometry offset"),
        "a_cmd": dict(world=wpt(kA).tolist(), how="A's entry command (k = 1), same form as c_star (A commands the grasp point itself: cmd_h = goal_h)"),
        "A_landing": dict(world=st["alone"]["hp"].tolist(), how="A's hand point at the close (wrist_yaw_link + R @ (0.13, 0, 0)), replayed; |it - p*| = record close_dist_cm (bit-equal)"),
        "R_landing": dict(world=st["reachable"]["hp"].tolist(), how="R's hand point at the close, replayed; |it - p*| = record close_dist_cm (bit-equal)"),
        "object_A_close": dict(world=st["alone"]["obj"].tolist(), how="candle body origin at A's close (simulator), replayed; |hand - it| = record close_obj_cm"),
        "object_R_close": dict(world=st["reachable"]["obj"].tolist(), how="candle body origin at R's close, replayed"),
    }
    out = dict(task=task, ep=ep, controller="scalebfm", cameras={}, points=pts,
               distances_cm=dict(A_close_dist=rec["alone"]["close_dist_cm"], R_close_dist=rec["reachable"]["close_dist_cm"],
                                 R_correction_cm=kR.get("corr_cm"), c_star_to_p_star=100 * float(np.linalg.norm(wpt(kR) - np.asarray(rec["reachable"]["plan"]["goal"])))))
    files = {}
    for cname, cam in cams.items():
        K = intrinsics(Mr, W, H)
        rnd.update_scene(st["alone"]["D"], camera=cam, scene_option=opt)
        ex = extrinsics(rnd)
        out["cameras"][cname] = dict(mjv=dict(type="free", lookat=list(map(float, cam.lookat)), azimuth_deg=float(cam.azimuth),
                                               elevation_deg=float(cam.elevation), distance=float(cam.distance),
                                               azimuth_offset_from_heading_deg=az_off, robot_heading_deg=yaw),
                                     intrinsics=K, extrinsics=ex)
        for k, v in pts.items():
            uv, zc = project(v["world"], ex, K)
            v.setdefault("pixel", {})[cname] = dict(u=uv[0], v=uv[1], depth_m=zc)
        imgs = {}
        sgA = seg_geom(render(rnd, st["alone"]["D"], cam, opt, key=False, seg=True))
        sgR = seg_geom(render(rnd, st["reachable"]["D"], cam, opt, key=False, seg=True))
        imgs["A_full"] = render(rnd, st["alone"]["D"], cam, opt)
        imgs["R_full"] = render(rnd, st["reachable"]["D"], cam, opt)
        imgs["overlay_A_Rghost"] = render(rnd, st["alone"]["D"], cam, opt, ghost=(Mg, st["reachable"]["D"]))
        voids = {k: composite_bg(imgs[k], sgA if k != "R_full" else sgR) for k in imgs}
        rob = G_ROBOT[np.clip(sgR, 0, None)] & (sgR >= 0)
        layer = np.dstack([imgs["R_full"], (rob * 255).astype(np.uint8)])
        imgs["R_robot_layer_rgba"] = layer
        out["cameras"][cname]["void_pixels_composited_to_grey"] = voids
        for k, im in imgs.items():
            fn = f"teaser_{cname}_{k}.png"
            Image.fromarray(im).save(f"{OUT}/{fn}")
            files[fn] = dict(size=list(im.shape))
        # projection check: markers at visible test points (each point moved 15 cm towards the camera, and lookat)
        chk = []
        for k in ("p_star", "c_star", "A_landing", "R_landing"):
            P = np.asarray(pts[k]["world"], float)
            cpos = np.asarray(ex["position_world"])
            Q = P + 0.15 * (cpos - P) / np.linalg.norm(cpos - P) if cname == "main" else P + 0.05 * (cpos - P) / np.linalg.norm(cpos - P)
            uv, _ = project(Q, ex, K)
            got = marker_check(rnd, st["alone"]["D"], cam, opt, Q)
            chk.append(dict(point=k + " moved towards the camera", projected=uv, marker_centroid=got,
                            err_px=None if got is None else float(np.hypot(got[0] - uv[0], got[1] - uv[1]))))
        out["cameras"][cname]["projection_check"] = chk
        print(cname, "projection check:", [(c["point"], None if c["err_px"] is None else round(c["err_px"], 2)) for c in chk], "void px", voids)
    out["render"] = dict(renderer=f"mujoco {mujoco.__version__} Renderer, MUJOCO_GL=egl", offsamples=int(Mr.vis.quality.offsamples),
                         shadowsize=int(Mr.vis.quality.shadowsize), geomgroups=list(GROUPS), model_lights=int(Mr.nlight),
                         key_light=dict(KEY, kind="directional, castshadow, from above, tilted 25 deg towards the camera's side", tilt_deg=KEY_TILT),
                         background=BG.tolist(), ghost_rgba=list(GHOST_RGBA), ghost_emission=0.25, size=[W, H],
                         same_model_A_R=same_model)
    out["files"] = files
    json.dump(out, open(f"{OUT}/teaser_points.json", "w"), indent=1)
    print(json.dumps({k: dict(world=np.round(v["world"], 5).tolist(), px={c: [round(v["pixel"][c]["u"], 1), round(v["pixel"][c]["v"], 1)] for c in v["pixel"]}) for k, v in pts.items()}, indent=0))


if __name__ == "__main__":
    if MODE == "search":
        search()
    elif MODE == "preview":
        preview([tuple(float(x) for x in c.split(",")) for c in sys.argv[3:]])
    elif MODE == "final":
        final(float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
