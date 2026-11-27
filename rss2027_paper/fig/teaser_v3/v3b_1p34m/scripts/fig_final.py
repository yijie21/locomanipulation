"""fig_final.py: Figure 1 v3b's view check and final panels (README section 4). notebook: skip figure render, not an
experiment.

    MUJOCO_GL=egl ... python fig_final.py <runs> <placement> check <out dir>
    MUJOCO_GL=egl ... python fig_final.py <runs> <placement> final <out dir> <az> <el> <d> <d_close>

Helpers (look, masks, cameras, segmentation) are fig_render.py's (v3a's contact sheet). States: each arm's snapshot at
the close command ("entry_end") and at the panel moment ("after3500ms", 3.50 s after it), set as fig_render.load does
(mj_setState + mj_forward, poses overwritten with the simulator's; the palm checked against the snapshot to the bit).
Main camera: v3a's rule (lookat over the midpoint of A's pelvis and vase at the close command, z 0.85; az from the
robot's heading, el down). Close-up camera: same az / el, lookat the grasp point p*, distance d_close.
check: main view at az -30, -40, -50 (el 45, d 2.0) at the panel moment for both arms (head >= 80 %, right hand >= 50 %,
vase >= 50 %, whole robot inside), the close command's numbers alongside; close-ups at 0.5 / 0.6 / 0.7 m (hand, vase
visible fraction and whether either touches the border).
final: A_full / R_full (panel moment) 2400 x 1600 and the hand close-ups (close command) 1600 x 1200, each rendered at
2x (4800 x 3200, 3200 x 2400; offsamples 16, shadowsize 4096) and averaged 2 x 2; robot layers RGBA (panel moment,
main camera): RGB = the full render, alpha = the robot's visible pixels (segmentation without multisampling, at 2x;
RGB averaged over the robot's sub-pixels only, alpha = their share); teaser_points.json: p*, c*, the landings, pixels in
each camera, intrinsics (output size) and extrinsics (mjv_averageCamera of the two eye cameras), a marker check.
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

RUNS, PI, MODE, OUT = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.argv = [sys.argv[0], str(RUNS), str(PI), "lib"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_render as FR  # noqa: E402

ARMS = ("A", "R")
CLOSE, PANEL = "entry_end", "after3500ms"
SS = 2  # supersampling factor
INTEG = mujoco.mjtState.mjSTATE_INTEGRATION


def load():
    xs = {a: RUNS / "scene" / f"{a}_p{PI:02d}" / "scene_locomani.xml" for a in ARMS}
    md5 = {a: FR.md5(xs[a]) for a in ARMS}
    assert md5["A"] == md5["R"], md5
    M = mujoco.MjModel.from_xml_path(str(xs["A"]))
    Z = {a: np.load(RUNS / "snap" / f"{a}_p{PI:02d}.npz") for a in ARMS}
    st = {}
    for tag in (CLOSE, PANEL):
        st[tag] = {}
        for a in ARMS:
            z = Z[a]
            D = mujoco.MjData(M)
            mujoco.mj_setState(M, D, z[f"{tag}__state"], INTEG)
            mujoco.mj_forward(M, D)
            D.xpos[:], D.xmat[:] = z[f"{tag}__xpos"], z[f"{tag}__xmat"]
            D.geom_xpos[:], D.geom_xmat[:] = z[f"{tag}__geom_xpos"], z[f"{tag}__geom_xmat"]
            wb = M.body("right_wrist_yaw_link").id
            palm = D.xpos[wb] + D.xmat[wb].reshape(3, 3) @ np.array([0.0415, 0.0, 0.0])
            st[tag][a] = dict(D=D, palm=palm, vase=D.xpos[M.body("vase").id].copy(),
                              t_after_close=float(z[f"{tag}__t"] - z[f"{CLOSE}__t"]),
                              palm_check_m=float(np.abs(palm - z[f"{tag}__palm"]).max()))
            print(f"{tag} {a}: t - t_close {st[tag][a]['t_after_close']:.3f} s; palm vs snapshot max |d| "
                  f"{st[tag][a]['palm_check_m']:.3g} m", flush=True)
    return M, Z, st, md5["A"]


def main_cam(M, st, az, el, d):
    S = {"A": st[CLOSE]["A"]}
    S["A"]["vase"] = st[CLOSE]["A"]["vase"]
    yaw = FR.heading(S)
    la = FR.lookat(M, S)
    return FR.cam(la, yaw, az, el, d), yaw, la


def vis_main(M, G, st, tag, la, yaw, views):
    return FR.visibility(M, G, {a: dict(D=st[tag][a]["D"]) for a in ARMS}, la, yaw, views)


def vis_close(M, G, st, p_star, yaw, az, el, dists, w=480, h=360):
    rf = mujoco.Renderer(FR.prep(M, G, w, h, 0), h, w)
    iso = {k: mujoco.Renderer(FR.prep(M, G, w, h, 0, keep=G[k]), h, w) for k in ("hand", "vase")}
    opt = FR.option()
    out = []
    for d in dists:
        c = FR.cam(p_star, yaw, az, el, d)
        r = dict(d=d)
        for a in ARMS:
            D = st[CLOSE][a]["D"]
            full = FR.seg_geom(FR.render(rf, D, c, opt, seg=True))
            v = {}
            for k in ("hand", "vase"):
                m = FR.seg_geom(FR.render(iso[k], D, c, opt, seg=True)) >= 0
                f = full[m]
                v[k] = float(((f >= 0) & G[k][np.clip(f, 0, None)]).sum()) / max(int(m.sum()), 1)
                ys, xs = np.where(m)
                v[k + "_touches_border"] = bool(len(xs) and (xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1))
            r[a] = v
        out.append(r)
    for x in [rf, *iso.values()]:
        x.close()
    return out


def check():
    M, Z, st, scene_md5 = load()
    G = FR.masks(M)
    res = dict(placement=PI, scene_md5=scene_md5, main={}, close={})
    for az in (-30, -40, -50):
        c, yaw, la = main_cam(M, st, az, 45, 2.0)
        for tag in (PANEL, CLOSE):
            r = vis_main(M, G, st, tag, la, yaw, [(az, 45, 2.0)])[0]
            res["main"][f"az{az}_{tag}"] = r
            print(f"main az {az} el 45 d 2.0 @ {tag}: valid {r['valid']} | " + " | ".join(
                f"{a} head {r[a]['head']:.2f} hand {r[a]['hand']:.2f} vase {r[a]['vase']:.2f} in {r[a]['in_frame']}" for a in ARMS), flush=True)
    p_star = Z["A"]["goal__entry"]
    for az in (-30, -40, -50):
        _, yaw, _ = main_cam(M, st, az, 45, 2.0)
        rows = vis_close(M, G, st, p_star, yaw, az, 45, (0.5, 0.6, 0.7))
        res["close"][f"az{az}"] = rows
        for r in rows:
            print(f"close-up az {az} d {r['d']}: " + " | ".join(
                f"{a} hand {r[a]['hand']:.2f}{'*' if r[a]['hand_touches_border'] else ''} vase {r[a]['vase']:.2f}"
                f"{'*' if r[a]['vase_touches_border'] else ''}" for a in ARMS) + "   (* touches the border)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "view_check.json", "w"), indent=1)


def down(img, k=SS):
    h, w = img.shape[:2]
    return img.reshape(h // k, k, w // k, k, -1).astype(np.float64).mean((1, 3))


def project(P, ex, K):
    R, t = np.asarray(ex["R_world_to_cam"]), np.asarray(ex["t_world_to_cam"])
    X = R @ np.asarray(P, float) + t
    return [K["fx"] * X[0] / X[2] + K["cx"], K["fy"] * X[1] / X[2] + K["cy"]], float(X[2])


def intrinsics(M, w, h):
    fovy = float(M.vis.global_.fovy)
    fy = 0.5 * h / np.tan(np.radians(fovy) / 2)
    return dict(fovy_deg=fovy, width=w, height=h, fx=fy, fy=fy, cx=w / 2, cy=h / 2,
                convention="pinhole, square pixels; u right, v down; (0, 0) = the image's top-left corner, pixel (i, j) covers "
                           "[i, i+1) x [j, j+1) (its centre is i + 0.5, j + 0.5); MuJoCo free camera, vis.global_.fovy; for the "
                           "delivered image size (rendered at 2x and averaged 2 x 2: hi-res pixel coordinates = 2 x these)")


def extrinsics(rnd):
    c = mujoco.mjv_averageCamera(rnd.scene.camera[0], rnd.scene.camera[1])
    pos, fwd, up = np.array(c.pos, float), np.array(c.forward, float), np.array(c.up, float)
    right = np.cross(fwd, up)
    R_wc = np.stack([right, -up, fwd])
    return dict(position_world=pos.tolist(), forward_world=fwd.tolist(), up_world=up.tolist(), R_world_to_cam=R_wc.tolist(),
                t_world_to_cam=(-R_wc @ pos).tolist(),
                frame="OpenCV camera frame: x right, y down, z forward; X_cam = R_world_to_cam @ X_world + t_world_to_cam",
                note="mjv_updateScene writes two eye cameras (+-vis.global_.ipd / 2 along the right axis); a mono render uses their "
                     "average (mjv_averageCamera): these are the average's",
                eye_cameras_pos=[list(map(float, rnd.scene.camera[i].pos)) for i in (0, 1)])


def marker_centroid(rnd, D, c, opt, P):
    rnd.update_scene(D, camera=c, scene_option=opt)
    a = rnd.render().astype(int)
    rnd.update_scene(D, camera=c, scene_option=opt)
    scn = rnd.scene
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0, 0]), np.asarray(P, float),
                        np.eye(3).ravel(), np.array([1.0, 0.0, 1.0, 1.0], np.float32))
    scn.geoms[scn.ngeom].emission = 1.0
    scn.ngeom += 1
    b = rnd.render().astype(int)
    m = np.abs(a - b).max(-1) > 40
    if not m.any():
        return None
    ys, xs = np.where(m)
    return [float(xs.mean() + 0.5), float(ys.mean() + 0.5), int(m.sum())]


def final(az, el, d, dc):
    M, Z, st, scene_md5 = load()
    G = FR.masks(M)
    OUT.mkdir(parents=True, exist_ok=True)
    cm, yaw, la = main_cam(M, st, az, el, d)
    p_star = np.asarray(Z["A"]["goal__entry"], float)
    assert np.array_equal(Z["A"]["goal__entry"], Z["R"]["goal__entry"]), "A's and R's grasp points differ"
    cc = FR.cam(p_star, yaw, az, el, dc)
    rows = {}
    for a in ARMS:
        for f in sorted(RUNS.glob(f"rows/{a}_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if int(r["index"]) == PI:
                    rows[a] = r
    sR = next(s for s in rows["R"]["stages"] if s["stage"] == "entry")
    sA = next(s for s in rows["A"]["stages"] if s["stage"] == "entry")
    c_star = np.asarray(sR["correct"]["cmd_w"], float)
    pts = {
        "p_star": dict(world=p_star.tolist(), how="the grasp point: simple_bench's entry goal (truth column: the placement's palm "
                                                  "pose in the vase's frame, at the vase's simulated pose); the same in A and R (checked)"),
        "c_star": dict(world=c_star.tolist(), how="R's entry command (the palm point it sent; R_D inversion on model D): row "
                                                  "stages[entry].correct.cmd_w"),
        "a_cmd": dict(world=np.asarray(sA["correct"]["cmd_w"], float).tolist(), how="A's entry command (= p* up to float rounding: A sends the goal)"),
        "A_landing": dict(world=st[CLOSE]["A"]["palm"].tolist(), how="A's palm point at the close command (snapshot entry_end); "
                                                                     "|it - p*| = row palm_cm"),
        "R_landing": dict(world=st[CLOSE]["R"]["palm"].tolist(), how="R's palm point at the close command (snapshot entry_end)"),
    }
    out = dict(placement=PI, scene_md5=scene_md5, controller="scalebfm", moments=dict(
        panels=dict(snapshot=PANEL, after_close_s={a: st[PANEL][a]["t_after_close"] for a in ARMS}),
        closeups_and_points=dict(snapshot=CLOSE, what="the close command (simple_bench sends the close; the judgment's moment)")),
        distances_cm=dict(A_close=100 * float(np.linalg.norm(st[CLOSE]["A"]["palm"] - p_star)),
                          R_close=100 * float(np.linalg.norm(st[CLOSE]["R"]["palm"] - p_star)),
                          A_row_palm_cm=rows["A"]["palm_cm"], R_row_palm_cm=rows["R"]["palm_cm"],
                          R_correction_c_star_to_p_star=100 * float(np.linalg.norm(c_star - p_star))),
        cameras={}, points=pts, files={})
    look = {}
    for cname, cam, (W, H), tag in (("main", cm, (2400, 1600), PANEL), ("closeup", cc, (1600, 1200), CLOSE)):
        Mr = FR.prep(M, G, W * SS, H * SS, 16)
        Mk = FR.prep(M, G, W, H, 0)  # marker check at the delivered size
        Ms = FR.prep(M, G, W * SS, H * SS, 0)  # segmentation, no multisampling
        rnd, rk, rs = mujoco.Renderer(Mr, H * SS, W * SS), mujoco.Renderer(Mk, H, W), mujoco.Renderer(Ms, H * SS, W * SS)
        opt = FR.option()
        K = intrinsics(Mr, W, H)
        rk.update_scene(st[CLOSE]["A"]["D"], camera=cam, scene_option=opt)
        ex = extrinsics(rk)
        out["cameras"][cname] = dict(mjv=dict(type="free", lookat=list(map(float, cam.lookat)), azimuth_deg=float(cam.azimuth),
                                              elevation_deg=float(cam.elevation), distance=float(cam.distance),
                                              azimuth_from_heading_deg=az, look_down_deg=el, robot_heading_deg=yaw),
                                     intrinsics=K, extrinsics=ex, moment=tag)
        for k, v in pts.items():
            uv, zc = project(v["world"], ex, K)
            v.setdefault("pixel", {})[cname] = dict(u=uv[0], v=uv[1], depth_m=zc)
        chk = []
        cpos = np.asarray(ex["position_world"])
        for k in ("p_star", "c_star", "A_landing", "R_landing"):
            P = np.asarray(pts[k]["world"], float)
            Q = P + (0.15 if cname == "main" else 0.05) * (cpos - P) / np.linalg.norm(cpos - P)
            uv, _ = project(Q, ex, K)
            got = marker_centroid(rk, st[CLOSE]["A"]["D"], cam, opt, Q)
            chk.append(dict(point=k + " moved towards the camera", projected=uv, marker_centroid=got,
                            err_px=None if got is None else float(np.hypot(got[0] - uv[0], got[1] - uv[1]))))
        out["cameras"][cname]["projection_check"] = chk
        print(cname, "projection check err px:", [None if c["err_px"] is None else round(c["err_px"], 2) for c in chk], flush=True)
        for a in ARMS:
            D = st[tag][a]["D"]
            img = FR.render(rnd, D, cam, opt)
            sg = FR.seg_geom(FR.render(rs, D, cam, opt, seg=True))
            void = int((sg < 0).sum())
            name = f"{a}_full" if cname == "main" else f"{a}_hand_closeup"
            im = np.clip(np.round(down(img)), 0, 255).astype(np.uint8)
            Image.fromarray(im).save(OUT / f"{name}.png")
            out["files"][f"{name}.png"] = dict(size=[W, H], camera=cname, moment=tag, void_pixels_hires=void)
            look[name] = im
            if cname == "main":
                rob = (sg >= 0) & G["robot"][np.clip(sg, 0, None)]
                m = down(rob[..., None].astype(np.float64))[..., 0]
                rgb = down(img.astype(np.float64) * rob[..., None])
                rgb = np.where(m[..., None] > 0, rgb / np.maximum(m[..., None], 1e-9), 0.0)
                layer = np.dstack([np.clip(np.round(rgb), 0, 255), np.round(255 * m)]).astype(np.uint8)
                Image.fromarray(layer, "RGBA").save(OUT / f"{a}_robot_layer_rgba.png")
                out["files"][f"{a}_robot_layer_rgba.png"] = dict(size=[W, H], camera=cname, moment=tag,
                                                                 alpha="the robot's visible pixels (occluders cut it out), 2 x 2 share")
            print(f"saved {name} ({W}x{H}), void hi-res px {void}", flush=True)
        for x in (rnd, rk, rs):
            x.close()
    out["render"] = dict(renderer=f"mujoco {mujoco.__version__} Renderer, MUJOCO_GL=egl", offsamples=16, shadowsize=4096,
                         supersampling=f"{SS}x{SS} box average", geomgroups=list(FR.GROUPS), look=FR.LOOK,
                         hidden="robot collision geoms (group 0), the chest logo mesh, all sites")
    json.dump(out, open(OUT / "teaser_points.json", "w"), indent=1)
    side = np.concatenate([look["A_full"], np.full((1600, 24, 3), 255, np.uint8), look["R_full"]], 1)
    Image.fromarray(side).resize((side.shape[1] // 3, side.shape[0] // 3), Image.LANCZOS).save(OUT / "preview_A_R.png")
    print(json.dumps({k: dict(world=np.round(v["world"], 5).tolist(), px={c: [round(v["pixel"][c]["u"], 1), round(v["pixel"][c]["v"], 1)]
                                                                           for c in v["pixel"]}) for k, v in pts.items()}))
    print("distances", json.dumps(out["distances_cm"]))


if __name__ == "__main__":
    if MODE == "check":
        check()
    else:
        final(*(float(x) for x in sys.argv[5:9]))
