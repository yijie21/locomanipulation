"""fig_render.py: Figure 1 v3 frames from fig_run.py's snapshots (MuJoCo offscreen EGL, mujoco.Renderer). notebook: skip
figure render, not an experiment.

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6 python fig_render.py <runs> <placement> look <out.png> az,el,d ...
    ... fig_render.py <runs> <placement> sheet <out dir>

State: each arm's scene (runs/scene/<arm>_p<i>/scene_locomani.xml, the run's own; A's and R's must be identical) and its
snapshot at the close (simple_bench sends the close: "entry_end"): mj_setState(INTEGRATION) + mj_forward, then the body
and geom poses overwritten with the ones the simulator held when the palm was measured (so the rendered palm is the
measured one to the bit; checked).

Look (render only, the simulation is untouched): the robot's collision geoms (group 0) and its chest logo mesh hidden,
no sites; lights re-balanced (LOOK); offsamples / shadowsize as asked.

Views: a free camera looking at LOOKAT_Z over the pelvis-vase midpoint; az = the camera's direction from the robot,
relative to its heading (0 = straight in front, + = towards the robot's left), MuJoCo azimuth = heading + 180 + az;
el = the look-down angle (MuJoCo elevation -el). sheet: grid az -60..60 step 10, el 35..55 step 5, d 2.0-3.2; a view
is valid when, for both arms, the head (head_link mesh) >= 80 % visible, the right hand (right_hand_* links) >= 50 %, the
vase (its ceramic geoms) >= 50 % (segmentation against isolated renders) and the whole robot (rendered alone) inside the
frame (2 % margin). 8 candidates spread over az (rule in pick_views), each ~600 px: A's frame with R's whole robot
overlaid translucent (alpha 0.30), labelled az / el / d.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUNS, PI, MODE = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
HIDE = 5
GROUPS = (1, 1, 1, 0, 0, 0)
GHOST_RGBA = (0 / 255, 114 / 255, 178 / 255, 0.30)
LOOKAT_Z = 0.85
LOOK = dict(headlight_ambient=0.30, headlight_diffuse=0.22, robot_top=0.15, sun_key=0.32, fill_left=0.18, rim_back=0.08,
            fill_dir=(-0.6, 0.4, -0.7), floor_rgb=0.88, oak=(0.70, 0.53, 0.36))
ARMS = ("A", "R")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def load():
    xs = {a: RUNS / "scene" / f"{a}_p{PI:02d}" / "scene_locomani.xml" for a in ARMS}
    same = md5(xs["A"]) == md5(xs["R"])
    assert same, "A's and R's scenes differ"
    M = mujoco.MjModel.from_xml_path(str(xs["A"]))
    st = {}
    for a in ARMS:
        z = np.load(RUNS / "snap" / f"{a}_p{PI:02d}.npz")
        D = mujoco.MjData(M)
        mujoco.mj_setState(M, D, z["entry_end__state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(M, D)
        D.xpos[:], D.xmat[:] = z["entry_end__xpos"], z["entry_end__xmat"]
        D.geom_xpos[:], D.geom_xmat[:] = z["entry_end__geom_xpos"], z["entry_end__geom_xmat"]
        wb = M.body("right_wrist_yaw_link").id
        palm = D.xpos[wb] + D.xmat[wb].reshape(3, 3) @ np.array([0.0415, 0.0, 0.0])
        g = z["goal__entry"]
        st[a] = dict(D=D, z=z, palm=palm, goal=g, vase=D.xpos[M.body("vase").id].copy(),
                     d_cm=100 * float(np.linalg.norm(palm - g)), palm_err=float(np.abs(palm - z["entry_end__palm"]).max()))
        print(f"{a}: palm from the rendered state vs the snapshot max |d| {st[a]['palm_err']:.3g} m; close distance "
              f"{st[a]['d_cm']:.6f} cm", flush=True)
    return M, st, md5(xs["A"])


def masks(M):
    names = [mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(M.nbody)]
    root = M.body_rootid
    robot_b = np.where(root == root[M.body("pelvis").id])[0]
    gb = M.geom_bodyid
    mesh = np.array([mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_MESH, int(i)) or "" if t == mujoco.mjtGeom.mjGEOM_MESH else ""
                     for t, i in zip(M.geom_type, M.geom_dataid)])
    vis = M.geom_group == 1
    robot = np.isin(gb, robot_b)
    G = dict(robot=robot & vis & (mesh != "logo_link"),
             head=robot & vis & (mesh == "head_link"),
             hand=robot & vis & np.isin(gb, [b for b in robot_b if names[b].startswith("right_hand_")]),
             vase=np.array([(mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("vase_") for g in range(M.ngeom)]),
             logo=robot & (mesh == "logo_link"), robot_coll=robot & (M.geom_group == 0))
    return G


def prep(M, G, w, h, samples=16, keep=None):
    R = copy.deepcopy(M)
    R.vis.global_.offwidth, R.vis.global_.offheight = w, h
    R.vis.quality.offsamples = samples
    R.vis.quality.shadowsize = 4096
    R.geom_group[G["robot_coll"] | G["logo"]] = HIDE
    if keep is not None:
        R.geom_group[~keep] = HIDE
    R.vis.headlight.ambient[:] = LOOK["headlight_ambient"]
    R.vis.headlight.diffuse[:] = LOOK["headlight_diffuse"]
    R.vis.headlight.specular[:] = 0.0
    for i in range(R.nlight):
        n = mujoco.mj_id2name(R, mujoco.mjtObj.mjOBJ_LIGHT, i) or "robot_top"
        R.light_diffuse[i] = LOOK.get(n, R.light_diffuse[i][0])
        R.light_specular[i] = 0.08
        R.light_castshadow[i] = n in ("sun_key", "robot_top")
        if n == "fill_left":
            fd = np.asarray(LOOK["fill_dir"], float)
            R.light_dir[i] = fd / np.linalg.norm(fd)
    R.mat_rgba[R.material("floor").id, :3] = LOOK["floor_rgb"]
    R.mat_rgba[R.material("oak").id, :3] = LOOK["oak"]
    return R


def option():
    o = mujoco.MjvOption()
    for g in range(6):
        o.geomgroup[g] = GROUPS[g]
    o.sitegroup[:] = 0
    return o


def heading(st):
    D = st["A"]["D"]
    Rp = D.xmat[1].reshape(3, 3)
    return float(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0])))


def lookat(M, st):
    pel = st["A"]["D"].xpos[M.body("pelvis").id]
    v = st["A"]["vase"]
    c = 0.5 * (pel + v)
    return np.array([c[0], c[1], LOOKAT_Z])


def cam(la, yaw, az, el, d):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = la
    c.azimuth, c.elevation, c.distance = yaw + 180.0 + az, -el, d
    return c


def add_ghost(scn, Mg, Dg, opt, rgba=GHOST_RGBA):
    n0 = scn.ngeom
    mujoco.mjv_addGeoms(Mg, Dg, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    for i in range(n0, scn.ngeom):
        g = scn.geoms[i]
        g.rgba[:] = rgba
        g.matid = -1
        g.emission = 0.25
        g.specular = 0.2
        g.reflectance = 0.0


def render(rnd, D, c, opt, ghost=None, seg=False):
    rnd.update_scene(D, camera=c, scene_option=opt)
    if ghost is not None:
        add_ghost(rnd.scene, *ghost, opt)
    if seg:
        rnd.enable_segmentation_rendering()
        img = rnd.render().copy()
        rnd.disable_segmentation_rendering()
        return img
    return rnd.render().copy()


def seg_geom(img):
    ids, ty = img[..., 0].astype(int), img[..., 1].astype(int)
    return np.where(ty == int(mujoco.mjtObj.mjOBJ_GEOM), ids, -1)


def visibility(M, G, st, la, yaw, views, w=480, h=320):
    rf = mujoco.Renderer(prep(M, G, w, h, 1), h, w)
    iso = {k: mujoco.Renderer(prep(M, G, w, h, 1, keep=G[k]), h, w) for k in ("head", "hand", "vase", "robot")}
    opt = option()
    out = []
    for az, el, d in views:
        c = cam(la, yaw, az, el, d)
        r = dict(az=az, el=el, d=d)
        ok = True
        for a in ARMS:
            D = st[a]["D"]
            full = seg_geom(render(rf, D, c, opt, seg=True))
            v = {}
            for k in ("head", "hand", "vase"):
                m = seg_geom(render(iso[k], D, c, opt, seg=True)) >= 0
                n = int(m.sum())
                f = full[m]
                v[k] = float(((f >= 0) & G[k][np.clip(f, 0, None)]).sum()) / max(n, 1) if n else 0.0
            rob = seg_geom(render(iso["robot"], D, c, opt, seg=True)) >= 0
            ys, xs = np.where(rob)
            v["in_frame"] = bool(len(xs) and xs.min() > 0.02 * w and xs.max() < 0.98 * w and ys.min() > 0.02 * h and ys.max() < 0.98 * h)
            r[a] = v
            ok &= v["head"] >= 0.80 and v["hand"] >= 0.50 and v["vase"] >= 0.50 and v["in_frame"]
        r["valid"] = bool(ok)
        out.append(r)
    for x in [rf, *iso.values()]:
        x.close()
    return out


def pick_views(rows):
    """8 candidates: for az in -50, -30, -10, 10, 30, 50 the valid view at el 45 (else the nearest valid el), the least
    valid distance; then two more at the az with the most valid views, el 35 and 55 (or the nearest valid). Fewer when
    fewer are valid."""
    good = [r for r in rows if r["valid"]]
    out = []

    def best(az, el):
        c = [r for r in good if r["az"] == az]
        if not c:
            return None
        e = min({r["el"] for r in c}, key=lambda x: (abs(x - el), x))
        return min((r for r in c if r["el"] == e), key=lambda r: r["d"])

    for az in (-50, -30, -10, 10, 30, 50):
        b = best(az, 45)
        if b is not None:
            out.append(b)
    if good:
        cnt = {}
        for r in good:
            cnt[r["az"]] = cnt.get(r["az"], 0) + 1
        az0 = max(sorted(cnt), key=lambda a: cnt[a])
        for el in (35, 55):
            b = best(az0, el)
            if b is not None and b not in out:
                out.append(b)
    return out[:8]


def label(img, text):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", max(14, img.shape[1] // 34))
    except OSError:
        f = ImageFont.load_default()
    x0, y0, x1, y1 = dr.textbbox((10, 8), text, font=f)
    dr.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(255, 255, 255))
    dr.text((10, 8), text, fill=(20, 20, 20), font=f)
    return np.asarray(im)


def main():
    M, st, scene_md5 = load()
    G = masks(M)
    yaw = heading(st)
    la = lookat(M, st)
    print(f"heading {yaw:.2f} deg, lookat {la.round(3).tolist()}, scene md5 {scene_md5}", flush=True)
    if MODE == "look":
        w, h = 900, 600
        Mr = prep(M, G, w, h, 8)
        Mg = prep(M, G, w, h, 8, keep=G["robot"])
        rnd = mujoco.Renderer(Mr, h, w)
        opt = option()
        tiles = []
        for v in sys.argv[5:]:
            az, el, d = (float(x) for x in v.split(","))
            c = cam(la, yaw, az, el, d)
            a = render(rnd, st["A"]["D"], c, opt)
            o = render(rnd, st["A"]["D"], c, opt, ghost=(Mg, st["R"]["D"]))
            tiles.append(np.concatenate([a, o], 1))
        Image.fromarray(np.concatenate(tiles, 0)).save(sys.argv[4])
        print("saved", sys.argv[4])
        return
    out = Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    grid = [(az, el, d) for az in range(-60, 61, 10) for el in (35, 40, 45, 50, 55) for d in (2.0, 2.4, 2.8, 3.2)]
    rows = visibility(M, G, st, la, yaw, grid)
    good = [r for r in rows if r["valid"]]
    print(f"views {len(rows)}, valid {len(good)}", flush=True)
    for el in (35, 40, 45, 50, 55):
        print(f"  el {el}: valid az " + " ".join(f"{az:+d}({min(r['d'] for r in good if r['az'] == az and r['el'] == el):.1f})"
                                             for az in range(-60, 61, 10) if any(r["az"] == az and r["el"] == el for r in good)))
    sel = pick_views(rows)
    W, H = 600, 400
    Mr = prep(M, G, W, H, 16)
    Mg = prep(M, G, W, H, 16, keep=G["robot"])
    rnd = mujoco.Renderer(Mr, H, W)
    opt = option()
    tiles = []
    for k, r in enumerate(sel):
        c = cam(la, yaw, r["az"], r["el"], r["d"])
        img = render(rnd, st["A"]["D"], c, opt, ghost=(Mg, st["R"]["D"]))
        tiles.append(label(img, f"{k + 1}: az {r['az']:+d} deg  down {r['el']} deg  d {r['d']:.1f} m"))
        print(f"  cand {k + 1}: az {r['az']:+d} el {r['el']} d {r['d']} | A head {r['A']['head']:.2f} hand {r['A']['hand']:.2f} "
              f"vase {r['A']['vase']:.2f} | R head {r['R']['head']:.2f} hand {r['R']['hand']:.2f} vase {r['R']['vase']:.2f}", flush=True)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)], 0)
    Image.fromarray(sheet).save(out / "contact_sheet.png")
    json.dump(dict(placement=PI, heading_deg=yaw, lookat=la.tolist(), scene_md5=scene_md5, look=LOOK, ghost_rgba=GHOST_RGBA,
                   close_cm={a: st[a]["d_cm"] for a in ARMS}, palm_check_m={a: st[a]["palm_err"] for a in ARMS},
                   grid=rows, candidates=sel), open(out / "views.json", "w"), indent=1)
    print("saved", out / "contact_sheet.png", sheet.shape)


if __name__ == "__main__":
    main()
