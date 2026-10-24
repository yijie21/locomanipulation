"""Check the plugin's "fresh_depth" mode (patches/urlab_fresh_depth.md) against MuJoCo, for vision RSI L3: Unreal's
depth of the visible surfaces should be MuJoCo's depth of the visual meshes the level was built from, pixel for pixel.

    grab     (bridge venv, the patched editor up with the apartment level) one state through UnrealView: colour by
             "fresh", then colour and depth by "fresh_depth" twice; writes the images, the depths (metres) and timings
    compare  (sim venv) MuJoCo's depth of the same state from the full scene model, visual geoms only (groups 0-2),
             against Unreal's: per camera the share of pixels within 1 / 2 / 5 cm, the median and 90th percentile of
             the difference, and the ratio at the image centre and corners (a ratio that grows towards the corners
             would mean Unreal's depth is the distance along the ray, not along the view axis); a side-by-side PNG

    PYTHONPATH=src UE_PROJECT=/workspace/ue_57/projects/URLabVision/URLabSmoke.uproject "$BRIDGE_PY" \\
        scripts/experiments/unreal_depth_check_20261024.py grab --qpos outputs/vision_rsi_20261023/l3_align/qpos.npy \\
        --out outputs/vision_rsi_20261023/depth_check
    MUJOCO_GL=egl PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/unreal_depth_check_20261024.py compare --dir outputs/vision_rsi_20261023/depth_check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SCENES = ROOT / "outputs/scene_discovery"
PHYS = SCENES / "adt_apartment_mars_physics/scene_locomani.xml"
FULL = SCENES / "adt_apartment_mars/scene_locomani.xml"
STREAMS = {"ego_view": "head_camera", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}  # Unreal -> MuJoCo


def grab(a) -> int:
    from PIL import Image

    from locomani.sim.unreal.view import UnrealView

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qpos = np.load(a.qpos)
    view = UnrealView(str(a.scene), a.level, a.repeats)
    try:
        view.warm_up(qpos)
        view.push(qpos)
        t0 = time.perf_counter()
        plain = view.grab()
        t1 = time.perf_counter()
        images, depth = view.grab_depth()
        t2 = time.perf_counter()
        images2, depth2 = view.grab_depth()
        t3 = time.perf_counter()
    finally:
        view.close()
    report = dict(fresh_s=round(t1 - t0, 3), fresh_depth_s=round(t2 - t1, 3), fresh_depth_again_s=round(t3 - t2, 3))
    for k in images:
        Image.fromarray(images[k]).save(out / f"ue_{k}.png")
        np.save(out / f"ue_depth_{k}.npy", depth[k])
        d, d2 = depth[k], depth2[k]
        both = np.isfinite(d) & np.isfinite(d2)
        report[k] = dict(
            shape=list(d.shape),
            depth_min_m=float(np.nanmin(d)),
            depth_max_m=float(np.nanmax(d)),
            repeat_max_abs_m=float(np.abs(d[both] - d2[both]).max()) if both.any() else None,
            colour_vs_plain_mean_abs=round(float(np.abs(images[k].astype(int) - plain[k].astype(int)).mean()), 3),
            colour_repeat_mean_abs=round(float(np.abs(images[k].astype(int) - images2[k].astype(int)).mean()), 3),
        )
    np.save(out / "qpos.npy", qpos)
    (out / "grab.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


def compare(a) -> int:
    import mujoco
    from PIL import Image

    from locomani.tasks.scene.perceive import intrinsics

    out = Path(a.dir)
    phys = mujoco.MjModel.from_xml_path(str(a.scene))
    m = mujoco.MjModel.from_xml_path(str(a.visual_scene))
    d = mujoco.MjData(m)
    qpos = np.load(out / "qpos.npy")
    for j in range(phys.njnt):  # the state is the physics model's: copy it joint by joint, by name
        name = phys.joint(j).name
        w = {0: 7, 1: 4}.get(int(phys.jnt_type[j]), 1)
        s, t = int(phys.jnt_qposadr[j]), int(m.jnt_qposadr[m.joint(name).id])
        d.qpos[t : t + w] = qpos[s : s + w]
    mujoco.mj_forward(m, d)
    report, tiles = {}, []
    for ue, cam in STREAMS.items():
        ud = np.load(out / f"ue_depth_{ue}.npy")
        h, w = ud.shape
        r = mujoco.Renderer(m, h, w, max_geom=int(m.ngeom) + 1000)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[:3] = 1  # visual groups only: the collision shapes (group 3) are not what Unreal draws
        r.update_scene(d, camera=m.camera(cam).id, scene_option=opt)
        r.enable_depth_rendering()
        md = r.render().copy()
        r.close()
        ok = (md > 0.05) & (md < a.max_range) & (ud > 0.05) & (ud < a.max_range) & np.isfinite(ud)
        diff = ud - md
        fx, fy, cx, cy = intrinsics(float(m.cam_fovy[m.camera(cam).id]), w, h)
        v, u = np.mgrid[0:h, 0:w]
        rad = np.hypot((u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy)  # tan of the angle off the axis
        centre, corner = ok & (rad < 0.15), ok & (rad > np.percentile(rad, 90))
        ad = np.abs(diff[ok])
        report[ue] = dict(
            valid=int(ok.sum()),
            within_1cm=round(float((ad < 0.01).mean()), 3),
            within_2cm=round(float((ad < 0.02).mean()), 3),
            within_5cm=round(float((ad < 0.05).mean()), 3),
            median_cm=round(100 * float(np.median(diff[ok])), 2),
            p90_abs_cm=round(100 * float(np.percentile(ad, 90)), 2),
            ratio_centre=round(float(np.median(ud[centre] / md[centre])), 4) if centre.any() else None,
            ratio_corner=round(float(np.median(ud[corner] / md[corner])), 4) if corner.any() else None,
            corner_sec=round(float(np.median(np.sqrt(1 + rad[corner] ** 2))), 4) if corner.any() else None,
        )

        def gray(x):
            return (255 * np.clip(x / a.max_range, 0, 1)).astype(np.uint8)

        err = np.zeros((h, w, 3), np.uint8)
        err[..., 0] = (255 * np.clip(np.where(ok, diff, 0) / 0.05, 0, 1)).astype(np.uint8)  # red: Unreal farther
        err[..., 2] = (255 * np.clip(np.where(ok, -diff, 0) / 0.05, 0, 1)).astype(np.uint8)  # blue: nearer
        rgb = np.asarray(Image.open(out / f"ue_{ue}.png"))
        tiles.append(np.concatenate([rgb, np.repeat(gray(ud)[..., None], 3, 2), np.repeat(gray(md)[..., None], 3, 2),
                                     err], axis=1))  # fmt: skip
    Image.fromarray(np.concatenate(tiles, axis=0)).save(out / "compare.png")
    (out / "compare.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grab")
    g.add_argument("--qpos", required=True, help="a physics-model qpos (.npy)")
    g.add_argument("--out", required=True)
    g.add_argument("--scene", default=str(PHYS))
    g.add_argument("--level", default="/Game/Levels/adt_apartment_mars")
    g.add_argument("--repeats", type=int, default=8)
    c = sub.add_parser("compare")
    c.add_argument("--dir", required=True)
    c.add_argument("--scene", default=str(PHYS))
    c.add_argument("--visual-scene", default=str(FULL))
    c.add_argument("--max-range", type=float, default=3.0)
    a = ap.parse_args(argv)
    return grab(a) if a.cmd == "grab" else compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
