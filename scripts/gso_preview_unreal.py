#!/usr/bin/env python3
"""Render each `logistics_room_1030_s<seed>` instance in its Unreal level (user 2026-11-04: the five scenes as Unreal
pictures): the platform from above beside the room view, the same two viewpoints as the MuJoCo preview
(scripts/gso_preview.py), with the drawn objects listed under them. The level's third-person rig (the camera
`export_mjcf.FILM_CAMERA`, 1280 x 960 at fovy 45) is placed at each viewpoint for one fresh capture, as
scripts/rsi/report/report_unreal.py films; the scene is shown at rest (the settled scene file's pose, the robot at its
start). Needs the Unreal editor (bash scripts/start_unreal.sh --gpu N) and every instance's level
(bash scripts/build_scene.sh logistics_room_1030_s<seed> --unreal; `--build` runs that for a missing or stale level).

    PYTHONPATH=src "$BRIDGE_PY" scripts/gso_preview_unreal.py --seeds 10 [--build] [--out outputs/gso_preview]
        [--scenes outputs/scene_discovery]

Writes `<out>/<scene>_unreal.png` per instance and `<out>/index_unreal.json`. About 30 s an instance (the level load
and PIE start; the two captures take a second).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from locomani import paths  # noqa: E402
from locomani.sim.unreal import export_mjcf  # noqa: E402
from locomani.sim.unreal.view import UnrealView  # noqa: E402

SCENE = "logistics_room_1030_s{seed}"
FILM = (1280, 960)  # the level's source.json `film`, export_mjcf
FOVY = 45.0
CAPTION_LINE_PX = 28
# the two viewpoints, as MuJoCo free-camera parameters (lookat, azimuth, elevation, distance): the platform (centre
# (0, -0.105), top 0.371 m) straight from above, and the room from the far corner as the scene's `reverse_room` camera
# sees it, pulled back for the rig's fovy 45 (reverse_room is 73.7); both stay inside the room polygon
# (x -3.9..1.65, y -3.4..2.5)
PLATFORM = (0.0, -0.105, 0.371)
VIEWS = {
    "platform_top": {"lookat": PLATFORM, "azimuth": 90.0, "elevation": -89.5, "distance": 2.0},
    "room": {"lookat": (0.0, -0.105, 0.45), "azimuth": -45.7, "elevation": -31.0, "distance": 3.4},
}


def pose(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """A MuJoCo free camera as the rig's pose: pos, wxyz quat, -Z forward, +Y up (report_unreal.pose)."""
    az, el = math.radians(cam["azimuth"]), math.radians(cam["elevation"])
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(cam["lookat"], float) - cam["distance"] * fwd
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([right, up, -fwd], axis=1).flatten())
    return pos, quat


def caption_rows(meta: dict) -> list[str]:
    """The free objects as `<key>: <phrase> (<kind>, <mass> kg, <band>)` (gso_preview.caption_rows)."""
    rows = []
    for key, o in meta["objects"].items():
        if not o.get("free"):
            continue
        sem, g = o.get("semantic") or {}, o.get("gso") or {}
        band = "edge band" if g.get("in_reach_band", True) else "middle"
        rows.append(f"{key}: {sem.get('phrase')} ({sem.get('graspable')}, {sem.get('mass_kg')} kg, {band})")
    return rows


def level_state(scene_dir: Path) -> str:
    """`ok`, `missing` or `stale` (built from another scene file) for the instance's level."""
    src = Path(paths.UNREAL_ASSETS) / scene_dir.name / "source.json"
    if not src.is_file():
        return "missing"
    want = hashlib.sha256((scene_dir / "scene_locomani.xml").read_bytes()).hexdigest()
    return "ok" if json.loads(src.read_text()).get("scene_xml_sha256") == want else "stale"


def capture(view: UnrealView, qpos, cam: dict) -> np.ndarray:
    """One fresh picture of `qpos` from the rig at `cam`; the rig goes back to its parking place after."""
    rig = view.film_adr[export_mjcf.RIG]
    pos, quat = pose(cam)
    view.data.qpos[rig : rig + 3], view.data.qpos[rig + 3 : rig + 7] = pos, quat
    try:
        view.push(qpos)
        blk = view._push({view.film_camera: "fresh"})["cameras"][view.film_camera]
    finally:
        view.data.qpos[rig : rig + 7] = view.client.model.qpos0[rig : rig + 7]
    px = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)[..., [2, 1, 0]]
    return np.ascontiguousarray(px)


def compose(scene_dir: Path, views: list[np.ndarray], meta: dict, out_png: Path) -> None:
    h, w = views[0].shape[:2]
    rows = caption_rows(meta)
    per_col = (len(rows) + 1) // 2
    band = CAPTION_LINE_PX * (per_col + 2)
    canvas = Image.new("RGB", (w * len(views), h + band), (20, 20, 20))
    for i, v in enumerate(views):
        canvas.paste(Image.fromarray(v), (w * i, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    gso = meta.get("gso", {})
    title = f"{scene_dir.name} (Unreal): seed {gso.get('seed')}, pool {gso.get('pool')}, {len(rows)} free objects on the platform"
    draw.text((16, h + 8), title, fill=(255, 255, 255), font=font)
    for j, row in enumerate(rows):
        x = 16 + (j // per_col) * w
        y = h + CAPTION_LINE_PX * (1 + j % per_col) + 8
        draw.text((x, y), row, fill=(220, 220, 220), font=font)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def render_instance(scene_dir: Path, out_png: Path, log=print) -> dict:
    """The instance's picture from its level; returns the index entry."""
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(scene_dir / "scene_locomani.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = UnrealView(
        str(scene_dir / "scene_locomani.xml"), f"/Game/Levels/{scene_dir.name}", repeats=8, streams=(), log=log
    )
    try:
        view.film_setup(*FILM, FOVY)
        views = [capture(view, data.qpos.copy(), cam) for cam in VIEWS.values()]
    finally:
        view.close()
    compose(scene_dir, views, meta, out_png)
    return {
        "scene": str(scene_dir), "png": str(out_png), "level": f"/Game/Levels/{scene_dir.name}", "views": list(VIEWS),
        "objects": {k: o["semantic"]["phrase"] for k, o in meta["objects"].items() if o.get("free")},
    }  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="instances s0 .. s<N-1>")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--scenes", type=Path, default=ROOT / "outputs/scene_discovery")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/gso_preview")
    ap.add_argument("--build", action="store_true", help="build (or rebuild) a missing or stale level first")
    a = ap.parse_args(argv)
    index = {}
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        scene_dir = a.scenes / SCENE.format(seed=seed)
        if not (scene_dir / "scene_locomani.xml").is_file():
            raise SystemExit(f"no instance {scene_dir}: $VENV_SIM_PY scripts/gso_preview.py --seeds {seed + 1}")
        state = level_state(scene_dir)
        if state != "ok":
            cmd = ["bash", str(ROOT / "scripts/build_scene.sh"), scene_dir.name, "--unreal"]
            if not a.build:
                raise SystemExit(f"{scene_dir.name}: Unreal level {state}; {' '.join(cmd)} (or --build)")
            print(f"[unreal] level {state}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
        t0 = time.time()
        entry = render_instance(scene_dir, a.out / f"{scene_dir.name}_unreal.png")
        entry["seconds"] = round(time.time() - t0, 1)
        index[scene_dir.name] = entry
        print(f"[unreal] {scene_dir.name}: {entry['png']} ({entry['seconds']} s)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "index_unreal.json").write_text(json.dumps(index, indent=1) + "\n")
    print(json.dumps({k: v["png"] for k, v in index.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
