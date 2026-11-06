#!/usr/bin/env python3
"""Build `logistics_room_1030_s<seed>` scene instances and render one picture of each (user Q25, 2026-11-03): the
platform straight from above (camera `platform_top`) beside the room view (`reverse_room`), with the drawn objects
listed under them. Every step runs in this process: the scene builder (scripts/prepare_logistics_scene.py), the
settle (scripts/scene_settle.py --write), the drop test (scripts/scene_drop_test.py, what the feasibility checker
reads), the lint (scripts/scene_lint.py, which must report 0 errors), the two task families
(scripts/rsi/gso_family.py) and the MuJoCo offscreen render (EGL: run with MUJOCO_GL=egl).

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 $VENV_SIM_PY scripts/gso_preview.py --seeds 10 [--out outputs/gso_preview]
        [--scenes outputs/scene_discovery] [--pool train] [--objects 10 --bimanual 4] [--families-dir ... | --no-families]

An instance already built under --scenes is reused as it is (delete its directory to draw it again). Writes
`<out>/<scene>.png` per instance and `<out>/index.json` (the objects, their kinds, the lint and drop-test results).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "rsi"))

import gso_family  # noqa: E402
import prepare_logistics_scene  # noqa: E402
import scene_drop_test  # noqa: E402
import scene_lint  # noqa: E402
import scene_settle  # noqa: E402

SCENE = "logistics_room_1030_s{seed}"
CAMERAS = (prepare_logistics_scene.PLATFORM_TOP_CAMERA, "reverse_room")
VIEW_W, VIEW_H = 960, 720
CAPTION_LINE_PX = 22


def build_instance(scene_dir: Path, seed: int, pool: str, objects: int, bimanual: int) -> dict:
    """The build pipeline of one instance into `scene_dir` (which must not exist): build, settle, drop test, lint.
    Returns {"lint_errors": n, "drop_failures": n}; raises SystemExit when the lint reports an error."""
    rel = str(scene_dir.relative_to(ROOT)) if scene_dir.is_relative_to(ROOT) else str(scene_dir)
    prepare_logistics_scene.main([rel, "--robot", "mars", "--rsi", "--gso-seed", str(seed), "--pool", pool,
                                  "--objects", str(objects), "--bimanual", str(bimanual)])  # fmt: skip
    scene_settle.main(["--scene", rel, "--write"])
    drop_failures = scene_drop_test.main(["--scene", str(Path(rel) / "scene_meta.json")])
    lint_errors = scene_lint.main(["--scene", rel])
    if lint_errors:
        raise SystemExit(f"{rel}: scene_lint.py reports errors (see {rel}/lint.json)")
    return {"lint_errors": int(lint_errors), "drop_failures": int(drop_failures)}


def caption_rows(meta: dict) -> list[str]:
    rows = []
    for key, o in meta["objects"].items():
        if not o.get("free"):
            continue
        sem, g = o.get("semantic") or {}, o.get("gso") or {}
        band = "edge band" if g.get("in_reach_band", True) else "middle"
        rows.append(f"{key}: {sem.get('phrase')} ({sem.get('graspable')}, {sem.get('mass_kg')} kg, {band})")
    return rows


def render(scene_dir: Path, out_png: Path, meta: dict) -> None:
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    model = mujoco.MjModel.from_xml_path(str(scene_dir / "scene_locomani.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, VIEW_H, VIEW_W) as r:
        views = []
        for cam in CAMERAS:
            r.update_scene(data, camera=cam)
            views.append(r.render().copy())
    rows = caption_rows(meta)
    per_col = (len(rows) + 1) // 2
    band = CAPTION_LINE_PX * (per_col + 2)
    canvas = Image.new("RGB", (VIEW_W * len(views), VIEW_H + band), (20, 20, 20))
    for i, v in enumerate(views):
        canvas.paste(Image.fromarray(np.ascontiguousarray(v)), (VIEW_W * i, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    gso = meta.get("gso", {})
    title = (
        f"{scene_dir.name}: seed {gso.get('seed')}, pool {gso.get('pool')}, {len(rows)} free objects on the platform"
    )
    draw.text((12, VIEW_H + 6), title, fill=(255, 255, 255), font=font)
    for j, row in enumerate(rows):
        x = 12 + (j // per_col) * VIEW_W
        y = VIEW_H + CAPTION_LINE_PX * (1 + j % per_col) + 6
        draw.text((x, y), row, fill=(220, 220, 220), font=font)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def preview(seeds, scenes: Path, out: Path, pool: str, objects: int, bimanual: int, families_dir: Path | None) -> dict:
    index = {}
    for seed in seeds:
        t0 = time.time()
        scene_dir = scenes / SCENE.format(seed=seed)
        if scene_dir.exists():
            print(f"[preview] reusing {scene_dir}", flush=True)
            checks = {"reused": True}
        else:
            checks = build_instance(scene_dir, seed, pool, objects, bimanual)
        meta = json.loads((scene_dir / "scene_meta.json").read_text())
        families = gso_family.write(scene_dir, families_dir) if families_dir else {}
        png = out / f"{scene_dir.name}.png"
        render(scene_dir, png, meta)
        index[scene_dir.name] = {
            "scene": str(scene_dir), "png": str(png), "seed": seed, "pool": pool, "families": families,
            "objects": {k: {"phrase": o["semantic"]["phrase"], "graspable": o["semantic"]["graspable"],
                            "in_reach_band": (o.get("gso") or {}).get("in_reach_band")}
                        for k, o in meta["objects"].items() if o.get("free")},
            "checks": checks, "seconds": round(time.time() - t0, 1),
        }  # fmt: skip
        print(f"[preview] {scene_dir.name}: {png} ({index[scene_dir.name]['seconds']} s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="instances s0 .. s<N-1>")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--scenes", type=Path, default=ROOT / "outputs/scene_discovery", help="where the instances live")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/gso_preview")
    ap.add_argument("--pool", choices=("train", "eval_unseen", "all"), default="train")
    ap.add_argument("--objects", type=int, default=10)
    ap.add_argument("--bimanual", type=int, default=4)
    ap.add_argument("--families-dir", type=Path, default=gso_family.FAMILIES)
    ap.add_argument("--no-families", action="store_true", help="do not write the two task families per instance")
    a = ap.parse_args(argv)
    if os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"):
        print("[preview] MUJOCO_GL is not egl or osmesa: the offscreen render needs one of them", file=sys.stderr)
    seeds = range(a.first_seed, a.first_seed + a.seeds)
    index = preview(seeds, a.scenes, a.out, a.pool, a.objects, a.bimanual, None if a.no_families else a.families_dir)
    print(
        json.dumps(
            {k: {"png": v["png"], "families": v["families"], "checks": v["checks"]} for k, v in index.items()}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
