"""Fill a backend.render offline run's dataset cameras with Isaac pictures, before convert.

    "$RENDER_PY" -m locomani.sim.isaac.render_dataset_cameras outputs/runs/<run> --gpu 1
    "$RENDER_PY" -m locomani.sim.isaac.render_dataset_cameras outputs/runs/<run> --dry-run

`scripts/render_dataset_cameras_isaac.sh` is one line over this. The pictures come from `view.IsaacView` on the
scene's Isaac stage and look (`run_context.isaac_scene`) -- the object and settings the eval mirror serves the
policy from. Which rows, which sessions and the marker (`renderer: isaac` plus the look) are
`locomani.sim.view_render`, shared with the Unreal re-render; this module owns Kit's lifetime.
Then the same view films every filled episode's third-person video that is missing (`view_render.film`,
`<run>/videos_isaac/`), after all the dataset images: filming stops the dataset cameras.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.sim import view_render
from locomani.sim.dataset_cameras import CAMERA_CONFIGS

ROOT = Path(paths.REPO)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory collection wrote")
    ap.add_argument("--gpu", default="1", help="the card Isaac renders on")
    ap.add_argument("--dry-run", action="store_true", help="say what would be rendered, render nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    isaac = run_context.isaac_scene(cfg) if cfg.get("backend", {}).get("render") == "offline" else None
    look = (
        {"isaac_stage": isaac["dir"], "rtx_settings": isaac["rtx_settings"], "supersample": isaac["supersample"]}
        if isaac
        else {}
    )
    planned = view_render.plan(run_dir, cfg, "isaac", look)
    if planned is None:
        return 0
    traces, todo = planned
    total = sum(len(v) for v in todo.values())
    films = len(view_render.film_todo(traces, list(todo), "isaac")) + total
    print(
        f"{run_dir.name}: {total} kept episode(s) to render in {len(todo)} finished session(s) "
        f"of {len(traces.sessions)}, then {films} third-person video(s), Isaac stage {isaac['dir']}",
        flush=True,
    )
    if args.dry_run or not films:
        return 0

    # before Kit starts: it takes the card from CUDA_VISIBLE_DEVICES
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES")
    from locomani.sim.isaac import stage as S
    from locomani.sim.isaac.view import IsaacView

    scene_xml = str(traces.scene_dir / "scene_locomani.xml")
    stage_cfg = S.StageCfg(
        scene_dir=isaac["dir"], robot_mjcf=scene_xml, absolute_camera=True, supersample=isaac["supersample"]
    )
    app = S.start_app(stage_cfg)
    try:
        view = IsaacView(
            stage_cfg, scene_xml, tuple(CAMERA_CONFIGS), isaac["rtx_settings"], log=lambda m: print(m, flush=True)
        )
        view_render.fill(view, traces, todo, "isaac", look, log=lambda m: print(m, flush=True))
        filming = view_render.film_todo(traces, list(todo), "isaac")
        view_render.film(view, traces, filming, "isaac", log=lambda m: print(m, flush=True))
    except BaseException:
        import sys
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)  # NOT app.close() first: it ends the process with status 0
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
