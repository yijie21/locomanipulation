"""Fill a backend.render offline run's dataset cameras with Isaac pictures, before convert.

    "$RENDER_PY" -m locomani.sim.isaac.render_dataset_cameras outputs/runs/<run> --gpu 1
    "$RENDER_PY" -m locomani.sim.isaac.render_dataset_cameras outputs/runs/<run> --dry-run

`scripts/render_dataset_cameras_isaac.sh` is one line over this. The pictures come from `view.IsaacView` on the
scene's Isaac stage and look (`run_context.isaac_scene`) -- the object and settings the eval mirror serves the
policy from. Which rows, which sessions and the marker (`renderer: isaac` plus the look) are
`locomani.sim.view_render`, shared with the Unreal re-render; this module owns Kit's lifetime.
The run's third-person videos are a separate, optional command (`scripts/render_third_person.sh`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.sim import robot_look, view_render

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
        {
            "isaac_stage": isaac["dir"],
            "rtx_settings": isaac["rtx_settings"],
            "supersample": isaac["supersample"],
            "robot_look": robot_look.NAME,  # stage.G1_LOOKS
            "robot": robot_look.scene_robot(ROOT / cfg["scene"]["dir"]),  # stage.robot_usd
        }
        if isaac
        else {}
    )
    planned = view_render.plan(run_dir, cfg, "isaac", look)
    if planned is None:
        return 0
    traces, todo = planned
    total = sum(len(v) for v in todo.values())
    print(
        f"{run_dir.name}: {total} kept episode(s) to render in {len(todo)} finished session(s) "
        f"of {len(traces.sessions)}, Isaac stage {isaac['dir']}",
        flush=True,
    )
    if args.dry_run or not total:
        return 0

    from locomani.sim.isaac.view import run_in_kit

    def work(view):
        view_render.fill(view, traces, todo, "isaac", look, log=lambda m: print(m, flush=True))

    run_in_kit(str(traces.scene_dir / "scene_locomani.xml"), isaac, args.gpu, work, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
