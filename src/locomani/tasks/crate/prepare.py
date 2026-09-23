"""Crate run preparation: the scene copy an expert reads and the refusals that come before any stack.

Moved out of the retired `pipeline/run_crate_pilot.py` so the collect stage (`locomani.stages.collect`)
and the tests share one implementation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from locomani import paths
from locomani.tasks.crate.config import CrateTaskConfig

EXPERTS = {
    "visual": ("tasks/crate/expert.py", "visual_crate"),
    "wall": ("tasks/crate/bimanual_wall.py", "wall"),
    "carry": ("tasks/crate/bimanual_carry.py", "bimanual"),
}


def check_profile(task_cfg: CrateTaskConfig, expert: str, record: bool) -> None:
    """Refuse combinations the experts cannot execute, before any resource is touched."""
    if expert not in EXPERTS:
        raise SystemExit(f"unknown crate expert {expert!r}; choices {sorted(EXPERTS)}")
    if expert == "wall" and task_cfg.carry.reach_control == "whole_body":
        raise SystemExit(
            "wall cannot execute carry.reach_control=whole_body; "
            "for the continuous assisted basket task pass --expert visual with "
            "task=crate_visual_low_reach (configs/task/crate_visual_low_reach.yaml) and a 30 cm scene. "
            "The bare runner defaults to the separate wall diagnostic."
        )
    if expert == "carry" and record and task_cfg.carry.reach_control == "whole_body":
        raise SystemExit("whole_body is a motion pilot; full-pose VLA action export is not validated")


def prepare_scene(src: Path, out: Path, task_cfg: CrateTaskConfig, expert: str, *, record: bool = False) -> Path:
    """Copy the frozen crate scene into the run (named after the source, the expert's lookups need that) and
    strip the marker paint for the assisted pose source; returns the copy."""
    check_profile(task_cfg, expert, record)
    src = Path(src)
    if not (src / "scene_meta.json").is_file():
        raise SystemExit(
            f"no crate scene at {src}; build it:\n  {paths.VENV_SIM_PY} {Path(paths.SRC) / 'tasks/crate/bimanual_scene.py'} {src}"
        )
    if expert == "visual" and task_cfg.visual.pose_source == "markers":
        if "crate_visual" not in json.loads((src / "scene_meta.json").read_text()):
            raise SystemExit("visual expert requires a scene built by locomani.tasks.crate.visual_scene")
    scene = Path(out) / src.name
    if not scene.exists():
        Path(out).mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, scene)
        if expert == "visual" and task_cfg.visual.pose_source == "assisted":
            import xml.etree.ElementTree as ET

            from locomani.sim.convex_boxes import convert_body_boxes
            from locomani.tasks.crate.visual_scene import remove_paint

            remove_paint(scene)
            xml = scene / "scene_locomani.xml"
            tree = ET.parse(xml)
            meta = json.loads((scene / "scene_meta.json").read_text())
            surface = meta["surfaces"][meta["objects"][meta.get("target_object") or "blue_crate"]["surface"]]
            convert_body_boxes(tree, surface["body"])
            tree.write(xml)
    return scene


def expert_argv(
    expert: str, task_cfg_path: Path, manifest: Path, rounds: int, seed: int, *, record: bool, trace_dir=None
) -> list:
    """The expert's own command line (the flags the retired pilot passed)."""
    script, prefix = EXPERTS[expert]
    argv = [paths.VENV_SIM_PY, "-u", Path(paths.SRC) / script, "--rounds", str(rounds), "--seed", str(seed)]
    if record:
        argv.append("--record")
    argv += ["--manifest", manifest, "--label", f"{prefix}_s{seed}", "--config", task_cfg_path]
    if trace_dir is not None and expert == "wall":
        argv += ["--ghost-trace", str(trace_dir)]
    return argv


def prepare(cfg: dict, run_dir: Path) -> Path:
    """Freeze the scene and the expert's config for a bimanual run; returns the scene copy.

    The single-arm counterpart is `locomani.tasks.bottle.prepare.prepare`, and the two are called
    through the same seam (`locomani.pipeline.lines`) so the shared stack starter carries no
    per-line branch. Idempotent: the stack starter and the round driver are separate processes
    since 2026-09-21 and either may be the one that gets here first.
    """
    task = CrateTaskConfig.from_dict(cfg["task"])
    source = Path(cfg["scene"]["dir"])
    scene = prepare_scene(
        source if source.is_absolute() else Path(paths.REPO) / source,
        Path(run_dir),
        task,
        task.expert,
        record=bool(cfg["collect"]["record"]),
    )
    task.dump(Path(run_dir) / "task_config.json")
    return scene


def collect_root(run_dir: Path) -> Path:
    """This line keeps manifests, videos and traces beside the run, not under `collect/`."""
    return Path(run_dir)


def trace_root(run_dir: Path) -> Path:
    """This line has always kept traces beside the run, and its expert is handed the path."""
    return Path(run_dir) / "trace"


def scene_name(cfg: dict, run_dir: Path) -> Path:
    """The copy keeps the SOURCE directory's name; the expert's own lookups depend on it."""
    return Path(run_dir) / Path(cfg["scene"]["dir"]).name


def eval_backend(evaluation) -> dict:
    """The eval stack's backend block: always a clean deploy, whatever `client` says.

    `EvalConfig` ties `clean_deploy` to `client == "async"`, but this line's evaluator pins
    `client="blocking"` AND runs its rounds clean (`vla_rounds --clean`, the default): a commander
    that drops every planner command, including the start that would engage the deploy. Only
    SONIC_AUTO_START engages it then. Taking the block unchanged started a deploy nothing engaged
    -- `eval_crate_closed_loop --attach` now refuses that pairing too (2026-09-23).
    """
    return {**evaluation.backend.to_dict(), "clean_deploy": True}
