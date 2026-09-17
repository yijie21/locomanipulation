#!/usr/bin/env python3
"""Collect single-arm episodes: one batch at a time, each on its own stack, until enough are kept.

    python scripts/collect_bottle.py configs/experiments/bottle_simple_200.yaml --gpu 1
    python scripts/collect_bottle.py configs/experiments/bottle_simple_gate8.yaml --gpu 1 --dry-run

Serves the bottle and pick-and-place lines: same expert, same gym, different task block. A batch is
one stack lifetime -- the exporter's threads grow with every episode and the container's pid ceiling
is 3840, so the stack is restarted between batches rather than stretched across the whole run.

Interrupted runs continue: `collect/inputs.json` records which batches finished and what they kept,
and rerunning the same name picks up from the next batch. It is not episode-level resume; check the
last batch's log and manifest before continuing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_common  # noqa: E402

from locomani import paths  # noqa: E402
from locomani.pipeline.experiment_cases import load_cases, save_cases  # noqa: E402
from locomani.pipeline.stages import Stage  # noqa: E402
from locomani.stack import Stack, session_root, spec_from_config  # noqa: E402
from locomani.tasks.bottle.config import BottleTaskConfig  # noqa: E402

SIM_PY = paths.VENV_SIM_PY
GYM = ROOT / "src/locomani/tasks/bottle/grasp_gym.py"


def prepare(cfg: dict, run_dir: Path) -> BottleTaskConfig:
    """The scene copy and the two files the expert process reads, written once per run."""
    scene = run_dir / "scene"
    if not scene.exists():
        source = Path(cfg["scene"]["dir"])
        shutil.copytree(source if source.is_absolute() else ROOT / source, scene)
    task = BottleTaskConfig.from_dict(cfg["task"])
    task.dump(run_dir / "task_config.json")
    task.reach.dump(run_dir / "reach_config.json")
    return task


def batch_cases(cfg: dict, run_dir: Path, batch: int, tag: str):
    """The fixed sample bank's slice for this batch, or None when every round is drawn fresh."""
    if not cfg["scene"].get("cases"):
        return None
    every = load_cases(ROOT / cfg["scene"]["cases"], run_dir / "scene" / "scene_meta.json")
    rounds, batches = cfg["collect"]["batch_rounds"], cfg["collect"]["max_batches"]
    if len(every) < rounds * batches:
        raise SystemExit(f"{cfg['scene']['cases']} holds {len(every)} cases, the budget needs {rounds * batches}")
    path = run_dir / "collect" / f"cases_{tag}.json"
    mine = every[batch * rounds : (batch + 1) * rounds]
    if path.exists():
        if load_cases(path, run_dir / "scene" / "scene_meta.json") != mine:
            raise SystemExit(f"{path} is not the slice this batch had; inspect before continuing")
    else:
        save_cases(path, mine, run_dir / "scene" / "scene_meta.json")
    return path


def gym_argv(cfg: dict, run_dir: Path, batch: int, kept: int, cases, manifest: Path) -> list:
    collect, spawn = cfg["collect"], cfg["task"]["spawn"]
    return [
        SIM_PY,
        GYM,
        "--task-config", run_dir / "task_config.json",
        "--reach-config", run_dir / "reach_config.json",
        "--targets", cfg["task"]["target"],
        "--rounds", str(collect["batch_rounds"]),
        *([] if cases else ["--until-ok", str(collect["successes"] - kept)]),
        "--seed", str(collect["seed0"] + 100 * batch),
        "--stand-retries", str(cfg["task"].get("stand_retries", 0)),
        *(["--record"] if collect["record"] else []),
        "--video-include-init",
        "--spawn-edge", *[str(v) for v in spawn["edge_range"]],
        *(["--face-object"] if spawn["face_object"] else []),
        "--spawn-yaw-err", str(spawn["yaw_err_deg"]),
        "--spawn-lateral", str(spawn["lateral"]),
        "--edge-jitter", str(spawn["edge_jitter"]),
        *(["--cases", cases] if cases else []),
        "--manifest", manifest,
    ]  # fmt: skip


def new_session(sessions: Path, before: set, timeout: float = 60.0) -> Path:
    """The exporter session this batch just created, paired with this batch's manifest by order."""
    end = time.time() + timeout
    while time.time() < end:
        fresh = [d for d in sessions.iterdir() if d not in before]
        if fresh:
            return max(fresh, key=lambda p: p.stat().st_mtime)
        time.sleep(1)
    raise SystemExit("no new exporter session appeared after the stack came up")


def run_batch(cfg: dict, run_dir: Path, gpu: str, state: dict, dry_run: bool) -> Path:
    batch = state["batches"]
    tag = f"{batch:02d}"
    collect_dir = run_dir / "collect"
    manifest = collect_dir / f"manifest_{tag}.jsonl"
    if manifest.exists() and str(manifest) not in state["manifests"]:
        raise SystemExit(f"{manifest} exists but no batch claimed it; inspect before continuing")
    spec = spec_from_config(
        cfg,
        run_dir=run_dir,
        gpu=gpu,
        video_dir=collect_dir / f"videos_{tag}",
        record=cfg["collect"]["record"],
    )
    env = dict(
        spec.settings(),
        SIM_STATE_TRACE_DIR=str(collect_dir / f"state_trace_{tag}"),
        TF_FRAME_AUDIT_PATH=str(collect_dir / f"frame_audit_{tag}.jsonl"),
        TF_EXPORTER_SCRIPT=str(ROOT / "src/locomani/pipeline/run_instance_exporter.py"),
        **dict(kv.split("=", 1) for kv in cfg["collect"].get("gym_env", [])),
    )
    argv = gym_argv(cfg, run_dir, batch, state["kept"], batch_cases(cfg, run_dir, batch, tag), manifest)
    if dry_run:
        run_common.show(f"stack_up_{tag}", ["locomani.stack", "up", str(spec.instance)], spec.settings())
        run_common.show(f"collect_{tag}", argv, {k: v for k, v in env.items() if k not in spec.settings()})
        return manifest
    sessions = session_root(spec.instance) / "outputs"
    sessions.mkdir(parents=True, exist_ok=True)
    before = set(sessions.iterdir())
    stage = Stage(run_dir)  # the run's commands.jsonl and timings.jsonl
    stack = Stack(spec, base_env=env, record_command=stage.record, run_dir=run_dir)
    env = stack.up()  # the expert runs in exactly the environment the children got
    try:
        state["sessions"].append(str(new_session(sessions, before)) if spec.record else None)
        state["manifests"].append(str(manifest))
        (collect_dir / "inputs.json").write_text(json.dumps(state, indent=1))
        stage.check(f"collect_{tag}", argv, env=env)
    finally:
        stack.down()
    return manifest


def kept_in(manifest: Path, record: bool) -> int:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return sum(bool(row.get("ok") and (row.get("produced", True) if record else True)) for row in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="run directory name; the experiment's own name by default")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    cfg, run_dir = run_common.open_run(args.experiment, args.name)
    collect = cfg["collect"]
    if not args.dry_run:
        run_common.require_headroom("collect", gpu=args.gpu)
        run_common.require_deploy_patches(cfg)
    prepare(cfg, run_dir)
    (run_dir / "collect").mkdir(exist_ok=True)
    inputs = run_dir / "collect" / "inputs.json"
    state = json.loads(inputs.read_text()) if inputs.exists() else dict(sessions=[], manifests=[], kept=0, batches=0)

    while state["kept"] < collect["successes"] and state["batches"] < collect["max_batches"]:
        manifest = run_batch(cfg, run_dir, args.gpu, state, args.dry_run)
        if args.dry_run:
            return 0
        state["kept"] += kept_in(manifest, collect["record"])
        state["batches"] += 1
        inputs.write_text(json.dumps(state, indent=1))
        print(f"collect: batch {state['batches'] - 1:02d} -> {state['kept']}/{collect['successes']} kept", flush=True)

    rows = [json.loads(line) for m in state["manifests"] for line in Path(m).read_text().splitlines() if line.strip()]
    run_common.summarise(
        run_dir,
        "collect",
        attempts=len(rows),
        kept=state["kept"],
        batches=state["batches"],
        yield_rate=round(state["kept"] / len(rows), 3) if rows else 0.0,
        failures=sorted({str(r.get("why") or "")[:70] for r in rows if not r.get("ok")}),
        sessions=state["sessions"],
        manifests=state["manifests"],
    )
    if state["kept"] < collect["successes"]:
        raise SystemExit(f"stopped at {state['kept']}/{collect['successes']} kept after {state['batches']} batches")
    print(f"collect: {state['kept']} kept in {state['batches']} batches -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
