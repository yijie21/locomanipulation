#!/usr/bin/env python3
"""Collect bimanual episodes: the crate experts, one batch per stack, until enough are kept.

    python scripts/collect_crate.py configs/experiments/crate_table30_record.yaml --gpu 1
    python scripts/collect_crate.py configs/experiments/crate_table30_record.yaml --dry-run

`task.expert` chooses which expert runs -- `visual` is the recording line, `wall` and `carry` are
diagnostics -- and the combination is checked against the task profile before any resource is
touched. The scene copy keeps the source's name because the expert's own lookups depend on it,
which is why this line's artifacts sit beside the run rather than under `collect/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_common  # noqa: E402

from locomani.pipeline.stages import Stage  # noqa: E402
from locomani.stack import Stack, session_root, spec_from_config  # noqa: E402
from locomani.tasks.crate.config import CrateTaskConfig  # noqa: E402
from locomani.tasks.crate.prepare import check_profile, expert_argv, prepare_scene  # noqa: E402


def prepare(cfg: dict, run_dir: Path) -> tuple[CrateTaskConfig, Path]:
    task = CrateTaskConfig.from_dict(cfg["task"])
    source = Path(cfg["scene"]["dir"])
    scene = prepare_scene(
        source if source.is_absolute() else ROOT / source,
        run_dir,
        task,
        task.expert,
        record=bool(cfg["collect"]["record"]),
    )
    task.dump(run_dir / "task_config.json")
    return task, scene


def run_batch(cfg: dict, run_dir: Path, scene: Path, gpu: str, state: dict, dry_run: bool) -> Path:
    batch = state["batches"]
    tag = f"{batch:02d}"
    manifest = run_dir / f"manifest_{tag}.jsonl"
    if manifest.exists() and str(manifest) not in state["manifests"]:
        raise SystemExit(f"{manifest} exists but no batch claimed it; inspect before continuing")
    collect, record = cfg["collect"], bool(cfg["collect"]["record"])
    spec = spec_from_config(
        cfg, run_dir=run_dir, gpu=gpu, scene_dir=scene, video_dir=run_dir / f"videos_{tag}", record=record
    )
    env = dict(spec.settings(), SIM_STATE_TRACE_DIR=str(run_dir / "trace"))
    if record:
        env["RECORD_INSTANCE"] = "1"  # the exporter runs on this instance, whichever one it is
    env.update(dict(kv.split("=", 1) for kv in collect.get("gym_env", [])))
    argv = expert_argv(
        cfg["task"]["expert"],
        run_dir / "task_config.json",
        manifest,
        collect["batch_rounds"],
        collect["seed0"] + 100 * batch,
        record=record,
        trace_dir=run_dir / "trace",
    )
    if dry_run:
        run_common.show(f"stack_up_{tag}", ["locomani.stack", "up", str(spec.instance)], spec.settings())
        run_common.show(
            f"{cfg['task']['expert']}_{tag}", argv, {k: v for k, v in env.items() if k not in spec.settings()}
        )
        return manifest

    (run_dir / "trace").mkdir(exist_ok=True)
    (run_dir / f"videos_{tag}").mkdir(exist_ok=True)
    sessions = session_root(spec.instance) / "outputs"
    sessions.mkdir(parents=True, exist_ok=True)
    before = set(sessions.iterdir())
    stage = Stage(run_dir)
    stack = Stack(spec, base_env=env, record_command=stage.record, run_dir=run_dir)
    env = stack.up()
    try:
        state["sessions"].append(str(_new_session(sessions, before)) if record else None)
        state["manifests"].append(str(manifest))
        (run_dir / "collect").mkdir(exist_ok=True)
        (run_dir / "collect" / "inputs.json").write_text(json.dumps(state, indent=1))
        stage.run(f"{cfg['task']['expert']}_{tag}", argv, env=env)  # a batch may end on a failed round
    finally:
        stack.down()
    return manifest


def _new_session(sessions: Path, before: set, timeout: float = 60.0) -> Path:
    end = time.time() + timeout
    while time.time() < end:
        fresh = [d for d in sessions.iterdir() if d not in before]
        if fresh:
            return max(fresh, key=lambda p: p.stat().st_mtime)
        time.sleep(1)
    raise SystemExit("no new exporter session appeared after the stack came up")


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

    cfg, run_dir = run_common.open_run(args.experiment, args.name, write=not args.dry_run)
    collect = cfg["collect"]
    if args.dry_run:
        check_profile(CrateTaskConfig.from_dict(cfg["task"]), cfg["task"]["expert"], bool(collect["record"]))
        source = Path(cfg["scene"]["dir"])
        scene = run_dir / (source if source.is_absolute() else ROOT / source).name
        run_batch(cfg, run_dir, scene, args.gpu, dict(sessions=[], manifests=[], kept=0, batches=0), True)
        return 0

    run_common.require_headroom("collect", gpu=args.gpu)
    run_common.require_deploy_patches(cfg)
    _, scene = prepare(cfg, run_dir)
    inputs = run_dir / "collect" / "inputs.json"
    state = json.loads(inputs.read_text()) if inputs.exists() else dict(sessions=[], manifests=[], kept=0, batches=0)

    while state["kept"] < collect["successes"] and state["batches"] < collect["max_batches"]:
        manifest = run_batch(cfg, run_dir, scene, args.gpu, state, False)
        state["kept"] += kept_in(manifest, bool(collect["record"]))
        state["batches"] += 1
        inputs.write_text(json.dumps(state, indent=1))
        print(f"collect: batch {state['batches'] - 1:02d} -> {state['kept']}/{collect['successes']} kept", flush=True)

    rows = [json.loads(line) for m in state["manifests"] for line in Path(m).read_text().splitlines() if line.strip()]
    run_common.summarise(
        run_dir,
        "collect",
        expert=cfg["task"]["expert"],
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
