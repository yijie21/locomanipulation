#!/usr/bin/env python3
"""Drive single-arm collection on a stack someone else started: batch after batch, until enough are kept.

    python -m locomani.pipeline.run_collect_single_arm configs/experiments/bottle_simple_200.yaml --gpu 1

`scripts/single_arm/run_agent_server.sh` is one line over this, and `scripts/collect_data.sh` is
what brought the stack up in the other terminal. This process attaches to that stack, so every
value the children were told comes back from `stack.json` rather than being rebuilt from the
config a second time -- rebuilding it is how two processes end up disagreeing about a field that
the file said once.

A batch is one exporter session. The exporter is restarted between batches for two reasons, and
only one of them is about resources: `convert` pairs exporter session k with manifest k BY
POSITION, so a single session spanning every batch would destroy the pairing; and the exporter's
threads grow with every episode against a container ceiling of 3840 pids. The sim and the deploy
live across the whole run.

Interrupted runs continue: `collect/inputs.json` records which batches finished and what they
kept, and rerunning the same name picks up from the next batch. It is not episode-level resume;
check the last batch's log and manifest before continuing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.pipeline.experiment_cases import load_cases, save_cases
from locomani.pipeline.lines import line_of
from locomani.pipeline.run_stack import frame_audit_path
from locomani.pipeline.stages import Stage
from locomani.stack import Stack, session_root

ROOT = Path(paths.REPO)
SIM_PY = paths.VENV_SIM_PY
GYM = ROOT / "src/locomani/tasks/bottle/grasp_gym.py"


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
        *(["--video-include-init"] if cfg["task"].get("video_include_init") else []),
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
    raise SystemExit("no new exporter session appeared after the exporter restarted")


def open_exporter_session(stack: Stack, collect_root: Path, batch: int) -> str | None:
    """Restart the exporter onto this batch's receipt file; return the session it opened."""
    if not stack.spec.record:
        return None
    sessions = session_root(stack.spec.instance) / "outputs"
    sessions.mkdir(parents=True, exist_ok=True)
    before = set(sessions.iterdir())
    stack.env["TF_FRAME_AUDIT_PATH"] = str(frame_audit_path(collect_root, batch))
    stack.restart_exporter()
    return str(new_session(sessions, before))


def kept_in(manifest: Path, record: bool) -> int:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return sum(bool(row.get("ok") and (row.get("produced", True) if record else True)) for row in rows)


def run_batch(cfg: dict, run_dir: Path, collect_root: Path, stack: Stack, state: dict) -> Path:
    batch = state["batches"]
    tag = f"{batch:02d}"
    manifest = collect_root / f"manifest_{tag}.jsonl"
    if manifest.exists() and str(manifest) not in state["manifests"]:
        raise SystemExit(f"{manifest} exists but no batch claimed it; inspect before continuing")
    argv = gym_argv(cfg, run_dir, batch, state["kept"], batch_cases(cfg, run_dir, batch, tag), manifest)
    state["sessions"].append(open_exporter_session(stack, collect_root, batch))
    state["manifests"].append(str(manifest))
    (collect_root / "inputs.json").write_text(json.dumps(state, indent=1))
    Stage(run_dir).check(f"collect_{tag}", argv, env=dict(stack.env))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="run directory name; the experiment's own name by default")
    ap.add_argument("--keep-stack", action="store_true", help="leave the stack running when the run ends")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    cfg, run_dir = run_context.open_run(args.experiment, args.name, write=not args.dry_run)
    collect, line = cfg["collect"], line_of(cfg)
    collect_root = line.collect_root(run_dir)

    if args.dry_run:
        state = dict(sessions=[], manifests=[], kept=0, batches=0)
        cases = collect_root / "cases_00.json" if cfg["scene"].get("cases") else None
        run_context.show("collect_00", gym_argv(cfg, run_dir, 0, 0, cases, collect_root / "manifest_00.jsonl"))
        print(f"\n  attaches to instance {collect['instance']}; {collect['max_batches']} batches at most")
        return 0

    run_context.require_headroom("collect", gpu=args.gpu, record=bool(collect["record"]))
    run_context.require_deploy_patches(cfg)
    line.prepare(cfg, run_dir)  # idempotent: the stack starter normally got here first
    collect_root.mkdir(parents=True, exist_ok=True)
    inputs = collect_root / "inputs.json"
    state = json.loads(inputs.read_text()) if inputs.exists() else dict(sessions=[], manifests=[], kept=0, batches=0)

    try:
        stack = Stack.attach(int(collect["instance"]))
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {collect['instance']}; start one in another terminal:\n"
            f"  bash scripts/collect_data.sh {args.experiment} --gpu {args.gpu}"
        )
    try:
        while state["kept"] < collect["successes"] and state["batches"] < collect["max_batches"]:
            manifest = run_batch(cfg, run_dir, collect_root, stack, state)
            state["kept"] += kept_in(manifest, collect["record"])
            state["batches"] += 1
            inputs.write_text(json.dumps(state, indent=1))
            print(
                f"collect: batch {state['batches'] - 1:02d} -> {state['kept']}/{collect['successes']} kept", flush=True
            )
    finally:
        if not args.keep_stack:
            stack.down()  # a recording stack left up holds ~1800 of the container's 3840 pids

    rows = [json.loads(line) for m in state["manifests"] for line in Path(m).read_text().splitlines() if line.strip()]
    run_context.summarise(
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
