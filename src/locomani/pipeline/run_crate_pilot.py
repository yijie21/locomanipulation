"""Crate (bimanual) pilot driver: the stack lifecycle in Python, the expert in its own process.

Why this file exists (2026-09-11). The crate line's only runners were
`scripts/run_wall_pilot.sh` and `scripts/run_bimanual_pilot.sh`: two ~80-line
bash scripts that were the same text apart from the python entry point and the
video directory, and each did four jobs at once -- pin the instance/GPU, start
the stack, run the expert, and stop the stack with its own 15-line `pkill`
reimplementation of `stop_instance.sh`. The single-arm line has had those jobs
separated since the refactor: `run_learn_pilot.py` owns the run directory and
the stack lifecycle, and `grasp_gym.py` is an expert process that assumes the
stack is up (its module docstring says so). This file is that same split for
the crate, so `bimanual_wall.py` / `bimanual_carry.py` are launched exactly the
way `grasp_gym.py` is.

What it fixes beyond the duplication:
  * the stack comes up through `pipeline/stack.py` -- decision 8's "one place a
    config becomes shell environment", whose `start_stack`/`stop_instance` had
    no caller at all before this file;
  * it goes down through `scripts/stack/stop_instance.sh`, the one scoped
    implementation. The bash runners matched processes with `pgrep -f`, which
    is what that script's own header warns against (memory
    pkill-self-match-in-shell-chains);
  * the port block is never re-derived here: `stop_instance.sh` and
    `locomani.sim.ports` hold the one `5555 + 10*INSTANCE` formula, and the
    runners carried a third copy of it;
  * every run gets its own directory and its own manifest. The runners pointed
    every run at `outputs/videos/wall_pilot/manifest.jsonl` and the experts
    open the manifest in APPEND mode, so two pilots silently mixed into one
    file -- and `scripts/gate.py record crate` reads that path.

NOT changed here: the experts, their CLI, and what the stack actually runs --
see `backend_config()` for the one knob that has to be spelled out.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # reach src/ for `python run_crate_pilot.py`
from locomani import paths  # noqa: E402
from locomani.backends.sonic.config import SonicBackendConfig  # noqa: E402
from locomani.pipeline import stack  # noqa: E402
from locomani.pipeline.provenance import file_digests, require_pose_ownership, write_config_snapshot  # noqa: E402
from locomani.pipeline.stages import Stage  # noqa: E402
from locomani.tasks.crate.config import CrateTaskConfig  # noqa: E402

# label prefix -> the expert script, keeping each expert's historical default
# label (`wall_s0` / `bimanual_s0`): scripts/render_wall_pilot.sh and
# scripts/gate.py both address rounds by that prefix.
EXPERTS = {
    "visual": ("tasks/crate/expert.py", "visual_crate"),
    "wall": ("tasks/crate/bimanual_wall.py", "wall"),
    "carry": ("tasks/crate/bimanual_carry.py", "bimanual"),
}
DEFAULT_SCENE = Path(paths.SCENE_DIR) / "crate_bimanual"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert", choices=tuple(EXPERTS), default="wall")
    ap.add_argument("--out", type=Path, required=True, help="run directory (manifests, videos, traces, logs)")
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="crate scene dir (LOCOMANI_SIM_DIR)")
    ap.add_argument("--config", type=Path, help="CrateTaskConfig JSON; defaults if omitted")
    ap.add_argument("--set", dest="set_", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--batch-rounds",
        type=int,
        default=0,
        help="restart the stack every N rounds (0 = one stack for the whole run, today's behaviour). "
        "The single-arm line restarts every 10 because the exporter leaks threads against the "
        "container pid cap; the crate line has no exporter yet (docs/crate_collection.md §11)",
    )
    ap.add_argument("--instance", type=int, default=1)
    ap.add_argument(
        "--allow-instance-0",
        action="store_true",
        help="instance 0 is the collection/eval stack, and the exporter only runs there -- which is what "
        "recording a crate dataset will need (docs/crate_collection.md §11). Refused without this flag",
    )
    ap.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID for the stack")
    ap.add_argument(
        "--no-state-trace",
        action="store_true",
        help="skip the per-round state trace. It is on by default because scripts/render_wall_pilot.sh "
        "reads <out>/trace/<label>_NN_state_trace.npz and the ghost hands come from the same directory "
        "(memory crate-video-ghost-hands-rule); the bash runners left it to an env var nobody set",
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="drive the data exporter and keep only the rounds that succeed -- the crate line's first "
        "route to a LeRobot dataset (docs/crate_collection.md \u00a711). Implies instance 0, the only one "
        "start_stack.sh runs an exporter on, and a batch size so the exporter's thread growth stays "
        "under the container pid cap",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing run directory at its first unfinished batch. Without it a directory that "
        "already holds batches is refused: the experts APPEND to their manifest, and a second run sharing "
        "the directory is how two pilots became one population in outputs/videos/wall_pilot/manifest.jsonl",
    )
    ap.add_argument(
        "--allow-no-encoder",
        action="store_true",
        help="permit a backend config with no_encoder=True. NO_ENCODER=1 is a token/VLA-only mode: it "
        "silently drops the arm and crouch references the crate expert drives through the planner "
        "(memory no-encoder-breaks-planner-arms)",
    )
    args = ap.parse_args(argv)
    if args.expert == "visual" and args.record:
        ap.error("visual currently records RGB/history video pilots; formal LeRobot export is not yet validated")
    if args.record:
        # Not a warning: a --record run on another instance completes every round
        # and produces no session at all (memory only-instance-0-can-record).
        args.instance, args.allow_instance_0 = 0, True
        if not args.batch_rounds:
            args.batch_rounds = 10  # the single-arm line's batch, for the same pid-cap reason
    if args.instance == 0 and not args.allow_instance_0:
        ap.error("refusing INSTANCE=0: that is the collection/eval stack (pass --allow-instance-0 to override)")
    if args.batch_rounds < 0:
        ap.error("--batch-rounds must be >= 0")
    if args.rounds < 1:
        ap.error("--rounds must be >= 1")
    return args


def backend_config(allow_no_encoder: bool = False) -> SonicBackendConfig:
    """The stack env for a crate pilot.

    `no_encoder=False` is spelled out rather than taken from the dataclass
    default, which is `True`: that default is the EVAL protocol's (the only
    caller of `stack.export_env` is eval_closed_loop.py, and both
    `configs/backends/sonic.json` and `configs/eval/closed_loop_100.json` set
    it true). The crate expert drives the arms and the crouch through the
    planner channel, which NO_ENCODER=1 silently drops -- and the bash runners
    never set the variable at all, so start_stack.sh's own `NO_ENCODER:-0`
    applied. This keeps that byte-for-byte while making it visible.

    Every other field's default already equals what the runners left unset:
    sim_vis "none", scene_random False, wrist_cams True (start_stack.sh's
    `WRIST_CAMS:-1`), lockstep off, sonic_dds_domain None so start_stack.sh
    keeps deriving the DDS domain from INSTANCE.
    """
    cfg = SonicBackendConfig(no_encoder=False)
    if allow_no_encoder:
        cfg = SonicBackendConfig(no_encoder=True)
    return cfg


def expert_argv(args, task_cfg_path: Path, manifest: Path, label: str, trace_dir: Path | None, seed=None) -> list:
    """The expert's own command line -- the same flags the bash runners passed."""
    seed = args.seed if seed is None else seed
    script, _ = EXPERTS[args.expert]
    argv = [
        paths.VENV_SIM_PY,
        "-u",
        Path(paths.SRC) / script,
        "--rounds",
        str(args.rounds if args.batch_rounds == 0 else args.batch_rounds),
        "--seed",
        str(seed),
        *(["--record"] if args.record else []),
        "--manifest",
        manifest,
        "--label",
        label,
        "--config",
        task_cfg_path,
    ]
    # bimanual_carry.py has no --ghost-trace: only the wall expert aims a pair
    # of hands at a rail, so only it has anything to draw a ghost for.
    if trace_dir is not None and args.expert == "wall":
        argv += ["--ghost-trace", str(trace_dir)]
    return argv


def prepare(args) -> tuple[Path, CrateTaskConfig, Path, Path]:
    """Make the run directory reproducible: scene copy, task config, provenance."""
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    task_cfg = CrateTaskConfig.load(args.config) if args.config else CrateTaskConfig()
    if args.set_:
        task_cfg = task_cfg.with_overrides(args.set_)
    if args.expert == "carry" and args.record and task_cfg.carry.reach_control == "whole_body":
        raise SystemExit("whole_body is a motion pilot; full-pose VLA action export is not validated")
    task_cfg_path = out / "task_config.json"
    if not task_cfg_path.exists():
        task_cfg.dump(task_cfg_path)
    if not (args.scene / "scene_meta.json").is_file():
        raise SystemExit(
            f"no crate scene at {args.scene}; build it:\n"
            f"  {paths.VENV_SIM_PY} {Path(paths.SRC) / 'tasks/crate/bimanual_scene.py'} {args.scene}"
        )
    if args.expert == "visual" and task_cfg.visual.pose_source == "markers":
        metadata = json.loads((args.scene / "scene_meta.json").read_text())
        if "crate_visual" not in metadata:
            raise SystemExit("visual expert requires a scene built by locomani.tasks.crate.visual_scene")
    # Keep the source directory's BASENAME: scripts/render_wall_pilot.sh resolves
    # the Isaac assets as assets/isaac/scene/$(basename $LOCOMANI_SIM_DIR), so a
    # copy named `scene` (run_learn_pilot.py's name) would break that lookup.
    scene = out / args.scene.name
    if not scene.exists():
        shutil.copytree(args.scene, scene)
        if args.expert == "visual" and task_cfg.visual.pose_source == "assisted":
            from locomani.tasks.crate.visual_scene import remove_paint

            remove_paint(scene)
            # Keep the basket shape while avoiding analytic box/box phantom contacts.
            import xml.etree.ElementTree as ET

            from locomani.sim.convex_boxes import convert_body_boxes

            xml = scene / "scene_locomani.xml"
            tree = ET.parse(xml)
            meta = json.loads((scene / "scene_meta.json").read_text())
            surface = meta["surfaces"][meta["objects"]["blue_crate"]["surface"]]
            convert_body_boxes(tree, surface["body"])
            tree.write(xml)
    if not (out / "config_snapshot.json").exists():
        write_config_snapshot(
            out,
            task_cfg=task_cfg,
            backend_cfg=backend_config(args.allow_no_encoder),
            started=time.time(),
            inputs=dict(scene=file_digests(scene)),
        )
    return out, task_cfg, task_cfg_path, scene


def main(argv=None) -> None:
    args = parse_args(argv)
    # Before anything is written: a second run in a directory that already holds
    # one is how two pilots became one manifest (see --resume).
    state_path = args.out.resolve() / "inputs.json"
    if state_path.exists() and not args.resume:
        raise SystemExit(f"{args.out} already holds a run ({state_path}); pass --resume or pick another --out")
    out, task_cfg, task_cfg_path, scene = prepare(args)
    backend = backend_config(args.allow_no_encoder)
    if args.expert in ("visual", "carry") and task_cfg.carry.reach_control == "whole_body":
        require_pose_ownership(stack.resolve_env(backend))
    stage = Stage(out)
    trace_dir = None if args.no_state_trace else out / "trace"
    if trace_dir is not None:
        trace_dir.mkdir(exist_ok=True)

    per_batch = args.batch_rounds or args.rounds
    batches = -(-args.rounds // per_batch)  # ceil
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(manifests=[], batches=0)
    print(
        f"[crate] {args.expert} expert, {args.rounds} rounds in {batches} batch(es), seed {args.seed}, "
        f"instance {args.instance}, gpu {args.gpu}, scene {scene} -> {out}",
        flush=True,
    )

    failures = 0
    for b in range(state["batches"], batches):
        tag = f"{b:02d}"
        manifest = out / f"manifest_{tag}.jsonl"
        # run_learn_pilot.py's guard, for the same reason: the experts append to
        # the manifest, so a leftover file from another run joins two pilots
        # into one file that gate.py then reads as a single population.
        if manifest.exists() and str(manifest) not in state["manifests"]:
            raise SystemExit(f"{manifest} exists but is not in inputs.json; inspect before continuing")
        videos = out / f"videos_{tag}"
        videos.mkdir(exist_ok=True)
        extra = {"LOCOMANI_SIM_DIR": str(scene)}  # the EGL device follows `gpu` inside stack.start_stack
        if trace_dir is not None:
            extra["SIM_STATE_TRACE_DIR"] = str(trace_dir)
        try:
            env = stack.start_stack(
                backend,
                task_cfg,
                instance=args.instance,
                gpu=args.gpu,
                video_dir=str(videos),
                extra_env=extra,
                runner=stage.run,
                name=f"start_stack_{tag}",
                receipt_path=out / "deploy_receipts.jsonl",
            )
        except RuntimeError as e:
            raise SystemExit(f"[crate] {e}; see {paths.session_root(args.instance)}") from None
        state["manifests"].append(str(manifest))
        state_path.write_text(json.dumps(state, indent=1))
        try:
            # run_learn_pilot.py's rule: batch b runs seed0 + 100*b, so two
            # batches never replay each other's rounds. With the default
            # (one batch) the label is the historical `wall_s<seed>`.
            seed = args.seed + 100 * b
            label = f"{EXPERTS[args.expert][1]}_s{seed}"
            argv_expert = expert_argv(args, task_cfg_path, manifest, label, trace_dir, seed)
            rc = stage.run(f"{args.expert}_{tag}", argv_expert, env=env)
            failures += bool(rc)
        finally:
            stack.stop_instance(args.instance, runner=stage.run)
        state["batches"] = b + 1
        state_path.write_text(json.dumps(state, indent=1))

    rows = [json.loads(x) for m in state["manifests"] for x in Path(m).read_text().splitlines() if x.strip()]
    ok = sum(bool(r.get("ok")) for r in rows)
    print(f"[crate] {ok}/{len(rows)} OK   manifests {out}/manifest_*.jsonl   videos {out}/videos_*", flush=True)
    if failures:
        raise SystemExit(f"[crate] {failures} batch(es) exited non-zero; see {out}/*.log")


if __name__ == "__main__":
    main()
