"""Learnability pilot: random-spawn ready-grasp collection -> convert -> train -> 100-round closed loop.

Differs from run_ready_pilot.py on purpose:
  * random spawns drawn from the CLOSED-LOOP EVAL distribution (edge 1.1-1.3 m,
    yaw 20 deg, lateral 0.20 m, jitter 0.20) instead of a frozen 0.9-1.0 m case
    bank -- the question is whether the policy is learnable on the benchmark
    distribution, not whether the expert passes a gate;
  * full stack restart every batch of rounds (exporter thread growth,
    container pid limit 3840);
  * evaluation is the standard fresh-round benchmark (seed 6000, same spawn
    args as base100), not a case bank, so the number sits next to 50/100.
Stage receipts make a rerun resume at the first incomplete stage.

Every semantic knob lives in `PilotConfig` (locomani/pipeline/pilot_config.py)
and `BottleTaskConfig` (locomani/tasks/bottle/config.py); see
`configs/pilot/learn_200.json` and `configs/tasks/bottle_visible.json` for the
canonical ready-visible protocol. `--set key=value` overrides one PilotConfig
field at a time.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # reach src/ for `python run_learn_pilot.py`
from locomani.backends.sonic.config import SonicBackendConfig  # noqa: E402
from locomani.pipeline import stack  # noqa: E402
from locomani.pipeline.experiment_cases import load_cases, save_cases  # noqa: E402
from locomani.pipeline.pilot_config import PilotConfig  # noqa: E402
from locomani.pipeline.provenance import digest, file_digests, write_config_snapshot  # noqa: E402
from locomani.pipeline.stages import Stage  # noqa: E402
from locomani.tasks.bottle.config import BottleTaskConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
DATA = ROOT / "third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python"
TRAIN = ROOT / "third_party/Isaac-GR00T/.venv/bin/python"
SRC = ROOT / "src/locomani"
SESSIONS = Path("/workspace/tmp/p03/outputs")


def spawn_argv(spawn) -> list[str]:
    """grasp_gym's --spawn-* flags for a BottleTaskConfig.spawn (SpawnConfig)."""
    return [
        "--spawn-edge",
        *[str(v) for v in spawn.edge_range],
        *(["--face-object"] if spawn.face_object else []),
        "--spawn-yaw-err",
        str(spawn.yaw_err_deg),
        "--spawn-lateral",
        str(spawn.lateral),
        "--edge-jitter",
        str(spawn.edge_jitter),
    ]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--config", type=Path, help="PilotConfig JSON; defaults if omitted")
    ap.add_argument("--task", type=Path, help="BottleTaskConfig JSON; defaults if omitted")
    ap.add_argument("--set", dest="set_", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True, help="frozen scene dir (LOCOMANI_SIM_DIR)")
    ap.add_argument(
        "--cases", type=Path, help="fixed sampling bank; scene identity must match, batches consume it in order"
    )
    ap.add_argument(
        "--initial-token-file", type=Path, default=ROOT / "configs/deployment/ready_grasp_initial_token.json"
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = PilotConfig.load(args.config) if args.config else PilotConfig()
    if args.set_:
        cfg = cfg.with_overrides(args.set_)
    if cfg.steps <= 0 or cfg.save_steps <= 0:
        raise ValueError("steps and save_steps must be positive")
    task_cfg = BottleTaskConfig.load(args.task) if args.task else BottleTaskConfig()
    SPAWN = spawn_argv(task_cfg.spawn)
    collect_backend = SonicBackendConfig(no_encoder=False)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reach_cfg_path = out / "reach_config.json"
    if not reach_cfg_path.exists():
        task_cfg.reach.dump(reach_cfg_path)
    task_cfg_path = out / "task_config.json"
    if not task_cfg_path.exists():
        task_cfg.dump(task_cfg_path)
    scene = out / "scene"
    if not scene.exists():
        shutil.copytree(args.scene, scene)
    cases = load_cases(args.cases, scene / "scene_meta.json") if args.cases else None
    if cases is not None and len(cases) < cfg.batch_rounds * cfg.max_batches:
        raise ValueError("fixed cases must cover batch_rounds * max_batches; no implicit resampling")
    if cases is not None and any("object_yaw" not in case for case in cases):
        raise ValueError("fixed cases need object_yaw; regenerate with grasp_gym --sample-only --write-cases")
    token = args.initial_token_file.resolve()

    stack_env = dict(kv.split("=", 1) for kv in cfg.stack_env)
    base_env = dict(
        os.environ,
        LOCOMANI_SIM_DIR=str(scene),
        SIM_VIS="none",
        SCENE_RANDOM="0",
        HF_HOME="/workspace/huggingface_cache",
        PYTHONUNBUFFERED="1",
        WANDB="0",
    )
    for k in ("NO_ENCODER", "CLEAN_DEPLOY", "SIM_VIDEO_DIR", "TF_INIT_TOKEN_FILE", "CUDA_VISIBLE_DEVICES"):
        base_env.pop(k, None)

    snapshot_env = stack.resolve_env(
        collect_backend,
        task_cfg,
        instance=0,
        gpu=stack_env.get("CUDA_VISIBLE_DEVICES", "0"),
        inherit=base_env,
        extra_env=dict(stack_env, **dict(kv.split("=", 1) for kv in cfg.gym_env)),
    )
    if task_cfg.reach.strategy == "whole_body":
        from locomani.pipeline.provenance import require_pose_ownership

        require_pose_ownership(snapshot_env, hand_handoff=True)
    if not (out / "config_snapshot.json").exists():
        write_config_snapshot(
            out,
            task_cfg=task_cfg,
            backend_cfg=SonicBackendConfig.from_dict(json.loads(snapshot_env["LOCOMANI_BACKEND_JSON"])),
            pilot_cfg=cfg,
            started=time.time(),
            runtime_env=snapshot_env,
            inputs=dict(
                scene=file_digests(scene),
                initial_token=digest(token),
                reach_config=digest(reach_cfg_path),
                **({"cases": digest(args.cases)} if args.cases else {}),
            ),
        )

    stages = Stage(out)

    def run(name, cmd, extra=None):
        stages.check(name, cmd, env=base_env if extra is None else dict(base_env, **extra))

    def start(name, backend, settings):
        return stack.start_stack(
            backend,
            task_cfg,
            instance=0,
            gpu=settings.get("CUDA_VISIBLE_DEVICES", "0"),
            inherit=base_env,
            extra_env=settings,
            runner=stages.run,
            name=name,
            receipt_path=out / "deploy_receipts.jsonl",
        )

    def done(stage):
        return (out / f"{stage}.done").exists()

    def mark(stage):
        (out / f"{stage}.done").write_text(str(time.time()))

    stack_up = False

    def stop():
        nonlocal stack_up
        if stack_up:
            stack.stop_instance(0, runner=stages.run)
            stack_up = False

    server = None
    try:
        # ---------------- collect ----------------
        collect = out / "collect"
        collect.mkdir(exist_ok=True)
        inputs = collect / "inputs.json"
        state = (
            json.loads(inputs.read_text()) if inputs.exists() else dict(sessions=[], manifests=[], kept=0, batches=0)
        )
        while not done("collect") and state["kept"] < cfg.successes and state["batches"] < cfg.max_batches:
            b = state["batches"]
            tag = f"{b:02d}"
            manifest = collect / f"manifest_{tag}.jsonl"
            case_argv = []
            if cases is not None:
                batch_cases = cases[b * cfg.batch_rounds : (b + 1) * cfg.batch_rounds]
                case_path = collect / f"cases_{tag}.json"
                if case_path.exists():
                    if load_cases(case_path, scene / "scene_meta.json") != batch_cases:
                        raise ValueError("fixed case batch changed on resume")
                else:
                    save_cases(case_path, batch_cases, scene / "scene_meta.json")
                case_argv = ["--cases", case_path]
            if manifest.exists():
                raise RuntimeError(f"{manifest} exists but is not in inputs.json; inspect before continuing")
            env = dict(
                SIM_VIDEO_DIR=str(collect / f"videos_{tag}"),
                SIM_STATE_TRACE_DIR=str(collect / f"state_trace_{tag}"),
                TF_FRAME_AUDIT_PATH=str(collect / f"frame_audit_{tag}.jsonl"),
                TF_EXPORTER_SCRIPT=str(SRC / "pipeline/run_instance_exporter.py"),
            )
            env.update(stack_env)
            env.update(dict(kv.split("=", 1) for kv in cfg.gym_env))
            before = set(SESSIONS.iterdir())
            env = start(f"start_stack_{tag}", collect_backend, env)
            stack_up = True
            # Wait for THIS stack's exporter dir. max(mtime) alone raced: a slow
            # teardown could leave the previous batch's dir newest, giving an
            # off-by-one session<->manifest join that only surfaces at convert.
            session = None
            for _ in range(60):
                fresh = [d for d in SESSIONS.iterdir() if d not in before]
                if fresh:
                    session = max(fresh, key=lambda p: p.stat().st_mtime)
                    break
                time.sleep(1)
            if session is None:
                raise RuntimeError("no new exporter session dir appeared after start_stack")
            state["sessions"].append(str(session))
            state["manifests"].append(str(manifest))
            inputs.write_text(json.dumps(state, indent=1))
            try:
                run(
                    f"collect_{tag}",
                    [
                        SIM,
                        SRC / "tasks/bottle/grasp_gym.py",
                        "--reach-config",
                        reach_cfg_path,
                        "--targets",
                        task_cfg.target,
                        "--task-config",
                        task_cfg_path,
                        "--rounds",
                        str(cfg.batch_rounds),
                        *(["--until-ok", str(cfg.successes - state["kept"])] if cases is None else []),
                        "--seed",
                        str(cfg.seed0 + 100 * b),
                        "--stand-retries",
                        "0",
                        "--record",
                        "--video-include-init",
                        *SPAWN,
                        *case_argv,
                        "--manifest",
                        manifest,
                    ],
                    env,
                )
            finally:
                stop()
            rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()] if manifest.exists() else []
            got = sum(bool(r.get("ok") and r.get("produced")) for r in rows)
            state["kept"] += got
            state["batches"] += 1
            inputs.write_text(json.dumps(state, indent=1))
            print(f"collect: batch {tag} {got}/{len(rows)} ok, total kept {state['kept']}/{cfg.successes}", flush=True)
        if not done("collect"):
            if state["kept"] < cfg.successes:
                raise RuntimeError(
                    f"collection stopped at {state['kept']}/{cfg.successes} after {state['batches']} batches"
                )
            mark("collect")
        if cfg.through == "collect":
            return

        # ---------------- convert ----------------
        train_ds, val_ds = out / "train_dataset", out / "val_dataset"
        if not done("convert"):
            for d in (train_ds, val_ds):
                if d.exists():
                    raise RuntimeError(f"partial conversion exists at {d}; inspect it and use a new output directory")
            run(
                "convert",
                [
                    DATA,
                    SRC / "pipeline/convert_to_hf_format.py",
                    "--sessions",
                    *state["sessions"],
                    "--manifest",
                    *state["manifests"],
                    "--trim-verification-tail",
                    "--out",
                    train_ds,
                    "--holdout",
                    str(cfg.holdout),
                    "--holdout-out",
                    val_ds,
                    "--jobs",
                    "4",
                ],
            )
            for label, ds in (("train", train_ds), ("val", val_ds)):
                run(
                    f"stats_{label}",
                    [
                        TRAIN,
                        "-m",
                        "gr00t.data.stats",
                        "--dataset-path",
                        ds,
                        "--embodiment-tag",
                        "UNITREE_G1_SONIC",
                        "--modality-config-path",
                        ROOT / "configs/gr00t_sonic_2cam.py",
                    ],
                )
                run(f"audit_{label}", [DATA, SRC / "pipeline/check_dataset.py", ds])
            mark("convert")
        if cfg.through == "convert":
            return

        # ---------------- train ----------------
        # The trainer always saves final weights to model/; periodic checkpoint
        # directories exist only when the final step lands on the save interval.
        ckpt = out / "model"
        if cfg.steps % min(cfg.save_steps, cfg.steps) == 0:
            ckpt = ckpt / f"checkpoint-{cfg.steps}"
        if not done("train"):
            if (out / "model").exists():
                raise RuntimeError("partial model exists; inspect it and use a new output directory")
            run(
                "train",
                ["bash", ROOT / "scripts/train_bottle.sh"],
                dict(
                    DATA=str(train_ds),
                    RUN="model",
                    OUT_ROOT=str(out),
                    MAX_STEPS=str(cfg.steps),
                    SAVE_STEPS=str(min(cfg.save_steps, cfg.steps)),
                    SAVE_LIMIT="1",
                    WANDB="0",
                    WORKERS="2",  # one checkpoint (disk)
                    GPUS="1",
                    SINGLE_GPU="1",
                    CUDA_VISIBLE_DEVICES=cfg.train_gpu,
                    # Another instance's MuJoCo EGL context sits on device 0 (~1.6-2.1 GB)
                    # whatever CUDA_VISIBLE_DEVICES says; the default 2048 MiB guard trips on it.
                    SQUAT_MAX=str(cfg.squat_max),
                ),
            )
            if not ckpt.exists():
                raise RuntimeError("no checkpoint after training")
            mark("train")
        if cfg.through == "train":
            return

        # ---------------- eval ----------------
        if not done("eval"):
            ev = out / "eval"
            ev.mkdir(exist_ok=True)
            eval_env = start(
                "start_eval_stack",
                SonicBackendConfig(dataset_fps=25.0),
                dict(stack_env, NO_ENCODER="1", SIM_VIDEO_DIR=str(ev / "videos")),
            )
            stack_up = True
            slog = (ev / "serve.log").open("a")
            server = subprocess.Popen(
                [str(TRAIN), str(SRC / "pipeline/serve_policy.py"), "--model-path", str(ckpt), "--port", "5590"],
                cwd=ROOT,
                env=dict(base_env, CUDA_VISIBLE_DEVICES=cfg.train_gpu),
                stdout=slog,
                stderr=subprocess.STDOUT,
            )
            t_end = time.monotonic() + 600
            while "listening" not in (ev / "serve.log").read_text():
                if server.poll() is not None or time.monotonic() > t_end:
                    raise RuntimeError("policy server did not come up")
                time.sleep(3)
            # configs/eval/closed_loop_100.json already encodes this stage's
            # protocol (seed 6000, async, 100 rounds, budget 90, replan 0.4,
            # ensemble 1, spawn 1.1-1.3); --set only for what this run
            # genuinely varies (round/seed CLI flags above) or that the
            # canonical file does not carry (stand_retries=0 here vs its
            # default 1; dataset_fps=25 for checkpoints trained on
            # re-converted data, vla_round.py's DATASET_FPS comment).
            run(
                "eval",
                [
                    SIM,
                    SRC / "pipeline/eval_closed_loop.py",
                    "--config",
                    ROOT / "configs/eval/closed_loop_100.json",
                    "--set",
                    f"rounds={cfg.eval_rounds}",
                    "--set",
                    f"seed={cfg.eval_seed}",
                    "--set",
                    "stand_retries=0",
                    "--set",
                    "backend.dataset_fps=25",
                    "--reach-config",
                    reach_cfg_path,
                    "--initial-token-file",
                    token,
                    "--out",
                    ev / "closed_loop.json",
                    "--trace-dir",
                    ev / "traces",
                ],
                eval_env,
            )
            mark("eval")
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        stop()
    print("pilot complete", flush=True)


if __name__ == "__main__":
    main()
