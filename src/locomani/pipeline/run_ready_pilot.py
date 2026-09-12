"""Bounded ready-grasp collection -> training -> exact-case evaluation.

Stage receipts cache only completed, fingerprint-identical work. Failed stages
stop the run. Logs and timings survive failures. Never resumes partial exports.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from locomani.backends.sonic.config import SonicBackendConfig  # noqa: E402
from locomani.pipeline import stack  # noqa: E402
from locomani.pipeline.stages import Stage  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
DATA = ROOT / "third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python"
TRAIN = ROOT / "third_party/Isaac-GR00T/.venv/bin/python"
SRC = ROOT / "src/locomani"
CONFIG = ROOT / "configs/collection/ready_grasp_coupled.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_session():
    return max(Path("/workspace/tmp/p03/outputs").iterdir(), key=lambda p: p.stat().st_mtime)


def evaluation_command(args, out: Path, reach_config: Path, initial_token_file: Path | None):
    """Prepare the exact config/argv consumed by normal and instrumented evaluation."""
    from locomani.pipeline.eval_config import EvalConfig

    config = EvalConfig(
        backend=SonicBackendConfig(dataset_fps=25.0),
        rounds=args.eval_rounds,
        stand_retries=args.stand_retries,
        seed=16800,
    )
    config_path = out / "eval_config.json"
    config.dump(config_path)
    entry = (
        [SIM, SRC / "rtc/diag_transition_live.py", out / "execution_capture"]
        if args.capture_execution
        else [SIM, SRC / "pipeline/eval_closed_loop.py"]
    )
    return [
        *entry,
        "--config",
        config_path,
        "--cases",
        out / "eval_cases.json",
        "--reach-config",
        reach_config,
        *(["--initial-token-file", initial_token_file] if initial_token_file else []),
        "--out",
        out / "eval.json",
        "--trace-dir",
        out / "traces",
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reach-config", type=Path, default=CONFIG)
    ap.add_argument(
        "--initial-token-file", type=Path, help="validated fixed token matching the collection starting posture"
    )
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--successes", type=int, default=24)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--collection-cases", type=Path, help="frozen case bank; replaces random collection batches")
    ap.add_argument("--eval-rounds", type=int, default=10)
    ap.add_argument(
        "--capture-execution", action="store_true", help="record predicted chunks, sends and WBC payload evidence"
    )
    ap.add_argument("--stand-retries", type=int, choices=(0, 1), default=1)
    ap.add_argument("--through", choices=("collect", "convert", "train", "eval"), default="eval")
    ap.add_argument("--reuse-data", type=Path, help="completed pilot whose converted datasets are reused")
    args = ap.parse_args(argv)
    if args.eval_rounds <= 0 or args.steps <= 0 or not 0 < args.holdout < args.successes or args.jobs < 1:
        ap.error("invalid budgets")
    collection_cases = args.collection_cases.resolve() if args.collection_cases else None
    case_rounds = len(json.loads(collection_cases.read_text())["cases"]) if collection_cases else None
    if case_rounds is not None and case_rounds < args.successes:
        ap.error("fixed case bank cannot meet requested success count")
    reach_config = args.reach_config.resolve()
    if not reach_config.is_file():
        ap.error("collection configuration missing")
    initial_token_file = args.initial_token_file.resolve() if args.initial_token_file else None
    if initial_token_file:
        sys.path.insert(0, str(ROOT / "src"))
        from locomani.backends.sonic.initial_pose import load_token_file

        load_token_file(initial_token_file)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reuse = args.reuse_data.resolve() if args.reuse_data else None
    reused_identity = None
    if reuse:
        receipt = reuse / "convert.done.json"
        if not receipt.exists():
            ap.error("reuse-data requires a completed conversion and audit")
        saved_receipt = json.loads(receipt.read_text())
        if saved_receipt.get("fingerprint") != digest(reuse / "spec.json"):
            ap.error("reuse-data receipt does not match original experiment specification")
        if not all(Path(p).exists() for p in saved_receipt.get("artifacts", [])):
            ap.error("reuse-data artifacts missing")
        original_spec = json.loads((reuse / "spec.json").read_text())
        previous_config = original_spec.get("collection_config", {}).get("sha256")
        if previous_config is None:
            if original_spec.get("policy") != reach_config.stem:
                ap.error("reused data belongs to a different collection policy")
            key = str(reach_config.relative_to(ROOT)) if reach_config.is_relative_to(ROOT) else str(reach_config)
            previous_config = original_spec.get("sources", {}).get(key)
        if previous_config != digest(reach_config):
            ap.error("reused data belongs to a different collection policy")
        reused_identity = {}
        for name in ("train_dataset", "val_dataset"):
            ds = reuse / name
            if not (ds / "meta/stats.json").exists():
                ap.error(f"missing dataset statistics: {ds}")
            for file in sorted(ds.rglob("*")):
                if file.is_file():
                    if file.stat().st_mtime > receipt.stat().st_mtime:
                        ap.error(f"dataset changed since completed audit: {file}")
                    reused_identity[str(file.relative_to(reuse))] = digest(file)
        if not (out / "scene").exists():
            shutil.copytree(reuse / "scene", out / "scene")
        if not (out / "eval_cases.json").exists():
            shutil.copy2(reuse / "eval_cases.json", out / "eval_cases.json")
    if not (out / "eval_cases.json").exists():
        ap.error("generate eval_cases.json and freeze scene/ first")
    env = dict(
        os.environ,
        LOCOMANI_SIM_DIR=str(out / "scene"),
        SIM_VIS="none",
        SCENE_RANDOM="0",
        HF_HOME="/workspace/huggingface_cache",
        PYTHONUNBUFFERED="1",
        WANDB="0",
    )
    for key in ("NO_ENCODER", "CLEAN_DEPLOY", "SIM_VIDEO_DIR", "TF_INIT_TOKEN_FILE"):
        env.pop(key, None)
    if collection_cases:
        env.update(
            TF_EXPORTER_SCRIPT=str(SRC / "pipeline/run_instance_exporter.py"),
            TF_FRAME_AUDIT_PATH=str(out / "frame_audit_00.jsonl"),
        )
    source_paths = [
        p
        for base in ("src/locomani", "src/gr00t", "configs", "scripts")
        for p in (ROOT / base).rglob("*")
        if p.suffix in (".py", ".json", ".sh")
    ]
    spec = dict(
        reuse_data=None if reuse is None else str(reuse),
        reused_identity=reused_identity,
        policy=reach_config.stem,
        collection_config=dict(path=str(reach_config), sha256=digest(reach_config)),
        base="nvidia/GR00T-N1.7-3B",
        steps=args.steps,
        successes=args.successes,
        holdout=args.holdout,
        collection_cases_sha256=digest(collection_cases) if collection_cases else None,
        collection_rounds=case_rounds,
        collection_chunk_rounds=10 if collection_cases else None,
        stand_retries=args.stand_retries,
        eval_rounds=args.eval_rounds,
        capture_execution=args.capture_execution,
        jobs=args.jobs,
        dataset_fps=25,
        eval_budget=90,
        eval_replan_interval=0.4,
        initial_token=None
        if initial_token_file is None
        else dict(path=str(initial_token_file), sha256=digest(initial_token_file)),
        eval_cases_sha256=digest(out / "eval_cases.json"),
        scene={p.name: digest(p) for p in (out / "scene").iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)): digest(p) for p in sorted(source_paths)},
    )
    spec_path = out / "spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise RuntimeError("experiment inputs changed; use a new output directory")
    spec_path.write_text(json.dumps(spec, indent=2))
    fingerprint = digest(spec_path)
    stack_active = False
    server = None

    stages = Stage(out)

    def run(name, cmd, extra=None):
        stages.check(name, cmd, env=env if extra is None else dict(env, **extra))

    def start(name, *, evaluation=False):
        nonlocal env
        env = stack.start_stack(
            SonicBackendConfig(no_encoder=evaluation, dataset_fps=25.0 if evaluation else 30.0),
            SimpleNamespace(prompt="pick up the water bottle"),
            instance=0,
            gpu=env.get("CUDA_VISIBLE_DEVICES", "0"),
            inherit=env,
            extra_env={key: env[key] for key in ("TF_EXPORTER_SCRIPT", "TF_FRAME_AUDIT_PATH") if key in env},
            runner=stages.run,
            name=name,
            receipt_path=out / "deploy_receipts.jsonl",
        )

    def cached(stage):
        receipt = out / f"{stage}.done.json"
        return (
            receipt.exists()
            and json.loads(receipt.read_text())["fingerprint"] == fingerprint
            and all(Path(p).exists() for p in json.loads(receipt.read_text())["artifacts"])
        )

    def done(stage, artifacts):
        (out / f"{stage}.done.json").write_text(
            json.dumps(
                dict(fingerprint=fingerprint, artifacts=[str(x) for x in artifacts], completed=time.time()), indent=2
            )
        )

    def stop():
        nonlocal stack_active
        if stack_active:
            stack.stop_instance(0, runner=stages.run)
            stack_active = False

    try:
        if reuse is None and not cached("collect"):
            collect = out / "collect"
            if collect.exists():
                raise RuntimeError("partial collection exists; inspect manifests before extending in a new run")
            collect.mkdir()
            start("start_collect_stack")
            stack_active = True
            sessions, manifests = [], []
            source_audits = []
            kept = 0
            if collection_cases:
                bank = json.loads(collection_cases.read_text())
                # Exporters accumulate threads per episode. Bound each process
                # lifetime without drawing replacements or changing case order.
                # Full stack resets also avoid carrying a previous gym's state
                # into a new process's first round. Both paired arms do this.
                for chunk, offset in enumerate(range(0, case_rounds, 10)):
                    if kept >= args.successes:
                        break
                    selected = bank["cases"][offset : offset + 10]
                    chunk_cases = collect / f"cases_{chunk:02d}.json"
                    chunk_cases.write_text(json.dumps(dict(bank, cases=selected), indent=2))
                    session = latest_session()
                    manifest = collect / f"manifest_{chunk:02d}.jsonl"
                    sessions.append(str(session))
                    manifests.append(str(manifest))
                    (collect / "inputs.json").write_text(
                        json.dumps(dict(sessions=sessions, manifests=manifests), indent=2)
                    )
                    run(
                        f"collect_fixed_{chunk:02d}",
                        [
                            SIM,
                            SRC / "tasks/bottle/grasp_gym.py",
                            "--reach-config",
                            reach_config,
                            "--targets",
                            "water_bottle",
                            "--rounds",
                            str(len(selected)),
                            "--until-ok",
                            str(args.successes - kept),
                            "--cases",
                            chunk_cases,
                            "--stand-retries",
                            str(args.stand_retries),
                            "--record",
                            "--manifest",
                            manifest,
                        ],
                    )
                    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
                    kept += sum(r["ok"] and r["produced"] for r in rows)
                    audit = out / f"frame_source_audit_{chunk:02d}.json"
                    run(
                        f"audit_sources_{chunk:02d}",
                        [
                            SIM,
                            SRC / "tools/audit_frame_sources.py",
                            out / f"frame_audit_{chunk:02d}.jsonl",
                            "--session",
                            session,
                            "--out",
                            audit,
                        ],
                    )
                    source_audits.append(audit)
                    if kept < args.successes and offset + len(selected) < case_rounds:
                        stop()
                        env["TF_FRAME_AUDIT_PATH"] = str(out / f"frame_audit_{chunk + 1:02d}.jsonl")
                        start(f"start_collect_stack_{chunk + 1:02d}")
                        stack_active = True
            else:
                for chunk in range(4):
                    if kept >= args.successes:
                        break
                    session = latest_session()
                    manifest = collect / f"manifest_{chunk:02d}.jsonl"
                    sessions.append(str(session))
                    manifests.append(str(manifest))
                    (collect / "inputs.json").write_text(
                        json.dumps(dict(sessions=sessions, manifests=manifests), indent=2)
                    )
                    run(
                        f"collect_{chunk:02d}",
                        [
                            SIM,
                            SRC / "tasks/bottle/grasp_gym.py",
                            "--reach-config",
                            reach_config,
                            "--targets",
                            "water_bottle",
                            "--rounds",
                            "10",
                            "--until-ok",
                            str(min(8, args.successes - kept)),
                            "--seed",
                            str(26800 + chunk * 100),
                            "--stand-retries",
                            str(args.stand_retries),
                            "--record",
                            "--spawn-edge",
                            "1.1",
                            "1.3",
                            "--face-object",
                            "--spawn-yaw-err",
                            "10",
                            "--spawn-lateral",
                            ".1",
                            "--edge-jitter",
                            ".1",
                            "--manifest",
                            manifest,
                        ],
                    )
                    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
                    kept += sum(r["ok"] and r["produced"] for r in rows)
                    print(f"collection: {kept}/{args.successes} kept", flush=True)
                    if kept + min(8, args.successes) * (3 - chunk) < args.successes:
                        break  # Remaining sessions cannot meet the declared success target.
                    if kept < args.successes:
                        run(
                            f"exporter_restart_{chunk:02d}",
                            ["bash", ROOT / "scripts/restart_exporter.sh"],
                            dict(TASK_PROMPT="pick up the water bottle"),
                        )
            stop()
            if kept < args.successes:
                raise RuntimeError(
                    f"expert gate failed: only {kept}/{args.successes} after bounded collection; inspect failures before training"
                )
            done("collect", [collect / "inputs.json", *manifests, *source_audits])
        if args.through == "collect":
            return
        data_root = reuse or out
        train_ds, val_ds = data_root / "train_dataset", data_root / "val_dataset"
        if reuse is None and not cached("convert"):
            if train_ds.exists() or val_ds.exists():
                raise RuntimeError("partial conversion exists; refusing overwrite")
            inputs = json.loads((out / "collect/inputs.json").read_text())
            run(
                "convert",
                [
                    DATA,
                    SRC / "pipeline/convert_to_hf_format.py",
                    "--sessions",
                    *inputs["sessions"],
                    "--manifest",
                    *inputs["manifests"],
                    "--trim-verification-tail",
                    "--out",
                    train_ds,
                    "--holdout",
                    str(args.holdout),
                    "--holdout-out",
                    val_ds,
                    "--jobs",
                    str(args.jobs),
                ],
            )
            for label, ds in [("train", train_ds), ("val", val_ds)]:
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
            done("convert", [train_ds, val_ds])
        if args.through == "convert":
            return
        checkpoint = out / "model" / f"checkpoint-{args.steps}"
        if not cached("train"):
            if (out / "model").exists():
                raise RuntimeError("partial training exists; saved model is not an exact training resume")
            run(
                "train",
                ["bash", ROOT / "scripts/train_bottle.sh"],
                dict(
                    DATA=str(train_ds),
                    RUN="model",
                    OUT_ROOT=str(out),
                    MAX_STEPS=str(args.steps),
                    SAVE_STEPS=str(args.steps),
                    SAVE_LIMIT="1",
                    WANDB="0",
                    WORKERS="2",
                ),
            )
            if not checkpoint.exists():
                raise RuntimeError("training returned without expected checkpoint")
            done("train", [checkpoint])
        if args.through == "train":
            return
        if not cached("eval"):
            if (out / "eval.json").exists():
                raise RuntimeError("partial evaluation exists; refusing to mix case results")
            start("start_eval_stack", evaluation=True)
            stack_active = True
            server_log = (out / "serve.log").open("a")
            server = subprocess.Popen(
                [str(TRAIN), str(SRC / "pipeline/serve_policy.py"), "--model-path", str(checkpoint), "--port", "5590"],
                cwd=ROOT,
                env=dict(env, CUDA_VISIBLE_DEVICES="1"),
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 300
            while "listening" not in (out / "serve.log").read_text():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("policy server failed to become ready")
                time.sleep(2)
            run("eval", evaluation_command(args, out, reach_config, initial_token_file))
            if args.capture_execution:
                run(
                    "audit_execution",
                    [
                        SIM,
                        SRC / "rtc/audit_prefix_execution.py",
                        out / "execution_capture",
                        "--out",
                        out / "execution_prefix_audit.json",
                    ],
                )
            done("eval", [out / "eval.json"])
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        stop()
    print("Completed. This is a new-policy learning pilot, not a paired improvement claim.", flush=True)


if __name__ == "__main__":
    main()
