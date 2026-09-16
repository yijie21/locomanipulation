"""eval: closed-loop rounds against the trained checkpoint (policy server + eval stack this stage owns)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from omegaconf import DictConfig

from locomani.backends.sonic.config import SonicBackendConfig
from locomani.pipeline import stack as stack_env
from locomani.pipeline.stages import Stage
from locomani.stack import Stack, StackSpec
from locomani.stages import common

TOKEN_FILE = common.ROOT / "configs/deployment/ready_grasp_initial_token.json"


def _serve(ckpt: Path, port: int, gpu: str, log: Path, modality: Path, tag: str) -> subprocess.Popen:
    server = subprocess.Popen(
        [
            str(common.TRAIN_PY),
            str(common.SRC / "pipeline/serve_policy.py"),
            "--model-path",
            str(ckpt),
            "--port",
            str(port),
            "--modality-config",
            str(modality),
            "--embodiment-tag",
            tag.upper(),
        ],
        cwd=common.ROOT,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            HF_HOME=os.environ.get("HF_HOME", os.path.join(common.paths.WORKSPACE, "huggingface_cache")),
        ),
        stdout=log.open("a"),
        stderr=subprocess.STDOUT,
    )
    t_end = time.monotonic() + 600
    while "listening" not in log.read_text():
        if server.poll() is not None or time.monotonic() > t_end:
            server.terminate()
            raise RuntimeError(f"policy server did not come up ({log})")
        time.sleep(3)
    return server


def _crate_eval(run_cfg, out: Path, ckpt: Path, ev: Path) -> None:
    """The bimanual evaluator owns its own stack and policy server (eval_crate_closed_loop)."""
    e = run_cfg.eval
    scene = next(p for p in out.iterdir() if p.is_dir() and (p / "scene_meta.json").exists())
    Stage(out).check(
        "eval",
        [
            common.SIM_PY,
            "-m",
            "locomani.pipeline.eval_crate_closed_loop",
            "--checkpoint",
            ckpt,
            "--scene",
            scene,
            "--out",
            ev,
            "--config",
            out / "task_config.json",
            "--rounds",
            str(e.rounds),
            "--seed",
            str(e.seed),
            "--instance",
            str(run_cfg.collect.instance),
            "--gpu",
            str(run_cfg.gpu),
            "--serve-gpu",
            str(run_cfg.gpu),
        ],
        env=dict(os.environ, PYTHONPATH=str(common.ROOT / "src")),
    )


def run(run_cfg, cfg: DictConfig, out: Path) -> dict:
    started, resources = time.time(), common.resources_now()
    up = common.require_upstream(out, "eval")
    ckpt = Path(up["outputs"]["checkpoint"])
    ev = common.stage_dir(out, "eval")
    e = run_cfg.eval
    if run_cfg.task.plugin == "crate":
        _crate_eval(run_cfg, out, ckpt, ev)
        result = json.loads((ev / "closed_loop.json").read_text())
        return common.write_manifest(
            out,
            "eval",
            cfg,
            ok=True,
            outputs=dict(closed_loop=ev / "closed_loop.json", checkpoint=ckpt),
            metrics={k: result[k] for k in ("rounds", "successes", "success_rate") if k in result},
            started=started,
            resources=resources,
        )
    backend = SonicBackendConfig(dataset_fps=25.0, no_encoder=True)
    env = stack_env.resolve_env(
        backend,
        run_cfg.task,
        instance=run_cfg.collect.instance,
        gpu=str(run_cfg.gpu),
        extra_env=dict(SIM_VIDEO_DIR=str(ev / "videos"), **dict(kv.split("=", 1) for kv in run_cfg.collect.stack_env)),
    )
    env["LOCOMANI_SIM_DIR"] = str(out / "scene")
    spec = StackSpec(
        instance=int(run_cfg.collect.instance),
        gpu=str(run_cfg.gpu),
        scene_dir=str(out / "scene"),
        prompt=run_cfg.task.prompt,
        record=False,
        encoder=False,
        video_dir=str(ev / "videos"),
        env={k: v for k, v in env.items() if k.startswith(("TF_", "LOCOMANI_BACKEND_JSON", "HAND"))},
        receipts=str(out / "deploy_receipts.jsonl"),
    )
    stages = Stage(out)
    stages.record("start_eval_stack", ["locomani.stack", "up", str(spec.instance)], env)
    stack = Stack(spec, base_env=env, record_command=stages.record)
    env = stack.up()
    server = None
    try:
        server = _serve(
            ckpt,
            e.policy_port,
            str(run_cfg.gpu),
            ev / "serve.log",
            common.ROOT / run_cfg.embodiment.gr00t_modality_config,
            run_cfg.embodiment.gr00t_tag,
        )
        stages.check(
            "eval",
            [
                common.SIM_PY,
                common.SRC / "pipeline/eval_closed_loop.py",
                "--set",
                f"rounds={e.rounds}",
                "--set",
                f"seed={e.seed}",
                "--set",
                f"stand_retries={e.stand_retries}",
                "--set",
                f"policy_port={e.policy_port}",
                "--set",
                "backend.dataset_fps=25",
                "--reach-config",
                out / "reach_config.json",
                "--initial-token-file",
                TOKEN_FILE,
                "--out",
                ev / "closed_loop.json",
                "--trace-dir",
                ev / "traces",
            ],
            env=env,
        )
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        stack.down()
        stages.record("stop_eval_stack", ["locomani.stack", "down", str(spec.instance)], {})
    result = json.loads((ev / "closed_loop.json").read_text())
    metrics = {k: result[k] for k in ("rounds", "successes", "success_rate") if k in result}
    return common.write_manifest(
        out,
        "eval",
        cfg,
        ok=True,
        outputs=dict(closed_loop=ev / "closed_loop.json", checkpoint=ckpt),
        metrics=metrics,
        started=started,
        resources=resources,
    )
