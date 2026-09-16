"""collect: batches of expert rounds on a stack this stage owns; manifest = sessions, manifests, yield.

The single-arm tasks (bottle, pick_place) run `tasks/bottle/grasp_gym.py` as the agent process; the
crate task runs `tasks/crate/expert.py`. Each batch gets a fresh stack (exporter thread growth,
container pid limit) started through `locomani.stack.Stack` on the run's GPU and instance.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from omegaconf import DictConfig

from locomani.backends.sonic.config import SonicBackendConfig
from locomani.pipeline import stack as stack_env
from locomani.pipeline.experiment_cases import load_cases, save_cases
from locomani.pipeline.stages import Stage
from locomani.stack import Stack, StackSpec, session_root
from locomani.stages import common


def _spawn_argv(spawn) -> list[str]:
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


def _prepare_scene(run, out: Path) -> Path:
    scene = out / "scene"
    if not scene.exists():
        src = Path(run.scene.dir)
        shutil.copytree(src if src.is_absolute() else common.ROOT / src, scene)
    return scene


def _stack_spec(run, out: Path, backend: SonicBackendConfig, batch_env: dict, prompt: str) -> tuple[StackSpec, dict]:
    """The stack for one batch and the environment the agent process gets (same settings, same instance)."""
    env = stack_env.resolve_env(
        backend,
        run.task,
        instance=run.collect.instance,
        gpu=str(run.gpu),
        extra_env=dict(batch_env, **dict(kv.split("=", 1) for kv in run.collect.gym_env)),
    )
    env["LOCOMANI_SIM_DIR"] = str(out / "scene")
    spec = StackSpec(
        instance=int(run.collect.instance),
        gpu=str(run.gpu),
        scene_dir=str(out / "scene"),
        prompt=prompt,
        record=bool(run.collect.record),
        encoder=not backend.no_encoder,
        lockstep=bool(backend.lockstep.enabled),
        wrist_cams=bool(backend.wrist_cams),
        video_dir=batch_env.get("SIM_VIDEO_DIR"),
        sim_vis=backend.sim_vis,
        scene_random=bool(backend.scene_random),
        dds_domain=backend.sonic_dds_domain,
        env={k: v for k, v in env.items() if k.startswith(("TF_", "LOCOMANI_BACKEND_JSON", "HAND", "EXPORTER_"))},
        receipts=str(out / "deploy_receipts.jsonl"),
    )
    return spec, env


def _wait_session(sessions: Path, before: set, timeout: float = 60.0) -> Path:
    end = time.time() + timeout
    while time.time() < end:
        fresh = [d for d in sessions.iterdir() if d not in before]
        if fresh:
            return max(fresh, key=lambda p: p.stat().st_mtime)
        time.sleep(1)
    raise RuntimeError("no new exporter session dir appeared after the stack came up")


def _bottle_batch(run, out: Path, stages: Stage, b: int, cases, state: dict, backend: SonicBackendConfig) -> Path:
    collect = out / "collect"
    tag = f"{b:02d}"
    manifest = collect / f"manifest_{tag}.jsonl"
    if manifest.exists():
        raise RuntimeError(f"{manifest} exists but is not in inputs.json; inspect before continuing")
    case_argv: list = []
    if cases is not None:
        batch_cases = cases[b * run.collect.batch_rounds : (b + 1) * run.collect.batch_rounds]
        case_path = collect / f"cases_{tag}.json"
        if case_path.exists():
            if load_cases(case_path, out / "scene" / "scene_meta.json") != batch_cases:
                raise ValueError("fixed case batch changed on resume")
        else:
            save_cases(case_path, batch_cases, out / "scene" / "scene_meta.json")
        case_argv = ["--cases", case_path]
    batch_env = dict(
        SIM_VIDEO_DIR=str(collect / f"videos_{tag}"),
        SIM_STATE_TRACE_DIR=str(collect / f"state_trace_{tag}"),
        TF_FRAME_AUDIT_PATH=str(collect / f"frame_audit_{tag}.jsonl"),
        TF_EXPORTER_SCRIPT=str(common.SRC / "pipeline/run_instance_exporter.py"),
        **dict(kv.split("=", 1) for kv in run.collect.stack_env),
    )
    spec, env = _stack_spec(run, out, backend, batch_env, run.task.prompt)
    sessions = session_root(spec.instance) / "outputs"
    sessions.mkdir(parents=True, exist_ok=True)
    before = set(sessions.iterdir())
    stages.record(f"start_stack_{tag}", ["locomani.stack", "up", str(spec.instance)], env)
    stack = Stack(spec, base_env=env, record_command=stages.record)
    env = stack.up()  # the agent runs in exactly the environment the stack's children got (DDS domain included)
    try:
        session = _wait_session(sessions, before) if spec.record else None
        state["sessions"].append(str(session) if session else None)
        state["manifests"].append(str(manifest))
        (collect / "inputs.json").write_text(json.dumps(state, indent=1))
        argv = [
            common.SIM_PY,
            common.SRC / "tasks/bottle/grasp_gym.py",
            "--reach-config",
            out / "reach_config.json",
            "--targets",
            run.task.target,
            "--task-config",
            out / "task_config.json",
            "--rounds",
            str(run.collect.batch_rounds),
            *(["--until-ok", str(run.collect.successes - state["kept"])] if cases is None else []),
            "--seed",
            str(run.collect.seed0 + 100 * b),
            "--stand-retries",
            "0",
            *(["--record"] if spec.record else []),
            "--video-include-init",
            *_spawn_argv(run.task.spawn),
            *case_argv,
            "--manifest",
            manifest,
        ]
        stages.check(f"collect_{tag}", argv, env=env)
    finally:
        stack.down()
        stages.record("stop_stack", ["locomani.stack", "down", str(spec.instance)], {})
    return manifest


def _crate_batches(run, out: Path, stages: Stage, state: dict) -> dict:
    """The crate experts, batch by batch on their own stack."""
    from locomani.tasks.crate.prepare import expert_argv, prepare_scene

    expert, record = run.task.expert, bool(run.collect.record)
    src = Path(run.scene.dir)
    scene = prepare_scene(src if src.is_absolute() else common.ROOT / src, out, run.task, expert, record=record)
    backend = run.backend if isinstance(run.backend, SonicBackendConfig) else SonicBackendConfig()
    if expert in ("visual", "carry") and run.task.carry.reach_control == "whole_body":
        from locomani.pipeline.provenance import require_pose_ownership

        require_pose_ownership(
            stack_env.resolve_env(backend, run.task, instance=run.collect.instance, gpu=str(run.gpu))
        )
    trace_dir = out / "trace"
    trace_dir.mkdir(exist_ok=True)
    while state["batches"] < run.collect.max_batches and state["kept"] < run.collect.successes:
        b = state["batches"]
        tag = f"{b:02d}"
        manifest = out / f"manifest_{tag}.jsonl"
        if manifest.exists() and str(manifest) not in state["manifests"]:
            raise RuntimeError(f"{manifest} exists but is not in inputs.json; inspect before continuing")
        videos = out / f"videos_{tag}"
        videos.mkdir(exist_ok=True)
        extra = dict(LOCOMANI_SIM_DIR=str(scene), SIM_STATE_TRACE_DIR=str(trace_dir))
        extra.update(dict(kv.split("=", 1) for kv in run.collect.stack_env))
        if record:
            extra["RECORD_INSTANCE"] = "1"
        env = stack_env.resolve_env(
            backend, run.task, instance=run.collect.instance, gpu=str(run.gpu), video_dir=str(videos), extra_env=extra
        )
        spec = StackSpec(
            instance=int(run.collect.instance),
            gpu=str(run.gpu),
            scene_dir=str(scene),
            prompt=run.task.prompt,
            record=record,
            encoder=not backend.no_encoder,
            wrist_cams=bool(backend.wrist_cams),
            video_dir=str(videos),
            env={k: v for k, v in env.items() if k.startswith(("TF_", "LOCOMANI_BACKEND_JSON", "HAND"))},
            receipts=str(out / "deploy_receipts.jsonl"),
        )
        sessions = session_root(spec.instance) / "outputs"
        sessions.mkdir(parents=True, exist_ok=True)
        before = set(sessions.iterdir())
        stages.record(f"start_stack_{tag}", ["locomani.stack", "up", str(spec.instance)], env)
        stack = Stack(spec, base_env=env, record_command=stages.record)
        env = stack.up()
        try:
            session = _wait_session(sessions, before) if record else None
            state["sessions"].append(str(session) if session else None)
            state["manifests"].append(str(manifest))
            (out / "collect" / "inputs.json").write_text(json.dumps(state, indent=1))
            seed = run.collect.seed0 + 100 * b
            argv = expert_argv(
                expert,
                out / "task_config.json",
                manifest,
                run.collect.batch_rounds,
                seed,
                record=record,
                trace_dir=trace_dir,
            )
            stages.run(f"{expert}_{tag}", argv, env=env)
        finally:
            stack.down()
            stages.record("stop_stack", ["locomani.stack", "down", str(spec.instance)], {})
        got = sum(bool(r.get("ok") and (r.get("produced", True) if record else True)) for r in common.rows(manifest))
        state["kept"] += got
        state["batches"] += 1
        (out / "collect" / "inputs.json").write_text(json.dumps(state, indent=1))
        print(f"collect: batch {tag} {got} kept, total {state['kept']}/{run.collect.successes}", flush=True)
    return state


def run(run_cfg, cfg: DictConfig, out: Path) -> dict:
    started, resources = time.time(), common.resources_now()
    out.mkdir(parents=True, exist_ok=True)
    scene = _prepare_scene(run_cfg, out) if run_cfg.task.plugin != "crate" else out
    task_path = out / "task_config.json"
    if not task_path.exists():
        run_cfg.task.dump(task_path)
    if hasattr(run_cfg.task, "reach") and not (out / "reach_config.json").exists():
        run_cfg.task.reach.dump(out / "reach_config.json")
    plugin = run_cfg.task.plugin
    collect = common.stage_dir(out, "collect")
    inputs = collect / "inputs.json"
    state = json.loads(inputs.read_text()) if inputs.exists() else dict(sessions=[], manifests=[], kept=0, batches=0)
    if plugin == "crate":
        state = _crate_batches(run_cfg, out, Stage(out), state)
    else:
        backend = run_cfg.backend if isinstance(run_cfg.backend, SonicBackendConfig) else SonicBackendConfig()
        if getattr(run_cfg.task.reach, "strategy", None) in ("whole_body", "simple"):
            # the streamed-pose experts need the patched deploy (hand ownership); refuse before any stack
            from locomani.pipeline.provenance import require_pose_ownership

            require_pose_ownership(
                stack_env.resolve_env(backend, run_cfg.task, instance=run_cfg.collect.instance, gpu=str(run_cfg.gpu)),
                hand_handoff=True,
            )
        cases = None
        if run_cfg.scene.cases:
            cases = load_cases(common.ROOT / run_cfg.scene.cases, scene / "scene_meta.json")
            if len(cases) < run_cfg.collect.batch_rounds * run_cfg.collect.max_batches:
                raise ValueError("fixed cases must cover batch_rounds * max_batches; no implicit resampling")
        stages = Stage(out)
        while state["kept"] < run_cfg.collect.successes and state["batches"] < run_cfg.collect.max_batches:
            manifest = _bottle_batch(run_cfg, out, stages, state["batches"], cases, state, backend)
            got = sum(bool(r.get("ok") and r.get("produced")) for r in common.rows(manifest))
            state["kept"] += got
            state["batches"] += 1
            inputs.write_text(json.dumps(state, indent=1))
            print(
                f"collect: batch {state['batches'] - 1:02d} {got} kept, total {state['kept']}/{run_cfg.collect.successes}",
                flush=True,
            )
    all_rows = [r for m in state["manifests"] for r in common.rows(Path(m))]
    ok = state["kept"] >= run_cfg.collect.successes
    metrics = dict(
        rounds=len(all_rows),
        kept=state["kept"],
        batches=state["batches"],
        yield_=round(state["kept"] / len(all_rows), 3) if all_rows else 0.0,
        failure_reasons=sorted({str(r.get("why") or "")[:70] for r in all_rows if not r.get("ok")}),
    )
    m = common.write_manifest(
        out,
        "collect",
        cfg,
        ok=ok,
        outputs=dict(sessions=json.dumps(state["sessions"]), manifests=json.dumps(state["manifests"]), scene=scene),
        metrics=metrics,
        started=started,
        resources=resources,
        error=None
        if ok
        else f"collection stopped at {state['kept']}/{run_cfg.collect.successes} after {state['batches']} batches",
    )
    if not ok:
        raise RuntimeError(m["error"])
    return m
