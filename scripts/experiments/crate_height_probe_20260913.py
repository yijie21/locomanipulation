"""Measure v1.1 squat height delivery away from the table, with both arms held.

Uses the shared stack lifecycle. This is a body-height diagnostic, not a grasp
attempt: starts far from the table and never walks or touches the crate.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from locomani import paths
from locomani.pipeline import stack
from locomani.pipeline.provenance import digest, file_digests, write_config_snapshot
from locomani.pipeline.run_crate_pilot import backend_config
from locomani.pipeline.stages import Stage
from locomani.tasks.crate.config import CrateTaskConfig


def worker(out):
    from locomani.backends.sonic.agent_server import LocomotionMode
    from locomani.backends.sonic.primitives import Primitives, upper_body_from_mj
    from locomani.sim.privileged_pub import EpisodeControlClient
    from locomani.tasks.crate import round_runner
    from locomani.tasks.crate.bimanual_carry import BimanualArms
    from locomani.tasks.crate.carry_config import DEFAULT_CARRY
    from locomani.tasks.crate.carry_phases import CarryRoundContext, _phase_reset
    from locomani.tasks.crate.motion_recording import MotionRecording

    meta = json.loads(Path(paths.SCENE_META).read_text())
    p, reset = Primitives(meta=meta), EpisodeControlClient()
    round_runner.prime(p, reset)
    rng = np.random.default_rng(11)
    for index, height in enumerate((0.65, 0.58, 0.53, 0.48)):
        label = f"height_{index:02d}"
        ctx = CarryRoundContext(p, reset, meta, index, rng, p.log, label, dict(ok=False, why="", produced=False))
        recording = None
        samples = []
        try:
            if _phase_reset(ctx) is not None:
                continue
            arms = BimanualArms(p, config=DEFAULT_CARRY)
            initial = p.obs()
            arms.start(q17=upper_body_from_mj(initial["joints"]), settle_seconds=0)
            held = arms.arm.q.copy()
            recording = MotionRecording(out / f"{label}_motion.jsonl", ctx.t_video)
            arms.motion_recording = recording
            recording.phase = "height_probe"
            start = time.time()
            initial_z = float(initial["pelvis"]["pos"][2])
            duration = (initial_z - height) / 0.04
            while time.time() - start < duration + 3:
                elapsed = time.time() - start
                arms.arm.mode = LocomotionMode.IDEL_SQUAT
                arms.arm.height = max(height, initial_z - 0.04 * elapsed)
                arms.push(held)
                state = p.obs()
                samples.append(
                    dict(
                        t=elapsed,
                        height_command=arms.arm.height,
                        pelvis=state["pelvis"],
                        seq=state["seq"],
                        fallen=state["fallen"],
                    )
                )
                if state["fallen"]:
                    ctx.m["why"] = "[height] fell"
                    break
            tail = [s for s in samples if s["t"] >= duration + 2]
            ctx.m.update(
                command_height=height,
                initial_z=initial_z,
                settled_z=float(np.median([s["pelvis"]["pos"][2] for s in tail])) if tail else None,
                ok=bool(tail) and not ctx.m["why"],
                meaning="ok means the diagnostic completed without falling; inspect measured height",
            )
        finally:
            if recording is not None:
                recording.close()
            reset.end_video()
            (out / f"{label}_height.json").write_text(json.dumps(samples, indent=2) + "\n")
            with (out / "manifest.jsonl").open("a") as f:
                f.write(json.dumps(dict(ctx.m, video=label)) + "\n")
            print(label, json.dumps(ctx.m), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker:
        worker(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    scene = out / args.scene.name
    shutil.copytree(args.scene, scene)
    (out / "trace").mkdir()
    (out / "videos").mkdir()
    task = CrateTaskConfig()
    backend = backend_config()
    write_config_snapshot(
        out,
        task_cfg=task,
        backend_cfg=backend,
        started=time.time(),
        inputs=dict(scene=file_digests(scene), script={Path(__file__).name: digest(__file__)}),
    )
    stage = Stage(out)
    env = stack.start_stack(
        backend,
        task,
        instance=1,
        gpu="0",
        video_dir=str(out / "videos"),
        extra_env=dict(LOCOMANI_SIM_DIR=str(scene), SIM_STATE_TRACE_DIR=str(out / "trace")),
        runner=stage.run,
        receipt_path=out / "deploy_receipts.jsonl",
    )
    try:
        rc = stage.run(
            "height_probe",
            [paths.VENV_SIM_PY, "-u", str(Path(__file__).resolve()), "--worker", "--out", str(out)],
            env=env,
        )
    finally:
        stack.stop_instance(1, runner=stage.run)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
