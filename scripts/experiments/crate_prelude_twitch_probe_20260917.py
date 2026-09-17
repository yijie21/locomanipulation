"""Measure the crate evaluation prelude: what the body does between reset and the first policy chunk.

Runs exactly what a round runs up to the point the policy is first queried --
placement, reset, IDLE hold, then initial_pose.prepare_token_pose -- with no
checkpoint and no policy server, because the lurch happens before any inference.
Samples the 29 joints, the pelvis pose and the deploy's own published reference
at ~50 Hz on its OWN subscriber (a zmq socket is not thread-safe; sharing the
Primitives sockets trips libzmq msg.cpp:367 the moment the prelude publishes).

The flags exist to separate the candidate causes; the measurements each one
produced are tabulated in docs/handoff/crate-eval-prelude-twitch-20260917.md.

    --warmup       one unrecorded round first (what vla_rounds.warm_up now does)
    --encoder      launch the deploy WITH the encoder (evaluation uses NO_ENCODER=1)
    --pin          slew the sim arm pin to the token posture and hold it through the switch
    --early        switch to the token right after start_control, no IDLE settle
    --prime-first  enter streamed mode once before the first reset

Ports: the module-level port constants resolve at IMPORT time from INSTANCE, and
scripts/paths.sh exports TF_* for instance 0, which wins over INSTANCE. So run it
with those unset:

    env -u TF_CAMERA_PORT -u TF_SONIC_PORT -u TF_OUT_PORT -u TF_KEYBOARD_PORT \
        -u TF_PRIVILEGED_PORT -u TF_EPISODE_PORT INSTANCE=2 \
        "$VENV_SIM_PY" scripts/experiments/crate_prelude_twitch_probe_20260917.py \
        --out outputs/<name> --instance 2 --gpu 1 --rounds 2 --warmup
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from locomani.backends.sonic import initial_pose, upper_ref  # noqa: E402
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode  # noqa: E402
from locomani.backends.sonic.config import SonicBackendConfig  # noqa: E402
from locomani.backends.sonic.primitives import Primitives  # noqa: E402
from locomani.pipeline import stack  # noqa: E402
from locomani.pipeline.stages import Stage  # noqa: E402
from locomani.sim.privileged_pub import EpisodeControlClient, PrivilegedSubscriber  # noqa: E402
from locomani.tasks.crate.config import CrateTaskConfig  # noqa: E402
from locomani.tasks.crate.visual_approach import initial_robot_xy  # noqa: E402


class ArmTrace:
    """Measured 14 arm joints at ~50 Hz, with a monotonic clock and a phase label.

    On its OWN subscriber: a zmq socket is not thread-safe, and sharing the
    Primitives' sockets with the sampling thread trips libzmq's msg.cpp:367
    assertion the moment the prelude starts publishing.
    """

    def __init__(self):
        self.sub = PrivilegedSubscriber()
        self.tap = upper_ref.shared()  # the deploy's own published reference (g1_debug body_q_target)
        self.rows, self.stop, self.label = [], False, "settled"

    def run(self):
        while not self.stop:
            state = self.sub.latest(timeout_ms=100)
            if state is not None:
                ref = self.tap.q29(max_age=0.5)
                self.rows.append(
                    dict(
                        t=time.monotonic(),
                        label=self.label,
                        arms=list(np.asarray(state["joints"], float)[15:29]),
                        ref_arms=None if ref is None else list(np.asarray(ref, float)[15:29]),
                        joints=list(map(float, state["joints"])),
                        pelvis_pos=list(map(float, state["pelvis"]["pos"])),
                        pelvis_quat=list(map(float, state["pelvis"]["quat"])),
                        pelvis_vel=list(map(float, state["pelvis"]["vel"])),
                    )
                )
            time.sleep(0.02)


def pin_arms_to(p, reset, target, *, rate: float = 1.0, dt: float = 0.02, settle: float = 0.3) -> None:
    """Slew the sim-side arm pin from the measured arms to `target` (14, MuJoCo order).

    A rejected candidate fix, kept as a comparison arm: the pin replaces the arm
    motors' torques (run_sim_locomani PIN_KP 120 / tau 30) so it works under
    NO_ENCODER, but it covers the arms only -- the legs and waist still step.
    """
    q = np.asarray(p.obs()["joints"], float)[15:29].copy()
    want = np.asarray(target, dtype=float)
    step = rate * dt
    while True:
        q += np.clip(want - q, -step, step)
        reset.pin_arms(left=q[:7].tolist(), right=q[7:].tolist())
        if np.abs(want - q).max() < 1e-9:
            break
        time.sleep(dt)
    time.sleep(settle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instance", type=int, default=2)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--scene", type=Path, default=ROOT / "outputs/runs/crate_twitch_probe/crate_table30_depth04")
    ap.add_argument("--task-config", type=Path, default=ROOT / "outputs/runs/crate_twitch_probe/task_config.json")
    ap.add_argument("--token", type=Path, default=ROOT / "configs/deployment/crate_initial_token.json")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warmup", action="store_true", help="run one full unrecorded round before the first recorded one")
    ap.add_argument(
        "--prime-first", action="store_true", help="enter streamed mode on the token once, before the first reset"
    )
    ap.add_argument(
        "--early", action="store_true", help="switch to the token right after start_control, no IDLE settle"
    )
    ap.add_argument(
        "--pin", action="store_true", help="slew the sim arm pin to the token posture and hold it through the switch"
    )
    ap.add_argument(
        "--encoder", action="store_true", help="launch the deploy WITH the encoder (eval uses NO_ENCODER=1)"
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    videos = args.out / "videos"
    videos.mkdir(exist_ok=True)
    stage = Stage(args.out)
    task_cfg = CrateTaskConfig.load(args.task_config)
    backend = SonicBackendConfig(no_encoder=not args.encoder, dataset_fps=25.0)

    # start_stack returns the child environment (the instance's port block). The
    # evaluator hands it to vla_rounds as a subprocess env; this probe drives the
    # stack in-process, so it has to adopt those ports itself or Primitives keeps
    # talking to instance 0's :5581.
    env = stack.start_stack(
        backend,
        task_cfg,
        instance=args.instance,
        gpu=str(args.gpu),
        video_dir=str(videos),
        extra_env={"LOCOMANI_SIM_DIR": str(args.scene), **({} if args.encoder else {"NO_ENCODER": "1"})},
        runner=stage.run,
        name="start_stack",
        receipt_path=args.out / "deploy_receipts.jsonl",
    )
    os.environ.update(env)
    try:
        initial_pose.INIT_TOKEN_FILE = str(args.token)
        meta = json.loads((args.scene / "scene_meta.json").read_text())
        p, reset = Primitives(), EpisodeControlClient()
        rng = np.random.default_rng(6000)
        cfg = task_cfg.visual
        records = []
        if args.prime_first:
            # The one PLANNER->STREAMED switch of the batch, with the robot wherever the stack left
            # it and nothing recording. Every round then finds the deploy already streamed.
            p.cmd.start_control()
            time.sleep(1.0)
            ok0 = initial_pose.prepare_token_pose(p, initial_pose.initial_arms(), hold_seconds=2.0)
            print(f"[probe] prime-first: token posture delivered={ok0}", flush=True)

        for index in range(-1 if args.warmup else 0, args.rounds):
            if index >= 0:
                reset.start_video(f"twitch_{index:03d}")
            jitter = rng.uniform(-cfg.position_jitter, cfg.position_jitter, size=2)
            yaw = np.pi / 2 + np.radians(rng.uniform(-cfg.yaw_jitter_deg, cfg.yaw_jitter_deg))
            start_xy = initial_robot_xy(meta, float(rng.uniform(cfg.start_edge_min, cfg.start_edge_max)))
            p.cmd.reset_pose()
            p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
            reset.pin_arms()
            reset.reset(
                objects={"blue_crate": {"dxy": jitter.tolist(), "yaw": float(yaw)}},
                robot={"dxy": (start_xy - meta["robot_start"]["xy"]).tolist(), "yaw": 0.0},
            )
            time.sleep(0.4)
            trace = ArmTrace()
            thread = threading.Thread(target=trace.run, daemon=True)
            thread.start()
            trace.label = "landing"
            p.cmd.start_control()
            if not args.early:
                p.hold_yaw = 0.0
                p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=(1, 0, 0))
                trace.label = "settled"
                time.sleep(1.0)  # a second of the settled planner pose: the baseline
            expected = initial_pose.initial_arms()
            if args.pin:
                trace.label = "pinramp"
                pin_arms_to(p, reset, expected)
            trace.label = "prelude"
            t0 = time.monotonic()
            ok = initial_pose.prepare_token_pose(p, expected)
            if args.pin:
                reset.pin_arms()  # release; the token owns the arms from here
            trace.label = "after"
            time.sleep(1.0)
            trace.stop = True
            thread.join(timeout=2.0)
            if index >= 0:
                reset.end_video()
            st = p.obs()
            arms = np.asarray(st["joints"], float)[15:29]
            print(
                f"[probe]   fallen={st['fallen']} arm_err={np.abs(arms - expected).max():.3f} pelvis_z={st['pelvis']['pos'][2]:.3f}",
                flush=True,
            )

            records.append(dict(round=index, ok=bool(ok), prelude_start=t0, rows=trace.rows))
            print(f"[probe] round {index}: prepare_token_pose ok={ok}, {len(trace.rows)} samples", flush=True)

        (args.out / "arm_trace.json").write_text(json.dumps(records))
        print(f"[probe] wrote {args.out / 'arm_trace.json'}", flush=True)
    finally:
        stack.stop_instance(args.instance, runner=stage.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
