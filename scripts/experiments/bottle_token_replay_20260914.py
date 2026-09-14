"""Replay a short recorded action segment through the student's token-only path.

Uses owned GPU 0 / instance 0 through the normal launcher. A fresh reset is not
the original episode's state: this verifies the live action route, not open-loop
task reproducibility. Source recordings remain untouched.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from locomani.backends.sonic.config import SonicBackendConfig
from locomani.backends.sonic.sonic_token import expand_hands, split_action
from locomani.pipeline import stack
from locomani.pipeline.stages import Stage
from locomani.tasks.bottle.config import BottleTaskConfig


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--rows", type=int, default=150)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.start_row < 0 or not 50 <= args.rows <= 250:
        raise ValueError("use a nonnegative start and a 1–5 second source segment")
    source = json.loads((args.run / "collect/inputs.json").read_text())
    if len(source["sessions"]) != 1:
        raise ValueError("this bounded probe takes a single collection batch")
    session = Path(source["sessions"][0])
    file = session / f"data/chunk-000/episode_{args.episode:06d}.parquet"
    table = pq.read_table(file).slice(args.start_row, args.rows)
    if len(table) != args.rows:
        raise ValueError("the requested source segment is incomplete")
    tokens = np.asarray(table["action.motion_token"].to_pylist())
    grips = np.asarray(table["teleop.right_hand_joints"].to_pylist())[:, :1]
    actions = np.concatenate([tokens, grips], axis=1)
    args.out.mkdir(parents=True, exist_ok=False)
    task = BottleTaskConfig.load(args.run / "task_config.json")
    stages = Stage(args.out)
    result = dict(source=str(file), start_row=args.start_row, rows=args.rows, protocol=4, no_encoder=True, trace=[])
    started = False
    p = reset = debug = None
    try:
        env = stack.start_stack(
            SonicBackendConfig(no_encoder=True),
            task,
            instance=0,
            gpu="0",
            runner=stages.run,
            extra_env=dict(LOCOMANI_SIM_DIR=str((args.run / "scene").resolve()), SIM_VIS="none", SCENE_RANDOM="0"),
            receipt_path=(args.out / "deploy_receipts.jsonl").resolve(),
        )
        started = True
        os.environ.update(env)
        recorded = json.loads((args.run / "deploy_receipts.jsonl").read_text().splitlines()[0])
        replayed = json.loads((args.out / "deploy_receipts.jsonl").read_text().splitlines()[0])
        for field in ("model_decoder.onnx_sha256", "observation_config_sha256"):
            if recorded[field] != replayed[field]:
                raise ValueError(f"recording/replay checkpoint mismatch: {field}")
        result["checkpoint_matches"] = True
        from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

        from locomani.backends.sonic.primitives import Primitives
        from locomani.sim import ports
        from locomani.sim.privileged_pub import EpisodeControlClient

        p, reset, debug = Primitives(), EpisodeControlClient(), ZMQStateSubscriber(port=ports.out())
        initial = np.asarray(json.loads(Path("configs/deployment/ready_grasp_initial_token.json").read_text()))
        left, right = expand_hands(0.0245)
        reset.reset(label="bottle_token_replay")
        time.sleep(0.3)
        p.cmd.start_control()
        for _ in range(30):
            p.cmd.token_frame(initial, left_hand=left, right_hand=right, frame_index=0)
            time.sleep(0.02)
        p.cmd.enter_streamed()
        for _ in range(100):
            p.cmd.token_frame(initial, left_hand=left, right_hand=right, frame_index=0)
            time.sleep(0.02)
        if p.obs()["fallen"]:
            raise RuntimeError("token-only initial posture fell")
        recent = deque(maxlen=12)
        deadline = time.monotonic()
        for i, action in enumerate(actions):
            token, gripper = split_action(action)
            left, right = expand_hands(gripper)
            p.cmd.token_frame(token, left_hand=left, right_hand=right, frame_index=i)
            recent.append((token, left, right))
            deadline += 0.02
            time.sleep(max(0, deadline - time.monotonic()))
            msg = debug.get_msg()
            state = p.state.latest(0)
            echo = None if msg is None else msg.get("token_state")
            result["trace"].append(
                dict(
                    row=i,
                    wall=time.time(),
                    commanded=action.tolist(),
                    echo=None if echo is None else np.asarray(echo).tolist(),
                    echo_matches_recent=None
                    if echo is None
                    else any(np.array_equal(np.asarray(echo).ravel(), t) for t, _, _ in recent),
                    consumed_left=None if msg is None else np.asarray(msg.get("last_left_hand_action", [])).tolist(),
                    consumed_right=None if msg is None else np.asarray(msg.get("last_right_hand_action", [])).tolist(),
                    hands_match_recent=None
                    if msg is None
                    else any(
                        np.array_equal(msg.get("last_left_hand_action"), l)
                        and np.array_equal(msg.get("last_right_hand_action"), r)
                        for _, l, r in recent
                    ),
                    state_seq=None if state is None else state["seq"],
                    joints=None if state is None else state["joints"],
                    fallen=None if state is None else state["fallen"],
                )
            )
            if state is not None and state["fallen"]:
                raise RuntimeError("fresh-state open-loop replay fell")
        observed = [r for r in result["trace"][12:] if r["echo_matches_recent"] is not None]
        result["echo_samples"] = len(observed)
        result["echo_match_fraction"] = (
            float(np.mean([r["echo_matches_recent"] for r in observed])) if observed else None
        )
        result["hands_match_fraction"] = (
            float(np.mean([r["hands_match_recent"] for r in observed])) if observed else None
        )
        if len(observed) < args.rows - 12 or min(result["echo_match_fraction"], result["hands_match_fraction"]) < 0.99:
            raise RuntimeError("token/hand consumption did not match the recorded action stream")
        result["complete"] = True
    except Exception as exc:
        result["error"] = repr(exc)
        raise
    finally:
        for obj in (debug, reset):
            if obj is not None:
                obj.close()
        if p is not None:
            p.cmd.close()
            p.state.close()
        if started:
            stack.stop_instance(0, runner=stages.run)
        (args.out / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
