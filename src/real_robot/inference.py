"""
VLA inference runner for the real G1 (Dex1 grippers, three RealSense cameras) -- NO ROS 2 DEPENDENCY.

    bash scripts/run_real_robot_inference.sh --host <policy server> --init-token-file configs/deployment/warehouse_initial_token.json
    bash scripts/run_real_robot_inference.sh --help

NVIDIA's `gear_sonic/scripts/run_vla_inference.py` (GR00T-WholeBodyControl, UPSTREAM.md), kept function for
function. Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack. All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses a PolicyClient (ZMQ REQ/REP, the Isaac-GR00T wire format; `real_robot.policy_client`) to talk to a running
PolicyServer (`scripts/run_gr00t_server.sh`).

What differs from upstream is this repository's data format: the observation is the dataset's (three cameras under
their own keys, the 60-d state with wrist poses, `real_robot.observation`), the action is a SONIC token snapped to
the FSQ codebook plus one Dex1 gripper scalar per hand (`real_robot.chunk`), broadcast to the 7-wide hand field
(`real_robot.gripper`). The served checkpoint's field order is checked against the schema before anything moves.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> blend smoothly to initial pose (or snap if no prior token) and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose (a Dex1 direction check: 5.5 open, 0 closed)
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
import zmq
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)

from locomani.schema import SONIC_BOTTLE, SONIC_CRATE
from real_robot.chunk import TOKEN_BOUND, ActionChunk
from real_robot.gripper import DEX1_CLOSED, DEX1_OPEN, hand_field
from real_robot.observation import build_observation, to_policy_observation
from real_robot.policy_client import PolicyClient, identity_mismatch

SCHEMAS = {"crate": SONIC_CRATE, "bottle": SONIC_BOTTLE}


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (scripts/run_gr00t_server.sh)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5590
    """The port of the Isaac-GR00T PolicyServer (upstream 5550; 5555-5580 are the robot's)."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz). Must be the training
    dataset's `dataset.fps`, or the actions replay at the wrong speed."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "*"
    """ZMQ host the action PUB socket binds (upstream localhost; * lets a deploy on another machine connect)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    schema: Literal["crate", "bottle"] = "crate"
    """Observation and action format: crate (ego + both wrists, 60-d state, token + two grippers) or bottle
    (ego + right wrist, one gripper)."""

    gripper_units: Literal["dex1", "sim_m"] = "dex1"
    """Gripper units the checkpoint predicts and observes: dex1 (0 closed .. 5.5 open) or sim_m (simulator metres,
    a checkpoint trained before the converter wrote Dex1 units)."""

    hand_wire: Literal["dex1", "sim"] = "dex1"
    """How the deploy reads the hand fields: dex1 (the value broadcast to all 7 slots, the robot) or sim (metres
    on the two jaw slots, this repository's simulation bridge)."""

    snap_tokens: bool = True
    """Project each motion token onto the FSQ codebook before it is sent."""

    identity_check: bool = True
    """Refuse a server whose checkpoint reports a field order different from the schema."""

    # Prompt / eval
    prompt: str = "pick up the box"
    """The language prompt for the VLA policy."""

    # Initial pose
    init_token_file: str | None = None
    """A run's `eval.initial_token` file. None: NVIDIA's standing token, a forward-leaning pose outside our data."""

    initial_pose_blend_duration: float = 1.0
    """Duration (seconds) for smooth interpolation to initial pose. The robot
    blends from its current motion token to the initial pose token over this
    period. Set to 0 to snap instantly (no blend)."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m", flush=True)


def load_initial_token(path: str | None) -> np.ndarray:
    """The token the deploy holds before the first chunk: a run's own standing token file, else NVIDIA's."""
    if path:
        from locomani.backends.sonic.initial_pose import load_token_file

        return np.asarray(load_token_file(path), dtype=np.float32)
    return np.asarray(LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32)


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(f"left_hand_joints must have shape [7], got {left_hand_joints.shape}")
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(f"right_hand_joints must have shape [7], got {right_hand_joints.shape}")
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    schema=SONIC_CRATE,
    gripper_units: str = "dex1",
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Upstream splits the 43-joint q into joint groups and adds projected gravity; this builds the dataset's
    observation instead (`real_robot.observation.build_observation`: the same groups plus wrist poses and the Dex1
    hand state, checked against `schema`) and nests it the way upstream's is nested.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    flat = build_observation(
        state_msg,
        camera_msg["images"],
        language_prompt,
        robot_model,
        schema,
        gripper_units=gripper_units,
        gravity=compute_projected_gravity,
    )
    if flat is None:
        if log_errors:
            missing = sorted(set(schema.video_keys) - set(camera_msg["images"]))
            print(f"[DEBUG] prepare_observation: waiting for cameras {missing}..", flush=True)
        return None
    return to_policy_observation(flat)


def run_policy_inference_and_process(policy, observation, robot_model, gripper_units="dex1", snap_tokens=True):
    """Run policy inference via the PolicyClient and process results.

    Returns:
        ActionChunk (token + Dex1 grippers) or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > TOKEN_BOUND:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > {TOKEN_BOUND}. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return ActionChunk.from_policy(processed_action, gripper_units=gripper_units, snap_tokens=snap_tokens)
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def initial_pose_hand_joints(closed: bool, hand_wire: str) -> np.ndarray:
    """The 7-wide hand field of the initial pose. Upstream: Dex3 zeros (open) or its IK closed pose; the Dex1 is
    one scalar, 5.5 open and 0 closed, broadcast (`real_robot.gripper.hand_field`)."""
    return hand_field(DEX1_CLOSED if closed else DEX1_OPEN, hand_wire)


def main(config: InferenceConfig):
    if config.action_publish_rate <= 0 or config.rate <= 0 or config.action_horizon <= 0:
        raise SystemExit("--action-publish-rate, --rate and --action-horizon must be positive")
    schema = SCHEMAS[config.schema]
    initial_motion_token = load_initial_token(config.init_token_file)
    pause_loop = True

    # the exporter's model (features_sonic_vla.get_g1_robot_model): the same FK the dataset's eef_state came from
    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        # Upstream warns and carries on; the field-order check below needs the server, and nothing should move
        # on the robot before it has passed.
        raise SystemExit(
            f"PolicyServer not reachable at {config.host}:{config.port}; "
            "start scripts/run_gr00t_server.sh --model-path <checkpoint> first"
        )
    mismatch = identity_mismatch(n1_policy.identity(), schema)
    if mismatch and config.identity_check:
        raise SystemExit(f"the served checkpoint's field order is not the {config.schema} schema's: {mismatch}")
    print_green(f"Schema {config.schema}: cameras {list(schema.video_keys)}, gripper units {config.gripper_units}")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(port=config.keyboard_zmq_port, host=config.keyboard_zmq_host)

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        print("Moving to initial pose")
        left_hand = initial_pose_hand_joints(initial_pose_left_hand_closed, config.hand_wire)
        right_hand = initial_pose_hand_joints(initial_pose_right_hand_closed, config.hand_wire)
        zmq_message = pack_latent_action_message(
            motion_token=initial_motion_token,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def blend_to_initial_pose(duration_s: float) -> bool:
        """Smoothly interpolate from the last sent motion token to the initial pose.

        Linearly blends over ``duration_s`` seconds at the action publish rate,
        sending intermediate tokens each loop iteration. Returns True if blend
        was performed, False if skipped (no previous token available).
        """
        nonlocal last_sent_motion_token
        if last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            publish_initial_pose()
            return False

        start_token = last_sent_motion_token.copy()
        target_token = initial_motion_token.copy()
        num_steps = max(1, round(config.action_publish_rate * duration_s))
        step_period = 1.0 / config.action_publish_rate

        left_hand = initial_pose_hand_joints(initial_pose_left_hand_closed, config.hand_wire)
        right_hand = initial_pose_hand_joints(initial_pose_right_hand_closed, config.hand_wire)

        print(f"Blending to initial pose over {duration_s:.2f}s ({num_steps} steps at {config.action_publish_rate} Hz)")

        for step in range(num_steps):
            t_step_start = time.monotonic()
            alpha = (step + 1) / num_steps
            blended_token = ((1.0 - alpha) * start_token + alpha * target_token).astype(np.float32)
            zmq_message = pack_latent_action_message(
                motion_token=blended_token,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
            zmq_socket.send(zmq_message)
            last_sent_motion_token = blended_token.copy()

            elapsed = time.monotonic() - t_step_start
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print_green("Initial pose blend complete.")
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk: ActionChunk | None = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0
    last_sent_motion_token: np.ndarray | None = None

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter, last_sent_motion_token

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX) :]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")

            pause_loop = True
            if config.initial_pose_blend_duration > 0 and last_sent_motion_token is not None:
                blend_to_initial_pose(config.initial_pose_blend_duration)
            else:
                publish_initial_pose()

            zmq_frame_counter = 0
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk, reset frame counter")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}")
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(f"Initial pose right hand: {'closed' if initial_pose_right_hand_closed else 'open'}")

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                schema=schema,
                gripper_units=config.gripper_units,
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
                gripper_units=config.gripper_units,
                snap_tokens=config.snap_tokens,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(f'New action chunk (prompt: "{language_prompt_ref[0]}", latency: {inference_delay:.3f}s)')
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                # One action per tick; ActionChunk.frame holds the last action past the horizon.
                motion_token, left_dex1, right_dex1 = cached_action_chunk.frame(action_chunk_index)
                left_hand_joints = hand_field(left_dex1, config.hand_wire)
                right_hand_joints = hand_field(right_dex1, config.hand_wire)

                frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                zmq_frame_counter += 1

                zmq_message = pack_latent_action_message(
                    motion_token,
                    frame_index,
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                )
                zmq_socket.send(zmq_message)
                last_sent_motion_token = np.asarray(motion_token, dtype=np.float32).copy()
                if zmq_frame_counter % 50 == 0:
                    print_green(
                        f"ZMQ: Sent latent action - frame: {frame_index[0]}, "
                        f"chunk index: {action_chunk_index}, grippers: {left_dex1:.2f} / {right_dex1:.2f}"
                    )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(context="VLA Inference Loop Missed", threshold=0.001)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        n1_policy.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
