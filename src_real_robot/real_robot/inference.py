"""Run a GR00T checkpoint on the real G1 through the SONIC C++ deploy.

    bash src_real_robot/run_inference.sh --host <policy server> --init-token-file configs/deployment/warehouse_initial_token.json
    bash src_real_robot/run_inference.sh --help

NVIDIA's `gear_sonic/scripts/run_vla_inference.py`, restructured (UPSTREAM.md): robot state over ZMQ from the C++
deploy (`g1_debug`), images from the camera server, the policy over ZMQ REQ/REP, latent actions back to the deploy as
protocol-v4 `pose` messages at the publish rate, keyboard commands from the ZMQ keyboard publisher. The observation
and action formats are this repository's (`real_robot.observation`, `real_robot.chunk`), the gripper is Dex1
(`real_robot.gripper`).

Keyboard (the standalone keyboard publisher):
  p  pause / resume the policy loop        k  start / stop the C++ control loop (planner mode)
  i  blend to the initial pose, POSE mode  t  change the prompt (`prompt:<text>`)
  [  toggle the left hand of the initial pose open / closed      ]  the right hand (a Dex1 direction check: 5.5 / 0)
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from locomani.schema import SONIC_BOTTLE, SONIC_CRATE
from real_robot.chunk import ActionChunk, latency_index, should_infer
from real_robot.gripper import DEX1_CLOSED, DEX1_OPEN, HAND_WIRES, MODEL_UNITS, hand_field
from real_robot.observation import build_observation
from real_robot.policy_client import PolicyClient, identity_mismatch

SCHEMAS = {"crate": SONIC_CRATE, "bottle": SONIC_BOTTLE}


@dataclass
class Settings:
    host: str = "localhost"  # the policy server (scripts/run_vla_server.sh <run>)
    port: int = 5590
    action_publish_rate: int = 50  # Hz, one token per tick to the C++ deploy
    action_horizon: int = 40  # actions per chunk the checkpoint predicts
    rate: float = 2.5  # Hz, how often a new chunk is requested
    camera_host: str = "localhost"
    camera_port: int = 5555
    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557
    action_zmq_host: str = "*"  # the PUB socket is bound here; the deploy connects
    action_zmq_port: int = 5556
    keyboard_zmq_host: str = "localhost"
    keyboard_zmq_port: int = 5580
    prompt: str = "pick up the box"
    schema: str = "crate"  # crate: ego + both wrists, 60-d state, token + two grippers; bottle: ego + right wrist
    gripper_units: str = "dex1"  # what the checkpoint predicts and observes: dex1 (0 closed .. 5.5 open) | sim_m
    hand_wire: str = "dex1"  # what the deploy reads from the hand fields: dex1 (broadcast) | sim (metres, jaw slots)
    init_token_file: str | None = None  # a run's eval.initial_token; None: NVIDIA's standing token
    initial_pose_blend_duration: float = 1.0
    no_snap: bool = False  # send the raw token instead of its codebook projection
    no_identity_check: bool = False  # do not refuse a server whose checkpoint field order differs from the schema
    verbose_timing: bool = False


def parse_args(argv: list[str] | None = None) -> Settings:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field, default in asdict(Settings()).items():
        flag = "--" + field.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, action="store_true", default=default)
        else:
            ap.add_argument(flag, type=type(default) if default is not None else str, default=default)
    args = ap.parse_args(argv)
    settings = Settings(**vars(args))
    if settings.schema not in SCHEMAS:
        ap.error(f"--schema must be one of {sorted(SCHEMAS)}")
    if settings.gripper_units not in MODEL_UNITS:
        ap.error(f"--gripper-units must be one of {MODEL_UNITS}")
    if settings.hand_wire not in HAND_WIRES:
        ap.error(f"--hand-wire must be one of {HAND_WIRES}")
    if settings.action_publish_rate <= 0 or settings.rate <= 0 or settings.action_horizon <= 0:
        ap.error("rates and the horizon must be positive")
    return settings


def green(text: str) -> None:
    print(f"\033[92m{text}\033[0m", flush=True)


def initial_token(path: str | None) -> np.ndarray:
    """The token the deploy holds before the first chunk: a run's own standing token file, else NVIDIA's."""
    if path:
        from locomani.backends.sonic.initial_pose import load_token_file

        return load_token_file(path)
    from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN

    return np.asarray(LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32)


class Runner:
    """The publish loop and its inference worker, over sockets handed in by `main` (tests hand in fakes)."""

    def __init__(self, s: Settings, *, robot_model, policy, state, camera, keyboard, send, init_token, gravity):
        self.s, self.rm, self.policy, self.state, self.camera, self.keyboard = (
            s,
            robot_model,
            policy,
            state,
            camera,
            keyboard,
        )
        self.send, self.init_token, self.gravity = send, init_token, gravity
        self.schema = SCHEMAS[s.schema]
        self.prompt = s.prompt
        self.paused = True
        self.cpp_running, self.cpp_mode = False, "OFF"
        self.init_hands_closed = {"left": False, "right": False}
        self.chunk: ActionChunk | None = None
        self.index = 0
        self.last_inference = 0.0
        self.frames = 0
        self.last_token: np.ndarray | None = None
        self.last_images: dict = {}
        self.requests: queue.Queue = queue.Queue(maxsize=1)
        self.results: queue.Queue = queue.Queue(maxsize=1)
        self.stop, self.busy = threading.Event(), threading.Event()

    # -- observation and inference (worker thread) ---------------------------------------------------------------

    def observe(self) -> dict | None:
        message = self.camera.read()
        if message is not None:
            self.last_images = message["images"]
        state = self.state.get_msg()
        if state is None or not self.last_images:
            return None
        return build_observation(
            state,
            self.last_images,
            self.prompt,
            self.rm,
            self.schema,
            gripper_units=self.s.gripper_units,
            gravity=self.gravity,
        )

    def infer(self, obs: dict) -> ActionChunk | None:
        try:
            return ActionChunk.from_policy(
                self.policy.get_action_raw(obs), gripper_units=self.s.gripper_units, snap_tokens=not self.s.no_snap
            )
        except ValueError as exc:  # NVIDIA's client skips a chunk outside the token bound the same way
            print(f"[chunk] skipped: {exc}", flush=True)
            return None

    def worker(self) -> None:
        while not self.stop.is_set():
            try:
                self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            self.busy.set()
            try:
                obs = self.observe()
                if obs is None:
                    print("[obs] waiting for the cameras and the robot state", flush=True)
                    continue
                t0 = time.monotonic()
                chunk = self.infer(obs)
                if chunk is not None:
                    while True:
                        try:
                            self.results.put_nowait((chunk, t0))
                            break
                        except queue.Full:
                            self.results.get_nowait()
            except Exception as exc:  # the loop keeps publishing the last chunk; the next request retries
                print(f"[worker] {exc!r}", flush=True)
            finally:
                self.busy.clear()

    # -- what goes to the deploy -----------------------------------------------------------------------------------

    def publish(self, token, left_dex1: float, right_dex1: float, frame_index: int) -> None:
        self.send(
            np.asarray(token, dtype=np.float32),
            frame_index,
            hand_field(left_dex1, self.s.hand_wire),
            hand_field(right_dex1, self.s.hand_wire),
        )
        self.last_token = np.asarray(token, dtype=np.float32).copy()

    def init_hands(self) -> tuple[float, float]:
        return tuple(DEX1_CLOSED if self.init_hands_closed[side] else DEX1_OPEN for side in ("left", "right"))

    def go_initial_pose(self) -> None:
        """Blend from the last sent token to the initial token over the configured duration (snap without one)."""
        left, right = self.init_hands()
        steps = max(1, round(self.s.action_publish_rate * self.s.initial_pose_blend_duration))
        if self.last_token is None or self.s.initial_pose_blend_duration <= 0:
            self.publish(self.init_token, 0, left, right)
            time.sleep(1.0)
            green("initial pose sent")
            return
        start, period = self.last_token.copy(), 1.0 / self.s.action_publish_rate
        for k in range(steps):
            t = time.monotonic()
            a = (k + 1) / steps
            self.publish((1 - a) * start + a * self.init_token, 0, left, right)
            rest = period - (time.monotonic() - t)
            if rest > 0:
                time.sleep(rest)
        green(f"blended to the initial pose over {steps} steps")

    def command(self, start: bool, planner: bool) -> None:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

        self.send_raw(build_command_message(start=start, stop=not start, planner=planner))
        self.cpp_running, self.cpp_mode = start, ("PLANNER" if planner else "POSE") if start else "OFF"
        green(f"C++ control loop: {'start' if start else 'stop'} ({self.cpp_mode})")

    def send_raw(self, message: bytes) -> None:  # replaced by main with the PUB socket's send
        raise NotImplementedError

    # -- keyboard ----------------------------------------------------------------------------------------------------

    def handle_key(self, key: str) -> None:
        if key.startswith("prompt:"):
            if key[7:]:
                green(f'prompt: "{self.prompt}" -> "{key[7:]}"')
                self.prompt = key[7:]
        elif key == "p":
            self.paused = not self.paused
            green("paused" if self.paused else "resumed")
        elif key == "k":
            self.command(start=not self.cpp_running, planner=True)
        elif key == "i":
            if self.cpp_running and self.cpp_mode == "PLANNER":
                self.command(start=True, planner=False)
            self.paused = True
            self.go_initial_pose()
            self.chunk, self.index, self.frames = None, 0, 0
        elif key in ("[", "]"):
            side = "left" if key == "[" else "right"
            self.init_hands_closed[side] = not self.init_hands_closed[side]
            green(f"initial pose {side} hand: {'closed' if self.init_hands_closed[side] else 'open'}")

    # -- the loop ----------------------------------------------------------------------------------------------------

    def tick(self, now: float) -> None:
        """One publish period: take a finished chunk, ask for the next one when due, publish one action."""
        try:
            chunk, t0 = self.results.get_nowait()
            delay = now - t0
            self.chunk, self.index, self.last_inference = (
                chunk,
                latency_index(delay, self.s.action_publish_rate, len(chunk)),
                now,
            )
            green(f'chunk ({len(chunk)} steps, latency {delay:.3f} s, prompt "{self.prompt}")')
        except queue.Empty:
            pass
        if should_infer(self.chunk is not None, self.busy.is_set(), now - self.last_inference, 1.0 / self.s.rate):
            try:
                self.requests.put_nowait(None)
            except queue.Full:
                pass
        if self.paused or self.chunk is None:
            return
        token, left, right = self.chunk.frame(self.index)
        self.publish(token, left, right, self.frames)
        self.frames += 1
        if self.frames % self.s.action_publish_rate == 0:
            print(
                f"[pose] frame {self.frames}, chunk index {self.index}, grippers {left:.2f} / {right:.2f}", flush=True
            )
        self.index = min(self.index + 1, self.s.action_horizon - 1)

    def run(self) -> None:
        thread = threading.Thread(target=self.worker, daemon=True)
        thread.start()
        period = 1.0 / self.s.action_publish_rate
        green(f"policy loop at {self.s.action_publish_rate} Hz, inference every {1.0 / self.s.rate:.2f} s; paused (p)")
        try:
            while True:
                t = time.monotonic()
                key = self.keyboard.read_msg()
                if key:
                    self.handle_key(key)
                self.tick(t)
                rest = period - (time.monotonic() - t)
                if rest > 0:
                    time.sleep(rest)
                elif self.s.verbose_timing:
                    print(f"[loop] late by {-rest * 1000:.1f} ms", flush=True)
        except KeyboardInterrupt:
            print("stopped by the user")
        finally:
            self.stop.set()
            thread.join(timeout=1.0)


def main(argv: list[str] | None = None) -> int:
    s = parse_args(argv)
    import zmq
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
    from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    schema = SCHEMAS[s.schema]
    policy = PolicyClient(s.host, s.port)
    if not policy.ping():
        raise SystemExit(f"no policy server at {s.host}:{s.port}; start scripts/run_vla_server.sh <run> first")
    mismatch = identity_mismatch(policy.identity(), schema)
    if mismatch and not s.no_identity_check:
        raise SystemExit(f"the served checkpoint's field order is not the {s.schema} schema's: {mismatch}")
    # the exporter's model (features_sonic_vla.get_g1_robot_model): the same FK the dataset's eef_state came from
    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{s.action_zmq_host}:{s.action_zmq_port}")
    time.sleep(0.1)

    def send(token, frame_index, left, right):
        data = {
            "token_state": token.reshape(1, -1),
            "frame_index": np.array([frame_index], dtype=np.int64),
            "left_hand_joints": left.reshape(1, -1),
            "right_hand_joints": right.reshape(1, -1),
        }
        pub.send(pack_pose_message(data, topic="pose", version=4))

    runner = Runner(
        s,
        robot_model=robot_model,
        policy=policy,
        state=ZMQStateSubscriber(host=s.state_zmq_host, port=s.state_zmq_port),
        camera=ComposedCameraClientSensor(server_ip=s.camera_host, port=s.camera_port),
        keyboard=ZMQKeyboardSubscriber(port=s.keyboard_zmq_port, host=s.keyboard_zmq_host),
        send=send,
        init_token=initial_token(s.init_token_file),
        gravity=compute_projected_gravity,
    )
    runner.send_raw = pub.send
    green(f"schema {s.schema}: cameras {schema.video_keys}, gripper units {s.gripper_units}, hand wire {s.hand_wire}")
    try:
        runner.run()
    finally:
        pub.close(linger=0)
        ctx.term()
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
