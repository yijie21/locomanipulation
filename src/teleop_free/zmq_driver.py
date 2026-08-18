"""Scripted replacement for the Pico teleop server.

Occupies exactly the seam `pico_manager_thread_server.py` occupies: a ZMQ PUB
socket on :5556 speaking the `command` / `planner` / `pose` topics, plus a
second PUB on :5580 carrying the data-exporter's record-control keys.

The C++ deploy stack subscribes to :5556 and does not care whether the
publisher is a VR headset or this script.

P0.3 usage (walk forward for 10 s and record one episode):

    source /workspace/code/locomanipulation/scripts/env_deploy.sh
    .venv_sim/bin/python /workspace/code/locomanipulation/src/teleop_free/zmq_driver.py \
        --duration 10 --record
"""

from __future__ import annotations

import argparse
import time
from enum import IntEnum

import zmq

import numpy as np

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

SONIC_PORT = 5556
KEYBOARD_PORT = 5580


class StreamMode(IntEnum):
    """Mirrors pico_manager_thread_server.StreamMode.

    run_data_exporter.py records `teleop.planner_*` only while stream_mode == 5
    (see run_data_exporter.py:693) and `teleop.smpl_*` only for 1 or 4. The mode
    is read from the `manager_state` topic, which nothing but the teleop server
    publishes — so a scripted commander has to publish it too, or the recorded
    episode silently loses every command field.
    """

    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


class LocomotionMode(IntEnum):
    """Mirrors pico_manager_thread_server.LocomotionMode."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3


class SonicCommander:
    """Publishes SONIC control messages the way the teleop server does."""

    def __init__(self, sonic_port: int = SONIC_PORT, keyboard_port: int = KEYBOARD_PORT):
        self._ctx = zmq.Context()

        self.sonic = self._ctx.socket(zmq.PUB)
        self.sonic.setsockopt(zmq.LINGER, 2000)  # else close() drops the last message
        self.sonic.bind(f"tcp://*:{sonic_port}")

        self.keyboard = self._ctx.socket(zmq.PUB)
        self.keyboard.setsockopt(zmq.LINGER, 2000)
        self.keyboard.bind(f"tcp://*:{keyboard_port}")

        # PUB sockets drop everything sent before a subscriber has connected.
        time.sleep(0.5)
        print(f"[commander] bound :{sonic_port} (sonic) and :{keyboard_port} (record keys)")

    # -- low level ---------------------------------------------------------

    def command(self, *, start: bool, stop: bool, planner: bool) -> None:
        self.sonic.send(build_command_message(start=start, stop=stop, planner=planner))

    def planner(
        self,
        mode: LocomotionMode,
        movement=(0.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed: float = -1.0,
        height: float = -1.0,
    ) -> None:
        self.sonic.send(
            build_planner_message(int(mode), list(movement), list(facing), speed, height)
        )

    def manager_state(
        self,
        stream_mode: StreamMode = StreamMode.PLANNER_VR_3PT,
        toggle_dc: bool = False,
        toggle_da: bool = False,
    ) -> None:
        """Publish the `manager_state` topic the data exporter gates on."""
        self.sonic.send(
            pack_pose_message(
                {
                    "stream_mode": np.array([int(stream_mode)], dtype=np.int32),
                    "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                    "toggle_data_abort": np.array([toggle_da], dtype=bool),
                },
                topic="manager_state",
            )
        )

    def key(self, k: str) -> None:
        """Send a data-exporter control key.

        run_data_exporter.py only implements two (see _check_recording_commands):
          'c'  toggle, cycling IDLE -> RECORDING -> NEED_TO_SAVE -> IDLE
          'x'  discard the episode being recorded
        There is no 's' / 'f' key in this version.
        """
        self.keyboard.send_string(k)

    # -- high level --------------------------------------------------------

    def idle(self) -> None:
        self.command(start=False, stop=False, planner=False)
        self.planner(LocomotionMode.IDLE)

    def start_control(self) -> None:
        self.command(start=True, stop=False, planner=True)

    def pause_control(self) -> None:
        """Stop driving without shutting the deploy process down.

        `stop=True` is a full shutdown: the C++ side prints "Stopping G1Deploy"
        and exits the process, so it must not be used between episodes.
        """
        self.command(start=False, stop=False, planner=True)

    def shutdown_deploy(self) -> None:
        """Terminate the C++ deploy process. Only for tearing the stack down."""
        self.command(start=False, stop=True, planner=True)

    def hold(
        self,
        mode: LocomotionMode,
        seconds: float,
        hz: float = 50.0,
        stream_mode: StreamMode = StreamMode.PLANNER_VR_3PT,
        **kwargs,
    ) -> None:
        """Stream one planner mode at a fixed rate for `seconds`."""
        dt = 1.0 / hz
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.manager_state(stream_mode)
            self.planner(mode, **kwargs)
            time.sleep(dt)

    def close(self) -> None:
        self.sonic.close()
        self.keyboard.close()
        self._ctx.term()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to walk forward")
    ap.add_argument("--settle", type=float, default=3.0, help="seconds to stand still first")
    ap.add_argument("--record", action="store_true", help="drive the data exporter too")
    ap.add_argument("--mode", type=str, default="WALK", choices=[m.name for m in LocomotionMode])
    args = ap.parse_args()

    c = SonicCommander()
    try:
        print("[commander] starting control in planner mode")
        c.start_control()
        c.hold(LocomotionMode.IDLE, args.settle)

        if args.record:
            print("[commander] record start")
            c.key("c")
            c.hold(LocomotionMode.IDLE, 1.0)

        mode = LocomotionMode[args.mode]
        print(f"[commander] {mode.name} forward for {args.duration}s")
        c.hold(mode, args.duration, movement=(1.0, 0.0, 0.0), facing=(1.0, 0.0, 0.0))

        print("[commander] settling")
        c.hold(LocomotionMode.IDLE, 2.0)

        if args.record:
            # One toggle only: RECORDING -> NEED_TO_SAVE, and the exporter's main
            # loop writes the episode and returns to IDLE by itself. A second 'c'
            # here would start recording a new episode instead of finishing this one.
            print("[commander] record stop -> save")
            c.key("c")
            c.hold(LocomotionMode.IDLE, 5.0)   # keep streaming while the episode is written

        print("[commander] pausing control (deploy stays up)")
        c.pause_control()
        time.sleep(1.0)
    finally:
        c.close()


if __name__ == "__main__":
    main()
