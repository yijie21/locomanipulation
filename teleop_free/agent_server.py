"""Scripted stand-in for the Pico teleop server (P1.1).

Occupies the same ZMQ seam `pico_manager_thread_server.py` occupies — PUB on
:5556 speaking `command` / `planner` / `manager_state`, plus :5580 for the data
exporter's record key — but drives it from privileged simulator state instead of
a human in a headset.

Why there is no calibration here
--------------------------------
`pico_manager_thread_server._apply_calibration` exists to map a *human's* VR
wrist positions into the robot's frame: CALIB_FULL captures the operator's
neutral pose, CALIB captures the offset between the operator's wrists and the
robot's own wrists computed by FK. Its output — see the neck reconstruction at
pico_manager_thread_server.py:1202-1205 — is a pose expressed in the robot's
root frame.

A scripted agent can produce root-frame wrist targets directly from the object
pose, so it has nothing to calibrate. With no calibration stored,
`_apply_calibration` returns its input unchanged, i.e. the wire format is
already what we want to emit.

Wire format (matches pico_manager_thread_server.py:1846-1848)
-------------------------------------------------------------
    vr_3pt pose      (3, 7)  rows are [L-wrist, R-wrist, neck]
                             columns are [x y z qw qx qy qz]
    vr_3pt_position  9   = 3 x xyz
    vr_3pt_orientation 12 = 3 x quaternion, scalar-first (wxyz)
    left/right_hand_position  7 joints each
"""

from __future__ import annotations

import os
import time
from enum import IntEnum

import numpy as np
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

SONIC_PORT = int(os.environ.get("TF_SONIC_PORT", "5556"))
KEYBOARD_PORT = int(os.environ.get("TF_KEYBOARD_PORT", "5580"))

# From pico_manager_thread_server.py:957-958 — used to place the neck point in
# the robot's root frame.
TORSO_LINK_OFFSET_Z = 0.05
NECK_LINK_LENGTH = 0.35

# pico_manager prints this as its startup value.
DEFAULT_COMPLIANCE = (0.5, 0.5, 0.0)

N_HAND_JOINTS = 7
HAND_OPEN = np.zeros(N_HAND_JOINTS, dtype=np.float32)
HAND_CLOSED = np.full(N_HAND_JOINTS, 1.0, dtype=np.float32)


class LocomotionMode(IntEnum):
    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3


class StreamMode(IntEnum):
    """run_data_exporter.py:693 only records `teleop.planner_*` at mode 5."""

    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


def nominal_neck() -> tuple[np.ndarray, np.ndarray]:
    """Neck point of an upright robot, in root frame."""
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pos = np.array(
        [0.0, 0.0, TORSO_LINK_OFFSET_Z + NECK_LINK_LENGTH], dtype=np.float32
    )

    return pos, quat


def rest_wrists() -> tuple[np.ndarray, np.ndarray]:
    """Arms down and slightly forward — the pose to hold while walking."""
    left = np.array([0.15, 0.22, -0.10], dtype=np.float32)
    right = np.array([0.15, -0.22, -0.10], dtype=np.float32)

    return left, right


class AgentCommander:
    """Publishes everything the C++ deploy stack and the exporter need."""

    def __init__(self, sonic_port: int = SONIC_PORT, keyboard_port: int = KEYBOARD_PORT):
        self._ctx = zmq.Context()

        self.sonic = self._ctx.socket(zmq.PUB)
        self.sonic.setsockopt(zmq.LINGER, 2000)  # else close() eats the last message
        self.sonic.bind(f"tcp://*:{sonic_port}")

        self.keyboard = self._ctx.socket(zmq.PUB)
        self.keyboard.setsockopt(zmq.LINGER, 2000)
        self.keyboard.bind(f"tcp://*:{keyboard_port}")

        # PUB drops anything sent before a subscriber has connected.
        time.sleep(0.5)
        print(f"[agent] bound :{sonic_port} (sonic) :{keyboard_port} (record)")

        # Latest upper-body target; `frame()` streams whatever is set here.
        self.reset_pose()

        # Yaw the deploy's planner heading frame is rotated by, measured by
        # diag_heading.py: every planner (re-)initialisation — boot, or any
        # STREAMED_MOTION -> PLANNER switch — anchors the deploy at the
        # robot's CURRENT yaw psi0, after which a commanded facing phi is
        # executed as phi + psi0 (and movement vectors rotate the same way).
        # At boot psi0 ~ 0, which is why this stayed invisible until streamed
        # mode entered the pipeline. Whoever triggers a re-anchor must store
        # the robot's yaw at that moment here; frame() undoes the rotation.
        self.heading_offset = 0.0

    def reset_pose(self) -> None:
        """Put the upper body back to rest — arms down, hands open.

        A batch reuses one commander (rebinding :5556 races with TIME_WAIT), so
        without this every episode inherits the previous one's last wrist
        target. After a grasp attempt that is the right hand down at table
        height, and the episode opens with the robot hunched over reaching at
        nothing — visible from the first frame of every `nav_*.mp4` except 000.
        """
        self.left_wrist_pos, self.right_wrist_pos = rest_wrists()
        self.left_wrist_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.right_wrist_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.left_hand = HAND_OPEN.copy()
        self.right_hand = HAND_OPEN.copy()
        self.upper_body = None
        # Which planner sub-mode `frame()` runs in. PLANNER_VR_3PT tracks the
        # wrist position targets above but ONLY while standing — the walking
        # policy ignores them and swings the arms. PLANNER_FROZEN_UPPER_BODY
        # instead pins the 17 upper-body joints to `upper_body` (waist 3 +
        # left/right-interleaved arms 14, the pico feedback-reader order)
        # through the walk as well.
        self.stream_mode = StreamMode.PLANNER_VR_3PT
        self.upper_body: np.ndarray | None = None

    # -- control plane -----------------------------------------------------

    def command(self, *, start: bool, stop: bool, planner: bool) -> None:
        self.sonic.send(build_command_message(start=start, stop=stop, planner=planner))

    def start_control(self) -> None:
        self.command(start=True, stop=False, planner=True)

    def pause_control(self) -> None:
        """`stop=True` terminates the deploy process; use this between episodes."""
        self.command(start=False, stop=False, planner=True)

    def shutdown_deploy(self) -> None:
        self.command(start=False, stop=True, planner=True)

    def key(self, k: str) -> None:
        """'c' toggles the exporter IDLE -> RECORDING -> NEED_TO_SAVE -> IDLE; 'x' discards."""
        self.keyboard.send_string(k)

    def manager_state(
        self,
        stream_mode: StreamMode = StreamMode.PLANNER_VR_3PT,
        toggle_dc: bool = False,
        toggle_da: bool = False,
    ) -> None:
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

    # -- upper body --------------------------------------------------------

    def set_wrists(self, left_pos=None, left_quat=None, right_pos=None, right_quat=None):
        if left_pos is not None:
            self.left_wrist_pos = np.asarray(left_pos, dtype=np.float32)
        if left_quat is not None:
            self.left_wrist_quat = np.asarray(left_quat, dtype=np.float32)
        if right_pos is not None:
            self.right_wrist_pos = np.asarray(right_pos, dtype=np.float32)
        if right_quat is not None:
            self.right_wrist_quat = np.asarray(right_quat, dtype=np.float32)

    def set_upper_body(self, q17) -> None:
        """17 joint targets for FROZEN_UPPER_BODY mode: waist y/r/p, then
        L/R-interleaved shoulder p, r, y, elbow, wrist r, p, y."""
        self.upper_body = np.asarray(q17, dtype=np.float32)

    def set_hands(self, left=None, right=None):
        if left is not None:
            self.left_hand = np.asarray(left, dtype=np.float32)
        if right is not None:
            self.right_hand = np.asarray(right, dtype=np.float32)

    def _vr_3pt(self) -> tuple[list[float], list[float]]:
        neck_pos, neck_quat = nominal_neck()
        pos = np.concatenate([self.left_wrist_pos, self.right_wrist_pos, neck_pos])
        quat = np.concatenate([self.left_wrist_quat, self.right_wrist_quat, neck_quat])

        return pos.astype(np.float32).tolist(), quat.astype(np.float32).tolist()

    # -- one control frame -------------------------------------------------

    def frame(
        self,
        mode: LocomotionMode = LocomotionMode.IDLE,
        movement=(0.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed: float = -1.0,
        height: float = -1.0,
        stream_mode: StreamMode | None = None,
        compliance=DEFAULT_COMPLIANCE,
    ) -> None:
        """Emit one full control frame: manager_state + planner.

        `stream_mode` defaults to the commander's `self.stream_mode`, so a
        session-wide choice (frozen upper body vs VR-3pt wrists) does not have
        to be threaded through every call site.
        """
        sm = self.stream_mode if stream_mode is None else stream_mode
        self.manager_state(sm)
        if self.heading_offset:
            c, s = np.cos(-self.heading_offset), np.sin(-self.heading_offset)
            facing = (c * facing[0] - s * facing[1],
                      s * facing[0] + c * facing[1], facing[2])
            movement = (c * movement[0] - s * movement[1],
                        s * movement[0] + c * movement[1], movement[2])
        kw = {}
        if (sm == StreamMode.PLANNER_FROZEN_UPPER_BODY
                and self.upper_body is not None):
            # Mirror the pico manager (pico_manager_thread_server.py:1832-38):
            # frozen mode carries explicit upper-body joint targets and no
            # VR fields.
            kw["upper_body_position"] = self.upper_body.tolist()
        else:
            vr_pos, vr_quat = self._vr_3pt()
            kw.update(vr_3pt_position=vr_pos, vr_3pt_orientation=vr_quat,
                      vr_3pt_compliance=list(compliance))
        self.sonic.send(
            build_planner_message(
                int(mode),
                list(movement),
                list(facing),
                speed=speed,
                height=height,
                left_hand_position=self.left_hand.tolist(),
                right_hand_position=self.right_hand.tolist(),
                **kw,
            )
        )

    # -- streamed motion (the channel Pico teleop actually uses) -----------

    def enter_streamed(self) -> None:
        """Switch the deploy to STREAMED_MOTION: it now imitates `pose` frames.

        Prime the pose buffer with `pose_chunk` BEFORE calling this —
        g1_deploy_onnx_ref.cpp:618 stops control if the mode is active with no
        motion frames buffered.
        """
        # Tell the manager the stream is POSE too: with a stale
        # FROZEN_UPPER_BODY manager state the deploy keeps the upper body
        # pinned to the frozen targets and the streamed reach stalls with
        # IK res 0.000 and a motionless palm.
        self.manager_state(StreamMode.POSE)
        self.command(start=True, stop=False, planner=False)

    def exit_streamed(self) -> None:
        self.command(start=True, stop=False, planner=True)

    def pose_chunk(self, joint_pos_isaac: np.ndarray, root_quat_wxyz,
                   frame_index_start: int, left_hand=None,
                   right_hand=None) -> None:
        """One `pose` message: an [N,29] IsaacLab-order joint chunk.

        Mirrors pico_manager_thread_server.py:1510-1542 (protocol v1: only
        joint_pos/joint_vel/body_quat/frame_index are required; SMPL fields
        are optional and omitted). `root_quat_wxyz` may be [4] (broadcast) or
        [N,4]; the deploy re-anchors its heading to the robot at mode switch,
        so a constant yaw-only quat means "hold the current heading".
        """
        jp = np.atleast_2d(np.asarray(joint_pos_isaac, dtype=np.float32))
        n = len(jp)
        quat = np.asarray(root_quat_wxyz, dtype=np.float32)
        if quat.ndim == 1:
            quat = np.tile(quat, (n, 1))
        data = {
            "joint_pos": jp,
            "joint_vel": np.zeros((n, 29), dtype=np.float32),
            "body_quat": quat.reshape(n, 1, 4),
            "frame_index": np.arange(
                frame_index_start, frame_index_start + n, dtype=np.int64),
        }
        if left_hand is not None:
            data["left_hand_joints"] = np.asarray(left_hand, dtype=np.float32)
        if right_hand is not None:
            data["right_hand_joints"] = np.asarray(right_hand, dtype=np.float32)
        self.sonic.send(pack_pose_message(data, topic="pose", version=1))

    def hold(self, seconds: float, hz: float = 50.0, **frame_kwargs) -> None:
        """Stream the current target at a fixed rate."""
        dt = 1.0 / hz
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.frame(**frame_kwargs)
            time.sleep(dt)

    def close(self) -> None:
        self.sonic.close()
        self.keyboard.close()
        self._ctx.term()
