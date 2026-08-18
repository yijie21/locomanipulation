"""Can SONIC's STREAMED_MOTION channel place the palm where IK says?

Planner-mode wrist targets go through the kinematic planner and arrive with a
25 cm warp, 1 s of lag, and 22 % orientation delivery — three weeks of
evidence says that channel cannot deliver grasp precision. Pico teleop never
uses it: it streams full 29-DoF joint references on the `pose` topic
(pico_manager_thread_server.py:1510-1542) and the policy imitates them.

This diagnostic drives that same channel from IK instead of a human:

  1. stand in planner mode, read the measured 29 joints
  2. switch to STREAMED_MOTION, stream the held pose (robot should just stand)
  3. per target: IK the right arm to a pelvis-frame palm position, stream a
     joint-space smoothstep to it, hold, measure the settled palm
  4. report palm error (the number planner mode could not get below ~25 cm
     feed-forward / 4 cm closed-loop) and joint-tracking error

    python src/teleop_free/diag_stream.py [--settle 3.0]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import LocomotionMode  # noqa: E402
from arm_ik import ArmIK  # noqa: E402
from g1_joints import ARM_MJ, mujoco_to_isaac  # noqa: E402
from primitives import DT, Primitives  # noqa: E402

# Pelvis-frame palm targets in the region the grasp uses. The last one also
# asks for orientation, to measure the rot channel on this path.
# Interior points of the arm-only reach envelope (IK residual < 0.05 cm),
# spread over the workspace. NOTE the envelope finding that picked them: with
# the shoulder at pelvis-frame z +0.29, the palm's max forward reach in the
# grasp band (z -0.25..0) is only ~0.20 m without waist lean — the planner
# era's "25 cm warp" was largely saturation against this boundary.
TARGETS = [
    {"pos": [0.236, -0.154, 0.090]},
    {"pos": [0.179, -0.243, 0.059]},
    {"pos": [0.292, -0.098, 0.170]},
    {"pos": [0.200, -0.166, 0.052]},
]

# Grasp-band targets — need the torso lean (--waist). These are where the can
# actually sits in the pelvis frame when standing at the table.
WAIST_TARGETS = [
    {"pos": [0.35, -0.20, -0.05]},
    {"pos": [0.30, -0.30, -0.15]},
    {"pos": [0.40, -0.20, -0.15]},
    {"pos": [0.45, -0.20, -0.10]},
]

JOINT_SPEED = 0.8  # rad/s, max joint rate for the streamed trajectory
CHUNK = 5          # frames per pose message, like the pico manager


def yaw_only(quat_wxyz):
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                    dtype=np.float32)


class PoseStreamer:
    """50 Hz sender of overlapping 5-frame chunks, like the pico manager."""

    def __init__(self, cmd, root_quat):
        self.cmd = cmd
        self.root_quat = root_quat
        self.buf: deque = deque(maxlen=CHUNK)
        self.i = 0
        self.next_t = time.time()

    def push(self, q_mj29):
        self.buf.append(mujoco_to_isaac(q_mj29).astype(np.float32))
        start = self.i - len(self.buf) + 1
        self.cmd.pose_chunk(np.stack(self.buf), self.root_quat, start)
        self.i += 1
        self.next_t += DT
        lag = self.next_t - time.time()
        if lag > 0:
            time.sleep(lag)

    def hold(self, q_mj29, seconds):
        for _ in range(int(round(seconds / DT))):
            self.push(q_mj29)

    def move(self, q_from, q_to, speed=JOINT_SPEED):
        dur = max(0.6, float(np.abs(q_to - q_from).max()) / speed)
        n = int(round(dur / DT))
        for k in range(1, n + 1):
            u = k / n
            s = 3 * u * u - 2 * u ** 3
            self.push(q_from + s * (q_to - q_from))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=float, default=4.0)
    ap.add_argument("--side", default="right")
    ap.add_argument("--waist", action="store_true",
                    help="grasp-band targets, torso lean in the IK chain")
    ap.add_argument("--squat", type=float, default=0.0,
                    help="squat angle a in rad (hip -a, knee +2a, ankle -a; "
                         "a=0.45 drops the pelvis ~6 cm, 0.6 ~11 cm); targets "
                         "are treated as world-fixed and shifted up in the "
                         "pelvis frame by the measured drop")
    args = ap.parse_args()
    side = args.side
    targets = WAIST_TARGETS if args.waist else TARGETS
    use_waist = args.waist

    from privileged_pub import EpisodeControlClient

    ik = ArmIK()
    p = Primitives()
    reset = EpisodeControlClient()
    # Same order collect_episode uses: reset the scene FIRST, then latch
    # control — the robot has been in uncontrolled free fall since stack boot.
    p.cmd.reset_pose()
    reset.reset(label="diag_stream")
    time.sleep(0.2)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)

    s = p.obs()
    if s["fallen"]:
        print(f"robot did not stand after reset (pelvis {s['pelvis']['pos'][2]:.3f})")
        p.close()
        return
    reset.start_video("stream_diag")
    q0 = np.array(s["joints"])
    if not np.any(q0):
        print("obs has no 'joints' — restart the stack (privileged_pub is stale)")
        p.close()
        return
    root_quat = yaw_only(s["pelvis"]["quat"])
    palm0 = p.world_to_root_full(s["palms"][side]["pos"], s)
    print(f"standing: pelvis_z {s['pelvis']['pos'][2]:.3f}  "
          f"palm {np.round(palm0, 3)}")

    # Prime the pose buffer BEFORE the mode switch (deploy stops control if
    # streamed mode goes active with an empty motion buffer), then switch.
    streamer = PoseStreamer(p.cmd, root_quat)
    streamer.hold(q0, 0.3)
    p.cmd.enter_streamed()
    print("switched to STREAMED_MOTION, holding current pose 2 s")
    streamer.hold(q0, 2.0)

    s = p.obs()
    print(f"  after hold: pelvis_z {s['pelvis']['pos'][2]:.3f}  fallen={s['fallen']}")

    z_stand = s["pelvis"]["pos"][2]
    dz = 0.0
    if args.squat > 0:
        # Deepen the standing flexion pattern (hip pitch, knee, ankle pitch on
        # both legs) — a squat reference for the policy to imitate, the way
        # teleop kneeling drives it.
        a = args.squat
        q_squat = q0.copy()
        for hip, knee, ank in ((0, 3, 4), (6, 9, 10)):
            q_squat[hip] -= a
            q_squat[knee] += 2 * a
            q_squat[ank] -= a
        q_squat = np.clip(q_squat, ik.lo, ik.hi)
        streamer.move(q0, q_squat, speed=0.6)
        streamer.hold(q_squat, 2.5)
        s = p.obs()
        dz = z_stand - s["pelvis"]["pos"][2]
        print(f"  squat x{args.squat}: pelvis_z {s['pelvis']['pos'][2]:.3f} "
              f"(dropped {dz * 100:.1f} cm)  fallen={s['fallen']}")
        if s["fallen"]:
            p.cmd.exit_streamed()
            p.cmd.pause_control()
            reset.end_video()
            p.close()
            return
        q0 = q_squat

    arm = ARM_MJ[side]
    q = q0.copy()
    results = []
    for t_i, tgt in enumerate(targets):
        want = np.array(tgt["pos"]) + np.array([0.0, 0.0, dz])
        q_next, ik_pos_err, ik_rot_err = ik.solve(
            side, want, tgt.get("rot"), q_mj29=q, use_waist=use_waist)
        print(f"\ntarget {t_i}: {np.round(want, 3)}  "
              f"(IK residual {ik_pos_err * 100:.2f} cm)")

        streamer.move(q, q_next)
        streamer.hold(q_next, args.settle)

        err = None
        for it in range(3):
            s = p.obs()
            palm = p.world_to_root_full(s["palms"][side]["pos"], s)
            meas = np.array(s["joints"])
            track_j = list(arm) + ([12, 13, 14] if use_waist else [])
            jerr = np.abs(meas[track_j] - q_next[track_j])
            worst = np.argsort(jerr)[::-1][:3]
            from g1_joints import MUJOCO_JOINT_NAMES
            wstr = "  ".join(
                f"{MUJOCO_JOINT_NAMES[track_j[w]].replace('_joint', '')}"
                f" {np.degrees(jerr[w]):.0f}deg" for w in worst)
            err = np.linalg.norm(palm - want)
            print(f"  it{it}: palm {np.round(palm, 3)}  err {err * 100:.2f} cm  "
                  f"worst joints: {wstr}  "
                  f"pelvis_z {s['pelvis']['pos'][2]:.3f}  fallen={s['fallen']}")
            if err < 0.010 or s["fallen"]:
                break
            # Re-aim by the CURRENT residual only. Accumulating corrections
            # across iterations diverges: a large it0 error is usually an
            # unsettled transient, not a static bias, and a correction carried
            # forward after the transient dies overshoots by its full size
            # (target 3 in the first run: 1.0 cm -> 13.3 cm).
            adj = want + (want - palm)
            q_sol, res2, _ = ik.solve(side, adj, tgt.get("rot"), q_mj29=q_next,
                                      use_waist=use_waist)
            if res2 > 0.02:
                print(f"    correction target unreachable (residual "
                      f"{res2 * 100:.1f} cm); keeping current")
                break
            q_prev = q_next
            q_next = q_sol
            streamer.move(q_prev, q_next)
            streamer.hold(q_next, 2.5)
        results.append(err)
        q = q_next
        if s["fallen"]:
            print("robot fell; stopping")
            break

    # back to rest and to planner mode
    streamer.move(q, q0, speed=0.8)
    p.cmd.exit_streamed()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.pause_control()
    reset.end_video()
    reset.close()
    p.close()

    if results:
        r = np.array(results) * 100
        print(f"\npalm error over {len(r)} targets: "
              f"mean {r.mean():.2f} cm  max {r.max():.2f} cm")
        print("planner-mode closed loop was ~3.7 cm; feed-forward 24.6 cm")


if __name__ == "__main__":
    main()
