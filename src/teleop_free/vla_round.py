"""One closed-loop VLA round: observation assembly + token streaming + verdict.

This module is what grasp_gym.VLA_POLICY gets installed with. A VLA round
therefore shares EVERYTHING with a scripted round -- the round sampling, the
spawn bisection, the teleport, the re-anchor, the appearance seed, the video --
and differs only from the point the robot stands settled at its spawn. That
sharing is what makes the result readable: the reference line is the scripted
policy's 200/210 = 95.2% on this exact distribution, measured by the
collection run itself.

OBSERVATION FIDELITY

Every observation channel is built the way the EXPORTER built it during data
collection, mostly by running the same code:

  images       taken verbatim from the camera SUB (:5555) message, no colour
               conversion -- the exporter wrote frame_data[feature] =
               images[key] with no conversion either, so whatever channel
               order training saw, this hands the policy the same one.
  joint state  the same 43-slot layout via
               robot_model.get_configuration_from_actuated_joints, sliced by
               the same group boundaries.
  wrist pose   the same pinocchio FK on the same 'right_wrist_yaw_link'
               frame, quaternion scalar-first, pelvis frame.
  gravity      the same compute_projected_gravity(base_quat).

The one intentional difference: the exporter recorded at 50 Hz and the dataset
was resampled to 30 Hz; here the policy is queried directly at the re-plan
points, each time with the freshest camera/state message.

TOKEN DELIVERY

Chunks come back 40 x 65. The token half is snapped to the FSQ grid
(sonic_token.snap -- training, this driver and the eval all snap the same
way), the gripper scalar becomes the two 7-wide protocol-v4 hand fields
(expand_hands), and frames go out at 30 Hz -- the chunk's native rate. Between
frames the token is zero-order held by the deploy itself (CONFLATE + its own
50 Hz loop). Only the first `execution_horizon` frames of each chunk are
streamed before re-planning, matching the open-loop metric's definition.

VERDICT

Same physical predicate the gym's scripted rounds use: the object's privileged
z (sim stream :5581, independent of the exporter) must rise > 6 cm above its
episode-start height and still be up when the stream ends. Failures are
CLASSIFIED, not just counted -- walk / reach / close / fell / timeout -- because
"the VLA fails where the script fails" (close) and "the VLA cannot even get
there" (walk) are different follow-ups: the first is inherited data bias, the
second is Q5's missing-history ablation trigger.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

from sonic_token import expand_hands, snap  # noqa: E402

TOKEN_DIM = 64
LIFT_M = 0.06            # the gym's own success threshold
NEAR_M = 0.60            # "got to the table" for failure classification
TIMEOUT_S = 90.0
REPLAN_HZ = 30.0         # the chunk's native frame rate


class ObsBuilder:
    """Assemble the policy's flat observation from the live stack."""

    def __init__(self, task: str):
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
        from gear_sonic.data.features_sonic_vla import get_g1_robot_model
        from gear_sonic.utils.data_collection.zmq_state_subscriber import (
            ZMQStateSubscriber,
        )

        cam_port = int(os.environ.get("TF_CAMERA_PORT", 5555))
        state_port = int(os.environ.get("TF_OUT_PORT", 5557))
        self.cam = ComposedCameraClientSensor(server_ip="localhost", port=cam_port)
        self.state = ZMQStateSubscriber(host="localhost", port=state_port)
        self.rm = get_g1_robot_model()
        self.task = task
        self._last_img = None
        self._last_proprio = None

    def _poll(self):
        m = self.cam.read()
        if m is not None:
            self._last_img = m["images"]
        s = self.state.get_msg()
        if s is not None:
            self._last_proprio = s

    def wait_ready(self, timeout: float = 10.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self._poll()
            if self._last_img is not None and self._last_proprio is not None:
                have = set(self._last_img)
                if {"ego_view", "right_wrist"} <= have:
                    return True
            time.sleep(0.05)
        return False

    def build(self) -> dict | None:
        from gear_sonic.utils.data_collection.transforms import (
            compute_projected_gravity,
        )
        from scipy.spatial.transform import Rotation as R

        self._poll()
        if self._last_img is None or self._last_proprio is None:
            return None
        pr = self._last_proprio
        body = np.asarray(pr["body_q"], dtype=np.float64)
        lh = np.asarray(pr["left_hand_q"], dtype=np.float64)
        rh = np.asarray(pr["right_hand_q"], dtype=np.float64)
        base_quat = np.asarray(pr["base_quat"], dtype=np.float64)

        whole = self.rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body,
            left_hand_actuated_joint_values=lh,
            right_hand_actuated_joint_values=rh)
        self.rm.cache_forward_kinematics(whole)
        pl = self.rm.frame_placement(self.rm.supplemental_info.hand_frame_names["right"])
        rw_pos = pl.translation[:3]
        rw_quat = R.from_matrix(pl.rotation).as_quat(scalar_first=True)

        f32 = lambda a: np.asarray(a, dtype=np.float32).reshape(1, -1)  # noqa: E731
        obs = {
            "state.left_leg": f32(body[0:6]),
            "state.right_leg": f32(body[6:12]),
            "state.waist": f32(body[12:15]),
            "state.left_arm": f32(body[15:22]),
            "state.right_arm": f32(body[22:29]),
            "state.left_hand": f32(lh),
            "state.right_hand": f32(rh),
            "state.projected_gravity": f32(compute_projected_gravity(base_quat)),
            "state.right_wrist_pos": f32(rw_pos),
            "state.right_wrist_abs_quat": f32(rw_quat),
            "annotation.human.task_description": self.task,
        }
        for key in ("ego_view", "right_wrist"):
            img = np.asarray(self._last_img[key])
            obs[f"video.{key}"] = img[None].astype(np.uint8)
        return obs

    def close(self):
        try:
            self.cam.close()
        except Exception:
            pass
        try:
            self.state.close()
        except Exception:
            pass


class PolicyLink:
    """Tiny msgpack/zmq client for serve_policy.py -- no gr00t imports."""

    def __init__(self, host="localhost", port=5590, timeout_ms=30000):
        import msgpack
        import msgpack_numpy as mnp
        import zmq

        self._msgpack, self._mnp = msgpack, mnp
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def _call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(self._msgpack.packb(req, default=self._mnp.encode))
        rep = self._msgpack.unpackb(self.sock.recv(), object_hook=self._mnp.decode)
        if isinstance(rep, dict) and "error" in rep:
            raise RuntimeError(f"policy server: {rep['error']}")
        return rep

    def ping(self) -> bool:
        try:
            self._call("ping")
            return True
        except Exception:
            return False

    def get_chunk(self, obs) -> tuple[np.ndarray, np.ndarray]:
        """-> (token [40,64] snapped, gripper [40])."""
        out = self._call("get_action_raw", {"obs": obs})
        tok = np.asarray(out["action.motion_token"], dtype=np.float64)
        grip = np.asarray(out["action.right_gripper"], dtype=np.float64).reshape(-1)
        return snap(tok), grip


def make_vla_policy(link: PolicyLink, task: str, execution_horizon: int = 16,
                    trace_dir: str | None = None):
    """Build the callable grasp_gym.VLA_POLICY expects.

    Called with the robot settled at its spawn, facing the object, video and
    (optionally) the exporter already recording. Must return (ok, why).
    """
    counter = {"n": 0}

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record):
        from primitives import LocomotionMode

        obs_src = ObsBuilder(task)
        if not obs_src.wait_ready():
            obs_src.close()
            return False, "[vla] camera/state streams never became ready"

        s0 = p.obs()
        obj0 = np.asarray(p.object_pos(obj_key, s0))
        trace = []
        ok, why = False, "[vla] timeout"
        dt = 1.0 / REPLAN_HZ
        commanded_close = False

        # Prime the buffer BEFORE the mode switch, then hold the first token
        # across it -- streamed mode over an empty buffer stops control
        # (g1_deploy_onnx_ref.cpp:618), and the first v4 frame is also what
        # flips is_using_encoder_ off. Same order diag_token_replay validated.
        obs = obs_src.build()
        tok0, grip0 = link.get_chunk(obs)
        l0, r0 = expand_hands(grip0[0])
        for _ in range(30):
            p.cmd.token_frame(tok0[0], right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(0.02)
        p.cmd.enter_streamed()
        for _ in range(30):
            p.cmd.token_frame(tok0[0], right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(0.02)

        t_start = time.time()
        frame_index = 0
        lifted_at = None
        try:
            while time.time() - t_start < TIMEOUT_S:
                obs = obs_src.build()
                if obs is None:
                    why = "[vla] observation stream dropped"
                    break
                tok, grip = link.get_chunk(obs)
                n = min(execution_horizon, len(tok))
                for j in range(n):
                    lhj, rhj = expand_hands(grip[j])
                    if grip[j] < 0.005:
                        commanded_close = True
                    p.cmd.token_frame(tok[j], right_hand=rhj, left_hand=lhj,
                                      frame_index=frame_index)
                    frame_index += 1
                    time.sleep(dt)

                st = p.obs()
                obj = np.asarray(p.object_pos(obj_key, st))
                pel = np.asarray(p.pelvis_xy(st))
                rise = float(obj[2] - obj0[2])
                dist = float(np.linalg.norm(obj[:2] - pel))
                trace.append(dict(t=time.time() - t_start, rise=rise, dist=dist,
                                  fallen=bool(st["fallen"]),
                                  grip=float(grip[:n].min())))
                if st["fallen"]:
                    why = "[vla] fell"
                    break
                if rise > LIFT_M:
                    if lifted_at is None:
                        lifted_at = time.time()
                    elif time.time() - lifted_at > 2.0:   # held, not a bounce
                        ok, why = True, "ok"
                        break
                else:
                    lifted_at = None
        finally:
            p.cmd.exit_streamed()
            p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
            obs_src.close()

        # Classify the failure by how far it got, in the manifest's own words.
        if not ok and not why.startswith("[vla] fell") and trace:
            min_dist = min(r["dist"] for r in trace)
            if min_dist > NEAR_M:
                why = f"[vla] walk: never closer than {min_dist:.2f} m"
            elif not commanded_close:
                why = f"[vla] reach: at the table (min {min_dist:.2f} m) but never commanded a close"
            else:
                moved = float(np.linalg.norm(
                    np.asarray(p.object_pos(obj_key, p.obs()))[:2] - obj0[:2]))
                why = (f"[vla] close: commanded a grasp but no lift "
                       f"(object moved {moved * 100:.0f} cm laterally)")

        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            with open(os.path.join(trace_dir, f"round_{counter['n']:03d}.json"), "w") as f:
                json.dump(dict(ok=ok, why=why, trace=trace), f)
        counter["n"] += 1
        return ok, why

    return vla_round
