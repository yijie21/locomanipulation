"""SONIC observation assembly and policy-server transport.

Runtime options are explicit immutable VlaConfig values.
"""

from __future__ import annotations

import os
import time

import numpy as np

from locomani.backends.sonic.sonic_token import snap
from locomani.backends.sonic.vla_config import DEFAULT_VLA, LIFT_M, NEAR_M, STALL_ABORT_M, VlaConfig
from locomani.schema import SONIC_BOTTLE


def _tstart_view(p, st, obj_key, prelude, gx=0.80, gy=0.90):
    """Is the target still in the ego frame when the trace starts?

    The gym gates ego visibility at the SPAWN -- after the settle, before the
    VLA fork (`[view] object not in ego view at spawn`, ~12% of rounds at
    --spawn-edge 1.1 1.3). Nothing re-checked it AFTER the prelude, and the
    prelude is where the 18-D path holds its pose open-loop, so a round could
    start blind and then be scored as a policy failure. This records where the
    target actually sits in the frame at t=0 (normalised, frame edge 1.0, live
    from the privileged publisher's camera) so such a round is identifiable
    offline instead of assumed away.

    Warned, not aborted: aborting here would have to unwind the feeder threads
    and the running recorder, and the honest treatment of a blind round in a
    PAIRED comparison is to drop it from both arms, which is an offline
    decision. tools/failure_taxonomy.py buckets it.

    Margin, measured on the pinned eval scene over 62 cover400 spawns
    (src/locomani/diag_spawn_view.py, 2026-09-04): the bottle leaves the image at a
    pelvis->object range of ~1.95 m and the conservative central-90% gate
    breaks at 1.68 m, while real spawns sit at 1.17-1.47 m (u_y 0.61-0.81).
    Sideways there is 78-105 cm of slack (|u_x| <= 0.40 against a 0.80 gate),
    so only BACKWARD drift can lose the target -- and the settle's measured
    direction is table-ward, which gains margin rather than spending it.
    """
    cam = ((st.get("objects") or {}).get(obj_key) or {}).get("cam")
    prelude["tstart_cam"] = None if cam is None else [round(float(v), 3) for v in cam]
    if cam is None:
        p.log(
            "[vla] WARNING target is BEHIND the ego camera at trace t=0 -- this round is blind and is not a policy test"
        )
    elif abs(cam[0]) > gx or abs(cam[1]) > gy:
        p.log(
            f"[vla] WARNING target outside the ego gate at trace t=0 "
            f"(cam=[{cam[0]:+.2f} {cam[1]:+.2f}], gate {gx}/{gy}) -- the "
            f"prelude drifted it off frame; not a policy test"
        )


def _obj_extra(st, obj_key):
    """Per-tick object facts for the failure taxonomy (2026-09-03): tilt from
    upright in degrees, which robot links touch the object, and its height.
    Old traces lack these; tools/failure_taxonomy.py degrades to xy displacement."""
    o = (st.get("objects") or {}).get(obj_key) or {}
    q = o.get("quat")
    tilt = None
    if q is not None:
        w_, x_, y_, z_ = (float(v) for v in q)
        r22 = 1.0 - 2.0 * (x_ * x_ + y_ * y_)
        tilt = float(np.degrees(np.arccos(float(np.clip(r22, -1.0, 1.0)))))
    pos = o.get("pos")
    return dict(
        obj_tilt=tilt,
        touch=list(o.get("touching") or []),
        touch_links=list(o.get("touch_links") or []),  # every robot link, not just the hands
        obj_z=(None if pos is None else float(pos[2])),
    )


class ObsBuilder:
    """Assemble the policy's flat observation from the live stack."""

    def __init__(self, task: str, schema=SONIC_BOTTLE):
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
        # Which TaskSchema this observation serves: SONIC_BOTTLE (two cameras,
        # right wrist, 53-d) or SONIC_CRATE (three cameras, both wrists, 60-d).
        self.schema = schema
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
                if set(self.schema.video_keys) <= have:
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
            body_actuated_joint_values=body, left_hand_actuated_joint_values=lh, right_hand_actuated_joint_values=rh
        )
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
        if "left_wrist_pos" in self.schema.state_keys:
            pl_left = self.rm.frame_placement(self.rm.supplemental_info.hand_frame_names["left"])
            obs["state.left_wrist_pos"] = f32(pl_left.translation[:3])
            obs["state.left_wrist_abs_quat"] = f32(R.from_matrix(pl_left.rotation).as_quat(scalar_first=True))
        for key in self.schema.video_keys:
            img = np.asarray(self._last_img[key])
            obs[f"video.{key}"] = img[None].astype(np.uint8)
        self.schema.validate_observation(obs)
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

    def __init__(self, host="localhost", port=5590, timeout_ms=30000, *, config: VlaConfig = DEFAULT_VLA):
        self.config = config
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

    def identity(self) -> dict:
        """Checkpoint content and model field order reported by the serving process."""
        return self._call("identity")

    def get_chunk(self, obs) -> tuple[np.ndarray, np.ndarray]:
        """-> (token [40,64], gripper [40]). Snapped unless SNAP is off."""
        out = self._call("get_action_raw", {"obs": obs})
        self.last_rtc = out.get("rtc.applied")  # [overlap, frozen] | None
        dump = self.config.backend.obs_dump
        if dump:  # diagnostic: what the policy saw + what it answered
            os.makedirs(dump, exist_ok=True)
            n = getattr(self, "_dump_n", 0)
            self._dump_n = n + 1
            np.savez_compressed(
                os.path.join(dump, f"obs_{n:05d}.npz"),
                wall=time.time(),
                tok=np.asarray(out["action.motion_token"]),
                grip=np.asarray(out["action.right_gripper"]),
                **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v)) for k, v in obs.items()},
            )
        tok = np.asarray(out["action.motion_token"], dtype=np.float64)
        grip = np.asarray(out["action.right_gripper"], dtype=np.float64).reshape(-1)
        return (snap(tok) if self.config.snap else tok), grip

    def get_chunk_bimanual(self, obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> (token [40,64], left gripper [40], right gripper [40]) for a SONIC_CRATE checkpoint."""
        out = self._call("get_action_raw", {"obs": obs})
        self.last_rtc = out.get("rtc.applied")
        tok = np.asarray(out["action.motion_token"], dtype=np.float64)
        left = np.asarray(out["action.left_gripper"], dtype=np.float64).reshape(-1)
        right = np.asarray(out["action.right_gripper"], dtype=np.float64).reshape(-1)
        return (snap(tok) if self.config.snap else tok), left, right
