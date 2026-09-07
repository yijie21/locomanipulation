"""Phase 2 of SIMPLE's replay/render contract: push a recorded MuJoCo state
back in, then render it.

SIMPLE's loop (docs/source/sonic-wbc/replay_render.md) is::

    for state in states:
        env.mjData.qpos[:] = state["qpos"]
        env.mjData.qvel[:] = state["qvel"]
        mujoco.mj_forward(env.mjModel, env.mjData)
        env.isaac.step(env.mujoco)
        frame = env._render_frame()

This module is the same loop against our own stage. The single point that
matters is that EVERYTHING Isaac needs comes out of ONE ``mj_forward`` on ONE
recorded state: the joint positions, the object world poses, and the camera
world pose. No second FK pass, no join between two recordings, no clock to
reconcile. Our earlier reconstruction failed precisely because it mixed the
exporter's proprioceptive joints with the pose sidecar's MuJoCo truth.
"""

from __future__ import annotations

import numpy as np


class MujocoReplay:
    """A MuJoCo model used as a state decoder, never stepped."""

    def __init__(self, scene_xml: str, object_keys=()):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.scene_xml = scene_xml
        self._obj = {}
        for k in object_keys:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{k}_object")
            if bid >= 0:
                self._obj[k] = bid

    # --- state ------------------------------------------------------------
    def set(self, qpos, qvel=None) -> None:
        if len(qpos) != self.model.nq:
            raise SystemExit(
                f"trace qpos has {len(qpos)} entries, the scene has nq="
                f"{self.model.nq}. The trace was recorded against a different "
                f"scene -- rendering it would place a correct trajectory in the "
                f"wrong world. Check the batch's scene snapshot.")
        self.data.qpos[:] = qpos
        if qvel is not None and len(qvel) == self.model.nv:
            self.data.qvel[:] = qvel
        self.mj.mj_forward(self.model, self.data)

    # --- readers, all from the same forward pass ---------------------------
    def dof_positions(self, names) -> np.ndarray:
        out = np.zeros(len(names))
        for i, n in enumerate(names):
            jid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                out[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return out

    def body_pose(self, name: str) -> np.ndarray:
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise SystemExit(f"no body '{name}' in {self.scene_xml}")
        return np.concatenate([self.data.xpos[bid], self.data.xquat[bid]])

    def object_poses(self) -> dict:
        return {k: np.concatenate([self.data.xpos[b], self.data.xquat[b]])
                for k, b in self._obj.items()}

    def camera_pose(self, name: str):
        """(pos, wxyz quat). MuJoCo's camera frame is -Z forward / +Y up, the
        same convention USD uses, so the quaternion transfers verbatim."""
        from scipy.spatial.transform import Rotation as R

        cid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise SystemExit(f"no camera '{name}' in {self.scene_xml}")
        m = self.data.cam_xmat[cid].reshape(3, 3)
        qx, qy, qz, qw = R.from_matrix(m).as_quat()
        return self.data.cam_xpos[cid].copy(), np.array([qw, qx, qy, qz])

    def camera_fovy(self, name: str) -> float:
        cid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_CAMERA, name)
        return float(self.model.cam_fovy[cid])
