"""Exact, differentiable forward kinematics of the mars G1 + Dex1 in the pelvis frame, read from the MuJoCo model:
what the robot knows about itself without learning anything. Replaces wm.py's learned kinematics head g.

Frames out (pelvis frame, batched over the 29 ScaleBFM joints in ScaleBFM order):
  jaw_right / jaw_left   jaw centre = wrist_yaw_link + R_wrist @ OFF, with the wrist rotation
  head_camera, right_wrist, left_wrist   camera position and rotation (columns = camera axes, MuJoCo convention:
                                          x right, y up, looking along -z)
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

OFF = np.array([0.1515, 0.0, 0.0])  # runtime.OFF: wrist frame -> jaw (pad) centre
BODIES = {"jaw_right": "right_wrist_yaw_link", "jaw_left": "left_wrist_yaw_link"}
CAMERAS = ("head_camera", "right_wrist", "left_wrist")


def quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


class FK:
    def __init__(self, model: mujoco.MjModel, qadr: np.ndarray, device="cpu"):
        """qadr: the qpos address of each of the 29 joints in ScaleBFM order (runtime.Sim.qadr)."""
        self.dev = device
        index = {int(a): i for i, a in enumerate(qadr)}
        self.links = {}  # body id -> (parent id, pos, rot, [(anchor, axis, joint index)])
        targets = {n: model.body(b).id for n, b in BODIES.items()}
        targets.update({c: int(model.cam_bodyid[model.camera(c).id]) for c in CAMERAS})
        need = set()
        for b in targets.values():
            while b != 0 and model.body_parentid[b] != 0:
                need.add(b)
                b = int(model.body_parentid[b])
            self.root = b  # the pelvis (child of world)
        for b in need:
            js = []
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
                assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, model.joint(j).name
                js.append((model.jnt_pos[j].copy(), model.jnt_axis[j].copy(), index[int(model.jnt_qposadr[j])]))
            self.links[b] = (int(model.body_parentid[b]), model.body_pos[b].copy(), quat_to_mat(model.body_quat[b]), js)
        self.order = sorted(need, key=lambda b: self.depth(b, model))
        self.targets = targets
        self.cams = {
            c: (model.cam_pos[model.camera(c).id].copy(), quat_to_mat(model.cam_quat[model.camera(c).id]))
            for c in CAMERAS
        }
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
        self.t = t
        self.T = {
            b: (p, t(pos), t(rot), [(t(an), t(ax), i) for an, ax, i in js])
            for b, (p, pos, rot, js) in self.links.items()
        }
        self.off = t(OFF)
        self.camT = {c: (t(p), t(R)) for c, (p, R) in self.cams.items()}

    @staticmethod
    def depth(b, model):
        d = 0
        while b != 0:
            b = int(model.body_parentid[b])
            d += 1
        return d

    @staticmethod
    def axis_angle(axis, ang):
        """(B,) angles about a fixed unit axis -> (B, 3, 3) rotations (Rodrigues)."""
        x, y, z = axis
        K = torch.stack([torch.stack([0 * x, -z, y]), torch.stack([z, 0 * x, -x]), torch.stack([-y, x, 0 * x])])
        s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
        eye = torch.eye(3, device=ang.device)
        return eye + s * K + (1 - c) * (K @ K)

    def __call__(self, q: torch.Tensor) -> dict:
        """q (B, 29) -> {name: (pos (B, 3), rot (B, 3, 3))} in the pelvis frame."""
        B = q.shape[0]
        eye = torch.eye(3, device=q.device).expand(B, 3, 3)
        pose = {self.root: (torch.zeros(B, 3, device=q.device), eye)}
        for b in self.order:
            parent, pos, rot, js = self.T[b]
            pp, pr = pose[parent]
            p = pp + torch.einsum("bij,j->bi", pr, pos)
            r = pr @ rot
            for anchor, axis, i in js:
                Rj = self.axis_angle(axis, q[:, i])
                a = torch.einsum("bij,j->bi", r, anchor)
                r_new = r @ Rj
                p = p + a - torch.einsum("bij,j->bi", r_new, anchor)
                r = r_new
            pose[b] = (p, r)
        out = {}
        for n in BODIES:
            p, r = pose[self.targets[n]]
            out[n] = (p + torch.einsum("bij,j->bi", r, self.off), r)
        for c in CAMERAS:
            p, r = pose[self.targets[c]]
            cp, cr = self.camT[c]
            out[c] = (p + torch.einsum("bij,j->bi", r, cp), r @ cr)
        return out


def check(n: int = 200) -> float:
    """Max position / rotation error against mj_kinematics over random joint vectors (pelvis at the origin)."""
    import sys
    from pathlib import Path

    sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "src")]
    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig

    here = Path(__file__).resolve().parents[3]
    cfg = ScaleBfmBackendConfig(scene_dir=str(here / "outputs" / "scene" / "walk_mars"), object_name="")
    meta = __import__("json").load(open(R.META))
    sim = R.Sim(meta, backend=cfg)
    m, d = sim.m, sim.d
    fk = FK(m, sim.qadr)
    rng = np.random.default_rng(0)
    lo, hi = m.jnt_range[[int(np.where(m.jnt_qposadr == a)[0][0]) for a in sim.qadr]].T
    worst_p = worst_r = 0.0
    qs = rng.uniform(lo, hi, (n, 29))
    out = fk(torch.tensor(qs, dtype=torch.float32))
    for k in range(n):
        d.qpos[:] = m.qpos0
        d.qpos[:3], d.qpos[3:7] = 0.0, [1, 0, 0, 0]
        d.qpos[sim.qadr] = qs[k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)
        for side, nm in (("right", "jaw_right"), ("left", "jaw_left")):
            b = m.body(f"{side}_wrist_yaw_link").id
            Rw = d.xmat[b].reshape(3, 3)
            jaw = d.xpos[b] + Rw @ OFF
            worst_p = max(worst_p, float(np.abs(out[nm][0][k].numpy() - jaw).max()))
            worst_r = max(worst_r, float(np.abs(out[nm][1][k].numpy() - Rw).max()))
        for c in CAMERAS:
            i = m.camera(c).id
            worst_p = max(worst_p, float(np.abs(out[c][0][k].numpy() - d.cam_xpos[i]).max()))
            worst_r = max(worst_r, float(np.abs(out[c][1][k].numpy() - d.cam_xmat[i].reshape(3, 3)).max()))
    print(f"max position error {worst_p * 1000:.4f} mm, max rotation entry error {worst_r:.2e} over {n} random poses")
    return worst_p


if __name__ == "__main__":
    check()
