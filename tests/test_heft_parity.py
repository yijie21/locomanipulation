"""TODO 51: the HEFT port (locomani/backends/scalebfm/heft.py) against LocoManiLab's own HEFT observation class and action
mapping, on the same reference frames and robot states. No simulator. LocoManiLab's `heft_policy_obs` is imported from its
checkout (third_party/LocoManiLab/external/g1_backend) with two packages it imports but never uses here stubbed
(`any4hdmi`, `mjhub`: dataset loading and asset hubs)."""

import importlib.abc
import importlib.machinery
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

from locomani.backends.scalebfm import heft as H
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

G1B = os.path.join(H.REPO, "third_party/LocoManiLab/external/g1_backend")
HAVE = os.path.exists(os.path.join(H.HEFT_DIR, "policy.yaml")) and os.path.isdir(G1B)


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("any4hdmi", "mjhub"):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = mock.MagicMock()
        m.__path__, m.__spec__, m.__name__ = [], spec, spec.name
        return m

    def exec_module(self, m):
        pass


def _lml():
    stub = _Stub()
    sys.meta_path.insert(0, stub)
    sys.path.insert(0, G1B)
    try:
        from sim2real.rl_policy.observations.heft import heft_policy_obs
        from sim2real.utils.strings import resolve_matching_names_values
    finally:
        sys.meta_path.remove(stub)
        sys.path.remove(G1B)
    return heft_policy_obs, resolve_matching_names_values


class _Env:
    """What heft_policy_obs reads of its host: the state processor and the motion source."""

    def __init__(self, man, F):
        n = len(MUJOCO_JOINT_NAMES)
        self.state_processor = types.SimpleNamespace(joint_names=list(MUJOCO_JOINT_NAMES))
        self.motion_config = {}
        self.motion_future_steps = np.asarray(man.future_steps)
        self.motion_joint_names = list(MUJOCO_JOINT_NAMES)  # a reference file's order, not the policy's
        self.motion_body_names = ["pelvis"]
        self.joint_names_simulation = list(man.sim_joint_names)
        self.body_names_simulation = ["pelvis"]
        self.policy_joint_names = list(man.joint_names)
        self.motion_data = None
        self.F, self.n = F, n

    def set_reference(self, joints_mj, pos, quat):
        F = self.F
        z = lambda *s: np.zeros(s)  # noqa: E731
        self.motion_data = types.SimpleNamespace(
            joint_pos=joints_mj[None], joint_vel=z(1, F, self.n),
            body_pos_w=pos[None, :, None, :], body_lin_vel_w=z(1, F, 1, 3),
            body_quat_w=quat[None, :, None, :], body_ang_vel_w=z(1, F, 1, 3),
        )  # fmt: skip


def _quat(rng, scale):
    v = rng.normal(size=3) * scale
    a = np.linalg.norm(v) + 1e-12
    return np.r_[np.cos(a / 2), np.sin(a / 2) * v / a]


@unittest.skipUnless(HAVE, "HEFT manifest or LocoManiLab checkout missing")
class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs_cls, resolve = _lml()
        cls.resolve = staticmethod(resolve)
        cls.man = H.HeftManifest()

    def _lml_obs(self):
        env = _Env(self.man, len(self.man.future_steps))
        sp = env.state_processor
        sp.joint_pos = sp.joint_vel = np.zeros(29)
        sp.root_quat_w = np.array([1.0, 0, 0, 0])
        sp.root_ang_vel_b = np.zeros(3)
        c = dict(
            future_steps=self.man.future_steps, root_angvel_history_steps=self.man.history_steps,
            projected_gravity_history_steps=self.man.history_steps, joint_pos_history_steps=self.man.history_steps,
            joint_vel_history_steps=self.man.history_steps, prev_action_steps=self.man.prev_action_steps,
            joint_names=self.man.joint_names, root_body_name="pelvis", boot_indicator_max=self.man.boot_indicator_max,
        )  # fmt: skip
        env.set_reference(np.zeros((env.F, 29)), np.zeros((env.F, 3)), np.tile([1.0, 0, 0, 0], (env.F, 1)))
        return env, self.obs_cls(env=env, **c)

    def test_the_four_input_groups_match_every_tick(self):
        rng = np.random.default_rng(7)
        env, theirs = self._lml_obs()
        ours = H.HeftObs(self.man)
        theirs.reset()
        ours.reset()
        sp, F = env.state_processor, len(self.man.future_steps)
        prev = np.zeros(29, np.float32)
        perm = np.array([MUJOCO_JOINT_NAMES.index(n) for n in self.man.joint_names])
        for tick in range(60):  # past the boot countdown (25) and the longest history (20)
            sp.joint_pos = rng.normal(size=29) * 0.4
            sp.joint_vel = rng.normal(size=29)
            sp.root_quat_w = _quat(rng, 0.3)
            sp.root_ang_vel_b = rng.normal(size=3)
            joints = rng.normal(size=(F, 29)) * 0.4
            pos = np.cumsum(rng.normal(size=(F, 3)) * 0.02, axis=0) + [1.0, -2.0, 0.78]
            quat = np.array([_quat(rng, 0.4) for _ in range(F)])
            env.set_reference(joints, pos, quat)
            theirs.update({"action": prev})
            want = theirs._build_obs_parts()
            got = ours.update(
                sp.root_quat_w,
                sp.root_ang_vel_b,
                sp.joint_pos[perm],
                sp.joint_vel[perm],
                joints[:, perm],
                pos,
                quat,
                prev,
            )
            self.assertEqual(list(want), list(got))
            for k in want:
                np.testing.assert_allclose(got[k], want[k], rtol=1e-6, atol=1e-7, err_msg=f"tick {tick} {k}")
            self.assertEqual([got[k].size for k in H.OBS_KEYS], [1, 114, 806, 808])
            prev = rng.normal(size=29).astype(np.float32)

    def test_action_mapping_and_gains_match_the_runtime_formulae(self):
        import yaml

        y = yaml.safe_load(open(os.path.join(H.HEFT_DIR, "policy.yaml")))
        names = y["policy_joint_names"]
        idx, _, default = self.resolve(
            y["default_joint_pos"], y["joint_names_simulation"], preserve_order=True, strict=False
        )
        d_sim = np.zeros(29)
        d_sim[idx] = default
        ids, _, values = self.resolve(y["action_scale"], names, preserve_order=True)
        scale = np.ones(29)
        scale[ids] = values  # as BasePolicy.__init__ scatters them
        ids, _, values = self.resolve(y["joint_kp"], y["joint_names_simulation"], preserve_order=True)
        kp = np.zeros(29)
        kp[ids] = values
        ids, _, values = self.resolve(y["joint_kd"], y["joint_names_simulation"], preserve_order=True)
        kd = np.zeros(29)
        kd[ids] = values
        # BasePolicy.setup_policy: q_target = default (sim order); q_target[controlled] += action * scale
        controlled = [y["joint_names_simulation"].index(n) for n in names]
        action = np.random.default_rng(3).normal(size=29).astype(np.float32)
        q = d_sim.copy()
        q[controlled] += action * np.asarray(scale)
        sim_index = {n: i for i, n in enumerate(y["joint_names_simulation"])}
        ours = H.target_of(self.man, action)
        np.testing.assert_allclose(ours, q[controlled], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(self.man.kp, np.asarray(kp)[[sim_index[n] for n in names]], rtol=0, atol=0)
        np.testing.assert_allclose(self.man.kd, np.asarray(kd)[[sim_index[n] for n in names]], rtol=0, atol=0)

    @unittest.skipUnless(os.path.exists(os.path.join(H.HEFT_DIR, "policy.onnx")), "weights not downloaded")
    def test_policy_call_returns_the_runtime_target_for_its_observation(self):
        import onnxruntime as ort

        pol = H.HeftPolicy()
        sess = ort.InferenceSession(os.path.join(H.HEFT_DIR, "policy.onnx"), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(11)
        env, theirs = self._lml_obs()
        theirs.reset()
        pol.reset(None)
        sp, F = env.state_processor, pol.F
        prev = np.zeros(29, np.float32)
        for _ in range(8):
            sp.joint_pos, sp.joint_vel = rng.normal(size=29) * 0.3, rng.normal(size=29)
            sp.root_quat_w, sp.root_ang_vel_b = _quat(rng, 0.2), rng.normal(size=3)
            joints = rng.normal(size=(F, 29)) * 0.3
            pos = np.cumsum(rng.normal(size=(F, 3)) * 0.01, axis=0) + [0.0, 0.0, 0.8]
            quat = np.array([_quat(rng, 0.2) for _ in range(F)])
            env.set_reference(joints, pos, quat)
            theirs.update({"action": prev})
            parts = theirs._build_obs_parts()
            action = sess.run(["action"], {k: parts[k][None, :] for k in H.OBS_KEYS})[0][0]
            fut = H.__dict__.get("Fut", None) or type("Fut", (dict,), {})()
            fut.poses = [(pos[i], quat[i], joints[i]) for i in range(F)]
            st = dict(root_quat=sp.root_quat_w, ang_vel=sp.root_ang_vel_b, dof_pos=sp.joint_pos[pol.mj_to_policy],
                      dof_vel=sp.joint_vel[pol.mj_to_policy])  # fmt: skip
            target = pol(st, fut)
            np.testing.assert_allclose(pol.last_action, action, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(target, H.target_of(self.man, action), rtol=1e-4, atol=1e-5)
            prev = action.astype(np.float32)


if __name__ == "__main__":
    unittest.main()
