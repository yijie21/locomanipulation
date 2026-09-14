"""Real MuJoCo sweeps catch mid-path contact, without altering IK state."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np

from locomani.backends.sonic.primitives import Primitives
from locomani.pipeline.execution_record import ExecutionRecord
from locomani.tasks.bottle import execution as sr
from locomani.tasks.bottle.transport_guard import SweepResult, TransportGuard

XML = """<mujoco><worldbody>
<body name="pelvis"><freejoint/><geom size=".01" mass="1" contype="0" conaffinity="0"/>
<body name="right_shoulder_pitch_link"><joint type="hinge" axis="0 0 1"/>
<geom type="sphere" pos="1 0 0" size=".05" mass="1"/></body></body>
<body name="target" pos="1 0 0"><freejoint/><geom type="sphere" size=".04" mass=".1"/></body>
</worldbody></mujoco>"""


class TransportGuardTests(unittest.TestCase):
    def setup_scene(self, q=-0.5, target=(1.0, 0.0, 0.0)):
        model = mujoco.MjModel.from_xml_string(XML)
        ik = SimpleNamespace(model=model, data=mujoco.MjData(model), base_qadr=0, qadr=np.array([7]))
        state = dict(
            joints=[q],
            pelvis=dict(pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]),
            palms={"right": dict(hand_q=[])},
            objects={"bottle": dict(pos=list(target), quat=[1.0, 0.0, 0.0, 0.0])},
        )
        return ik, state

    def test_clear_endpoints_do_not_imply_clear_transport(self):
        ik, state = self.setup_scene()
        guard = TransportGuard(ik, "bottle", "target")
        self.assertTrue(guard.check(state, [-0.5], "right").clear)
        end = deepcopy(state)
        end["joints"] = [0.5]
        self.assertTrue(guard.check(end, [0.5], "right").clear)
        result = guard.check(state, [0.5], "right")
        self.assertFalse(result.clear)
        self.assertTrue(0 < result.fraction < 1)
        self.assertEqual(result.link, "right_shoulder_pitch_link")

    def test_base_translation_checks_midpath_even_when_joint_reference_is_constant(self):
        ik, state = self.setup_scene(0.0, (1.0, 0.0, 0.0))
        state["pelvis"]["pos"] = [-0.3, 0.0, 0.0]
        guard = TransportGuard(ik, "bottle", "target")
        self.assertTrue(guard.check(state, [0.0], "right").clear)
        hit = guard.check(state, [0.0], "right", candidate_base=[0.3, 0, 0, 1, 0, 0, 0])
        self.assertFalse(hit.clear)
        self.assertTrue(0 < hit.fraction < 1)
        self.assertGreater(hit.samples, 2)
        self.assertEqual(state["pelvis"]["pos"], [-0.3, 0.0, 0.0])

    def test_margin_and_private_model(self):
        ik, state = self.setup_scene(0.0, (1.093, 0.0, 0.0))
        saved = ik.model.geom_margin.copy()
        q = ik.data.qpos.copy()
        self.assertTrue(TransportGuard(ik, "bottle", "target", margin=0.0).check(state, [0.0], "right").clear)
        self.assertFalse(TransportGuard(ik, "bottle", "target", margin=0.004).check(state, [0.0], "right").clear)
        np.testing.assert_array_equal(ik.model.geom_margin, saved)
        np.testing.assert_array_equal(ik.data.qpos, q)

    def test_finger_only_capture_does_not_exempt_arm_target_collision(self):
        ik, state = self.setup_scene(0.0)
        guard = TransportGuard(ik, "bottle", "target")
        # Opt-in contact semantics: a capture allowance must not turn the
        # whole shoulder/forearm/wrist into allowed bottle contact geometry.
        guard.finger_only_target_contact = True
        self.assertFalse(guard.check(state, [0.0], "right", allow_target_contact=True).clear)

    def test_legacy_target_contact_allowance_is_preserved(self):
        ik, state = self.setup_scene(0.0)
        self.assertTrue(
            TransportGuard(ik, "bottle", "target").check(state, [0.0], "right", allow_target_contact=True).clear
        )

    def test_finger_only_capture_allows_the_actual_finger_contact(self):
        ik, state = self.setup_scene(0.0)
        xml = XML.replace(
            '<geom type="sphere" pos="1 0 0" size=".05" mass="1"/>',
            '<body name="right_dex1_finger_1_link"><geom type="sphere" pos="1 0 0" size=".05" mass="1"/></body>',
        )
        ik.model = mujoco.MjModel.from_xml_string(xml)
        guard = TransportGuard(ik, "bottle", "target", finger_only_target_contact=True)
        self.assertTrue(guard.check(state, [0.0], "right", allow_target_contact=True).clear)
        self.assertFalse(guard.check(state, [0.0], "right").clear)

    def test_changed_table_height_is_used_by_offline_collision_query(self):
        xml = XML.replace(
            "</worldbody>", '<body name="table" pos="1 0 -.5"><geom type="box" size=".1 .1 .05"/></body></worldbody>'
        )
        ik, state = self.setup_scene(0.0, (4.0, 0.0, 0.0))
        ik.model = mujoco.MjModel.from_xml_string(xml)
        guard = TransportGuard(ik, "bottle", "target", ["table"])
        self.assertTrue(guard.check(state, [0.0], "right").clear)
        guard.model.body_pos[guard.model.body("table").id, 2] = 0.0
        self.assertFalse(guard.check(state, [0.0], "right").clear)

    def test_pregrasp_shortcut_requires_measured_delivery(self):
        st = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
        palm = np.array(st["palms"]["right"]["pos"])
        plan = SimpleNamespace(rot=np.eye(3), wrist=palm + [0.3 + sr.SAFE_GAP, 0.0, 0.0])
        robot = SimpleNamespace(obs=lambda: st, object_pos=lambda k, s: np.array(s["objects"][k]["pos"]))
        arm = SimpleNamespace(_push=Mock())
        ok, why = sr._servo_continuous(robot, arm, "water_bottle", plan, "right", start_at_pregrasp=True)
        self.assertFalse(ok)
        self.assertIn("not delivered", why)
        arm._push.assert_not_called()

    def test_servo_rejects_before_publishing_colliding_reference(self):
        st = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
        root = Primitives._rot(st["pelvis"]["quat"])
        pel = np.array(st["pelvis"]["pos"])
        palm = np.array(st["palms"]["right"]["pos"])
        plan = SimpleNamespace(rot=np.eye(3), wrist=palm + [0.3 + sr.SAFE_GAP, 0.0, 0.0])
        p = SimpleNamespace(
            obs=lambda: st,
            state=SimpleNamespace(latest=lambda _: st),
            object_pos=lambda key, s: np.array(s["objects"][key]["pos"]),
            world_to_root_full=lambda point, s: root.T @ (point - pel),
            log=lambda _: None,
        )
        arm = SimpleNamespace(
            q=np.array(st["joints"]),
            ik=SimpleNamespace(solve=lambda *a, **kw: (np.array(st["joints"]), 0.0, 0.0)),
            _push=Mock(),
        )
        guard = SimpleNamespace(check=lambda *a: SweepResult(False, 0.002, "right_elbow_link", "target", 0.5, 4))
        record = ExecutionRecord()
        ok, why = sr._servo_continuous(p, arm, "water_bottle", plan, "right", execution=record, transport_guard=guard)
        self.assertFalse(ok)
        self.assertIn("right_elbow_link", why)
        arm._push.assert_not_called()
        self.assertEqual(record.metrics["transport_guard"]["checks"], 1)


if __name__ == "__main__":
    unittest.main()
