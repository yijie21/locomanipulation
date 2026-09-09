"""Exercise the real servo at a captured pre-grasp handover pose."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from locomani.backends.sonic import stream_reach as sr
from locomani.backends.sonic.primitives import Primitives


class FirstCommand(Exception):
    pass


class TransportRotationTests(unittest.TestCase):
    def test_far_pregrasp_keeps_tips_up_even_with_tiny_tracking_error(self):
        st = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
        rotation = Primitives._rot(st["pelvis"]["quat"])
        pel = np.array(st["pelvis"]["pos"])
        palm = np.array(st["palms"]["right"]["pos"])
        # Horizontal final approach, with the pregrasp 30 cm away. The
        # first incremental reference is only millimetres from the palm.
        pre = palm + np.array([0.30, 0.0, 0.0])
        plan = SimpleNamespace(rot=np.eye(3), wrist=pre + np.array([sr.SAFE_GAP, 0.0, 0.0]))
        p = SimpleNamespace(
            obs=lambda: st,
            state=SimpleNamespace(latest=lambda _: st),
            object_pos=lambda key, s: np.array(s["objects"][key]["pos"]),
            world_to_root_full=lambda point, s: rotation.T @ (point - pel),
            log=lambda _: None,
        )
        target = []

        def solve(side, pos, rot, **kw):
            target.append(rotation @ rot)
            return np.array(st["joints"]), 0.0, 0.0

        def push(q):
            raise FirstCommand()

        arm = SimpleNamespace(q=np.array(st["joints"]), ik=SimpleNamespace(solve=solve), _push=push)
        with self.assertRaises(FirstCommand):
            sr._servo_continuous(p, arm, "water_bottle", plan, "right")
        self.assertGreater(
            float(target[0][2, 0]), 0.7, "far from pregrasp: fingertip axis must point up, independent of tracking lag"
        )


if __name__ == "__main__":
    unittest.main()
