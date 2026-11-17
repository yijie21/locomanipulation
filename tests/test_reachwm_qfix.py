"""The self-model's "plan joints minus start joints" feature (reachwm data.py, REACHWM_QFIX): the start joints come in
ScaleBFM's metadata order, the plan's in MuJoCo order, and the switch reorders the first into the second. Runs a fresh
interpreter per switch value, since data.py reads its switches at import."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REACH = ROOT / "scripts" / "experiments" / "reachwm_20261030"

PROBE = """
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import data as DT
import plan_invert as PI
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
names = json.load(open(DT.E.REPO / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json"))["joint_names"]
q0 = np.arange(29, dtype=np.float32)  # joint k of S's order holds the value k
rec = dict(ref_q=np.zeros(29, np.float32), ref_pel=np.r_[0, 0, 0, 1, 0, 0, 0].astype(np.float32), ref_s=1.0)
f = PI.plan_features(rec, q0, np.zeros(4, np.float32))
moved = -f[29:58]  # ref - q0 with ref = 0: the start joints as the feature orders them
want = [float(names.index(n)) for n in MUJOCO_JOINT_NAMES]
print(json.dumps(dict(qfix=DT.QFIX, as_mujoco=moved.tolist() == want, unchanged=moved.tolist() == q0.tolist())))
"""


def probe(qfix: str) -> dict:
    env = dict(os.environ, REACHWM_QFIX=qfix, REACHWM_GOAL_REF="1")
    out = subprocess.check_output([sys.executable, "-c", PROBE, str(REACH)], env=env, cwd=ROOT, text=True)
    return json.loads(out.strip().splitlines()[-1])


@unittest.skipUnless((ROOT / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json").exists(), "needs assets/")
class QFix(unittest.TestCase):
    def test_off_keeps_the_old_feature(self):
        r = probe("0")
        self.assertFalse(r["qfix"])
        self.assertTrue(r["unchanged"])

    def test_on_puts_the_start_joints_in_mujoco_order(self):
        r = probe("1")
        self.assertTrue(r["qfix"])
        self.assertTrue(r["as_mujoco"])
        self.assertFalse(r["unchanged"])

    def test_a_model_refuses_the_other_feature(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import data as DT\n"
            "class M: pass\n"
            "m = M(); m.qfix = True\n"
            "try:\n    DT.check_qfix(m); print('accepted')\nexcept AssertionError:\n    print('refused')\n"
        )
        env = dict(os.environ, REACHWM_QFIX="0")
        out = subprocess.check_output([sys.executable, "-c", code, str(REACH)], env=env, cwd=ROOT, text=True)
        self.assertEqual(out.strip().splitlines()[-1], "refused")


if __name__ == "__main__":
    unittest.main()
