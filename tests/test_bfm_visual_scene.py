import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/teleop_free/scalebfm'))
import run_bottle as B
from visual_scene import make_sim, scene_directory


class SceneTests(unittest.TestCase):
    def test_taller_table_collision_matches_metadata_and_supports_bottle(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = scene_directory(tmp, 'high_near')
            policy = SimpleNamespace(meta=json.loads(Path(B.R.META).read_text()))
            sim = make_sim(policy, directory)
            sim.reset([-1.5, 0], 0, 0)
            geom = sim.m.geom('dining_table_dining_table_top').id
            top = sim.d.geom_xpos[geom, 2] + sim.m.geom_size[geom, 2]
            self.assertAlmostEqual(top, sim.table_top, places=5)
            initial = sim.crate()[0][2]
            for _ in range(200):
                mujoco.mj_step(sim.m, sim.d)
            mujoco.mj_forward(sim.m, sim.d)
            self.assertLess(abs(sim.crate()[0][2] - initial), .005)


if __name__ == '__main__':
    unittest.main()
