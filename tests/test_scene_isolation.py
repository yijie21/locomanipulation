"""Scene selection is shared by path consumers and the IK default."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SceneIsolationTests(unittest.TestCase):
    def test_private_scene_and_ik_path_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            env = dict(
                os.environ,
                LOCOMANI_SIM_DIR=folder,
                PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src/teleop_free"),
            )
            data = json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-c",
                        "import json,paths,arm_ik;print(json.dumps([paths.SCENE_META,paths.SCENE_XML,arm_ik.ArmIK.__init__.__defaults__[0]]))",
                    ],
                    env=env,
                    text=True,
                )
            )
            self.assertEqual(
                data, [folder + "/scene_meta.json", folder + "/scene_locomani.xml", folder + "/scene_locomani.xml"]
            )


if __name__ == "__main__":
    unittest.main()
