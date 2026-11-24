"""TODO 49.2: SimpleConfig keeps today's values, takes --set, rejects unknown fields; the CLI refuses switch values outside
the loops' set and ids outside the practice pools."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from locomani.bench.simple import __main__ as cli
from locomani.bench.simple.config import SimpleConfig

ESC = "/workspace/code/locomanipulation-esc/outputs/escalation_20261116"
MAIN = "/workspace/code/locomanipulation"


class Config(unittest.TestCase):
    def test_defaults_are_the_old_beds_values(self):
        c = SimpleConfig()
        self.assertEqual(c.practice_dir, ESC + "/practice2")
        self.assertEqual(c.dev_dir, ESC + "/practice")
        self.assertEqual(c.cache_dir, ESC + "/official/cloud_q")
        self.assertEqual(c.slots_dir, ESC + "/official/slots")  # shared with the esc bed
        self.assertEqual(c.slots, (1, 2, 3))
        self.assertEqual((c.isaac_gpu, c.depth_gpu, c.graspgen_gpu), (0, 1, 1))
        self.assertEqual((c.sam3_host, c.sam3_port, c.sam3()), ("127.0.0.1", 18832, ("127.0.0.1", 18832)))
        self.assertFalse(c.lazy_render)
        self.assertEqual(c.kinds, "Q")
        self.assertEqual(c.xdir, "/nonexistent")
        self.assertEqual(c.env_names, ESC + "/official/env_names.json")
        self.assertEqual(c.simple_python, MAIN + "/third_party/SIMPLE/.venv/bin/python")
        self.assertEqual(c.sim_python, MAIN + "/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python")
        self.assertEqual(c.sam3d_python, "/workspace/miniconda3/envs/sam3d/bin/python")
        self.assertEqual(c.graspgen_python, MAIN + "/third_party/GraspGenX/.venv/bin/python")
        self.assertEqual(c.hf_home, "/workspace/huggingface_cache")
        self.assertEqual((c.batch_tries, c.look_tries), (4, 3))

    def test_set_overrides_and_roundtrip(self):
        c = SimpleConfig().with_overrides(["slots=2,3", "lazy_render=true", "kinds=PQ", "sam3_port=18900"])
        self.assertEqual((c.slots, c.lazy_render, c.kinds, c.sam3_port), ((2, 3), True, "PQ", 18900))
        self.assertEqual(SimpleConfig.from_dict(json.loads(json.dumps(c.to_dict()))), c)

    def test_unknown_field_and_wrong_type_rejected(self):
        with self.assertRaises(ValueError):
            SimpleConfig().with_overrides(["no_such_field=1"])
        with self.assertRaises(ValueError):
            SimpleConfig.from_dict(dict(SimpleConfig().to_dict(), nope=1))
        with self.assertRaises(ValueError):
            SimpleConfig.from_dict(dict(SimpleConfig().to_dict(), isaac_gpu="zero"))


class Cli(unittest.TestCase):
    def run_cli(self, *argv):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as t, redirect_stderr(err):
            ids = Path(t) / "ids.txt"
            ids.write_text(self.ids)
            rc = cli.main(["run", "XMovePickTeleop", "--ids", str(ids), "--out", str(Path(t) / "rows.jsonl"), *argv])
        return rc, err.getvalue()

    ids = "XMovePickTeleop/Q/0\n"

    def test_switch_values_outside_the_loops_are_refused_with_a_pointer(self):
        for kv, pointer in (
            ("target=truth", "49.2 step 6"),
            ("sensor=rgbd", "49.2 step 6"),
            ("depth_noise=realsense", "49.2 step 6"),
            ("self_pose=truth", "49.3"),
            ("controller_root=truth", "49.3"),
        ):
            sets = ["--bench-set", kv]
            if kv.startswith("depth_noise"):
                sets += ["--bench-set", "sensor=rgbd"]
            rc, err = self.run_cli(*sets)
            self.assertEqual(rc, 2, kv)
            self.assertIn(pointer, err, kv)

    def test_released_and_foreign_ids_are_refused_before_any_slot_is_taken(self):
        type(self).ids = "XMovePickTeleop/L0/e0\n"
        try:
            rc, err = self.run_cli("--set", "slots_dir=/nonexistent/never")
            self.assertEqual(rc, 2)
            self.assertIn("exam", err)
            type(self).ids = "XMovePickTeleop/P/0\n"  # the dev pool is not open unless kinds says so
            rc, err = self.run_cli("--set", "slots_dir=/nonexistent/never")
            self.assertEqual(rc, 2)
            self.assertIn("kinds", err)
        finally:
            type(self).ids = "XMovePickTeleop/Q/0\n"

    def test_bad_arguments(self):
        rc, err = self.run_cli("--set", "bogus=1")
        self.assertEqual(rc, 2)
        rc, err = self.run_cli("--setting", "L2")
        self.assertEqual(rc, 2)
        rc, err = self.run_cli("--setting", "{not json")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
