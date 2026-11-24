"""TODO 49.2: the skill package is loaded by path under a unique module name; two copies load side by side; the look cache key."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from locomani.bench.simple import run as bed
from locomani.bench.simple.config import SEED_DIR


class Loading(unittest.TestCase):
    def copy(self, t, name, edit=None):
        d = Path(t) / name
        shutil.copytree(SEED_DIR, d, ignore=shutil.ignore_patterns("__pycache__"))
        if edit:
            p = d / edit[0]
            p.write_text(p.read_text().replace(edit[1], edit[2]))
        return d

    def test_two_copies_side_by_side(self):
        with tempfile.TemporaryDirectory() as t:
            a = self.copy(t, "a")
            b = self.copy(t, "b", ("skill.py", "reach1_speed=0.15", "reach1_speed=0.20"))
            before = list(sys.path)
            pa, pb = bed.load_package(a), bed.load_package(b)
            self.assertEqual(sys.path, before)  # nothing is put on sys.path
        self.assertNotEqual(pa.__name__, pb.__name__)
        self.assertEqual(pa.skill.K_ORIG["reach1_speed"], 0.15)
        self.assertEqual(pb.skill.K_ORIG["reach1_speed"], 0.20)
        self.assertIsNot(pa.pick, pb.pick)
        self.assertIsNot(pa.skill.P, pb.skill.P)  # each copy's `from . import pick` is its own
        self.assertEqual(pa.parameters("CloseDoorTeleop")["art_obj"], "door")

    def test_missing_file_is_an_error_and_leaves_nothing_loaded(self):
        with tempfile.TemporaryDirectory() as t:
            d = self.copy(t, "c")
            (d / "pick.py").unlink()
            n = len(sys.modules)
            with self.assertRaises(FileNotFoundError):
                bed.load_package(d)
            self.assertEqual(len(sys.modules), n)

    def test_failed_import_cleans_up(self):
        with tempfile.TemporaryDirectory() as t:
            d = self.copy(t, "d", ("skill.py", "SIDE = ", "raise RuntimeError('boom')\nSIDE = "))
            with self.assertRaises(RuntimeError):
                bed.load_package(d)
            self.assertFalse(
                [k for k in sys.modules if k.startswith("_simple_skill_") and k.endswith("skill") and "boom" in k]
            )

    def test_the_seed_covers_all_nine_tasks(self):
        pkg = bed.load_package(SEED_DIR)
        S = pkg.skill
        for task in ("BendPickMP", "XMovePickTeleop", "LocomotionPickBetweenTablesTeleop", "HandoverTeleop"):
            self.assertIn("stand", S.defaults(task, S.K_CUT))
        self.assertEqual(
            set(S.ART_TASK), {"OpenOvenTeleop", "CloseDoorTeleop", "OpenFaucetTeleop", "OpenTrashCanTeleop"}
        )
        for task in S.ART_TASK:
            self.assertIn("art_sweep", S.defaults(task, S.K_CUT))
        self.assertTrue(hasattr(pkg.camera_view.CameraView, "art"))  # the articulated additions
        self.assertNotIn("art_sweep", S.defaults("XMovePickTeleop", S.K_CUT))

    def test_cache_key_follows_the_camera_view_and_the_look_parameters(self):
        with tempfile.TemporaryDirectory() as t:
            a = bed.load_package(self.copy(t, "a"))
            b = bed.load_package(self.copy(t, "b", ("camera_view.py", "search = True", "search = True  # edited")))
            k = bed.cache_key(a, "XMovePickTeleop", {})
            self.assertEqual(len(k), 10)
            self.assertEqual(
                k, bed.cache_key(a, "XMovePickTeleop", {"lift": 0.2})
            )  # a parameter the look does not read
            self.assertNotEqual(k, bed.cache_key(a, "XMovePickTeleop", {"start_hold": 2.0}))
            self.assertNotEqual(k, bed.cache_key(a, "XMovePickTeleop", {"feed_torque": True}))
            self.assertNotEqual(k, bed.cache_key(b, "XMovePickTeleop", {}))

    def test_comply_fields(self):
        r = bed.comply_fields("XMovePickTeleop", False, None)
        self.assertEqual(r["comply"], dict(ok=False, why="not_success"))
        self.assertEqual(set(r["comply_facts"]), set(bed.COMPLY_FACTS))
        self.assertTrue(set(bed.ART_FACTS) <= set(bed.comply_fields("CloseDoorTeleop", False, None)["comply_facts"]))

    def test_scope(self):
        from locomani.bench import options

        ok = options.resolve("simple", allow_planned=True).options
        bed.check_scope(ok)
        for kv in ("controller_root=truth", "self_pose=truth"):  # TODO 49.3: both switches take both values
            bed.check_scope(options.resolve("simple", [kv], allow_planned=True).options)


if __name__ == "__main__":
    unittest.main()
