"""Regression tests for the 2026-09-10 review findings (docs/ledger/refactor.md).

Each test reproduces the defect the review found, so a reintroduction fails here on CPU:
Spec 1 config fail-fast, Spec 2 the ScaleBFM source fingerprint, Spec 3 the backend scene
directory, Standards 3 classify_failure over round.py's real `why` wording.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.pipeline.eval_config import EvalConfig
from locomani.pipeline.pilot_config import PilotConfig
from locomani.tasks.bottle_visual import classify_failure
from locomani.tasks.bottle_visual.config import BottleVisualTaskConfig
from locomani.tasks.bottle_visual.pipeline.provenance import line_sources, source_relpath


class ConfigFailFastTests(unittest.TestCase):
    def test_unknown_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown field"):
            EvalConfig.from_dict({"round": 100})

    def test_schema_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            EvalConfig.from_dict({"schema_version": 99})

    def test_scalar_type_is_checked(self):
        with self.assertRaisesRegex(ValueError, "rtc"):
            EvalConfig.from_dict({"rtc": "false"})
        with self.assertRaisesRegex(ValueError, "rounds"):
            EvalConfig.from_dict({"rounds": "20"})

    def test_enum_fields_reject_typos(self):
        with self.assertRaisesRegex(ValueError, "client"):
            EvalConfig().with_overrides(["client=asnyc"])
        with self.assertRaisesRegex(ValueError, "through"):
            PilotConfig().with_overrides(["through=collec"])
        with self.assertRaisesRegex(ValueError, "facing"):
            BottleVisualTaskConfig().with_overrides(["grasp.facing=table"])

    def test_valid_round_trips_still_pass(self):
        for cfg in (EvalConfig(), PilotConfig(), BottleVisualTaskConfig(), ScaleBfmBackendConfig()):
            self.assertEqual(type(cfg).from_dict(cfg.to_dict()), cfg)
        self.assertEqual(EvalConfig().with_overrides(["client=lockstep", "rtc=true"]).client, "lockstep")


class ScaleBfmProvenanceTests(unittest.TestCase):
    def test_line_sources_cover_backend_and_task(self):
        names = {p.name for p in line_sources()}
        self.assertTrue({"runtime.py", "network.py", "action.py", "round.py", "planner.py"} <= names)
        rels = [source_relpath(p) for p in line_sources()]
        self.assertEqual(len(rels), len(set(rels)))
        self.assertIn(Path("backends/scalebfm/config.py"), rels)
        self.assertIn(Path("tasks/bottle_visual/config.py"), rels)


class ScaleBfmSceneFromBackendTests(unittest.TestCase):
    def test_make_sim_uses_the_backend_scene_dir(self):
        from locomani.tasks.bottle_visual import scene

        seen = {}

        class FakeSim:
            def __init__(self, meta, video=False, backend=None):
                seen["backend"] = backend
                self.m = type("M", (), {"ngeom": 0})()

        with patch.object(scene.R, "Sim", FakeSim):
            scene.make_sim(type("P", (), {"meta": {}})(), None, ScaleBfmBackendConfig(scene_dir="/tmp/review-scene"))
        self.assertEqual(seen["backend"].scene_dir, "/tmp/review-scene")

    def test_scene_directory_reads_the_backend_scene(self):
        from locomani.tasks.bottle_visual import scene

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "scene_meta.json").write_text(
                json.dumps({"surfaces": {"A": {"top": scene.STRATA["low_near"][0]}}})
            )
            out = scene.scene_directory(Path(d) / "variants", "low_near", ScaleBfmBackendConfig(scene_dir=d))
        self.assertEqual(Path(out), Path(d))


class BottleVisualFailureLabelsTests(unittest.TestCase):
    def test_round_py_wordings_get_real_labels(self):
        # the literal `why` strings round.py emits (grep 'why =' round.py)
        self.assertEqual(classify_failure("timeout"), "timeout")
        self.assertEqual(classify_failure("physics warning/fall"), "fall")
        self.assertEqual(classify_failure("bottle height changed before reach: scene/contact setup"), "setup")
        self.assertEqual(classify_failure("bottle knocked before close"), "knock")
        self.assertEqual(classify_failure(""), "ok")
        self.assertEqual(classify_failure("something new"), "other")


if __name__ == "__main__":
    unittest.main()
