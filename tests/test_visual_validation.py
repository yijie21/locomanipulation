"""Integration regression against a real exported visual-validation fixture."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

# torch tier (tests/conftest.py): the GR00T venv's packages, skipped elsewhere.
pd = pytest.importorskip("pandas")
pytest.importorskip("gr00t")

ROOT = Path(__file__).resolve().parents[1]
for path in ("src", "configs"):
    sys.path.insert(0, str(ROOT / path))

import gr00t_bfm_visual_2cam

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from locomani.tasks.bottle_visual.pipeline.eval import validation_diagnostic

FIXTURE = Path(
    os.environ.get(
        "SCALEBFM_VALIDATION_FIXTURE", str(ROOT / "outputs/experiments/scalebfm_visual_goal_20260908/dataset_A_val")
    )
)


@unittest.skipUnless(
    (FIXTURE / "meta/info.json").exists(),
    "Set SCALEBFM_VALIDATION_FIXTURE to an exported 20-episode validation dataset",
)
class VisualValidationTest(unittest.TestCase):
    def test_real_loader_retains_labels_without_exposing_them_to_predictor(self):
        calls = []

        def predict(observation):
            self.assertTrue(any(k.startswith("video.") for k in observation))
            self.assertTrue(any(k.startswith("state.") for k in observation))
            self.assertFalse(any(k.startswith("action.") for k in observation))
            calls.append(1)
            return np.zeros((40, 29), dtype=np.float32)

        predict.modality_config = MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1_SONIC.value]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "validation.json"
            validation_diagnostic(predict, FIXTURE, out)
            report = json.loads(out.read_text())
        self.assertEqual(len(calls), 60)
        self.assertEqual(report["samples"], 60)
        self.assertIn("action", predict.modality_config)
        for case in report["cases"]:
            episode = case["episode"]
            raw = pd.read_parquet(FIXTURE / f"data/chunk-000/episode_{episode:06d}.parquet")
            start = case["frame"]
            truth = np.stack(raw["action.visual_command"].iloc[start : start + 40])
            self.assertTrue(case["finite"])
            np.testing.assert_allclose(case["per_dimension_mae"], abs(truth).mean(0), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
