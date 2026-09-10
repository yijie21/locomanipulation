"""Exercise the real structured-config serialization used at train startup."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/teleop_free/scalebfm'))

from omegaconf import OmegaConf
from train_visual_vla import build_config


class TrainingConfigTest(unittest.TestCase):
    def test_fresh_and_resume_configs_serialize_at_real_run_boundary(self):
        for resume in (None, '/tmp/visual-training-test/checkpoint-500'):
            with self.subTest(resume=resume):
                config = build_config('/tmp/visual-dataset', '/tmp/visual-training-test', 17, resume)
                saved = OmegaConf.create(config.__dict__)
                self.assertIs(saved.training.resume_from_checkpoint, bool(resume))
                self.assertEqual(saved.training.max_steps, 3000)


if __name__ == '__main__':
    unittest.main()
