"""Every registered Task must pass the same contract (decision 3, 13; docs/reference/standards.md).

Parametrised over `locomani.tasks.registry.TASKS` with `subTest`, so adding a
third task automatically gets these checks with no new test code. Skips
nothing: a task with a gap (e.g. no separate acceptance function) must still
satisfy every assertion here, not opt out of one.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path

from locomani.tasks.bottle import grasp_gym
from locomani.tasks.registry import TASKS

REPO = Path(__file__).resolve().parents[1]


class TaskContractTests(unittest.TestCase):
    def test_config_cls_round_trips(self):
        for name, task in TASKS.items():
            with self.subTest(task=name):
                cfg = task.config_cls()
                # to_dict / from_dict
                self.assertEqual(task.config_cls.from_dict(cfg.to_dict()), cfg)
                # dump / load, through a real file
                path = Path(self._make_tmp_path(name))
                cfg.dump(path)
                self.assertEqual(task.config_cls.load(path), cfg)

    def _make_tmp_path(self, name: str) -> str:
        fd, path = tempfile.mkstemp(prefix=f"task_contract_{name}_", suffix=".json")
        os.close(fd)
        self.addCleanup(os.remove, path)
        return path

    def test_fields_are_callable_where_declared(self):
        for name, task in TASKS.items():
            with self.subTest(task=name):
                for field in ("sample_scene", "run_round", "acceptance", "classify_failure"):
                    self.assertTrue(callable(getattr(task, field)), f"{name}.{field} is not callable")

    def test_schema_modality_file_exists(self):
        for name, task in TASKS.items():
            with self.subTest(task=name):
                self.assertTrue(
                    (REPO / task.schema.modality_config_path).exists(),
                    f"{name}: {task.schema.modality_config_path} does not exist",
                )
                self.assertTrue(task.schema.video_keys)
                self.assertTrue(task.schema.state_keys)
                self.assertTrue(task.schema.action_keys)

    def test_classify_failure_labels_every_failure_reason(self):
        for name, task in TASKS.items():
            with self.subTest(task=name):
                self.assertTrue(task.failure_reasons, f"{name}: failure_reasons is empty")
                for reason in task.failure_reasons:
                    why = f"[{reason}] synthetic failure for the contract test"
                    label = task.classify_failure(why)
                    self.assertIsNotNone(label, f"{name}: classify_failure({why!r}) returned None")
                    self.assertNotEqual(label, "unknown", f"{name}: classify_failure({why!r}) returned 'unknown'")

    def test_bottle_run_round_signature_accepts_the_task_config(self):
        params = inspect.signature(grasp_gym.run_round).parameters
        self.assertIn("reach_config", params, "grasp_gym.run_round must accept a reach_config (BottleTaskConfig.reach)")


if __name__ == "__main__":
    unittest.main()
