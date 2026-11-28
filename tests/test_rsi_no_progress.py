"""TODO 56.3 (user 2026-11-27, main's TODO 54): the loop stops, normally, after `stop_no_progress` iterations in a
row in which no gate call was judged on the full pool; voided iterations are not counted, a `stop` row restarts the
count, a `pause` does not; the stop holds in an alternating run; the setting comes from the command line, the family
file or the resume."""

import os
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scene_loop():
    names = ("gate", "batch", "family", "scene_round", "digest", "loop", "merge", "verdict")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return _load("scene_rsi_loop_56_3", HERE / "loop.py")
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def gate(n, **kw):
    return dict(step="gate", iter=n, hash=f"h{n}", **kw)


FULL = dict(accepted=False, reason="net wins 1 < 3", result=dict(b=5, c=4, candidate_ok=10))
STAGE1 = dict(accepted=False, reason="stage 1: net wins 0 <= 0 on 30 pairs", stage1=dict(b=2, c=2))


class Progress(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()

    def test_progress_is_a_verdict_on_the_full_pool(self):
        p = self.lp.progressed
        self.assertTrue(p([gate(1, **FULL)], 1))
        self.assertTrue(p([gate(1, accepted=True, reason="accepted", result=dict(b=9, c=0))], 1))
        self.assertTrue(p([gate(1, **STAGE1), gate(1, shot=2, **FULL)], 1))  # the second shot counts
        self.assertFalse(p([gate(1, **STAGE1)], 1))
        self.assertFalse(p([gate(1, accepted=False, void=True, reason="pairing incomplete: ...", result=None)], 1))
        self.assertFalse(p([gate(1, accepted=False, dry_run=True, reason="dry run", result=dict(b=1, c=0))], 1))
        self.assertFalse(p([gate(2, **FULL)], 1))  # another iteration's
        self.assertFalse(p([dict(step="compose", iter=1, decision="keep")], 1))

    def test_the_stop_counts_the_newest_unvoided_iterations_since_the_last_stop_row(self):
        lp = self.lp
        rows = [dict(step="review", iter=0), gate(0, **FULL), dict(step="review", iter=1), gate(1, **STAGE1),
                dict(step="review", iter=2), dict(step="compose", iter=2, decision="keep")]  # fmt: skip
        with mock.patch.object(lp, "STOP_NO_PROGRESS", 2):
            self.assertEqual(lp.no_progress_stop(rows),
                             "no progress: iterations 1, 2 had no candidate past stage 1 (2 in a row)")  # fmt: skip
            self.assertIsNone(lp.no_progress_stop(rows[:4]))  # iteration 1 alone: the streak is 1
            self.assertIsNone(lp.no_progress_stop([*rows, dict(step="review", iter=3), gate(3, **FULL)]))
            # a voided iteration is not counted: 1 (no progress), 2 void, 3 (no progress) -> 2 in a row
            voided = [*rows[:4], dict(step="review", iter=2), dict(step="void_iter", iter=2, role="manager"),
                      dict(step="review", iter=3), gate(3, **STAGE1)]  # fmt: skip
            self.assertEqual(lp.no_progress_stop(voided),
                             "no progress: iterations 1, 3 had no candidate past stage 1 (2 in a row)")  # fmt: skip
            self.assertIsNone(lp.no_progress_stop(voided[:-2]))  # 1 and a void 2: the streak is still 1
            # a `pause` does not reset the count; a `stop` row does
            paused = [*rows[:4], dict(step="pause", reason="--max-iters 1 done"), *rows[4:]]
            self.assertTrue(lp.no_progress_stop(paused))
            stopped = [*rows[:4], dict(step="stop", reason="manager: stop"), *rows[4:]]
            self.assertIsNone(lp.no_progress_stop(stopped))
        with mock.patch.object(lp, "STOP_NO_PROGRESS", 0):
            self.assertIsNone(lp.no_progress_stop(rows))
        with mock.patch.object(lp, "STOP_NO_PROGRESS", 3):
            self.assertIsNone(lp.no_progress_stop(rows))
            self.assertTrue(lp.no_progress_stop([*rows, dict(step="review", iter=3), gate(3, **STAGE1)]))


class NoProgressStop(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        root = Path(self.d.name)
        (root / "pool_diag").mkdir()
        hid = root / "outputs" / "scene_rsi_20261016_hidden" / "fam"
        hid.mkdir(parents=True)
        for p in (mock.patch.multiple(self.lp, STATE=root / "state", LOOP=root, HIDDEN=hid, GATE_DIR=hid / "gate",
                                      accept_pool=lambda: root, STOP_NO_PROGRESS=2),
                  mock.patch.dict(os.environ)):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("RSI_MAX_WALL_H", None)
        (root / "state").mkdir()
        self.meta = dict(started="2026-11-27T00:00:00Z", started_s=time.time())

    def test_two_iterations_without_progress_stop_the_loop_normally_also_when_alternating(self):
        lp = self.lp
        lp.ledger(dict(step="review", iter=0))
        lp.ledger(gate(0, **STAGE1))
        self.assertIsNone(lp.stop_reason(self.meta))
        lp.ledger(dict(step="review", iter=1))
        lp.ledger(dict(step="compose", iter=1, decision="keep"))
        why = lp.stop_reason(self.meta)
        self.assertEqual(why, "no progress: iterations 0, 1 had no candidate past stage 1 (2 in a row)")
        self.assertFalse(lp.is_abnormal(why))  # a normal stop: the sealed set runs
        with mock.patch.object(lp, "ALTERNATING", True):
            self.assertEqual(lp.stop_reason(self.meta), why)
        with mock.patch.object(lp, "STOP_NO_PROGRESS", 0):
            self.assertIsNone(lp.stop_reason(self.meta))

    def test_the_setting_comes_from_the_cli_the_family_file_or_the_resume(self):
        lp = self.lp
        none = Namespace(stop_no_progress=None)
        with mock.patch.object(lp, "family_spec", lambda: {}):
            self.assertEqual(lp.resolve_settings(none, {}, False)["stop_no_progress"], 0)
            self.assertEqual(
                lp.resolve_settings(none, dict(started="x", stop_no_progress=3), True)["stop_no_progress"], 3
            )
            self.assertEqual(
                lp.resolve_settings(none, dict(started="x", stop_no_progress=3), False)["stop_no_progress"], 0
            )
            self.assertEqual(lp.resolve_settings(Namespace(stop_no_progress=4), {}, False)["stop_no_progress"], 4)
        with (
            mock.patch.object(lp, "family_spec", lambda: dict(stop_no_progress=2)),
            mock.patch.object(lp, "FAMILY", "f"),
        ):
            self.assertEqual(lp.resolve_settings(none, {}, False)["stop_no_progress"], 2)
            self.assertEqual(lp.resolve_settings(Namespace(stop_no_progress=0), {}, False)["stop_no_progress"], 0)
        seen = {}
        with (
            mock.patch.object(lp, "cmd_run", lambda a: seen.update(vars(a)) or 0),
            mock.patch.object(lp, "configure", lambda f: None),
        ):
            lp.main(["run", "gso_single_grasp", "--stop-no-progress", "3"])
            self.assertEqual(seen["stop_no_progress"], 3)
        for fam in ("gso_single_grasp", "gso_bimanual_lift"):
            import json

            self.assertEqual(json.loads((HERE / "families" / f"{fam}.json").read_text())["stop_no_progress"], 2)


if __name__ == "__main__":
    unittest.main()
