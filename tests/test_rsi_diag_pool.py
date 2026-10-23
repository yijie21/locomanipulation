"""TODO 33.23 (user 2026-10-23 "先做 33.23"): the diagnostic pool is re-drawn when the incumbent saturates it --
fresh starts from the diagnostic seed range, the incumbent runs on them, the failing ones replace passing starts until
half the pool fails; every task keeps two starts and a passing one, parked starts stay, retired starts and their rounds
are set aside, not deleted."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DP = _load("scene_rsi_diag_pool", HERE / "diag_pool.py")
TASKS = [f"pick_{t}" for t in ("A", "B", "C")]


def _start(pool: Path, label: str) -> None:
    pool.mkdir(parents=True, exist_ok=True)
    (pool / f"{label}.json").write_text(json.dumps(dict(task=DP.task_of(label), start={"seed": DP.seed_of(label)})))


def _round(d: Path, label: str, ok: bool, npz: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{label}.json").write_text(json.dumps(dict(label=label, task=DP.task_of(label), ok=ok)))
    if npz:
        (d / f"{label}.npz").write_bytes(b"trace")


def _fake_draw(fails_at):
    """family.draw stand-in: one start per call slot, tasks rotating from start_i, seeds counting up."""

    def draw(fam, pool, out, seed=None, n=None, tasks=None, start_i=0):
        out.mkdir(parents=True, exist_ok=True)
        written = 0
        for k in range(int(n)):
            t = tasks[(start_i + k) % len(tasks)]
            _start(out, f"{t}_s{seed}")
            seed += 1
            written += 1
        s = dict(family="f", pool=pool, written=written, unavailable={}, skipped=[], next_seed=seed)
        (out / "_pool.json").write_text(json.dumps(s))
        return s

    return draw


def _fake_run(fails_at):
    """The incumbent on a pool: a start fails when its seed satisfies `fails_at`."""

    def run(pool_dir: Path, out: Path) -> None:
        for x in DP.labels(pool_dir):
            _round(out, x, ok=not fails_at(DP.seed_of(x)))

    return run


class Pick(unittest.TestCase):
    def test_same_task_first_then_the_fullest_task_never_a_frozen_or_failing_start(self):
        ok = {"pick_A_s1": True, "pick_A_s2": True, "pick_A_s3": False, "pick_B_s4": True, "pick_B_s5": True,
              "pick_C_s6": True, "pick_C_s7": True, "pick_C_s8": True, "pick_C_s9": True}  # fmt: skip
        # a new A failure: A keeps 2 starts and a passing one after s1 goes
        self.assertEqual(DP.pick_retire(ok, "pick_A", set()), "pick_A_s1")
        # A's passing starts both frozen: the fullest task (C) gives one
        self.assertEqual(DP.pick_retire(ok, "pick_A", {"pick_A_s1", "pick_A_s2"}), "pick_C_s6")
        # B would drop below two starts: C again
        self.assertEqual(DP.pick_retire({k: v for k, v in ok.items() if k != "pick_B_s5"}, "pick_B", set()),
                         "pick_C_s6")  # fmt: skip
        # a task whose only passing start would go keeps it
        self.assertIsNone(DP.pick_retire({"pick_A_s1": True, "pick_A_s2": False}, "pick_A", set()))

    def test_plan_stops_at_the_target_and_respects_the_per_task_cap(self):
        ok = {f"pick_{t}_s{i}": True for t in "ABC" for i in range(3)}  # 9 starts, all passing
        new = [f"pick_A_s{100 + i}" for i in range(10)]
        added, retired = DP.plan(ok, new, set(), target=4, per_task_max=6)
        self.assertEqual(len(added), 4)
        self.assertEqual(len(retired), 4)
        # A took its own passing starts first (two can go: A keeps 2 starts and ... no passing start would be left)
        self.assertTrue(all(DP.task_of(x) == "pick_A" for x in added))
        state = {k: v for k, v in ok.items() if k not in retired}
        state.update({x: False for x in added})
        per = {t: sum(DP.task_of(x) == f"pick_{t}" for x in state) for t in "ABC"}
        self.assertEqual(sum(per.values()), 9)  # the pool keeps its size
        self.assertTrue(all(v >= 2 for v in per.values()))
        self.assertTrue(all(any(state[x] for x in state if DP.task_of(x) == f"pick_{t}") for t in "ABC"))
        # the cap: with per_task_max 3, A (already 3) takes nothing
        self.assertEqual(DP.plan(ok, new, set(), target=4, per_task_max=3), ([], []))


class Refresh(unittest.TestCase):
    def _pool(self, root: Path, n_per_task: int = 3):
        pool = root / "pool_diag"
        inc = root / "diag" / "incumbent_abc"
        seed = 410000
        for i in range(n_per_task):
            for t in TASKS:
                _start(pool, f"{t}_s{seed}")
                _round(inc, f"{t}_s{seed}", ok=True)
                seed += 1
        (pool / "_pool.json").write_text(json.dumps(dict(written=seed - 410000, skipped=[])))
        return pool, inc

    def test_a_saturated_pool_takes_failing_fresh_starts_and_sets_the_retired_aside(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pool, inc = self._pool(root)
            cand = root / "diag" / "candidate_xyz"
            for x in DP.labels(pool):
                _round(cand, x, ok=True, npz=False)
            fam = dict(tasks=TASKS, pools={"diag": {"seed": 410000, "n": 9}})
            fails_at = lambda s: s % 2 == 0  # noqa: E731 -- half the fresh starts fail
            row = DP.refresh(fam, pool, inc, _fake_run(fails_at), root / "scratch", set(), set(), min_fails=3,
                             target=4, draw_max=30, draw=_fake_draw(fails_at), batch=6)  # fmt: skip
            self.assertEqual((row["fails_before"], row["fails_after"], row["n"]), (0, 4, 9))
            self.assertEqual(len(row["added"]), 4)
            self.assertEqual(len(row["retired"]), 4)
            self.assertEqual(row["seed_from"], 410009)  # one past the pool's last seed
            self.assertLessEqual(row["drawn"], 12)  # two batches of six at most
            labels = DP.labels(pool)
            self.assertEqual(len(labels), 9)
            self.assertTrue(set(row["added"]) <= set(labels))
            self.assertFalse(set(row["retired"]) & set(labels))
            # retired starts and their rounds are set aside, not deleted
            self.assertEqual(sorted(DP.labels(root / DP.RETIRED_DIR)), sorted(row["retired"]))
            for x in row["retired"]:
                self.assertTrue((inc / f"_retired_{x}.json").exists())
                self.assertTrue((inc / f"_retired_{x}.npz").exists())
                self.assertTrue((cand / f"_retired_{x}.json").exists())
                self.assertFalse((inc / f"{x}.json").exists())
            # the added starts came in with the incumbent's rounds on them (json and trace)
            for x in row["added"]:
                self.assertTrue((inc / f"{x}.json").exists() and (inc / f"{x}.npz").exists())
                self.assertFalse(json.loads((inc / f"{x}.json").read_text())["ok"])
            # the incumbent's outcomes over the pool: exactly the target failing, all tasks represented
            out = DP.outcomes(inc)
            self.assertEqual(sum(not out[x]["ok"] for x in labels), 4)
            self.assertEqual({DP.task_of(x) for x in labels}, set(TASKS))
            meta = json.loads((pool / "_pool.json").read_text())
            self.assertEqual(len(meta["redraws"]), 1)
            self.assertEqual(meta["written"], 9)
            self.assertEqual(meta["next_seed"], row["seed_to"] + 1)
            # a second redraw starts after the first one's seeds and does nothing while enough starts fail
            self.assertEqual(DP.next_seed(pool), row["seed_to"] + 1)
            self.assertIsNone(DP.refresh(fam, pool, inc, _fake_run(fails_at), root / "scratch2", set(), set(),
                                         min_fails=3, target=4, draw_max=30, draw=_fake_draw(fails_at)))  # fmt: skip

    def test_parked_tasks_and_frozen_starts_are_left_alone_and_the_draw_budget_holds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pool, inc = self._pool(root)
            fam = dict(tasks=TASKS, pools={"diag": {"seed": 410000, "n": 9}})
            frozen = {"pick_A_s410000", "pick_A_s410003", "pick_A_s410006"}  # every A start parked
            drawn_tasks = []

            def draw(fam_, pool_, out, seed=None, n=None, tasks=None, start_i=0):
                drawn_tasks.extend(tasks)
                return _fake_draw(None)(fam_, pool_, out, seed=seed, n=n, tasks=tasks, start_i=start_i)

            row = DP.refresh(fam, pool, inc, _fake_run(lambda s: True), root / "scratch", {"pick_C"}, frozen,
                             min_fails=3, target=8, draw_max=4, draw=draw, batch=3)  # fmt: skip
            self.assertNotIn("pick_C", drawn_tasks)  # a parked task draws no fresh starts
            self.assertEqual(row["drawn"], 4)  # the budget: 3 + 1
            self.assertFalse(set(row["retired"]) & frozen)  # frozen starts never go
            labels = set(DP.labels(pool))
            self.assertTrue(frozen <= labels)  # ... they are still in the pool
            self.assertEqual(
                sorted(x for x in labels if DP.task_of(x) == "pick_C"),
                ["pick_C_s410002", "pick_C_s410005", "pick_C_s410008"],
            )  # the parked task's starts stay
            self.assertTrue(all(DP.task_of(x) == "pick_B" for x in row["retired"]))  # only B had room
            self.assertEqual(len(labels), 9)

    def test_nothing_happens_below_the_threshold_or_with_the_switch_off(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pool, inc = self._pool(root)
            fam = dict(tasks=TASKS, pools={"diag": {"seed": 410000, "n": 9}})
            for x in DP.labels(pool)[:3]:
                _round(inc, x, ok=False)
            calls = []
            run = lambda p, o: calls.append(p)  # noqa: E731
            self.assertIsNone(DP.refresh(fam, pool, inc, run, root / "s", set(), set(), min_fails=3, target=5,
                                         draw_max=30, draw=_fake_draw(None)))  # fmt: skip
            self.assertIsNone(DP.refresh(fam, pool, inc, run, root / "s", set(), set(), min_fails=10, target=5,
                                         draw_max=0, draw=_fake_draw(None)))  # fmt: skip
            self.assertEqual(calls, [])
            self.assertEqual(len(DP.labels(pool)), 9)


class LoopSettings(unittest.TestCase):
    def test_the_driver_exposes_the_three_switches_and_the_ledger_line(self):
        names = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "loop")
        saved = {k: sys.modules.pop(k, None) for k in names}
        sys.path.insert(0, str(HERE))
        try:
            lp = _load("scene_rsi_loop_33_23", HERE / "loop.py")
        finally:
            sys.path.remove(str(HERE))
            for k, v in saved.items():
                sys.modules.pop(k, None)
                if v is not None:
                    sys.modules[k] = v
        from argparse import Namespace

        cfg = lp.resolve_settings(Namespace(diag_min_fails=0, diag_target_fails=None, diag_draw_max=12), {}, False)
        self.assertEqual((cfg["diag_min_fails"], cfg["diag_target_fails"], cfg["diag_draw_max"]), (0, 15, 12))
        cfg = lp.resolve_settings(Namespace(), dict(started="x", diag_min_fails=4), True)
        self.assertEqual(cfg["diag_min_fails"], 4)
        line = lp.iteration_line(3, [dict(step="diag_pool", iter=3, fails_before=0, fails_after=15, n=30,
                                          added=list("abc"), retired=list("xyz"), drawn=60)])  # fmt: skip
        self.assertIn("diag pool redrawn: 0 -> 15 failing of 30 (+3 -3, 60 fresh starts run)", line)


if __name__ == "__main__":
    unittest.main()
