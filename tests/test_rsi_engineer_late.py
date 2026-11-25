"""TODO 37.3 (D, user 2026-10-23: "B 到 E 按顺序做"): the Engineer soft deadline -- once half the drafts are in, the
rest get --engineer-late-min minutes, then the driver stops them (run_role kills the session, RoleLate), the draft is a
late void row in the ledger and the iteration goes on with the drafts it has. Run C's slowest Engineer took 22-37 min
against a median draft of about 14, waiting on its own one-process rounds."""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import rsi_loop_ns as NS  # noqa: E402

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
    names = ("gate", "batch", "family", "scene_round", "digest", "loop")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return NS.fresh_loop()
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


class EngineerDeadline(unittest.TestCase):
    def test_run_role_raises_role_late_when_its_stop_event_is_set(self):
        lp = _scene_loop()
        role = "import sys, time\nsys.stdin.read()\ntime.sleep(600)\n"
        stop = threading.Event()
        threading.Timer(0.5, stop.set).start()
        with mock.patch.object(lp, "transcript", lambda s: None):
            t0 = time.time()
            with self.assertRaises(lp.RoleLate) as c:
                lp.run_role([sys.executable, "-c", role], "prompt", dict(os.environ), "s", grace=1.0, poll=0.2,
                            stop=stop)  # fmt: skip
        self.assertLess(time.time() - t0, 20)
        self.assertIn("stopped by the driver", str(c.exception))
        self.assertTrue(issubclass(lp.RoleLate, lp.RoleFailed))  # multi_engineer_step's except clause catches it

    def test_wait_drafts_stops_the_straggler_once_half_are_in_and_zero_waits(self):
        lp = _scene_loop()
        stops = {k: threading.Event() for k in (1, 2, 3)}
        threads = {1: threading.Thread(target=lambda: None), 2: threading.Thread(target=lambda: None),
                   3: threading.Thread(target=lambda: stops[3].wait(30))}  # fmt: skip
        for t in threads.values():
            t.start()
        with mock.patch.object(lp, "ENGINEER_LATE_MIN", 0.01):  # 0.6 s after the second draft
            t0 = time.time()
            late = lp.wait_drafts(threads, stops, poll=0.1)
        self.assertEqual(late, [3])
        self.assertTrue(stops[3].is_set() and not stops[1].is_set() and not stops[2].is_set())
        self.assertLess(time.time() - t0, 10)
        stops2 = {1: threading.Event(), 2: threading.Event()}
        threads2 = {1: threading.Thread(target=lambda: None), 2: threading.Thread(target=lambda: time.sleep(0.8))}
        for t in threads2.values():
            t.start()
        with mock.patch.object(lp, "ENGINEER_LATE_MIN", 0):
            t0 = time.time()
            self.assertEqual(lp.wait_drafts(threads2, stops2, poll=0.1), [])
        self.assertGreater(time.time() - t0, 0.7)  # waited for the slow one instead of stopping it
        self.assertFalse(stops2[2].is_set())

    def test_a_stopped_engineer_is_a_late_void_draft_and_the_iteration_goes_on(self):
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            st = root / "loop"
            it_dir = st / "iter_00"
            it_dir.mkdir(parents=True)
            inc_dir = root / "inc"
            (inc_dir / "skills").mkdir(parents=True)
            (inc_dir / "skills" / "__init__.py").write_text("X = 1\n")
            (inc_dir / "candidate.json").write_text('{"set": {}}')
            inc = dict(dir=str(inc_dir), hash=lp.candidate_hash(inc_dir), set={})
            hyps = dict(hypotheses=[dict(id=h, claim="c", instead="i", delta={"set": {}}) for h in ("H1", "H2", "H3")])
            dec = dict(decision="engineer", hypotheses=["H1", "H2", "H3"], reason="r")
            rows, gated = [], []

            def fake_claude(role, prompt, schema, it_dir_, tools, stub=None, stop=None):
                if role == "engineer_3":
                    stop.wait(30)
                    raise lp.RoleLate("stopped by the driver 1 s in (--engineer-late-min)")
                if role.startswith("engineer_"):
                    k = int(role.split("_")[1])
                    (it_dir / f"cand_{k}" / "skills" / "__init__.py").write_text(f"X = {k + 1}\n")
                    return (dict(mechanism=f"m{k}", files_changed=["__init__.py"], set={}, tests="t"),
                            dict(cost_usd=1.0, session=f"s{k}", seconds=1))  # fmt: skip
                if role == "composer":
                    return (dict(decision="adopt", adopted=1, notes="n", mechanism="m1"),
                            dict(cost_usd=1.0, session="sc", seconds=1))  # fmt: skip
                raise AssertionError(role)

            common = dict(RULES="r", PLAN="p", SIM_PY="py", POOL="pool", FAMILY="fam", FAMILY_FILE="ff", N_DIAG=30,
                          LIBRARY="", DIAG_INC="d", TMP="t")  # fmt: skip
            with (
                mock.patch.multiple(lp, STATE=st, LOOP=root, FAMILY="fam", CANDIDATES=3, ENGINEER_LATE_MIN=0.01,
                                    claude=fake_claude, ledger=rows.append, transcript_leaks=lambda s: [],
                                    revert_strays=lambda b: [], put_skills=lambda p: None,
                                    named_objects=lambda s, names=None: [], context=lambda *a, **k: {},
                                    pool_view=lambda *a, **k: [],
                                    gate_candidate=lambda n, it, cand, stub, **k: gated.append(str(cand)) or None),
                mock.patch.object(lp.B, "run_jobs", lambda *a, **k: {}),
                mock.patch.object(lp, "flips", lambda a, b: dict(ok=0, incumbent_ok=0, n=0, fixed=[], broken=[])),
            ):  # fmt: skip
                t0 = time.time()
                why = lp.multi_engineer_step(0, it_dir, inc, hyps, [], dec, common, [], stub=False)
            self.assertIsNone(why)
            self.assertLess(time.time() - t0, 30)
            eng = [r for r in rows if r["step"] == "engineer"]
            self.assertEqual([(r["candidate"], bool(r.get("void")), bool(r.get("late"))) for r in eng],
                             [(1, False, False), (2, False, False), (3, True, True)])  # fmt: skip
            self.assertIn("stopped by the driver", eng[2]["error"])
            self.assertEqual([r["decision"] for r in rows if r["step"] == "compose"], ["adopt"])
            self.assertEqual(len(gated), 1)
            self.assertTrue(gated[0].endswith("candidate_1"))

    def test_the_flag_reaches_the_settings(self):
        from argparse import Namespace

        lp = _scene_loop()
        seen = {}
        with (
            mock.patch.object(lp, "cmd_run", lambda a: seen.update(vars(a)) or 0),
            mock.patch.object(lp, "configure", lambda f: None),
        ):
            lp.main(["run", "fam", "--engineer-late-min", "4"])
        self.assertEqual(seen["engineer_late_min"], 4.0)
        none = Namespace(candidates=None, engineer_late_min=None)
        self.assertEqual(lp.resolve_settings(none, {}, False)["engineer_late_min"], 10.0)
        self.assertEqual(
            lp.resolve_settings(none, dict(started="x", engineer_late_min=0), True)["engineer_late_min"], 0.0
        )
        self.assertEqual(
            lp.resolve_settings(Namespace(engineer_late_min=7), dict(started="x", engineer_late_min=0), True)[
                "engineer_late_min"
            ],
            7.0,
        )


if __name__ == "__main__":
    unittest.main()
