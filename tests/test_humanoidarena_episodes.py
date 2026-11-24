"""TODO 49.5b: HumanoidArenaConfig keeps its values, takes --set, rejects unknown fields; the pools never use the benchmark's evaluation
groups {0, 1, 2}, and no derived seed of a pool collides with the 60 derived evaluation seeds of its task; the evaluation episodes are
refused in every layer (the config, the CLI before a server starts, check_id, reset_msg, the server's own check)."""

import hashlib
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from locomani.bench.humanoidarena import __main__ as cli
from locomani.bench.humanoidarena import config as C
from locomani.bench.humanoidarena import episodes as EP
from locomani.bench.humanoidarena.config import HumanoidArenaConfig
from locomani.bench.humanoidarena.episodes import EpisodeRefused

TASK = "pp_box"
SERVER = Path(EP.__file__).with_name("server.py")


def server_module():
    spec = importlib.util.spec_from_file_location("humanoidarena_server_under_test", SERVER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Config(unittest.TestCase):
    def test_defaults_and_overrides(self):
        c = HumanoidArenaConfig()
        self.assertEqual((c.pools, c.max_steps, c.gpu, c.cores), ("P", 0, 1, "28-31"))
        c2 = c.with_overrides(["pools=PA", "gpu=1", "max_steps=300"])
        self.assertEqual((c2.pools, c2.max_steps), ("PA", 300))
        self.assertEqual(HumanoidArenaConfig.from_dict(c2.to_dict()), c2)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            c.with_overrides(["nonsense=1"])

    def test_paths_point_into_the_install(self):
        c = HumanoidArenaConfig()
        self.assertTrue(c.ha_python.endswith("third_party/HumanoidArena/venv/bin/python"))
        self.assertTrue(c.server_script().endswith("humanoidarena/server.py"))
        self.assertTrue(c.lock_path().endswith("humanoidarena.lock"))

    def test_scope_refuses_the_evaluation_pool_and_unknown_letters(self):
        for pools in ("E", "PE", "", "X"):
            with self.assertRaises(ValueError, msg=pools):
                HumanoidArenaConfig(pools=pools).validate_scope()
        HumanoidArenaConfig(pools="PAS").validate_scope()

    def test_pools_never_include_an_evaluation_group(self):
        self.assertEqual(C.pool_ranges_ok(), [])
        for k, (lo, n) in C.POOLS.items():
            self.assertTrue(all(not (lo <= g < lo + n) for g in C.EVAL_GROUPS), k)

    def test_a_pool_that_included_one_would_be_caught(self):
        old = dict(C.POOLS)
        try:
            C.POOLS["S"] = (0, 10)
            self.assertTrue(any("evaluation group" in p for p in C.pool_ranges_ok()))
            with self.assertRaises(ValueError):
                HumanoidArenaConfig(pools="S").validate_scope()
        finally:
            C.POOLS.clear()
            C.POOLS.update(old)


class Seeds(unittest.TestCase):
    def test_the_formula_is_the_benchmarks(self):
        # eval_vla_suite.py:_derive_episode_seed, restated independently here
        want = (
            int.from_bytes(hashlib.sha256(b"Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby|1|7").digest()[-4:], "big")
            & 0x7FFFFFFF
        )
        self.assertEqual(EP.derive_seed("pp_box", 1, 7), want)
        self.assertEqual(EP.derive_seed("pp_box", 1, 7), server_module().derive_seed(EP.GYM["pp_box"], 1, 7))

    def test_sixty_evaluation_seeds_per_task_and_the_server_agrees(self):
        srv = server_module()
        for t in EP.TASKS:
            self.assertEqual(len(EP.eval_seeds(t)), 60, t)
            self.assertEqual(set(EP.eval_seeds(t)), srv.eval_seed_set(EP.GYM[t]), t)

    def test_no_pool_seed_collides_with_an_evaluation_seed(self):
        cfg = HumanoidArenaConfig(pools="PAS")
        for t in EP.TASKS:
            ev = EP.eval_seeds(t)
            for pool, (first, groups) in C.POOLS.items():
                for n in list(range(0, 60)) + [groups * C.PER_GROUP - 1]:
                    self.assertNotIn(EP.check_id(cfg, f"{t}/{pool}/{n}"), ev, (t, pool, n))

    def test_a_colliding_seed_is_refused(self):
        # force a pool episode's derived seed onto an evaluation seed: check_id and reset_msg must both refuse it
        real = EP.derive_seed
        ev = sorted(EP.eval_seeds(TASK))[0]
        cfg = HumanoidArenaConfig(pools="P")
        try:
            EP.derive_seed = lambda task, g, r: ev
            with self.assertRaises(EpisodeRefused):
                EP.check_id(cfg, f"{TASK}/P/3")
            with self.assertRaises(EpisodeRefused):
                EP.reset_msg(cfg, f"{TASK}/P/3")
        finally:
            EP.derive_seed = real


class Ids(unittest.TestCase):
    cfg = HumanoidArenaConfig(pools="PAS")

    def test_groups_and_repeats(self):
        self.assertEqual(EP.group_repeat(f"{TASK}/P/0"), (1000, 0))
        self.assertEqual(EP.group_repeat(f"{TASK}/P/45"), (1002, 5))
        self.assertEqual(EP.group_repeat(f"{TASK}/A/19"), (2000, 19))
        self.assertEqual(EP.group_repeat(f"{TASK}/S/20"), (3001, 0))

    def test_refused(self):
        for eid, why in [
            (f"{TASK}/E/0", "evaluation pool"),
            (f"{TASK}/P/20000", "past the pool"),
            (f"{TASK}/A/20000", "past the pool"),
            (f"{TASK}/Z/0", "unknown pool"),
            (f"{TASK}/P/-1", "not an integer"),
            (f"{TASK}/P/1.5", "not an integer"),
            (f"{TASK}/P", "two parts"),
            ("football/P/0", "task out of scope"),
            ("sit_sofa/P/0", "task out of scope"),
        ]:
            with self.assertRaises(EpisodeRefused, msg=why):
                EP.check_id(self.cfg, eid)

    def test_a_pool_the_run_was_not_given_is_refused(self):
        with self.assertRaises(EpisodeRefused):
            EP.check_id(HumanoidArenaConfig(pools="P"), f"{TASK}/A/0")

    def test_reset_msg_carries_group_and_repeat(self):
        self.assertEqual(EP.reset_msg(self.cfg, f"{TASK}/P/23"), ("reset", TASK, 1001, 3))

    def test_reset_msg_rechecks_the_group_it_sends(self):
        real = EP.group_repeat
        try:
            EP.group_repeat = lambda eid: (1, 4)  # a broken mapping upstream
            with self.assertRaises(EpisodeRefused):
                EP.reset_msg(self.cfg, f"{TASK}/P/3")
        finally:
            EP.group_repeat = real

    def test_ids_of_a_spec_and_a_file(self):
        self.assertEqual(EP.ids_of(f"{TASK}/P/0-2"), [f"{TASK}/P/{i}" for i in range(3)])
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(f"{TASK}/P/5\n\n{TASK}/A/1\n")
        self.assertEqual(EP.ids_of(f.name), [f"{TASK}/P/5", f"{TASK}/A/1"])


class Cli(unittest.TestCase):
    def run_cli(self, *argv):
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(list(argv))
        return code, err.getvalue()

    def test_the_evaluation_pool_is_refused_before_a_server_starts(self):
        for pool_set in ([], ["--set", "pools=PAS"]):
            code, err = self.run_cli("run", TASK, "--ids", f"{TASK}/E/0", "--out", "/dev/null", *pool_set)
            self.assertEqual(code, cli.EXIT_ARGS)
            self.assertIn("never run", err)

    def test_a_pool_past_its_end_is_refused(self):
        code, err = self.run_cli("check", "--ids", f"{TASK}/S/19999", "--set", "pools=S")
        self.assertEqual(code, 0, err)
        code, err = self.run_cli("check", "--ids", f"{TASK}/S/20000", "--set", "pools=S")
        self.assertEqual(code, cli.EXIT_ARGS)

    def test_an_unsupported_switch_is_refused(self):
        code, err = self.run_cli("check", "--ids", f"{TASK}/P/0", "--bench-set", "target=perception")
        self.assertEqual(code, cli.EXIT_ARGS)
        self.assertIn("perception", err)

    def test_ids_of_another_task_are_refused(self):
        code, err = self.run_cli("run", TASK, "--ids", "open_door/P/0", "--out", "/dev/null")
        self.assertEqual(code, cli.EXIT_ARGS)


class ServerWall(unittest.TestCase):
    """The last layer: the environment server derives the seed itself and refuses an evaluation episode before it boots Isaac."""

    def test_server_refuses_the_evaluation_groups_and_their_seeds(self):
        srv = server_module().Server("/nonexistent", 1, "cpu")
        for task in EP.TASKS:
            for g in (0, 1, 2):
                with self.assertRaises(ValueError, msg=(task, g)):
                    srv.reset(task, g, 0)
        with self.assertRaises(ValueError):
            srv.reset("football", 1000, 0)

    def test_server_refuses_a_group_whose_seed_is_an_evaluation_seed(self):
        m = server_module()
        srv = m.Server("/nonexistent", 1, "cpu")
        ev = sorted(m.eval_seed_set(EP.GYM[TASK]))[0]
        m.derive_seed = lambda gym, g, r: ev
        with self.assertRaises(ValueError):
            srv.reset(TASK, 1000, 0)

    def test_the_client_starts_the_server_pinned_on_the_chosen_gpu(self):
        from locomani.bench.humanoidarena import client

        cfg = HumanoidArenaConfig()
        cmd = client.server_command(cfg, "/tmp/x.sock")
        self.assertEqual(cmd[:3], ["taskset", "-c", "28-31"])
        self.assertEqual(cmd[cmd.index("--kit-gpu") + 1], "1")
        self.assertEqual(client.server_env(cfg)["CUDA_VISIBLE_DEVICES"], "1")
        self.assertTrue(
            client.server_env(cfg)["ROBOT_USD_OVERRIDE"].endswith("g1_29dof_with_dex3_rev_1_0_m2_thumd.usd")
        )


if __name__ == "__main__":
    unittest.main()
