"""TODO 49.5: BiGym2Config keeps its values, takes --set, rejects unknown fields; the pools never touch BiGym's evaluation block
[620000, 630000) or the demonstration seeds; the block is refused in every layer (the config, the CLI before a server starts,
check_id, reset_msg, the server's own check)."""

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from locomani.bench.bigym2 import __main__ as cli
from locomani.bench.bigym2 import config as C
from locomani.bench.bigym2 import episodes as EP
from locomani.bench.bigym2.config import BiGym2Config
from locomani.bench.bigym2.episodes import EpisodeRefused

TASK = "reach_target_single"
SERVER = Path(EP.__file__).with_name("server.py")


def server_module():
    spec = importlib.util.spec_from_file_location("bigym2_server_under_test", SERVER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Config(unittest.TestCase):
    def test_defaults_and_overrides(self):
        c = BiGym2Config()
        self.assertEqual((c.pools, c.depth, c.render_shape, c.max_steps), ("P", False, (240, 320), 0))
        c2 = c.with_overrides(["depth=true", "pools=PA", "render_shape=120,160"])
        self.assertTrue(c2.depth)
        self.assertEqual((c2.pools, tuple(c2.render_shape)), ("PA", (120, 160)))
        self.assertEqual(BiGym2Config.from_dict(c2.to_dict()), c2)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            c.with_overrides(["nonsense=1"])

    def test_paths_point_into_the_clone(self):
        c = BiGym2Config()
        self.assertTrue(c.bigym_python.endswith("third_party/BiGym2/.venv/bin/python"))
        self.assertTrue(c.server_script().endswith("bigym2/server.py"))
        self.assertTrue(c.lock_path().endswith("bigym2.lock"))

    def test_scope_refuses_the_official_pool_and_unknown_letters(self):
        for pools in ("E", "PE", "", "X"):
            with self.assertRaises(ValueError, msg=pools):
                BiGym2Config(pools=pools).validate_scope()
        BiGym2Config(pools="PAS").validate_scope()

    def test_pools_never_overlap_the_block_or_the_demo_seeds(self):
        self.assertEqual(C.pool_ranges_ok(), [])
        for k, (lo, n) in C.POOLS.items():
            hi = lo + n
            self.assertTrue(hi <= C.EVAL_BLOCK[0] or lo >= C.EVAL_BLOCK[1], k)
            self.assertTrue(lo >= C.DEMO_SEEDS[1], k)

    def test_a_pool_that_overlapped_would_be_caught(self):
        old = dict(C.POOLS)
        try:
            C.POOLS["S"] = (619990, 100)
            self.assertTrue(any("evaluation block" in p for p in C.pool_ranges_ok()))
            with self.assertRaises(ValueError):
                BiGym2Config(pools="S").validate_scope()
        finally:
            C.POOLS.clear()
            C.POOLS.update(old)


class Ids(unittest.TestCase):
    cfg = BiGym2Config(pools="PAS")

    def test_seeds_of_the_pools(self):
        self.assertEqual(EP.check_id(self.cfg, f"{TASK}/P/0"), 1000)
        self.assertEqual(EP.check_id(self.cfg, f"{TASK}/P/8999"), 9999)
        self.assertEqual(EP.check_id(self.cfg, f"{TASK}/A/7"), 100007)
        self.assertEqual(EP.check_id(self.cfg, f"{TASK}/S/0"), 700000)

    def test_refused(self):
        for eid, why in [
            (f"{TASK}/E/0", "official pool"),
            (f"{TASK}/P/9000", "past the pool"),
            (f"{TASK}/A/1000", "past the pool"),
            (f"{TASK}/Z/0", "unknown pool"),
            (f"{TASK}/P/-1", "not an integer"),
            (f"{TASK}/P/1.5", "not an integer"),
            (f"{TASK}/P", "two parts"),
        ]:
            with self.assertRaises(EpisodeRefused, msg=why):
                EP.check_id(self.cfg, eid)

    def test_a_pool_the_run_was_not_given_is_refused(self):
        with self.assertRaises(EpisodeRefused):
            EP.check_id(BiGym2Config(pools="P"), f"{TASK}/A/0")

    def test_no_id_of_any_pool_has_a_seed_in_the_block(self):
        for pool, (first, size) in C.POOLS.items():
            for n in (0, 1, size - 1):
                seed = EP.check_id(self.cfg, f"{TASK}/{pool}/{n}")
                self.assertFalse(EP.in_eval_block(seed), (pool, n))

    def test_reset_msg_carries_the_checked_seed(self):
        self.assertEqual(EP.reset_msg(self.cfg, f"{TASK}/P/3"), ("reset", TASK, 1003))

    def test_reset_msg_rechecks_the_seed_it_sends(self):
        real = EP.check_id
        try:
            EP.check_id = lambda cfg, eid: 625000  # a broken check upstream
            with self.assertRaises(EpisodeRefused):
                EP.reset_msg(self.cfg, f"{TASK}/P/3")
        finally:
            EP.check_id = real

    def test_ids_of_a_spec_and_a_file(self):
        self.assertEqual(EP.ids_of(f"{TASK}/P/0-2"), [f"{TASK}/P/{i}" for i in range(3)])
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(f"{TASK}/P/5\n\n{TASK}/A/1\n")
        self.assertEqual(EP.ids_of(f.name), [f"{TASK}/P/5", f"{TASK}/A/1"])

    def test_seed_ignoring_tasks(self):
        for t in ("dishwasher_close", "wall_cupboard_open", "wall_cupboard_close"):
            self.assertTrue(EP.ignores_seed(t))
        self.assertFalse(EP.ignores_seed("drawer_top_open"))


class Cli(unittest.TestCase):
    def run_cli(self, *argv):
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(list(argv))
        return code, err.getvalue()

    def test_the_block_is_refused_before_a_server_starts(self):
        for pool_set in ([], ["--set", "pools=PAS"]):
            code, err = self.run_cli("run", TASK, "--ids", f"{TASK}/E/0", "--out", "/dev/null", *pool_set)
            self.assertEqual(code, cli.EXIT_ARGS)
            self.assertIn("never run", err)

    def test_a_seed_in_the_block_cannot_be_named_by_any_id(self):
        # the nearest ids: sealed pool ends far below the block; a made-up pool letter is refused too
        code, err = self.run_cli("check", "--ids", f"{TASK}/S/9999", "--set", "pools=S")
        self.assertEqual(code, 0, err)
        code, err = self.run_cli("check", "--ids", f"{TASK}/S/10000", "--set", "pools=S")
        self.assertEqual(code, cli.EXIT_ARGS)

    def test_an_unsupported_switch_is_refused(self):
        code, err = self.run_cli("check", "--ids", f"{TASK}/P/0", "--bench-set", "target=perception")
        self.assertEqual(code, cli.EXIT_ARGS)
        self.assertIn("perception", err)

    def test_ids_of_another_task_are_refused(self):
        code, err = self.run_cli("run", TASK, "--ids", "drawer_top_open/P/0", "--out", "/dev/null")
        self.assertEqual(code, cli.EXIT_ARGS)


class ServerWall(unittest.TestCase):
    """The last layer: the environment server refuses the block itself, before it builds anything."""

    def test_server_refuses_the_block_and_negative_seeds(self):
        srv = server_module().Server((620000, 630000), False, (240, 320))
        for seed in (620000, 620099, 625000, 629999, -1):
            with self.assertRaises(ValueError, msg=seed):
                srv.reset(TASK, seed)
        for seed in ("1000", 1000.0, True):
            with self.assertRaises(ValueError, msg=repr(seed)):
                srv.reset(TASK, seed)

    def test_the_client_starts_the_server_with_the_block(self):
        from locomani.bench.bigym2 import client

        cmd = client.server_command(BiGym2Config(), "/tmp/x.sock")
        self.assertEqual((cmd[cmd.index("--refuse-lo") + 1], cmd[cmd.index("--refuse-hi") + 1]), ("620000", "630000"))
        self.assertNotIn("--depth", cmd)
        self.assertIn("--depth", client.server_command(BiGym2Config(depth=True), "/tmp/x.sock"))


if __name__ == "__main__":
    unittest.main()
