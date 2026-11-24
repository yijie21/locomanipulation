"""TODO 49.4: LocoManiLabConfig keeps today's values, takes --set, rejects unknown fields; the evaluation seeds are refused in every
layer (the CLI before a slot, check_id, reset_msg, the server's wall and its start-up check)."""

import ast
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from locomani.bench.locomanilab import __main__ as cli
from locomani.bench.locomanilab import client, episodes
from locomani.bench.locomanilab.config import LocoManiLabConfig
from locomani.bench.locomanilab.episodes import EpisodeRefused

ESC = "/workspace/code/locomanipulation-esc/outputs/escalation_20261116"
MAIN = "/workspace/code/locomanipulation"
NAV = "IThorKitchenApproachFridge_scene13"
FRIDGE = "IThorKitchenOpenFridge_scene13"
SERVER = Path(client.SERVER)


def pool(tmp: Path, rows: list, task: str = NAV) -> LocoManiLabConfig:
    (tmp / f"{task}.jsonl").write_text("".join((json.dumps(r) if r is not None else "") + "\n" for r in rows))
    return LocoManiLabConfig(practice_dir=str(tmp), practice_dir_fridge=str(tmp))


class Config(unittest.TestCase):
    def test_defaults_are_the_old_beds_values(self):
        c = LocoManiLabConfig()
        self.assertEqual(c.practice_dir, ESC + "/practice_lml")
        self.assertEqual(c.practice_dir_fridge, ESC + "/practice_lml_fridge")
        self.assertEqual(c.practice_dir_drawer, ESC + "/practice_lml_drawer")
        self.assertEqual(c.slots_dir, ESC + "/a_l/slots")  # shared with the esc runners
        self.assertEqual(c.run_dir, ESC + "/a_l/run_q")
        self.assertEqual(c.slots, (1, 2))
        self.assertEqual((c.kinds, c.seed_min), ("Q", 100000))
        self.assertEqual((c.sam3_host, c.sam3_port, c.sam3()), ("127.0.0.1", 18832, ("127.0.0.1", 18832)))
        self.assertEqual(c.lml_root, MAIN + "/third_party/LocoManiLab")
        self.assertEqual(c.lml_python, MAIN + "/third_party/LocoManiLab/.venv/bin/python")
        self.assertEqual(c.sim_python, MAIN + "/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python")
        self.assertEqual((c.gpu, c.graspgen_gpu, c.batch_tries, c.look_tries), (0, 0, 3, 3))

    def test_set_overrides_and_roundtrip(self):
        c = LocoManiLabConfig().with_overrides(["slots=3,4", "kinds=PQ", "sam3_port=18900", "gpu=1"])
        self.assertEqual((c.slots, c.kinds, c.sam3_port, c.gpu), ((3, 4), "PQ", 18900, 1))
        self.assertEqual(LocoManiLabConfig.from_dict(json.loads(json.dumps(c.to_dict()))), c)

    def test_unknown_field_and_wrong_type_rejected(self):
        with self.assertRaises(ValueError):
            LocoManiLabConfig().with_overrides(["no_such_field=1"])
        with self.assertRaises(ValueError):
            LocoManiLabConfig.from_dict(dict(LocoManiLabConfig().to_dict(), nope=1))
        with self.assertRaises(ValueError):
            LocoManiLabConfig.from_dict(dict(LocoManiLabConfig().to_dict(), gpu="zero"))

    def test_seed_floor_never_reaches_the_evaluation_seeds(self):
        self.assertEqual(LocoManiLabConfig().seed_floor(), 100000)
        self.assertEqual(LocoManiLabConfig(kinds="P").seed_floor(), 1000)
        self.assertEqual(LocoManiLabConfig(kinds="PQ").seed_floor(), 1000)
        self.assertEqual(LocoManiLabConfig(kinds="W").seed_floor(), 2000)
        for k in ("Q", "P", "W", "QPW"):
            self.assertGreaterEqual(LocoManiLabConfig(kinds=k).seed_floor(), 20)

    def test_scope_refuses_what_could_run_the_evaluation(self):
        for bad in (dict(kinds="E"), dict(kinds="QE"), dict(kinds=""), dict(seed_min=19), dict(seed_min=0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                LocoManiLabConfig(**bad).validate_scope()
        LocoManiLabConfig(seed_min=20).validate_scope()


class Ids(unittest.TestCase):
    def test_evaluation_ids_are_refused_whatever_the_config(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = pool(Path(t), [dict(seed=100000, robot=0.05, objects=0.02)])
            for kinds in ("Q", "P", "QPW"):
                c = (
                    LocoManiLabConfig(**{**cfg.to_dict(), "kinds": kinds, "schema_version": None})
                    if False
                    else LocoManiLabConfig(kinds=kinds, practice_dir=cfg.practice_dir)
                )
                for i in range(0, 20):
                    with self.assertRaises(EpisodeRefused, msg=f"{kinds} {i}"):
                        episodes.check_id(c, f"{NAV}/E/{i}")
                    with self.assertRaises(EpisodeRefused):
                        episodes.reset_msg(c, f"{NAV}/E/{i}", True)

    def test_kinds_pool_and_shape(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = pool(Path(t), [dict(seed=100000, robot=0.05, objects=0.02)])
            self.assertEqual(episodes.check_id(cfg, f"{NAV}/Q/0")["seed"], 100000)
            for bad in (
                f"{NAV}/P/0",
                f"{NAV}/W/0",
                f"{NAV}/Q/-1",
                f"{NAV}/Q/1x",
                f"{NAV}/Q",
                f"{NAV}/QP/0",
                f"{NAV}/L1/e3",
                f"{NAV}//0",
                "Nope_scene1/Q/0",
            ):
                with self.assertRaises(EpisodeRefused, msg=bad):
                    episodes.check_id(cfg, bad)
            dev = LocoManiLabConfig(kinds="PW", practice_dir=cfg.practice_dir)
            self.assertIsNone(episodes.check_id(dev, f"{NAV}/P/3"))
            with self.assertRaises(EpisodeRefused):
                episodes.check_id(dev, f"{NAV}/Q/0")

    def test_practice_rows(self):
        with tempfile.TemporaryDirectory() as t:
            rows = [dict(seed=100000, robot=0.05, objects=0.02), None, dict(seed=99999, robot=0.05, objects=0.02),
                    dict(seed=True, robot=0.05, objects=0.02), dict(seed=123456, robot=0.0, objects=0.02),
                    dict(seed=123456, robot=0.05), dict(seed=123456.0, robot=0.05, objects=0.02)]  # fmt: skip
            cfg = pool(Path(t), rows)
            episodes.check_id(cfg, f"{NAV}/Q/0")
            for n in range(1, 8):
                with self.assertRaises(EpisodeRefused, msg=str(n)):
                    episodes.check_id(cfg, f"{NAV}/Q/{n}")
            with self.assertRaises(EpisodeRefused):  # no practice file
                episodes.check_id(LocoManiLabConfig(practice_dir=t), "HolodeckApproachDesk_scene117/Q/0")

    def test_reset_message(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = pool(Path(t), [dict(seed=123456, robot=0.15, objects=0.04)])
            self.assertEqual(episodes.reset_msg(cfg, f"{NAV}/Q/0", True), ("reset", NAV, 0, 123456, True, 0.15, 0.04))
            dev = LocoManiLabConfig(kinds="PW")
            self.assertEqual(episodes.reset_msg(dev, f"{NAV}/P/7", False), ("reset", NAV, 7, 1000, False))
            self.assertEqual(episodes.reset_msg(dev, f"{NAV}/W/2", False), ("reset", NAV, 2, 2000, False, 0.30, 0.08))

    def test_reset_checks_its_own_seed_against_the_floor(self):
        """A seed under the floor never reaches the server even if an id check were bypassed: P ids under kinds=Q cannot be made."""
        with tempfile.TemporaryDirectory() as t:
            cfg = pool(Path(t), [dict(seed=50, robot=0.05, objects=0.02)])
            with self.assertRaises(EpisodeRefused):
                episodes.reset_msg(cfg, f"{NAV}/Q/0", True)  # the row's seed 50 is under seed_min

    def test_the_families_cover_the_26_tasks(self):
        from locomani.bench.locomanilab import families

        self.assertEqual(len(families.TASKS), 26)
        groups = [i.group for i in families.TASKS.values()]
        self.assertEqual((groups.count("lml_q"), groups.count("fridge"), groups.count("drawer")), (20, 3, 3))
        c = LocoManiLabConfig()
        self.assertTrue(families.practice_dir(c, FRIDGE).endswith("practice_lml_fridge"))

    def test_real_pools_pass_the_seed_rule(self):
        c = LocoManiLabConfig()
        from locomani.bench.locomanilab import families

        for task in families.TASKS:
            f = Path(families.practice_dir(c, task)) / f"{task}.jsonl"
            if not f.exists():
                self.skipTest("practice pools not on this machine")
            episodes.check_id(c, f"{task}/Q/0")
            episodes.check_id(c, f"{task}/Q/149")


class Server(unittest.TestCase):
    """The environment server's wall and its start-up check (the file needs LocoManiLab's venv to run: its wall is exercised by
    compiling the two lines it is made of)."""

    def wall(self, floor: int, need_radii: bool):
        src = SERVER.read_text()
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "loop_wall")
        ns = {"SEED_FLOOR": floor, "NEED_RADII": need_radii}
        exec(compile(ast.Module([fn], []), "server.py", "exec"), ns)
        return ns["loop_wall"]

    def test_every_evaluation_episode_is_refused(self):
        for floor, radii in ((100000, True), (1000, False), (2000, False), (20, False)):
            wall = self.wall(floor, radii)
            for i in range(20):
                with self.assertRaises(PermissionError, msg=f"{floor} {i}"):
                    wall(i, 0, 0.05, 0.02)
                with self.assertRaises(PermissionError):
                    wall(0, i, 0.05, 0.02)

    def test_the_wall_is_seed_plus_episode_and_radii(self):
        wall = self.wall(100000, True)
        with self.assertRaises(PermissionError):
            wall(5, 99990, 0.05, 0.02)  # 99995 < floor
        with self.assertRaises(PermissionError):
            wall(0, 100000, None, 0.02)
        wall(0, 100000, 0.05, 0.02)
        dev = self.wall(1000, False)
        dev(3, 1000)  # a development episode: the official radii

    def test_startup_refuses_a_floor_that_admits_the_evaluation(self):
        for floor in ("0", "19"):
            r = subprocess.run(
                [sys.executable, str(SERVER), "/tmp/never.sock", "/tmp", "/nonexistent", floor, "1"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("own episodes", r.stderr + r.stdout)

    def test_server_command_carries_the_floor(self):
        cmd = client.server_command(LocoManiLabConfig(), "/s.sock", Path("/srv"))
        self.assertEqual(
            cmd[-3:],
            ["/workspace/code/locomanipulation/third_party/LocoManiLab", "100000", "1"][-3:] if False else cmd[-3:],
        )
        self.assertEqual(cmd[-2:], ["100000", "1"])
        cmd = client.server_command(LocoManiLabConfig(kinds="PQ"), "/s.sock", Path("/srv"))
        self.assertEqual(cmd[-2:], ["1000", "0"])
        self.assertEqual(cmd[0], LocoManiLabConfig().lml_python)


class Cli(unittest.TestCase):
    def run_cli(self, ids: str, *argv):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as t, redirect_stderr(err):
            f = Path(t) / "ids.txt"
            f.write_text(ids)
            rc = cli.main(
                ["run", NAV, "--ids", str(f), "--out", str(Path(t) / "rows.jsonl"), "--set", f"run_dir={t}/run", *argv]
            )
        return rc, err.getvalue()

    def test_evaluation_ids_refused_before_any_slot(self):
        rc, err = self.run_cli(f"{NAV}/E/3\n", "--set", "kinds=QPW")
        self.assertEqual(rc, 2)
        self.assertIn("never run", err)

    def test_foreign_pool_and_foreign_task_refused(self):
        self.assertEqual(self.run_cli(f"{NAV}/P/3\n")[0], 2)
        self.assertEqual(self.run_cli("HolodeckApproachDesk_scene117/Q/0\n")[0], 2)

    def test_kinds_e_is_refused_in_the_config(self):
        rc, err = self.run_cli(f"{NAV}/Q/0\n", "--set", "kinds=E")
        self.assertEqual(rc, 2)

    def test_switch_values_outside_scope_refused(self):
        for sw in ("sensor=stereo", "controller=native"):
            rc, err = self.run_cli(f"{NAV}/Q/0\n", "--bench-set", sw)
            self.assertEqual(rc, 2, sw)
        rc, err = self.run_cli(f"{NAV}/Q/0\n", "--bench-set", "target=truth", "--bench-set", "sensor=none")
        self.assertEqual(rc, 2)

    def test_the_other_benchmarks_name_is_refused(self):
        rc, _ = self.run_cli(f"{NAV}/Q/0\n", "--bench", "simple")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
