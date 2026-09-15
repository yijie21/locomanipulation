"""run_crate_pilot.py: the crate line's stack lifecycle, separated from its expert.

The four things these cover are the four the bash runners got wrong or left to
the caller (2026-09-11): the encoder mode, the port block, the manifest's
append hazard, and instance 0.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from locomani.pipeline import run_crate_pilot as R


class BackendTests(unittest.TestCase):
    def test_no_encoder_is_off_for_the_crate(self):
        """NO_ENCODER=1 drops the planner's arm and crouch references, which is
        the channel the crate expert drives (memory no-encoder-breaks-planner-arms).
        SonicBackendConfig's own default is True -- the EVAL protocol's -- so the
        crate driver has to say otherwise, and the shell env it produces must say 0."""
        from locomani.pipeline.stack import backend_env

        self.assertFalse(R.backend_config().no_encoder)
        self.assertEqual(backend_env(R.backend_config())["NO_ENCODER"], "0")
        self.assertEqual(backend_env(R.backend_config(allow_no_encoder=True))["NO_ENCODER"], "1")

    def test_backend_matches_what_the_bash_runners_left_unset(self):
        cfg = R.backend_config()
        env = __import__("locomani.pipeline.stack", fromlist=["backend_env"]).backend_env(cfg)
        self.assertEqual(env["SIM_VIS"], "none")  # the runners exported SIM_VIS=none
        self.assertEqual(env["SCENE_RANDOM"], "0")
        self.assertEqual(env["WRIST_CAMS"], "1")  # start_stack.sh's own WRIST_CAMS:-1
        self.assertEqual(env["LOCKSTEP"], "0")
        self.assertNotIn("SONIC_DDS_DOMAIN", env)  # start_stack.sh derives it from INSTANCE


class ArgvTests(unittest.TestCase):
    def _args(self, *extra):
        return R.parse_args(["--out", "/tmp/x", *extra])

    def test_expert_choice_selects_script_and_label_prefix(self):
        for expert, script, label in (("wall", "bimanual_wall.py", "wall"), ("carry", "bimanual_carry.py", "bimanual")):
            argv = R.expert_argv(self._args("--expert", expert), Path("/t/task.json"), Path("/t/m.jsonl"), "x", None)
            self.assertTrue(str(argv[2]).endswith(script), argv[2])
            self.assertEqual(R.EXPERTS[expert][1], label)

    def test_ghost_trace_only_for_the_wall_expert(self):
        """bimanual_carry.py has no --ghost-trace flag; passing it would abort the round."""
        wall = R.expert_argv(self._args("--expert", "wall"), Path("/t/t.json"), Path("/t/m"), "l", Path("/t/trace"))
        carry = R.expert_argv(self._args("--expert", "carry"), Path("/t/t.json"), Path("/t/m"), "l", Path("/t/trace"))
        self.assertIn("--ghost-trace", wall)
        self.assertNotIn("--ghost-trace", carry)

    def test_task_config_is_the_snapshot_not_the_users_path(self):
        """The expert reads <out>/task_config.json, so the round's knobs are the
        ones the provenance snapshot recorded even if the source file changes."""
        argv = [str(x) for x in R.expert_argv(self._args(), Path("/run/task_config.json"), Path("/m"), "l", None)]
        self.assertEqual(argv[argv.index("--config") + 1], "/run/task_config.json")

    def test_batch_rounds_bounds_the_expert_not_the_run(self):
        argv = [
            str(x)
            for x in R.expert_argv(
                self._args("--rounds", "30", "--batch-rounds", "10"), Path("t"), Path("m"), "l", None
            )
        ]
        self.assertEqual(argv[argv.index("--rounds") + 1], "10")

    def test_instance_zero_is_refused_without_the_flag(self):
        with self.assertRaises(SystemExit):
            R.parse_args(["--out", "/tmp/x", "--instance", "0"])
        self.assertEqual(R.parse_args(["--out", "/tmp/x", "--instance", "0", "--allow-instance-0"]).instance, 0)


class RunDirTests(unittest.TestCase):
    def test_whole_body_profile_cannot_silently_run_the_wall_default(self):
        args = R.parse_args(["--out", "/tmp/unused-crate-run", "--config", "configs/tasks/crate_visual_low_reach.json"])
        with self.assertRaisesRegex(SystemExit, "pass --expert visual"):
            R.prepare(args)

    def test_a_second_run_in_the_same_directory_is_refused(self):
        """The experts open the manifest in APPEND mode and gate.py reads the
        directory, so two pilots sharing it become one population."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "inputs.json").write_text(json.dumps(dict(manifests=[], batches=1)))
            with self.assertRaises(SystemExit):
                R.main(["--out", d, "--rounds", "1"])


if __name__ == "__main__":
    unittest.main()


class RecordTests(unittest.TestCase):
    def test_visual_pilot_records_on_its_own_instance(self):
        """--record no longer forces instance 0: start_stack.sh starts the exporter
        on any instance given RECORD_INSTANCE=1, so the visual expert records on
        the instance/GPU it holds. The batch size is the single-arm line's, for the
        exporter's thread growth against the container pid cap."""
        a = R.parse_args(["--out", "/tmp/x", "--expert", "visual", "--record", "--instance", "1"])
        self.assertEqual((a.instance, a.expert, a.record), (1, "visual", True))
        self.assertEqual(a.batch_rounds, 10)

    def test_record_reaches_the_expert(self):
        def argv(*extra):
            args = R.parse_args(["--out", "/t", *extra])
            return [str(x) for x in R.expert_argv(args, Path("t"), Path("m"), "l", None)]

        self.assertIn("--record", argv("--record"))
        self.assertNotIn("--record", argv())
