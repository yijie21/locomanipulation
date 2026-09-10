"""The instance port formula and the exporter probe (locomani.sim.ports)."""

import os
import unittest
from unittest.mock import patch

from locomani.sim import ports


class PortFormulaTests(unittest.TestCase):
    def test_instance_0_is_the_historical_layout(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ports.sonic(), 5556)
            self.assertEqual(ports.privileged(), 5581)
            self.assertEqual(ports.camera(), 5555)

    def test_instance_shifts_the_whole_block_by_ten(self):
        # The gym used to keep instance-0 numbers while start_stack.sh moved the
        # stack, and the only symptom was "no privileged state on :5581".
        with patch.dict(os.environ, {"INSTANCE": "1"}, clear=True):
            self.assertEqual(ports.sonic(), 5566)
            self.assertEqual(ports.privileged(), 5591)
            self.assertEqual(ports.keyboard(), 5590)

    def test_an_exported_port_still_wins(self):
        with patch.dict(os.environ, {"INSTANCE": "1", "TF_SONIC_PORT": "7777"}, clear=True):
            self.assertEqual(ports.sonic(), 7777)

    def test_unparsable_instance_falls_back_to_zero(self):
        with patch.dict(os.environ, {"INSTANCE": ""}, clear=True):
            self.assertEqual(ports.privileged(), 5581)


class ExporterProbeTests(unittest.TestCase):
    def test_both_entry_points_count_as_the_exporter(self):
        # start_stack.sh runs gear_sonic's run_data_exporter.py by default and
        # the pipeline overrides TF_EXPORTER_SCRIPT to run_instance_exporter.py;
        # matching only the default refused every run_learn_pilot collect on a
        # live stack (2026-09-10).
        for name in (b"run_data_exporter.py", b"pipeline/run_instance_exporter.py"):
            self.assertTrue(ports._is_exporter_cmdline(b"/venv/bin/python\x00" + name))

    def test_an_unrelated_process_does_not(self):
        self.assertFalse(ports._is_exporter_cmdline(b"/usr/bin/zsh\x00-c\x00echo exporter"))

    def test_require_exporter_names_the_instance_when_none_is_up(self):
        with (
            patch.object(ports, "exporter_pid", return_value=None),
            patch.dict(os.environ, {"INSTANCE": "1"}, clear=True),
        ):
            with self.assertRaises(SystemExit) as caught:
                ports.require_exporter("--record")
            self.assertIn("instance 1", str(caught.exception))
            self.assertIn("5565", str(caught.exception))

    def test_require_exporter_passes_when_one_is_up(self):
        with patch.object(ports, "exporter_pid", return_value=1234):
            ports.require_exporter("--record")


if __name__ == "__main__":
    unittest.main()
