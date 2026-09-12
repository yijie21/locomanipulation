"""The active SONIC checkpoint is one decision, made in one place.

2026-09-11 (user ruling): the default is `policy/sonic_v1_1`, not
`policy/release`. The failure this pins down is not "the wrong model runs" --
that would be visible -- but the SPLIT: the deploy booting one checkpoint while
an offline encoder/decoder resolves another. The token is a checkpoint-private
latent, so a split produces numbers that look fine and mean nothing.
"""

import importlib
import os
import re
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from locomani import paths

REPO = Path(__file__).resolve().parents[1]
DEFAULT = "policy/sonic_v1_1"
SHELL_DEFAULTS = {
    "scripts/paths.sh": r'export SONIC_POLICY="\$\{SONIC_POLICY:-([^}"]+)\}"',
    "scripts/stack/start_stack.sh": r'SONIC_POLICY="\$\{SONIC_POLICY:-([^}"]+)\}"',
    "scripts/restart_deploy.sh": r'SONIC_POLICY="\$\{SONIC_POLICY:-([^}"]+)\}"',
}


class DefaultCheckpointTests(unittest.TestCase):
    def test_every_shell_entry_point_defaults_to_the_same_checkpoint(self):
        for rel, pattern in SHELL_DEFAULTS.items():
            found = re.findall(pattern, (REPO / rel).read_text())
            self.assertEqual(found, [DEFAULT], f"{rel} default is {found}, expected [{DEFAULT!r}]")

    def test_python_half_agrees_with_the_shell_half(self):
        env = {k: v for k, v in os.environ.items() if k != "SONIC_POLICY"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            fresh = importlib.reload(paths)
            try:
                self.assertEqual(fresh.SONIC_POLICY, DEFAULT)
                self.assertTrue(fresh.SONIC_POLICY_DIR.endswith(DEFAULT))
                self.assertTrue(fresh.sonic_policy_file("model_decoder.onnx").endswith("model_decoder.onnx"))
            finally:
                importlib.reload(paths)  # restore the module other tests imported

    def test_the_env_override_still_works(self):
        with unittest.mock.patch.dict(os.environ, {"SONIC_POLICY": "policy/release"}):
            fresh = importlib.reload(paths)
            try:
                self.assertTrue(fresh.SONIC_POLICY_DIR.endswith("policy/release"))
            finally:
                importlib.reload(paths)

    def test_the_deploy_launchers_check_the_checkpoint_exists(self):
        # A foreign deploy tree (SIMPLE's own lockstep build) has policy/release
        # only, and $SONIC_POLICY resolves against the deploy's cwd -- without a
        # guard that reads as a stack timeout, not a missing file.
        for rel in ("scripts/stack/start_stack.sh", "scripts/restart_deploy.sh"):
            text = (REPO / rel).read_text()
            self.assertIn('if [ ! -f "$SONIC_POLICY/model_decoder.onnx" ]; then', text, rel)


class EncoderLayoutTests(unittest.TestCase):
    """The layout comes from the checkpoint's own config, and the two differ."""

    def _layout(self, policy):
        from locomani.backends.sonic.sonic_encoder import EncoderLayout

        cfg = Path(paths.SONIC_DEPLOY) / policy / "observation_config.yaml"
        if not cfg.is_file():
            self.skipTest(f"{cfg} not downloaded")
        return EncoderLayout.from_config(str(cfg))

    def test_v1_1_layout(self):
        lay = self._layout("policy/sonic_v1_1")
        self.assertEqual(lay.dim, 1751)
        self.assertEqual(lay.live_width(), 644)
        self.assertEqual(lay.anchor_block, "motion_anchor_orientation_heading_10frame_step5")
        self.assertEqual(lay.offsets[lay.anchor_block], (584, 60))
        self.assertTrue(lay.heading_anchor)

    def test_release_layout_is_still_readable_and_different(self):
        lay = self._layout("policy/release")
        self.assertEqual(lay.dim, 1762)
        self.assertEqual(lay.live_width(), 644)
        self.assertEqual(lay.offsets[lay.anchor_block], (601, 60))
        self.assertFalse(lay.heading_anchor)


class HeadingAnchorTests(unittest.TestCase):
    """orientation_mode 1 (cpp:684) takes the robot's heading, not its attitude."""

    def test_heading_quat_keeps_yaw_and_drops_pitch(self):
        from locomani.backends.sonic.sonic_encoder import heading_quat

        yaw = 0.4
        qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        np.testing.assert_allclose(heading_quat(qz), qz, atol=1e-12)
        pitch = 0.3
        qy = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
        np.testing.assert_allclose(heading_quat(qy), [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_the_two_anchor_conventions_disagree_under_pitch(self):
        from locomani.backends.sonic.sonic_encoder import anchor_6d

        pitch = 0.25
        base = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
        ref = np.array([1.0, 0.0, 0.0, 0.0])
        full = anchor_6d(base, ref, heading_only=False)
        head = anchor_6d(base, ref, heading_only=True)
        self.assertGreater(np.abs(full - head).max(), 0.1)
        # heading-only on a pure-pitch base is the identity relative rotation
        np.testing.assert_allclose(head, [1, 0, 0, 1, 0, 0], atol=1e-12)


class ProvenanceTests(unittest.TestCase):
    def test_the_snapshot_records_which_checkpoint_ran(self):
        from locomani.pipeline.provenance import sonic_policy_identity

        ident = sonic_policy_identity()
        self.assertEqual(ident["policy"], paths.SONIC_POLICY)
        self.assertTrue(ident["dir"].endswith(paths.SONIC_POLICY))


if __name__ == "__main__":
    unittest.main()


class GaitBandTests(unittest.TestCase):
    """The approach's speeds follow the checkpoint, not a carried-over literal.

    Stage A (ledger bottle.md 21): the command -> velocity map is a step whose
    position differs by 0.065 between the checkpoints, so `APPR_COARSE = 0.30`
    delivers 0.203 m/s on release and 0.044 on v1.1.
    """

    def test_release_bands_are_the_historical_production_triple(self):
        from locomani.backends.sonic import gait_speeds

        # Changing this row silently re-dates every pre-2026-09-11 dataset.
        self.assertEqual(gait_speeds.approach_bands("release"), (0.30, 0.215, 0.20))

    def test_v1_1_coarse_is_above_its_own_gait_step(self):
        from locomani.backends.sonic import gait_speeds

        coarse, mid, shuffle = gait_speeds.approach_bands("sonic_v1_1")
        self.assertEqual((mid, shuffle), (0.215, 0.20), "only COARSE was re-picked in Stage B")
        # v1.1 steps between 0.28 (0.007 m/s) and 0.30 (0.044): COARSE must land
        # ABOVE the step and deliver about what release's COARSE delivered.
        self.assertGreaterEqual(gait_speeds.delivered(coarse, "sonic_v1_1"), 0.15)
        release_coarse = gait_speeds.delivered(0.30, "release")
        self.assertGreater(gait_speeds.delivered(coarse, "sonic_v1_1"), 0.8 * release_coarse)

    def test_an_unmeasured_checkpoint_is_refused_not_defaulted(self):
        from locomani.backends.sonic import gait_speeds

        with self.assertRaises(KeyError) as cm:
            gait_speeds.approach_bands("low_latency")
        self.assertIn("diag_shuffle", str(cm.exception))

    def test_the_approach_reads_the_table_instead_of_literals(self):
        src = (REPO / "src/locomani/tasks/bottle/round_phases_approach.py").read_text()
        self.assertIn("gait_speeds.approach_bands()", src)
        self.assertNotIn("APPR_COARSE, APPR_MID, APPR_SHUFFLE = 0.30", src)

    def test_every_dormant_speed_site_reads_the_table(self):
        # A dormant path keeping release-era literals is how a checkpoint switch
        # comes back to bite months later (creep: no production caller;
        # recovery: recovery_attempts 0 in the collection profiles).
        for rel in (
            "src/locomani/tasks/bottle/execution.py",
            "src/locomani/tasks/bottle/recoverable_grasp_policy.py",
        ):
            src = (REPO / rel).read_text()
            self.assertIn("gait_speeds.approach_bands()", src, rel)
            self.assertNotIn("speed = 0.215", src, rel)
            self.assertNotIn("speed=0.20 if distance", src, rel)
