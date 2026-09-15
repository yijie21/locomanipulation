"""`replay_obs` must send the eval's observation, not a lookalike.

The tool exists to answer "would the policy have closed here?" offline. That
answer is only worth anything if the observation it sends is the one
`ObsBuilder.build()` sends -- same keys, same shapes, same meaning. These tests
pin the parts that can drift silently: the key set, the dataset's state slicing
(which is NOT the live builder's ordering), and the substitution used to bisect
the train/deploy seam.

The network half is not tested here: it needs a loaded 3B checkpoint on a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomani.pipeline import replay_obs  # noqa: E402
from locomani.schema import SONIC_BOTTLE  # noqa: E402


def _obs(**over):
    obs = {
        "state.left_leg": np.zeros((1, 6), np.float32),
        "state.right_leg": np.zeros((1, 6), np.float32),
        "state.waist": np.zeros((1, 3), np.float32),
        "state.left_arm": np.zeros((1, 7), np.float32),
        "state.right_arm": np.zeros((1, 7), np.float32),
        "state.left_hand": np.zeros((1, 7), np.float32),
        "state.right_hand": np.zeros((1, 7), np.float32),
        "state.projected_gravity": np.zeros((1, 3), np.float32),
        "state.right_wrist_pos": np.zeros((1, 3), np.float32),
        "state.right_wrist_abs_quat": np.zeros((1, 4), np.float32),
        "annotation.human.task_description": "pick up the water bottle",
        "video.ego_view": np.zeros((1, 4, 4, 3), np.uint8),
        "video.right_wrist": np.zeros((1, 4, 4, 3), np.uint8),
    }
    obs.update(over)
    return obs


def test_the_key_set_matches_the_live_observation_builder():
    """Drift between this tool's key list and vla_obs.build() must fail loudly."""
    src = (Path(__file__).resolve().parents[1] / "src/locomani/backends/sonic/vla_obs.py").read_text()
    build = src[src.index("obs = {") : src.index('"annotation.human.task_description": self.task,')]
    live_state = {line.split('"')[1] for line in build.splitlines() if line.strip().startswith('"state.')}
    assert live_state == set(replay_obs.STATE_KEYS), (
        "ObsBuilder.build() and replay_obs disagree about the state keys; "
        "replaying a different observation than the eval sends answers a different question"
    )
    # The live builder takes its camera names from the task schema (SONIC_BOTTLE by default).
    assert tuple(replay_obs.VIDEO_KEYS) == tuple(f"video.{k}" for k in SONIC_BOTTLE.video_keys)


def test_check_keys_rejects_a_missing_or_extra_key():
    replay_obs.check_keys(_obs())  # the complete observation passes
    short = _obs()
    del short["state.right_hand"]
    with pytest.raises(SystemExit, match="missing="):
        replay_obs.check_keys(short)
    with pytest.raises(SystemExit, match="extra="):
        replay_obs.check_keys(_obs(**{"state.nonsense": np.zeros((1, 1), np.float32)}))


def test_dataset_slices_cover_the_43_wide_state_without_overlap():
    """The dataset interleaves the hands; the live `body` vector does not.

    left_arm, left_hand, right_arm, right_hand -- not left_arm, right_arm then
    the hands. Getting this wrong feeds the model an arm where it expects a
    gripper and is invisible downstream, so pin the exact spans.
    """
    spans = replay_obs.DATASET_STATE_SLICES
    assert spans["state.left_arm"] == (15, 22) and spans["state.left_hand"] == (22, 29)
    assert spans["state.right_arm"] == (29, 36) and spans["state.right_hand"] == (36, 43)
    covered = sorted(i for a, b in spans.values() for i in range(a, b))
    assert covered == list(range(43)), "the seven groups must tile observation.state exactly once"


def test_substitute_swaps_only_the_named_keys():
    base, donor = _obs(), _obs(**{"state.right_hand": np.full((1, 7), 9.0, np.float32)})
    out = replay_obs.substitute(base, donor, ["state.right_hand"])
    assert np.all(out["state.right_hand"] == 9.0)
    assert np.all(out["state.left_hand"] == 0.0), "substitution must not touch unnamed keys"
    assert np.all(base["state.right_hand"] == 0.0), "the base observation must not be mutated"
    with pytest.raises(SystemExit, match="no such key"):
        replay_obs.substitute(base, donor, ["state.nope"])


def test_close_threshold_agrees_with_the_round_code():
    """One number decides 'commanded a close'; two copies of it would diverge."""
    src = (Path(__file__).resolve().parents[1] / "src/locomani/backends/sonic/vla_round.py").read_text()
    assert f"{replay_obs.CLOSE_THRESHOLD}" in src, (
        "replay_obs.CLOSE_THRESHOLD must be the threshold vla_round uses to classify a close"
    )
