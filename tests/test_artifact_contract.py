"""Every stage still leaves behind what it used to.

The contracts in `tests/contracts/` were recorded from runs we accept (see each file's
`recorded_from`). They are the net for the one failure shape that has no other alarm: an artifact
that quietly stops being produced. On 2026-09-17 the eval stage rendered no round video for a day
while the stage reported ok and every test passed, because nothing asserted that `eval/videos` had
anything in it.

These tests check the contracts themselves -- that they are loadable, that they describe artifacts
rather than a single run, and that a missing file is reported. Checking a real run against one is
`python scripts/artifact_contract.py check <run_dir> <name>`, which each stage's acceptance runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from artifact_contract import CONTRACTS, check, load, pattern_for  # noqa: E402

NAMES = sorted(p.stem for p in CONTRACTS.glob("*.json"))
CRITICAL = {  # the artifact each contract exists to protect, named so a deletion has to be deliberate
    "bottle_collect": "collect/videos/*.mp4",
    "bottle_convert": "train_dataset/meta/info.json",
    "bottle_convert_val": "val_dataset/meta/info.json",
    "bottle_train": "model/checkpoint-*/config.json",
    "bottle_eval": "eval/videos/*.mp4",
    "crate_collect": "videos/*.mp4",
    "crate_eval": "eval/videos/*.mp4",
}


def test_every_stage_has_a_contract():
    assert set(NAMES) == set(CRITICAL)


def test_an_unverified_contract_says_so():
    """crate_eval was written from the old path's layout and has never been checked against a run.

    The bimanual checkpoint was deleted before the flat script existed, so the first real bimanual
    evaluation is what confirms this one -- until then the file has to admit it.
    """
    assert "UNVERIFIED" in json.loads((CONTRACTS / "crate_eval.json").read_text())["note"]


@pytest.mark.parametrize("name", NAMES)
def test_contract_is_loadable_and_carries_its_provenance(name):
    doc = json.loads((CONTRACTS / f"{name}.json").read_text())
    assert doc["recorded_from"] and doc["note"]
    assert doc["required"], "an empty contract asserts nothing"
    for entry in doc["required"]:
        assert entry["min"] >= 1
        assert not entry["pattern"].startswith("/"), "patterns are relative to the run directory"


@pytest.mark.parametrize("name", NAMES)
def test_contract_protects_its_critical_artifact(name):
    assert CRITICAL[name] in [e["pattern"] for e in load(name)]


@pytest.mark.parametrize("name", NAMES)
def test_contract_generalises_over_batches_and_timestamps(name):
    """A contract taken from a 30-batch run has to fit a 3-batch one."""
    for entry in load(name):
        for part in entry["pattern"].split("/"):
            assert not (part[:1].isdigit() or part.rstrip(".jsonlmp4xy").endswith("_00")), (
                f"{name}: {entry['pattern']} pins one run's numbering"
            )


def test_check_reports_what_is_missing(tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "closed_loop.json").write_text("{}")
    missing = check(tmp_path, load("bottle_eval"))
    assert any("eval/videos/*.mp4" in line for line in missing)  # the one that went unnoticed for a day
    assert not any("eval/closed_loop.json" in line for line in missing)


def test_recorded_patterns_survive_renumbering():
    assert pattern_for(Path("collect/videos_07/gym_1789579704.mp4")) == "collect/videos_*/*.mp4"
    assert pattern_for(Path("model/checkpoint-10000/config.json")) == "model/checkpoint-*/config.json"
    assert pattern_for(Path("collect/inputs.json")) == "collect/inputs.json"  # a name a rerun repeats stays exact
