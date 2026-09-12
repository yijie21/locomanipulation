"""A matching case ID/seed is insufficient when delivered object yaw differs."""

import copy
import json

import pytest

from scripts.experiments.maintenance_gpu_regression_20260912 import (
    check_pairing,
    crate_coverage,
    prepare_replay,
    write_crate_acceptance,
)


def row(index, *, ok=False):
    return dict(
        case_id=str(index),
        ok=ok,
        init={"robot_xy": [0.0, 1.0]},
        appearance=index,
        target="water_bottle",
        table_top=0.6,
        obj_height=0.16,
        obj_diameter=0.06,
        execution={"metrics": {"object_yaw": index * 0.01}},
    )


@pytest.mark.parametrize("field", ["yaw", "init", "missing", "duplicate"])
def test_pairing_rejects_a_changed_or_incomplete_delivered_population(field):
    baseline = [row(0), row(1)]
    candidate = copy.deepcopy(baseline)
    if field == "yaw":
        candidate[0]["execution"]["metrics"]["object_yaw"] = 1.0
    elif field == "init":
        candidate[0]["init"]["robot_xy"][0] = 0.1
    elif field == "missing":
        candidate.pop()
    else:
        candidate.append(copy.deepcopy(candidate[0]))
    with pytest.raises(ValueError):
        check_pairing(baseline, candidate)


def test_replay_keeps_every_reference_case_and_replays_observed_yaw(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    output = tmp_path / "replay"
    output.mkdir()
    (reference / "protocol.json").write_text("{}")
    for strategy, count in [("coupled", 30), ("teleop", 10)]:
        rows = [row(i, ok=i % 3 == 0) for i in range(count)]
        bank = {
            "schema_version": 1,
            "scene_sha256": "scene",
            "cases": [{"case_id": str(i), "appearance": i} for i in range(count)],
        }
        (reference / f"{strategy}_baseline_cases.json").write_text(json.dumps(bank))
        batch = reference / f"{strategy}_baseline_0"
        batch.mkdir()
        (batch / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    prepare_replay(output, reference)
    for strategy, count in [("coupled", 30), ("teleop", 10)]:
        cases = [c for p in sorted(output.glob(f"{strategy}_cases_*.json")) for c in json.loads(p.read_text())["cases"]]
        assert len(cases) == count
        assert cases == [{"case_id": str(i), "appearance": i, "object_yaw": i * 0.01} for i in range(count)]


def test_pairing_allows_different_outcomes_with_identical_delivered_inputs():
    before, after = row(0, ok=False), row(0, ok=True)
    assert check_pairing([before], [after]) == 1


def test_crate_carry_schema_and_failed_wall_attempts_are_incomplete():
    assert not crate_coverage([dict(ok=True, yaw_err_deg=1.0)])["complete"]
    coverage = crate_coverage(
        [dict(ok=False, stand_off_meas=0.22, depth=0.11, yaw_err_deg=1.0, reach_s=3.7, descend_s=2.6)]
    )
    assert not coverage["complete"]
    assert coverage["metric_samples"]["stand_off_meas"] == 1
    assert coverage["metric_samples"]["reach_s"] == 0


def test_raw_gate_pass_is_not_accepted_when_both_crate_populations_have_no_successes(tmp_path):
    for label in ("baseline", "candidate"):
        folder = tmp_path / f"crate_{label}"
        folder.mkdir()
        (folder / "manifest_00.jsonl").write_text(json.dumps(dict(ok=False, yaw_err_deg=1.0)) + "\n")
    (tmp_path / "crate_comparison.json").write_text(json.dumps({"verdict": "PASS"}))
    result = write_crate_acceptance(tmp_path)
    assert result["verdict"] == "INCOMPLETE"
    assert result["raw_gate_verdict"] == "PASS"
    assert not result["accepted"]


def test_ready_wall_success_does_not_require_dataset_recording_or_a_walk_in_reach_time():
    rows = [dict(ok=True, produced=False, stand_off_meas=0.22, depth=0.11, yaw_err_deg=1.0, descend_s=2.6)]
    coverage = crate_coverage(rows, require_reach_time=False)
    assert coverage["complete"] and coverage["successful"] == 1
    assert coverage["not_applicable"] == ["reach_s"]
    assert not crate_coverage(rows, require_reach_time=True)["complete"]
