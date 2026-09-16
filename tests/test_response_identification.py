import json
from pathlib import Path

import numpy as np
import pytest

from locomani.pipeline.response_identification import (
    Episode,
    IdentificationInputError,
    identify_response,
    load_episode,
)


def _synthetic_episode(index: int, *, n: int = 240, delay: int = 3) -> Episode:
    rng = np.random.default_rng(100 + index)
    t = np.arange(n) * 0.02
    reference = np.zeros((n, 29))
    for joint in range(6):
        reference[:, joint] = (
            0.16 * np.sin((0.7 + 0.11 * joint) * t * 2 * np.pi + 0.4 * index)
            + 0.07 * np.sin((0.17 + 0.03 * index) * t * 2 * np.pi)
            + 0.02 * rng.normal(size=n)
        )
    alpha = np.linspace(0.60, 0.82, 29)
    bias = np.linspace(-0.025, 0.035, 29)
    measured = np.empty_like(reference)
    measured[0] = reference[0] + bias
    for k in range(1, n):
        selected = reference[max(0, k - delay)]
        measured[k] = alpha * measured[k - 1] + (1 - alpha) * (selected + bias)
    measured += rng.normal(scale=2e-4, size=measured.shape)
    wall = 1_000.0 + index * 10 + t
    return Episode(
        episode_id=f"episode-{index}",
        seq=np.arange(index * 10_000, index * 10_000 + n),
        simulation_time=t,
        state_wall=wall,
        reference_received_wall=wall - 0.006,
        reference_sample_id=np.arange(index * 1_000, index * 1_000 + n),
        reference=reference,
        measured=measured,
        pose_send_wall=wall - 0.004,
        sent_reference=reference.copy(),
        sent_frame_index=np.arange(n),
        segment_id=np.zeros(n, dtype=int),
    )


def test_identifies_known_bias_delay_and_first_order_lag():
    report = identify_response([_synthetic_episode(i) for i in range(5)], max_delay_steps=6)

    assert report["identified"] is True
    assert report["model"]["delay_steps"] == 3
    joints = report["model"]["joints"]
    np.testing.assert_allclose([joint["alpha"] for joint in joints[:6]], np.linspace(0.60, 0.82, 29)[:6], atol=0.025)
    np.testing.assert_allclose(
        [joint["static_bias_rad"] for joint in joints[:6]], np.linspace(-0.025, 0.035, 29)[:6], atol=0.003
    )
    assert report["timing"]["reference_timestamp_semantics"] == "agent_receive_wall_no_controller_ack"
    assert "no rectification" in report["signed_measured_velocity"]["note"]
    json.dumps(report)


def test_rejects_insufficient_excitation_without_model_parameters():
    episodes = [_synthetic_episode(i) for i in range(3)]
    episodes = [
        Episode(
            episode_id=episode.episode_id,
            seq=episode.seq,
            simulation_time=episode.simulation_time,
            state_wall=episode.state_wall,
            reference_received_wall=episode.reference_received_wall,
            reference_sample_id=episode.reference_sample_id,
            reference=np.zeros_like(episode.reference),
            measured=np.zeros_like(episode.measured),
            pose_send_wall=episode.pose_send_wall,
            sent_reference=np.zeros_like(episode.sent_reference),
            sent_frame_index=episode.sent_frame_index,
            segment_id=episode.segment_id,
        )
        for episode in episodes
    ]

    report = identify_response(episodes)

    assert report["identified"] is False
    assert "insufficient_reference_excitation" in report["rejection_reasons"]
    assert "model" not in report


def test_rejects_episode_inconsistent_response_instead_of_averaging_parameters():
    episodes = [_synthetic_episode(i, delay=3) for i in range(6)]
    for index in (4, 5):
        episode = episodes[index]
        rng = np.random.default_rng(900 + index)
        reference = episode.reference.copy()
        reference[:, :6] = rng.normal(scale=0.15, size=(len(reference), 6))
        measured = reference + np.linspace(-0.025, 0.035, 29)
        episodes[index] = Episode(
            episode_id=episode.episode_id,
            seq=episode.seq,
            simulation_time=episode.simulation_time,
            state_wall=episode.state_wall,
            reference_received_wall=episode.reference_received_wall,
            reference_sample_id=episode.reference_sample_id,
            reference=reference,
            measured=measured,
            pose_send_wall=episode.pose_send_wall,
            sent_reference=reference.copy(),
            sent_frame_index=episode.sent_frame_index,
            segment_id=episode.segment_id,
        )

    report = identify_response(
        episodes,
        max_delay_steps=6,
        training_ids={"episode-0", "episode-1", "episode-2", "episode-3"},
        heldout_ids={"episode-4", "episode-5"},
    )

    assert report["identified"] is False
    assert "model" not in report
    assert any(
        reason
        in {
            "held_out_static_bias_improvement_below_threshold",
            "held_out_persistence_improvement_below_threshold",
            "held_out_free_run_improvement_not_repeatable",
            "delay_not_consistent_across_held_out_episodes",
            "delay_at_search_boundary",
        }
        for reason in report["rejection_reasons"]
    )


def test_prescribed_split_fits_training_and_reports_independent_holdout():
    episodes = [_synthetic_episode(i) for i in range(6)]
    report = identify_response(
        episodes,
        max_delay_steps=6,
        training_ids={"episode-0", "episode-1", "episode-2", "episode-3"},
        heldout_ids={"episode-4", "episode-5"},
    )

    assert report["identified"] is True
    assert report["model"]["delay_steps"] == 3
    assert report["validation"]["method"] == "prescribed_episode_holdout"
    assert report["validation"]["heldout_episode_ids"] == ["episode-4", "episode-5"]
    assert report["residual_diagnostics"]["source"] == "training episodes only"


def _write_join_inputs(
    tmp_path: Path,
    *,
    mismatch: bool = False,
    duplicate_seq: bool = False,
    stream_mode: int = 1,
    reference_receive_offset_s: float = -0.005,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sidecar = tmp_path / "episode.jsonl"
    trace = tmp_path / "episode.npz"
    rows = []
    n = 30
    for i in range(n):
        seq = i if not duplicate_seq or i != n - 1 else i - 1
        wall = 100.0 + i * 0.02
        joints = np.full(29, i * 0.01)
        rows.append(
            {
                "seq": seq,
                "t": i * 0.02,
                "wall": wall,
                "joints": joints.tolist(),
                "deploy_reference": {
                    "schema_version": 1,
                    "source": "deploy_current_motion_output",
                    "timestamp_semantics": "agent_receive_wall_no_controller_ack",
                    "sample_id": i,
                    "received_wall": wall + reference_receive_offset_s,
                    "body_q_target": (joints + 0.02).tolist(),
                    "base_quat_target": [1, 0, 0, 0],
                },
                "reference_command_state": {
                    "control_owner_requested": "streamed",
                    "stream_mode_requested": stream_mode,
                    "pose_send": {
                        "sent_wall": wall - 0.003,
                        "timestamp_semantics": "agent_post_send_wall_no_deploy_ack",
                        "joint_order": "isaaclab",
                        "joint_pos": (joints + 0.01).tolist(),
                        "frame_index": i,
                    },
                },
            }
        )
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))
    trace_seq = np.arange(n)
    qpos = np.stack([np.full(29, i * 0.01) for i in range(n)])
    if mismatch:
        qpos[5, 2] += 0.1
    np.savez_compressed(trace, seq=trace_seq, qpos=qpos)
    return sidecar, trace


def test_load_episode_matches_real_trace_by_seq_with_preceding_snapshots(tmp_path):
    sidecar, trace = _write_join_inputs(tmp_path)

    episode = load_episode(sidecar, trace, np.arange(29))

    np.testing.assert_array_equal(episode.seq, np.arange(30))
    assert np.all(episode.reference_received_wall <= episode.state_wall)
    np.testing.assert_allclose(episode.measured[:, 0], np.arange(30) * 0.01)


def test_load_episode_excludes_future_snapshots_and_frozen_upper_mode(tmp_path):
    sidecar, trace = _write_join_inputs(tmp_path / "future", reference_receive_offset_s=0.005)
    episode = load_episode(sidecar, trace, np.arange(29))
    assert episode.seq[0] == 1
    assert episode.reference_sample_id[0] == 0
    assert np.all(episode.reference_received_wall <= episode.state_wall)

    sidecar, trace = _write_join_inputs(tmp_path / "frozen", stream_mode=3)
    with pytest.raises(IdentificationInputError, match="no fresh streamed-POSE"):
        load_episode(sidecar, trace, np.arange(29))


def test_load_episode_rejects_stale_publications(tmp_path):
    sidecar, trace = _write_join_inputs(tmp_path)
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    stale = dict(rows[0]["deploy_reference"], received_wall=99.0)
    for row in rows:
        row["deploy_reference"] = stale
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(IdentificationInputError, match="no fresh streamed-POSE"):
        load_episode(sidecar, trace, np.arange(29))


def test_load_episode_rejects_trace_mismatch_and_reset_collision(tmp_path):
    sidecar, trace = _write_join_inputs(tmp_path / "mismatch", mismatch=True)
    with pytest.raises(IdentificationInputError, match="sidecar/trace joint mismatch"):
        load_episode(sidecar, trace, np.arange(29))

    sidecar, trace = _write_join_inputs(tmp_path / "reset", duplicate_seq=True)
    with pytest.raises(IdentificationInputError, match="duplicate sidecar sequence"):
        load_episode(sidecar, trace, np.arange(29))
