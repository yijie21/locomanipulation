"""World object height from the run's state trace, keyed by the exporter's priv_seq."""

import json

import numpy as np
import pytest

from locomani.pipeline.world_lift import RunTraces, resample_selection


@pytest.mark.parametrize("n", [1, 2, 3, 4, 50, 101, 946])
def test_resample_selection_is_the_nearest_row_pick_the_converter_uses(n):
    # convert_to_hf_format imports this function, so the converter and the
    # trace lookup cannot drift apart; here only the pick's shape is pinned.
    sel = resample_selection(n)
    assert sel[0] == 0 and sel[-1] <= n - 1 and len(sel) == int(np.floor((n - 1) / 2)) + 1
    assert np.all(np.diff(sel) == 2) if n > 2 else True


def test_world_z_joins_manifest_label_priv_seq_and_trace(tmp_path):
    run, session = tmp_path / "run", tmp_path / "session"
    (run / "scene").mkdir(parents=True)
    (run / "trace").mkdir()
    (session / "priv_seq").mkdir(parents=True)
    (run / "scene/scene_meta.json").write_text(
        json.dumps({"target_object": "crate", "objects": {"crate": {"qpos_adr": 5}}})
    )
    (run / "inputs.json").write_text(
        json.dumps({"sessions": [str(session)], "manifests": [str(run / "manifest_00.jsonl")]})
    )
    rows = [
        {"label": "a", "produced": True},
        {"label": "skipped", "produced": False},  # never recorded: not an episode
        {"label": "b", "produced": True},
    ]
    (run / "manifest_00.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    seq = np.arange(100, 130)
    qpos = np.zeros((30, 9))
    qpos[:, 7] = 0.3 + 0.01 * np.arange(30)  # object z = qpos[adr + 2]
    np.savez(run / "trace/b_state_trace.npz", seq=seq, qpos=qpos)
    (session / "priv_seq/episode_000001.json").write_text(json.dumps({"priv_seq": list(range(110, 120))}))
    z = RunTraces(run).world_z(str(session), 1, kept_rows=8)
    # 8 kept source rows -> 4 converted rows at seqs 110, 112, 114, 116
    np.testing.assert_allclose(z, 0.3 + 0.01 * np.array([10, 12, 14, 16]))
    # One or two rows before the trace start take the first sample (the 'c' key
    # and the trace start race across processes); a larger gap is a join error.
    (session / "priv_seq/episode_000001.json").write_text(json.dumps({"priv_seq": [99, 100, 101, 102]}))
    np.testing.assert_allclose(
        RunTraces(run).world_z(str(session), 1), [0.3, 0.3 + 0.01]
    )  # rows 99 -> edge (seq 100), 101
    with pytest.raises(ValueError, match="not all present"):
        (session / "priv_seq/episode_000001.json").write_text(json.dumps({"priv_seq": [90, 100, 101, 102]}))
        RunTraces(run).world_z(str(session), 1)
    with pytest.raises(ValueError, match="not all present"):
        (session / "priv_seq/episode_000001.json").write_text(json.dumps({"priv_seq": [128, 130, 135]}))
        RunTraces(run).world_z(str(session), 1)


def test_trace_rows_holds_the_tail_the_exporter_records_after_the_trace_stops(tmp_path):
    run, session = tmp_path / "run", tmp_path / "session"
    (run / "scene").mkdir(parents=True)
    (run / "trace").mkdir()
    (session / "priv_seq").mkdir(parents=True)
    (run / "scene/scene_meta.json").write_text(
        json.dumps({"target_object": "box", "objects": {"box": {"qpos_adr": 0}}})
    )
    (run / "inputs.json").write_text(
        json.dumps({"sessions": [str(session)], "manifests": [str(run / "manifest_00.jsonl")]})
    )
    (run / "manifest_00.jsonl").write_text(json.dumps({"label": "a", "produced": True}) + "\n")
    np.savez(run / "trace/a_state_trace.npz", seq=np.arange(100, 110), qpos=np.zeros((10, 7)))
    seqs = np.array([100, 101, 101, 109, 115, 140])  # a repeated seq, then rows past the trace's end
    with pytest.raises(ValueError, match="not all present"):
        RunTraces(run).trace_rows(str(session), 0, seqs)
    _, idx = RunTraces(run).trace_rows(str(session), 0, seqs, hold_tail=True)
    assert idx.tolist() == [0, 1, 1, 9, 9, 9]


def test_join_by_seq_window_past_an_empty_current_layout_dir(tmp_path):
    # Single-arm manifests carry no label, so the episode is joined by the trace
    # whose seq window holds it. bottle_simple_200_20260916 has its traces in
    # collect/state_trace_NN plus an EMPTY collect/state_trace a later stack made;
    # the empty one used to win the layout check and every join failed.
    run, session = tmp_path / "run", tmp_path / "session"
    (run / "scene").mkdir(parents=True)
    (run / "collect/state_trace").mkdir(parents=True)
    (run / "collect/state_trace_00").mkdir()
    (session / "priv_seq").mkdir(parents=True)
    (run / "scene/scene_meta.json").write_text(
        json.dumps({"target_object": "water_bottle", "objects": {"water_bottle": {"qpos_adr": 5}}})
    )
    (run / "collect/inputs.json").write_text(
        json.dumps({"sessions": [str(session)], "manifests": [str(run / "collect/manifest_00.jsonl")]})
    )
    (run / "collect/manifest_00.jsonl").write_text(json.dumps({"produced": True}) + "\n")
    for first in (0, 200):  # two traces of one batch, disjoint seq windows
        seq = np.arange(first, first + 50)
        qpos = np.tile(seq[:, None].astype(float), (1, 9))
        np.savez(run / f"collect/state_trace_00/g{first}_state_trace.npz", seq=seq, qpos=qpos, qvel=-qpos[:, :8])
    (session / "priv_seq/episode_000000.json").write_text(json.dumps({"priv_seq": list(range(210, 216))}))
    # 6 source rows -> 3 converted rows at seqs 210, 212, 214, from the second trace
    np.testing.assert_array_equal(RunTraces(run).world_z(str(session), 0), [210, 212, 214])
