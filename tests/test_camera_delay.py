"""images.camera_delay (user, 2026-10-06): the offline re-render draws each row from an earlier state, as late as a
real camera's picture. Pure numpy and the render stages' fakes; no renderer."""

import json

import numpy as np
import pytest

from locomani.pipeline import camera_delay as C
from locomani.pipeline import run_context
from locomani.sim import view_render as R
from locomani.sim.dataset_cameras import CAMERA_CONFIGS

DELAY = {"fps": 30, "latency_ms": [20, 30]}


def test_parse_takes_the_documented_form_and_refuses_the_rest():
    assert C.parse(None) is None
    assert C.parse(DELAY) == C.CameraDelay(30.0, (20.0, 30.0))
    for bad in ({"fps": 30}, {"fps": 0, "latency_ms": [0, 1]}, {"fps": 60, "latency_ms": [0, 1]},
                {"fps": 30, "latency_ms": [30, 20]}, {"fps": 30, "latency_ms": [-1, 5]},
                {"fps": 30, "latency_ms": [20, 30], "jitter": 1}):  # fmt: skip
        with pytest.raises(SystemExit, match="images.camera_delay"):
            C.parse(bad)


def test_each_picture_is_a_latency_plus_at_most_a_frame_old_and_never_newer_than_the_one_before():
    tseq = np.arange(500, 1500)  # one seq per 20 ms
    idx = np.arange(len(tseq))
    delay = C.parse(DELAY)
    cam, draw = C.delayed_rows(delay, tseq, idx, "session/3")
    latency = draw["latency_ms"] / 1e3
    assert 0.020 <= latency <= 0.030 and 0 <= draw["phase_ms"] < 1e3 / 30
    assert np.all(cam <= idx) and np.all(np.diff(cam) >= 0)
    age = (tseq[idx] - tseq[cam]) * C.TICK_S
    settled = idx * C.TICK_S > 0.1  # past the trace start, where the latency would reach before the first sample
    assert np.all(age[settled] >= latency - 1e-9)
    assert np.all(age[settled] < latency + 1 / 30 + C.TICK_S + 1e-9)
    # a 30 fps camera: about 30 distinct pictures a second of 50 Hz rows
    assert abs(len(np.unique(cam[100:150])) - 30) <= 1
    # the same episode draws the same rows; another episode its own latency and phase
    assert np.array_equal(C.delayed_rows(delay, tseq, idx, "session/3")[0], cam)
    assert C.delayed_rows(delay, tseq, idx, "session/4")[1] != draw


def test_no_delay_moves_nothing():
    idx = np.array([0, 0, 1, 2, 3])
    cam, draw = C.delayed_rows(None, np.arange(10, 14), idx, "s/0")
    assert np.array_equal(cam, idx) and draw == {}


def test_the_config_takes_camera_delay_only_beside_an_offline_render():
    cfg = {"backend": {"render": "offline"}, "images": {"source": "unreal", "camera_delay": DELAY}}
    assert run_context.migrate_images(cfg)["images"]["camera_delay"] == DELAY
    with pytest.raises(SystemExit, match="live collection records the sim's own frames"):
        run_context.migrate_images(
            {"backend": {"render": "live"}, "images": {"source": "mujoco", "camera_delay": DELAY}}
        )
    with pytest.raises(SystemExit, match="images must be"):
        run_context.migrate_images({"backend": {"render": "offline"}, "images": {"source": "unreal", "delay": DELAY}})


def test_the_view_render_draws_the_delayed_rows_and_records_the_draw(tmp_path, monkeypatch):
    from test_view_render import FakeEncoder, FakeTraces, FakeView

    session = tmp_path / "session"
    (session / "meta").mkdir(parents=True)
    n = 60
    (session / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": n}) + "\n")
    info = {"fps": 25, "chunks_size": 1000, "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4"}
    monkeypatch.setattr(R, "encoder", FakeEncoder)
    tseq = list(range(100, 100 + n))
    plain, delayed = FakeView(), FakeView()
    R.render_episode(plain, FakeTraces(tseq, tseq), session, 0, info, tuple(CAMERA_CONFIGS))
    receipt = R.render_episode(
        delayed, FakeTraces(tseq, tseq), session, 0, info, tuple(CAMERA_CONFIGS), delay=C.parse(DELAY)
    )
    assert delayed.rendered != plain.rendered and max(delayed.rendered) < max(plain.rendered)
    assert receipt["camera_delay"]["age_ms"][1] >= 20


def test_a_session_drawn_with_another_delay_is_never_topped_up(tmp_path, monkeypatch):
    from test_view_render import film_run

    import locomani.pipeline.world_lift as world_lift

    traces, session = film_run(tmp_path, filled=False)
    traces.sessions = [str(session)]
    monkeypatch.setattr(world_lift, "RunTraces", lambda run: traces)
    cfg = {"backend": {"render": "offline"}, "images": {"source": "unreal", "camera_delay": DELAY}}
    look = {"unreal_level": "lv", **R.delay_look(cfg)}
    # a marker from before the delay existed drew pictures in step with the state
    (session / R.MARKER).write_text(json.dumps({"episodes": [0], "renderer": "unreal", "unreal_level": "lv"}))
    with pytest.raises(SystemExit, match="another unreal look .*camera_delay"):
        R.plan(traces.run, cfg, "unreal", look)
    assert list(R.plan(traces.run, cfg, "unreal", {**look, "camera_delay": None})[1]) == [session]


def test_the_audit_expects_the_configured_lag_and_still_refuses_a_missing_one():
    from locomani.pipeline.check_dataset import video_aligned

    expected = C.mean_lag_frames(C.parse(DELAY))
    assert 1.0 < expected < 1.6  # 20-30 ms + half of a 30 fps frame + half a 20 ms tick, at 25 fps
    assert C.mean_lag_frames(None) == 0.0
    assert C.mean_lag_frames(C.parse(DELAY), 50) == pytest.approx(2 * expected)  # a 50 fps dataset counts twice
    delayed, in_step = [1, 1, 2, 2, 2, 2, 2, 3], [0, 0, 0, 0, 0, 0, 1, 2]  # 2026-10-06's two audits
    assert video_aligned(delayed, expected) and not video_aligned(delayed)
    assert video_aligned(in_step) and not video_aligned(in_step, expected)
