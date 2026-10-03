"""The view-based dataset-camera stage (locomani.sim.view_render, Isaac and Unreal): which rows it renders, which sessions it touches, what lands in the videos.

Kit is not needed: the view and the video encoder are fakes, the way test_isaac_view drives the view.
"""

import json

import numpy as np

from locomani.pipeline import sim_video
from locomani.pipeline.world_lift import resample_selection
from locomani.sim import view_render as R
from locomani.sim.dataset_cameras import CAMERA_CONFIGS


def test_rendered_rows_cover_every_prefix_the_verification_trim_can_keep():
    full = R.rendered_rows(101)
    for kept in (1, 2, 37, 100, 101):
        assert set(resample_selection(kept).tolist()) <= full


def test_only_finished_batches_are_touched(tmp_path):
    (tmp_path / "collect").mkdir()
    (tmp_path / "collect" / "inputs.json").write_text(json.dumps({"batches": 2}))
    assert R.finished_sessions(tmp_path, ["a", "b", "c"]) == ["a", "b"]


class FakeView:
    def __init__(self):
        self.rendered, self.warmed = [], []

    def warm_up(self, qpos):
        self.warmed.append(float(qpos[0]))

    def render(self, qpos):
        self.rendered.append(float(qpos[0]))
        return {k: np.full((c["height"], c["width"], 3), int(qpos[0]), np.uint8) for k, c in CAMERA_CONFIGS.items()}


class FakeTraces:
    def __init__(self, priv, tseq):
        self.priv, self.tseq = np.asarray(priv), np.asarray(tseq)

    def priv_seq(self, session, ep):
        return self.priv

    def trace_rows(self, session, ep, seqs, hold_tail=False):
        seqs = np.minimum(seqs, self.tseq[-1]) if hold_tail else seqs
        return {"seq": self.tseq, "qpos": self.tseq[:, None].astype(float)}, np.searchsorted(self.tseq, seqs)


class FakeEncoder:
    def __init__(self, dst, width, height, fps):
        self.dst, self.frames = dst, []
        outer = self

        class Stdin:
            def write(self, data):
                outer.frames.append(data[0])

            def close(self):
                outer.dst.parent.mkdir(parents=True, exist_ok=True)
                outer.dst.write_bytes(bytes(outer.frames))

        self.stdin = Stdin()

    def wait(self):
        return 0


def test_render_episode_renders_the_picked_rows_and_repeats_them_between(tmp_path, monkeypatch):
    session = tmp_path / "session"
    (session / "meta").mkdir(parents=True)
    # rows 0..6 used seqs 10,10,11,12,13,14,15; the trace ends at 14, so row 6 holds its last sample
    (session / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 3, "length": 7}) + "\n")
    info = {"fps": 50, "chunks_size": 1000, "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4"}
    monkeypatch.setattr(R, "encoder", FakeEncoder)
    view = FakeView()
    receipt = R.render_episode(
        view, FakeTraces([10, 10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14]), session, 3, info, tuple(CAMERA_CONFIGS)
    )
    # the pick keeps rows 0, 2, 4, 6 -> seqs 10, 11, 13, 14 (held)
    assert view.warmed == [10.0] and view.rendered == [10.0, 11.0, 13.0, 14.0]
    assert receipt["renders"] == 4 and receipt["rows"] == 7 and receipt["held_tail_rows"] == 1
    for key in CAMERA_CONFIGS:
        video = session / f"videos/observation.images.{key}/episode_000003.mp4"
        assert list(video.read_bytes()) == [10, 10, 11, 11, 13, 13, 14]


def test_fill_does_not_open_a_session_with_nothing_pending(tmp_path):
    gone = tmp_path / "gone"  # never created: an empty batch's session whose instance root was cleaned
    R.fill(FakeView(), FakeTraces([0], [0]), {gone: []}, "isaac", {})
    assert not gone.exists()


# --- the third-person video (film) ---------------------------------------------------------------------------------

SHOT_SCENE = (
    '<mujoco><worldbody><body name="pelvis" pos="0 0 0.8"><freejoint/><geom size="0.1"/></body>'
    '<camera name="reverse_room" pos="-1.4 1.3 1.7" xyaxes="1 -1 0 0.3 0.3 1" fovy="60"/></worldbody></mujoco>'
)


class FilmTraces:
    """A run with one kept, filled episode (label L) of 12 trace rows; parquet rows use seqs 100..111 once each."""

    def __init__(self, run):
        self.run = run
        self.scene_dir = run / "scene"
        self.trace_dir = run / "trace"
        self.scene_dir.mkdir(parents=True)
        self.trace_dir.mkdir()
        (self.scene_dir / "scene_locomani.xml").write_text(SHOT_SCENE)
        (self.scene_dir / "scene_meta.json").write_text(json.dumps({"video_camera": "reverse_room"}))
        qpos = np.tile([0, 0, 0.8, 1, 0, 0, 0], (12, 1)).astype(float)
        qpos[:, 0] = np.arange(12) / 10
        np.savez(self.trace_dir / "L_state_trace.npz", qpos=qpos, seq=np.arange(100, 112))
        pos = np.stack([np.full((2, 3), s) for s in range(100, 112)]).astype(float)
        np.savez(
            self.trace_dir / "L_ghost.npz", seq=np.arange(100, 112), pos=pos, quat=np.tile([1.0, 0, 0, 0], (12, 2, 1))
        )

    def label(self, session, ep):
        return "L" if ep == 0 else None

    def priv_seq(self, session, ep):
        return np.arange(100, 112)

    def trace_rows(self, session, ep, seqs, hold_tail=False):
        return None, np.asarray(seqs) - 100


class FilmView:
    def __init__(self):
        self.setup, self.warmed, self.filmed = None, [], []

    def film_setup(self, width, height, fovy):
        self.setup = (width, height, fovy)

    def film_warm_up(self, qpos, camera, ghosts):
        self.warmed.append(float(qpos[0]))

    def film(self, qpos, camera, ghosts):
        self.filmed.append((round(float(qpos[0]), 3), float(ghosts[0][0][0])))
        return np.full((sim_video.HEIGHT, sim_video.WIDTH, 3), 7, np.uint8)


def film_run(tmp_path, filled):
    run = tmp_path / "run"
    traces = FilmTraces(run)
    session = run / "session"
    (session / "meta").mkdir(parents=True)
    (session / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 12}) + "\n")
    (session / "meta" / "info.json").write_text(json.dumps({"fps": 50}))
    if filled:
        (session / R.MARKER).write_text(json.dumps({"episodes": [0], "renderer": "isaac"}))
    return traces, session


def test_only_rounds_this_renderer_filled_take_their_insets_from_the_dataset(tmp_path, monkeypatch):
    import locomani.pipeline.world_lift as world_lift

    traces, session = film_run(tmp_path, filled=False)
    traces.sessions = [str(session), str(tmp_path / "gone")]  # a vanished session is not an error
    monkeypatch.setattr(world_lift, "RunTraces", lambda run: traces)
    assert R.filled_insets(traces.run, "isaac", ["L"]) == {}  # not filled: film_rounds renders them
    (session / R.MARKER).write_text(json.dumps({"episodes": [0], "renderer": "unreal"}))
    assert R.filled_insets(traces.run, "isaac", ["L"]) == {}  # another renderer's pictures never stand in
    (session / R.MARKER).write_text(json.dumps({"episodes": [0], "renderer": "isaac"}))
    assert set(R.filled_insets(traces.run, "isaac", ["L"])) == {"L"}
    assert R.filled_insets(traces.run, "isaac", ["other"]) == {}


def test_a_session_drawn_with_another_robot_look_is_never_topped_up(tmp_path, monkeypatch):
    import pytest

    import locomani.pipeline.world_lift as world_lift
    from locomani.sim import robot_look

    traces, session = film_run(tmp_path, filled=False)
    traces.sessions = [str(session)]
    monkeypatch.setattr(world_lift, "RunTraces", lambda run: traces)
    cfg = {"backend": {"render": "offline"}, "images": {"source": "isaac"}}
    look = {"isaac_stage": "stage", "robot_look": robot_look.NAME}
    marker = {"episodes": [0], "renderer": "isaac", "isaac_stage": "stage"}
    for old in ({}, {"robot_look": "unitree_20260929"}):  # a marker from before robot_look, and another look
        (session / R.MARKER).write_text(json.dumps({**marker, **old}))
        with pytest.raises(SystemExit, match="another isaac look .*robot_look"):
            R.plan(traces.run, cfg, "isaac", look)
    (session / R.MARKER).write_text(json.dumps({**marker, "robot_look": robot_look.NAME}))
    assert list(R.plan(traces.run, cfg, "isaac", look)[1]) == [session]


def test_a_session_drawn_with_another_robot_is_never_topped_up(tmp_path, monkeypatch):
    import pytest

    import locomani.pipeline.world_lift as world_lift
    from locomani.sim import robot_look

    traces, session = film_run(tmp_path, filled=False)
    traces.sessions = [str(session)]
    monkeypatch.setattr(world_lift, "RunTraces", lambda run: traces)
    cfg = {"backend": {"render": "offline"}, "images": {"source": "unreal"}}
    marker = {"episodes": [0], "renderer": "unreal", "unreal_level": "lv", "robot_look": robot_look.NAME}
    # a marker from before the robot was recorded drew the stock robot: it tops up a stock run, never a mars one
    (session / R.MARKER).write_text(json.dumps(marker))
    look = {"unreal_level": "lv", "robot_look": robot_look.NAME}
    assert list(R.plan(traces.run, cfg, "unreal", {**look, "robot": "stock"})[1]) == [session]
    with pytest.raises(SystemExit, match="another unreal look .*robot"):
        R.plan(traces.run, cfg, "unreal", {**look, "robot": "mars"})
    (session / R.MARKER).write_text(json.dumps({**marker, "robot": "mars"}))
    assert list(R.plan(traces.run, cfg, "unreal", {**look, "robot": "mars"})[1]) == [session]


def test_the_insets_are_the_filled_frames_of_the_first_parquet_row_at_or_past_each_filmed_row(tmp_path):
    import cv2

    traces, session = film_run(tmp_path, filled=True)
    info = {"fps": 50, "chunks_size": 1000, "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4"}
    for key, c in CAMERA_CONFIGS.items():
        path = session / f"videos/observation.images.{key}/episode_000000.mp4"
        path.parent.mkdir(parents=True)
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 50, (c["width"], c["height"]))
        for row in range(12):
            out.write(np.full((c["height"], c["width"], 3), 20 * row, np.uint8))
        out.release()
    got = R.inset_frames(traces, session, 0, info, [3, 7, 11])
    assert set(got) == {c["mjcf_name"] for c in CAMERA_CONFIGS.values()}
    for images in got.values():
        assert [round(float(img.mean()) / 20) for img in images] == [3, 7, 11]


# --- film_rounds: one video per round, insets from the dataset or rendered first --------------------------------


class RenderingView(FilmView):
    """Renders the dataset cameras until film_setup, as the Isaac view (which cannot switch back) must."""

    def __init__(self):
        super().__init__()
        self.rendered = []

    def warm_up(self, qpos):
        assert self.setup is None

    def render(self, qpos):
        assert self.setup is None, "a dataset render after film_setup"
        self.rendered.append(round(float(qpos[0]), 3))
        return {k: np.full((c["height"], c["width"], 3), 60, np.uint8) for k, c in CAMERA_CONFIGS.items()}


def test_film_rounds_renders_missing_insets_first_then_films_every_recorder_row_with_its_ghosts(tmp_path, monkeypatch):
    from locomani.pipeline import sim_video as sv

    run = tmp_path / "run"
    traces = FilmTraces(run)  # scene/ and a 12-row trace L with its ghosts
    shown = {}

    def filled(rows):  # the dataset's own frames, as filled_insets hands them over
        shown["rows"] = list(rows)
        return iter([{name: np.full((h, w, 3), 200, np.uint8) for name, _l, w, h, _x, _y in sv.INSETS}] * len(rows))

    made = []
    import locomani.pipeline.video_preview as vp

    monkeypatch.setattr(vp, "make_previews", lambda src, stem: made.append((src.name, stem)) or [(src, 0.0, 1.0)])
    trace = traces.trace_dir / "L_state_trace.npz"
    out, previews = run / "videos_isaac", run / "previews_isaac"
    out.mkdir()
    view = RenderingView()
    R.film_rounds(view, traces.scene_dir, [("A", trace, filled), ("B", trace, None)], "isaac", out, previews, log=print)
    assert view.rendered == [0.3, 0.7, 1.1]  # B's insets only, all before film_setup
    assert view.setup == (sv.WIDTH, sv.HEIGHT, 60.0) and shown["rows"] == [3, 7, 11]
    assert view.filmed == [(0.3, 103.0), (0.7, 107.0), (1.1, 111.0)] * 2  # every 4th row, each row's own ghosts
    assert view.warmed == [0.3, 0.3]
    assert sorted(p.name for p in out.iterdir()) == ["A.mp4", "B.mp4"]  # the staged insets are gone
    assert made == [("A.mp4", previews / "A"), ("B.mp4", previews / "B")]
