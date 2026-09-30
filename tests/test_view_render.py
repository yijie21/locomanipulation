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


def test_only_filled_episodes_without_a_video_are_filmed(tmp_path):
    traces, session = film_run(tmp_path, filled=False)
    assert R.film_todo(traces, [session], "isaac") == []  # its insets are its filled frames: wait for the fill
    (session / R.MARKER).write_text(json.dumps({"episodes": [0], "renderer": "isaac"}))
    assert R.film_todo(traces, [session], "isaac") == [(session, 0, "L")]
    (traces.run / "videos_isaac").mkdir()
    (traces.run / "videos_isaac" / "L.mp4").write_bytes(b"x")
    assert R.film_todo(traces, [session], "isaac") == []


def test_film_draws_every_recorder_row_with_its_ghosts_and_the_dataset_frames_as_insets(tmp_path, monkeypatch):
    from locomani.pipeline import sim_video as sv

    traces, session = film_run(tmp_path, filled=True)
    monkeypatch.setattr(R, "encoder", FakeEncoder)
    shown = {}

    def insets(traces_, session_, ep, info, rows):
        shown["rows"] = list(rows)
        return {name: [np.full((h, w, 3), 200, np.uint8)] * len(rows) for name, _l, w, h, _x, _y in sv.INSETS}

    monkeypatch.setattr(R, "inset_frames", insets)
    view = FilmView()
    R.film(view, traces, R.film_todo(traces, [session], "isaac"), "isaac", log=lambda m: None)
    assert view.setup == (sv.WIDTH, sv.HEIGHT, 60.0)
    assert shown["rows"] == [3, 7, 11] and view.warmed == [0.3]
    assert view.filmed == [(0.3, 103.0), (0.7, 107.0), (1.1, 111.0)]  # each row's own ghost targets
    video = traces.run / "videos_isaac" / "L.mp4"
    assert video.exists() and not (traces.run / "videos_isaac" / "L.rendering.mp4").exists()


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


# --- a closed-loop evaluation's rounds (film_eval) -----------------------------------------------------------------


def eval_run(tmp_path):
    run = tmp_path / "run"
    traces = FilmTraces(run)  # scene/ and a 12-row trace L with its ghosts
    (run / "eval" / "state_trace").mkdir(parents=True)
    for label in ("crate_vla_000", "crate_vla_001"):
        (run / "eval" / "state_trace" / f"{label}_state_trace.npz").hardlink_to(run / "trace" / "L_state_trace.npz")
    rows = [{"video": "crate_vla_000", "ok": False}, {"video": "crate_vla_001", "ok": True}]
    (run / "eval" / "closed_loop.manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, traces.scene_dir


def test_eval_rounds_are_the_successful_ones_unless_all_and_never_a_filmed_one(tmp_path):
    run, _scene = eval_run(tmp_path)
    assert R.eval_rounds(run, "unreal", every=False) == ["crate_vla_001"]
    assert R.eval_rounds(run, "unreal", every=True) == ["crate_vla_000", "crate_vla_001"]
    (run / "eval" / "videos_unreal").mkdir()
    (run / "eval" / "videos_unreal" / "crate_vla_001.mp4").write_bytes(b"x")
    assert R.eval_rounds(run, "unreal", every=True) == ["crate_vla_000"]


class EvalView(FilmView):
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


def test_film_eval_renders_every_rounds_insets_first_then_films_and_leaves_videos_and_previews(tmp_path, monkeypatch):
    run, scene = eval_run(tmp_path)
    made = []
    import locomani.pipeline.video_preview as vp

    monkeypatch.setattr(vp, "make_previews", lambda src, stem: made.append((src.name, stem)) or [(src, 0.0, 1.0)])
    view = EvalView()
    R.film_eval(view, run, scene, ["crate_vla_000", "crate_vla_001"], "unreal", log=lambda m: None)
    assert view.rendered == [0.3, 0.7, 1.1] * 2  # both rounds' dataset cameras, before any filming
    assert [x for x, _g in view.filmed] == [0.3, 0.7, 1.1] * 2
    out = run / "eval" / "videos_unreal"
    for label in ("crate_vla_000", "crate_vla_001"):
        assert (out / f"{label}.mp4").exists()
    assert not (out / "insets").exists()  # the pass-1 videos go once their round is filmed
    assert made == [
        (f"{label}.mp4", run / "eval" / "previews_unreal" / label) for label in ("crate_vla_000", "crate_vla_001")
    ]
