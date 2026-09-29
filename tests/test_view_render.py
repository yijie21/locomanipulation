"""The view-based dataset-camera stage (locomani.sim.view_render, Isaac and Unreal): which rows it renders, which sessions it touches, what lands in the videos.

Kit is not needed: the view and the video encoder are fakes, the way test_isaac_view drives the view.
"""

import json

import numpy as np

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
