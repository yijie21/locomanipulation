"""The opening card of the report videos (user 2026-11-25): MuJoCo videos keep it, Unreal videos open on the round."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts" / "rsi" / "report"), str(REPO / "src")]

import report_media as RM  # noqa: E402
import report_unreal as RU  # noqa: E402

SIZE = (160, 120)
FRAMES = 10


def frames():
    for k in range(FRAMES):
        yield np.full((SIZE[1], SIZE[0], 3), 250, np.uint8), 0.1 * k


def decoded(mp4: Path) -> np.ndarray:
    """Every frame of the video as RGB."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout  # fmt: skip
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(mp4)], check=True, capture_output=True, text=True).stdout  # fmt: skip
    w, h = (int(x) for x in probe.strip().split(","))
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


class Compose(unittest.TestCase):
    def render(self, d: Path, **kw) -> np.ndarray:
        out = RM.compose(None, d / "v.mp4", [frames()], [[("final", (255, 255, 255))]], ["title", "detail", "pool"], 3,
                         SIZE, note="", **kw)  # fmt: skip
        self.assertTrue(out.with_suffix(".jpg").exists())
        return decoded(out)

    def test_the_card_is_six_seconds_by_default_and_none_at_zero(self):
        with tempfile.TemporaryDirectory() as d:
            with_card = self.render(Path(d) / "card")
            without = self.render(Path(d) / "none", card_s=0)
        hold = int(2 * RM.dv.FPS)  # the last frame, held
        self.assertEqual(len(with_card), int(6 * RM.dv.FPS) + FRAMES + hold)
        self.assertEqual(len(without), FRAMES + hold)
        # the card is dark text on black; the round's frames are near white below the caption strip
        self.assertLess(with_card[0, -20:].mean(), 60)
        self.assertGreater(without[0, -20:].mean(), 200)


class UnrealVideo(unittest.TestCase):
    def test_report_unreal_composes_without_the_card(self):
        row = {"video": "x/final_01_pick_a.mp4"}
        j = dict(panels=[("panel0", object(), {})], caps=[[("final", (255, 255, 255))]], speed=3, compare=None)
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(RU, "UnrealView"),
            mock.patch.object(RU.V, "Scene"),
            mock.patch.object(RU, "scene_entry", return_value=(Path(d), "/Game/Levels/x")),
            mock.patch.object(RU, "media_index", return_value={"final_01_pick_a": row}),
            mock.patch.object(RU, "media_job", return_value=j),
            mock.patch.object(RU, "film_round", return_value=[]),
            mock.patch.object(RU.RM, "compose") as compose,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(RU.main(["all", "--media", d, "--out", str(Path(d) / "out"), "--scene", "x"]), 0)
        self.assertEqual(compose.call_count, 1)
        self.assertEqual(compose.call_args.kwargs.get("card_s"), 0)
        self.assertEqual(compose.call_args.args[1], Path(d) / "out" / "final_01_pick_a.mp4")


if __name__ == "__main__":
    unittest.main()
