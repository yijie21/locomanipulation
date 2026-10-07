"""Dataset pictures as late as a real camera's: the offline re-render draws each row from an earlier state.

An offline re-render (`render_dataset_cameras`, `view_render`) draws every row's cameras from that row's own state,
so the dataset's pictures are exactly as old as its proprioception. A real robot's are not: a RealSense captures on
its own clock (6/15/30/60 fps, never the 25 Hz dataset rate) and a frame reaches the policy some tens of
milliseconds after capture, while the joint state is ~1 ms old. A policy trained on in-sync pictures has never seen
that lag (user, 2026-10-06: add it to the training pictures).

`images.camera_delay: {fps: 30, latency_ms: [20, 30]}` models that camera. Per episode, a latency drawn uniformly
from `latency_ms` and a capture phase drawn uniformly over one frame; a row at time t shows the last capture at or
before t - latency, drawn from the last 50 Hz state at or before it. So a picture is latency + 0..1/fps older than
its row (20-63 ms with the values above), never newer than the one before it, and the same row of the same episode
always draws the same state (the draw is seeded by the session and episode). Without the key nothing moves.

The dataset's offline re-render reads it (a live collection records the sim's own frames, which nothing re-draws:
run_context.migrate_images refuses the pair), and so does the lockstep closed loop (2026-10-07, user: the policy
sees pictures as old in evaluation as in training): `camera_seq` gives, per observation, the state its pictures are
drawn from, by the same per-episode draw keyed by the round's seed (`tasks/crate/vla_lockstep.py`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from locomani.pipeline.world_lift import DST_FPS, SRC_FPS

TICK_S = 1.0 / SRC_FPS  # one state-trace seq: the sim publishes its privileged state every 20 ms


@dataclass(frozen=True)
class CameraDelay:
    fps: float
    latency_ms: tuple[float, float]

    def spec(self) -> dict:
        """The JSON form the config gives and a render marker records."""
        return {"fps": self.fps, "latency_ms": list(self.latency_ms)}


def parse(value) -> CameraDelay | None:
    """`images.camera_delay` as given (None: no delay); raises on anything else."""
    if value is None:
        return None
    usage = "images.camera_delay must be {fps: <frames/s>, latency_ms: [low, high]}"
    if not isinstance(value, dict) or set(value) != {"fps", "latency_ms"}:
        raise SystemExit(f"{usage}, not {value!r}")
    fps, latency = value["fps"], value["latency_ms"]
    if not isinstance(fps, (int, float)) or not 0 < fps <= SRC_FPS:
        raise SystemExit(f"{usage}: fps {fps!r} must be in (0, {SRC_FPS}], the state trace's rate")
    if (
        not isinstance(latency, (list, tuple))
        or len(latency) != 2
        or not all(isinstance(v, (int, float)) for v in latency)
        or not 0 <= latency[0] <= latency[1]
    ):
        raise SystemExit(f"{usage}: latency_ms {latency!r} must be [low, high] with 0 <= low <= high")
    return CameraDelay(float(fps), (float(latency[0]), float(latency[1])))


def episode_draw(delay: CameraDelay, key: str) -> tuple[float, float]:
    """(latency s, capture phase s) of one episode, seeded by `key` (session/episode) so a re-render repeats it."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    latency = rng.uniform(*delay.latency_ms) / 1e3
    phase = rng.uniform(0.0, 1.0 / delay.fps)
    return float(latency), float(phase)


def _wanted_seq(t, latency: float, phase: float, fps: float):
    """The 50 Hz state seq at or before the last capture at or before `t - latency` on a `fps` clock of `phase`."""
    period = 1.0 / fps
    capture = phase + np.floor((t - latency - phase) / period + 1e-9) * period
    return np.floor(capture / TICK_S + 1e-9).astype(np.int64)


def camera_seq(delay: CameraDelay, draw: tuple[float, float], seq: int) -> int:
    """The closed loop's form: the privileged seq whose state the pictures show when the policy looks at state
    `seq` (one seq per 20 ms control tick, as in the trace), under the episode's `draw` (`episode_draw`)."""
    latency, phase = draw
    return int(min(_wanted_seq(seq * TICK_S, latency, phase, delay.fps), seq))


def delayed_rows(delay: CameraDelay | None, tseq: np.ndarray, idx: np.ndarray, key: str) -> tuple[np.ndarray, dict]:
    """Per row, the trace index its pictures are drawn from, and the episode's draw for the marker.

    `tseq` is the state trace's seq column (one per 20 ms), `idx` each row's own trace index. A row before the
    trace's start (latency reaching past it) takes the first sample."""
    idx = np.asarray(idx)
    if delay is None:
        return idx, {}
    latency, phase = episode_draw(delay, key)
    t = np.asarray(tseq, dtype=np.float64)[idx] * TICK_S
    want = _wanted_seq(t, latency, phase, delay.fps)
    cam = np.clip(np.searchsorted(tseq, want, side="right") - 1, 0, None)
    cam = np.minimum(cam, idx)  # never a newer state than the row's own
    age_ms = (t - np.asarray(tseq, dtype=np.float64)[cam] * TICK_S) * 1e3
    return cam, {
        "latency_ms": round(latency * 1e3, 1),
        "phase_ms": round(phase * 1e3, 1),
        "age_ms": [round(float(age_ms.min()), 1), round(float(np.median(age_ms)), 1), round(float(age_ms.max()), 1)],
    }


def describe(delay: CameraDelay | None) -> str:
    if delay is None:
        return "pictures in step with the state"
    lo, hi = delay.latency_ms
    return (
        f"pictures {lo:g}-{hi + 1e3 / delay.fps:.0f} ms older than the state "
        f"({delay.fps:g} fps camera, latency {lo:g}-{hi:g} ms per episode)"
    )


def mean_lag_frames(delay: CameraDelay | None, dataset_fps: int = DST_FPS) -> float:
    """How many dataset frames (`dataset.fps`) the pictures trail the state on average: what check_dataset's
    video/table alignment should find. Mean latency, half a capture period, and half a 20 ms state tick (the
    capture time is floored onto the trace)."""
    if delay is None:
        return 0.0
    return (sum(delay.latency_ms) / 2e3 + 0.5 / delay.fps + TICK_S / 2) * dataset_fps
