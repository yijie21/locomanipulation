"""Task-level scene information with explicit sampling, latency and occlusion.

The tracker accepts observations, never a simulator. Object ground truth may be
used by a separate proxy sensor during expert collection, not by VLA execution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneMeasurement:
    captured_at: float
    bottle_position: tuple[float, float, float] | None
    table_height: float | None
    edge_point: tuple[float, float, float] | None
    inward_normal: tuple[float, float, float] | None
    visible_pixels: tuple[int, int] = (0, 0)
    visible_fraction: tuple[float, float] = (0.0, 0.0)

    def visible(self, min_pixels=32, min_fraction=0.2):
        return any(p >= min_pixels and f >= min_fraction for p, f in zip(self.visible_pixels, self.visible_fraction))


@dataclass(frozen=True)
class SceneEstimate:
    bottle_position: tuple[float, float, float] | None
    table_height: float | None
    edge_point: tuple[float, float, float] | None
    inward_normal: tuple[float, float, float] | None
    observed_at: float | None
    confidence: float
    currently_visible: bool

    @property
    def precision_allowed(self):
        return self.currently_visible and self.confidence >= 0.5


class SceneTracker:
    """Sample-and-hold estimates; hidden motion cannot refresh remembered position.

    Call capture_due(now) before obtaining a measurement. Ingest rejects an early
    refresh instead of silently allowing a privileged high-rate feedback loop.
    """

    def __init__(self, interval=0.4, latency=0.08, memory_s=1.0):
        if not np.isfinite([interval, latency, memory_s]).all() or interval <= 0 or latency < 0 or memory_s <= latency:
            raise ValueError("Invalid observation cadence/latency/memory")
        self.interval, self.latency, self.memory_s = interval, latency, memory_s
        self.next_capture = 0.0
        self.pending = deque()
        self.last_visible = None
        self.last_delivered = None
        self.last_now = -np.inf

    def capture_due(self, now):
        return now + 1e-9 >= self.next_capture

    def ingest(self, measurement: SceneMeasurement):
        t = measurement.captured_at
        if not np.isfinite(t) or t < max(0, self.last_now) or not self.capture_due(t):
            raise ValueError("Measurement violates observation cadence")
        if measurement.visible():
            if any(
                x is None
                for x in (
                    measurement.bottle_position,
                    measurement.table_height,
                    measurement.edge_point,
                    measurement.inward_normal,
                )
            ):
                raise ValueError("Visible measurement missing geometry")
            values = (
                *measurement.bottle_position,
                measurement.table_height,
                *measurement.edge_point,
                *measurement.inward_normal,
            )
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite scene measurement")
        else:
            # Even a misbehaving sensor cannot leak an occluded object's new pose.
            measurement = SceneMeasurement(
                t, None, None, None, None, measurement.visible_pixels, measurement.visible_fraction
            )
        self.pending.append((t + self.latency, measurement))
        self.next_capture = t + self.interval

    def estimate(self, now):
        if not np.isfinite(now) or now < self.last_now:
            raise ValueError("Observation time must be finite and monotonic")
        self.last_now = now
        while self.pending and self.pending[0][0] <= now + 1e-9:
            _, self.last_delivered = self.pending.popleft()
            if self.last_delivered.visible():
                self.last_visible = self.last_delivered
        m = self.last_visible
        if m is None:
            return SceneEstimate(None, None, None, None, None, 0.0, False)
        age = max(0.0, now - m.captured_at)
        confidence = float(np.clip(1 - age / self.memory_s, 0, 1))
        visible = bool(self.last_delivered.visible()) and age < self.interval + self.latency + 1e-8
        return SceneEstimate(
            m.bottle_position, m.table_height, m.edge_point, m.inward_normal, m.captured_at, confidence, visible
        )
