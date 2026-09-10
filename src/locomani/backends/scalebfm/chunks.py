"""Delayed action-chunk switching without dropping the currently active plan."""

from collections import deque

import numpy as np

from locomani.backends.scalebfm.action import ACTION_DIM


class ChunkScheduler:
    def __init__(self, latency_ticks=4, action_stride=2):
        if latency_ticks < 0 or action_stride < 1:
            raise ValueError("Invalid chunk timing")
        self.latency_ticks = latency_ticks
        self.stride = action_stride
        self.pending = deque()
        self.active = None
        self.origin = None
        self.last_capture = -1
        self.last_tick = -1

    def submit(self, capture_tick, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or not len(chunk) or not np.isfinite(chunk).all():
            raise ValueError("Prediction must be a finite N x29 action chunk")
        if capture_tick <= self.last_capture or capture_tick % self.stride:
            raise ValueError("Observation captures must increase on the action grid")
        self.pending.append((capture_tick + self.latency_ticks, capture_tick, chunk.copy()))
        self.last_capture = capture_tick

    def row(self, tick):
        if tick < self.last_tick:
            raise ValueError("Execution clock must be monotonic")
        self.last_tick = tick
        while self.pending and self.pending[0][0] <= tick:
            _, self.origin, self.active = self.pending.popleft()
        if self.active is None:
            return None
        index = min((tick - self.origin) // self.stride, len(self.active) - 1)
        return self.active[index].copy()
