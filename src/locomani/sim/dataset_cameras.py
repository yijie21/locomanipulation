"""The cameras a recording stores, and their stand-in when the sim renders nothing.

`CAMERA_CONFIGS` is the contract between the sim and the exporter: the sim renders each entry through its
`mjcf_name` with default visual options and publishes it under its key, and the exporter resolves
`observation.images.<key>` by that key. `backend.render: offline` breaks the stream on purpose -- the sim
renders nothing -- so both of its readers get `PlaceholderCamera` instead: the assisted expert (its pose and
visibility are geometric) and the exporter, which otherwise waits for an image forever and records no row
at all (2026-09-24: every offline episode saved an empty `priv_seq`). The black frames the exporter then
stores are replaced from the state trace afterwards (`locomani.pipeline.render_dataset_cameras`).

Plain numpy: the exporter's venv imports this too.
"""

from __future__ import annotations

import time

import numpy as np

CAMERA_CONFIGS = {
    "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
    "left_wrist": {"height": 480, "width": 640, "mjcf_name": "left_wrist"},
    "right_wrist": {"height": 480, "width": 640, "mjcf_name": "right_wrist"},
}


class PlaceholderCamera:
    """backend.render offline: every read is three black frames at the published camera size.

    Frames change at the live stream's ~30 Hz, not on every read, so `fresh_image` toggles like live. The stamp is
    the read time: the consumer must sample its own `now` after reading, or the frame is "from the future" and the
    reach rejects it (per-read stamps ended every offline round `vision_lost`; the 30 Hz stamp still froze the
    reach clock at 0 for two offline pilots, 2026-09-23/24)."""

    shapes = {name: (c["height"], c["width"], 3) for name, c in CAMERA_CONFIGS.items()}
    period = 1 / 30

    def __init__(self):
        self.stamp = 0.0

    def read(self):
        now = time.time()
        if now - self.stamp >= self.period:
            self.stamp = now
        return {
            "images": {n: np.zeros(s, np.uint8) for n, s in self.shapes.items()},
            "timestamps": dict.fromkeys(self.shapes, self.stamp),
        }

    def close(self):
        pass


def install_exporter_placeholder(collector_cls) -> None:
    """Swap the exporter's camera client for `PlaceholderCamera` once it is built (the upstream constructor
    makes the client itself; there is no argument to hand one in)."""
    original = collector_cls.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._image_subscriber = PlaceholderCamera()
        print("[exporter] backend.render offline: black placeholder frames, filled from the trace later", flush=True)

    collector_cls.__init__ = patched
