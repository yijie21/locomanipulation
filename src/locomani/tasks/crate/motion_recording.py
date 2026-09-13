"""Record commanded references and measured body motion for carry diagnostics."""

import json
from pathlib import Path

import numpy as np


class MotionRecording:
    def __init__(self, path, started):
        self.file = Path(path).open("x")
        self.started = started
        self.phase = "approach"

    def sample(self, now, reference, velocity, state, *, body_height=None):
        self.file.write(
            json.dumps(
                dict(
                    t=now - self.started,
                    phase=self.phase,
                    seq=state.get("seq"),
                    state_wall_time=state.get("wall"),
                    reference=np.asarray(reference).tolist(),
                    velocity=None if velocity is None else np.asarray(velocity).tolist(),
                    joints=np.asarray(state["joints"]).tolist(),
                    base_quat=state["pelvis"]["quat"],
                    base_position=state["pelvis"]["pos"],
                    body_height=body_height,
                )
            )
            + "\n"
        )

    def close(self):
        self.file.close()
