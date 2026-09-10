"""Per-attempt raw student observations/actions, with privileged audit elsewhere.

Records 25 Hz float32 action inputs before the common executor; does not apply
quality selection or export a training dataset. Failed attempts remain auditable.
"""

import subprocess
from pathlib import Path

import numpy as np

from locomani.backends.scalebfm.camera import CAMERAS, RobotCameraObserver
from locomani.backends.scalebfm.state import STATE_DIMS


class RawEpisodeRecorder:
    def __init__(self, sim, out, width=320, height=240):
        self.out = Path(out)
        if (self.out / "student.npz").exists():
            raise ValueError("Student recording already exists; do not overwrite")
        self.observer = RobotCameraObserver(sim, width, height)
        self.processes = {}
        for camera in CAMERAS:
            path = self.out / f"{camera}.partial.mp4"
            if path.exists() or path.with_name(f"{camera}.mp4").exists():
                raise ValueError("Video recording already exists; inspect incomplete attempt")
            self.processes[camera] = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    "25",
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(path),
                ],
                stdin=subprocess.PIPE,
            )
        self.states, self.actions, self.times = [], [], []
        self.closed = False

    def capture(self, t, previous_action, blackout=False):
        if len(self.states) != len(self.actions):
            raise ValueError("Previous observation has no action")
        _, state, images = self.observer.capture(t, previous_action, blackout=blackout)
        self.states.append(np.concatenate([state[key] for key in STATE_DIMS]))
        self.times.append(t)
        for key, image in images.items():
            self.processes[key].stdin.write(np.ascontiguousarray(image).tobytes())

    def append_action(self, action):
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("Action has no matching observation")
        self.actions.append(np.asarray(action, dtype=np.float32).copy())

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        for key, process in self.processes.items():
            process.stdin.close()
            code = process.wait()
            if code:
                errors.append(f"{key}: ffmpeg exit {code}")
            else:
                (self.out / f"{key}.partial.mp4").rename(self.out / f"{key}.mp4")
        self.observer.close()
        if len(self.states) != len(self.actions):
            errors.append("Observation/action count mismatch")
        if errors:
            raise RuntimeError("; ".join(errors))
        temporary = self.out / "student.partial.npz"
        np.savez_compressed(
            temporary,
            state=np.asarray(self.states, dtype=np.float32),
            action=np.asarray(self.actions, dtype=np.float32),
            simulation_time=np.asarray(self.times),
            fps=25,
            state_keys=np.array(list(STATE_DIMS)),
            state_widths=np.array(list(STATE_DIMS.values())),
        )
        temporary.rename(self.out / "student.npz")
