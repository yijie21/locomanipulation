"""Record the RGB frames and robot/command history actually consumed by a video pilot."""

import json
from pathlib import Path

import cv2
import numpy as np


class VisualRecording:
    def __init__(self, directory: Path, fps=25.0):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.log = (directory / "observations.jsonl").open("w")
        self.writers = {}
        self.fps = fps
        self.frames = 0

    def append(self, images, row):
        for name, rgb in images.items():
            if name not in self.writers:
                height, width = rgb.shape[:2]
                writer = cv2.VideoWriter(
                    str(self.directory / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open video writer for {name}")
                self.writers[name] = writer
            self.writers[name].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.log.write(
            json.dumps(
                {"frame": self.frames, **row}, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)
            )
            + "\n"
        )
        self.frames += 1

    def close(self):
        for writer in self.writers.values():
            writer.release()
        self.log.close()
