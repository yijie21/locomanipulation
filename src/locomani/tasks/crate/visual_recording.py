"""Record the RGB frames and robot/command history actually consumed by a video pilot."""

import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from locomani.pipeline.execution_record import RoundResult


def record_attempt(manifest: Path, label: str, seed: int, execute) -> RoundResult:
    """Journal before setup, then persist every completed or exceptional attempt."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    directory = manifest.parent / "observations" / label
    if directory.exists():
        raise FileExistsError(f"attempt already exists: {directory}")
    with (manifest.parent / "attempts.jsonl").open("a") as handle:
        handle.write(json.dumps({"label": label, "seed": seed, "started_wall": time.time()}) + "\n")
    try:
        result = execute()
    except Exception:
        result = RoundResult(False, "runtime_error", False, {"metrics": {"error": traceback.format_exc(), "frames": 0}})
    row = {
        **result.execution["metrics"],
        "label": label,
        "seed": seed,
        "ok": result.accepted(),
        "why": result.why,
        "produced": result.produced,
    }
    with manifest.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(row, indent=2))
    return result


class VisualRecording:
    def __init__(self, directory: Path, fps=25.0):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.log = (directory / "observations.jsonl").open("w")
        self.commands = (directory / "commands.jsonl").open("w")
        self.command_count = 0
        self.context = {}
        self.training_tail = None
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

    def observe_reference(self, channel, payload):
        """Called after each successful send, including POSE handoff/priming frames.

        POSE payloads retain their buffered frame indices, roots and velocities;
        planner payloads retain their actual 17-joint order and movement fields.
        This journals sends, not acknowledgement or execution by the deploy.
        """
        self.commands.write(
            json.dumps(
                dict(
                    command_id=self.command_count,
                    sent_wall_time=time.time(),
                    **self.context,
                    channel=channel,
                    payload=payload,
                ),
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x),
            )
            + "\n"
        )
        self.command_count += 1

    def mark_training_end(self, state):
        """Tentative inclusive boundary; publish only if the following hold succeeds."""
        self.training_tail = dict(
            end_seq=int(state["seq"]),
            simulation_time=float(state["t"]),
            end_command_id=self.command_count - 1,
            end_frame=self.frames - 1,
            reason="upright_loaded_hold_verification",
        )

    def close(self):
        for writer in self.writers.values():
            writer.release()
        self.log.close()
        self.commands.close()
