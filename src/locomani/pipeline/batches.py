"""Exporter sessions and kept-episode counting: the part of a collection run that is not the task.

    session = open_exporter_session(stack, collect_root, batch)
    kept += kept_in(manifest, record)

A batch is one exporter session. The exporter is restarted between batches for two reasons, and
only one of them is about resources: `convert` pairs exporter session k with manifest k BY
POSITION, so a single session spanning every batch would destroy the pairing; and the exporter's
threads grow with every episode against a container ceiling of 3840 pids. The sim and the deploy
live across the whole run.

Both lines have always shared this -- until 2026-09-22 by `run_collect_bimanual` importing it out
of `run_collect_single_arm`, which made one line's driver a dependency of the other's.

TWO RULES, because this module is where a deleted driver could quietly grow back
(docs/handoff/flat-functional-scripts-20260922.md, decision 6):

  * it must not import anything under `locomani.tasks`, and
  * it must not assemble a command line.

It answers "which exporter session does this batch record into" and "how many episodes did this
manifest keep". Anything that needs to know WHICH expert is running belongs in that expert's line.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from locomani.pipeline.run_stack import frame_audit_path
from locomani.stack import Stack, session_root


def new_session(sessions: Path, before: set, timeout: float = 60.0) -> Path:
    """The exporter session this batch just created, paired with this batch's manifest by order."""
    end = time.time() + timeout
    while time.time() < end:
        fresh = [d for d in sessions.iterdir() if d not in before]
        if fresh:
            return max(fresh, key=lambda p: p.stat().st_mtime)
        time.sleep(1)
    raise SystemExit("no new exporter session appeared after the exporter restarted")


def alive_exporter(stack: Stack) -> bool:
    pid = stack.pids.get("exporter")
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def open_exporter_session(stack: Stack, collect_root: Path, batch: int) -> str | None:
    """The exporter session this batch records into; return it, or None when nothing records.

    Batch 0 uses the session the stack starter's exporter already opened. Restarting it there
    would subscribe a fresh exporter to a camera stream that has been publishing for a minute,
    and that costs a frame: two runs that restarted at batch 0 put 50% and 38% of their
    wrist-camera measurements at lag 0 with a median of +1, where the pre-refactor path -- whose
    exporter came up with the sim -- put 86% at lag 0 with a median of +0 (2026-09-21).
    Later batches must restart: `convert` pairs session k with manifest k by position.
    """
    if not stack.spec.record:
        return None
    sessions = session_root(stack.spec.instance) / "outputs"
    sessions.mkdir(parents=True, exist_ok=True)
    if batch == 0:
        existing = sorted(sessions.iterdir(), key=lambda p: p.stat().st_mtime)
        if existing and alive_exporter(stack):
            return str(existing[-1])
    before = set(sessions.iterdir())
    stack.env["TF_FRAME_AUDIT_PATH"] = str(frame_audit_path(collect_root, batch))
    stack.restart_exporter()
    return str(new_session(sessions, before))


def kept_in(manifest: Path, record: bool) -> int:
    """Episodes this batch kept: a round that succeeded, and that produced an episode when recording."""
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return sum(bool(row.get("ok") and (row.get("produced", True) if record else True)) for row in rows)
