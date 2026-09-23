"""Exporter sessions and kept-episode counting: the part of a collection run that is not the task.

    state = resume_state(collect_root / "inputs.json")
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

It answers "which exporter session does this batch record into", "how many episodes did this
manifest keep" and "which batches finished, so where does an interrupted run continue". Anything that needs to know WHICH expert is running belongs in that expert's line.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from locomani.pipeline.run_stack import frame_audit_path
from locomani.stack import Stack


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

    The session is the one the stack NAMED when it started the exporter (`Stack.session`). Until
    2026-09-23 batch 0 took the newest directory by mtime, which in a directory every run on the
    instance shares can be the previous run's. A stack whose state predates that has no name to
    offer, and gets a restart instead of a guess.
    """
    if not stack.spec.record:
        return None
    if batch == 0 and stack.session is not None and alive_exporter(stack):
        return str(stack.session)
    stack.env["TF_FRAME_AUDIT_PATH"] = str(frame_audit_path(collect_root, batch))
    stack.restart_exporter()  # names the new session and waits until the exporter has opened it
    return str(stack.session)


def resume_state(inputs: Path) -> dict:
    """The batch bookkeeping to continue from: only batches that FINISHED count.

    A batch claims its session and manifest in `inputs.json` before it runs, so an interrupted
    batch leaves one entry more in each list than `batches` counts. Continuing from that appended
    the same batch a second time, and its manifest -- written in append mode -- then held both
    attempts' rows, so `convert`'s by-position pairing of session k with manifest k slipped by one
    from there on (`crate_env2_so45` sat in exactly this state; found 2026-09-23). The unfinished
    batch's manifest is moved aside under a name no `manifest_*.jsonl` glob matches, and its
    session is dropped from the list: that batch runs again from the start, into a new session.
    """
    if not inputs.exists():
        return dict(sessions=[], manifests=[], kept=0, batches=0)
    state = json.loads(inputs.read_text())
    done = int(state["batches"])
    for manifest in state["manifests"][done:]:
        path = Path(manifest)
        if path.exists():
            aside = path.with_name(f"interrupted_{path.stem}_{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
            path.rename(aside)
            print(f"collect: batch {done:02d} did not finish; its manifest moved aside to {aside.name}", flush=True)
    state["sessions"] = state["sessions"][:done]
    state["manifests"] = state["manifests"][:done]
    inputs.write_text(json.dumps(state, indent=1))
    return state


def kept_in(manifest: Path, record: bool) -> int:
    """Episodes this batch kept: a round that succeeded, and that produced an episode when recording."""
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return sum(bool(row.get("ok") and (row.get("produced", True) if record else True)) for row in rows)
