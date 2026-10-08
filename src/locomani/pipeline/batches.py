"""Exporter sessions and kept-episode counting: the part of a collection run that is not the task.

    state = resume_state(collect_root / "inputs.json")
    session = open_exporter_session(stack, collect_root, batch)
    wait_saved(session, manifest)
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
manifest keep", "which batches finished, so where does an interrupted run continue" and, for a run
that measures a success rate rather than collecting (`collect.stop_rule`), "has it measured enough". Anything that needs to know WHICH expert is running belongs in that expert's line.

A run that collects in several lanes (`locomani.pipeline.lanes`) runs several batches at once, one per lane process.
They share `inputs.json` through `claim_batch` / `finish_batch`, which hold a file lock: a lane claims the lowest
batch number nobody has finished or is running (its seed and manifest name follow from the number), and only a
FINISHED batch enters `sessions` / `manifests`, so those two stay pairs in batch order of finishing and
`sessions[:batches]` is still "the finished ones" for every reader. Claims in flight sit in `running`.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
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


def open_exporter_session(stack: Stack, collect_root: Path, batch: int, *, starter_batch: int = 0) -> str | None:
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

    `starter_batch` is the batch the starter's exporter was opened for (its frame-audit file is named after it): 0 for
    a run's one stack, the lane's index for a lane's (`run_stack.collect_spec`); -1 when the starter's session is
    spent. Only that batch takes the session over.
    """
    if not stack.spec.record:
        return None
    if batch == starter_batch and stack.session is not None and alive_exporter(stack):
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
    # Batches claimed and not finished: the single-lane form listed them past `batches`, the lane form in `running`.
    unfinished = [(done + i, m) for i, m in enumerate(state["manifests"][done:])]
    unfinished += [(int(b), claim["manifest"]) for b, claim in sorted(state.pop("running", {}).items())]
    for batch, manifest in unfinished:
        path = Path(manifest)
        if path.exists():
            aside = path.with_name(f"interrupted_{path.stem}_{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
            path.rename(aside)
            print(f"collect: batch {batch:02d} did not finish; its manifest moved aside to {aside.name}", flush=True)
    state["sessions"] = state["sessions"][:done]
    state["manifests"] = state["manifests"][:done]
    if "numbers" in state:
        state["numbers"] = state["numbers"][:done]
    inputs.write_text(json.dumps(state, indent=1))
    return state


@contextlib.contextmanager
def locked_state(inputs: Path):
    """`inputs.json` read, handed over and written back under an exclusive lock: what the lanes share."""
    inputs = Path(inputs)
    inputs.parent.mkdir(parents=True, exist_ok=True)
    with open(inputs.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = (
            json.loads(inputs.read_text()) if inputs.exists() else dict(sessions=[], manifests=[], kept=0, batches=0)
        )
        state.setdefault("numbers", list(range(int(state["batches"]))))  # before lanes, batch k finished k-th
        state.setdefault("running", {})
        yield state
        inputs.write_text(json.dumps(state, indent=1))


def claim_batch(inputs: Path, lane: int, *, target: int, max_batches: int, manifest_for) -> int | None:
    """The batch number this lane runs next, recorded as running; None when the run has what it needs.

    Nothing more is claimed once the finished batches keep `target` episodes, a stop rule decided, or every number
    below `max_batches` is finished or running. Batches running elsewhere still finish, so a run in several lanes
    can end with more than `target` kept."""
    with locked_state(inputs) as state:
        if state["kept"] >= target or state.get("stop"):
            return None
        taken = set(state["numbers"]) | {int(b) for b in state["running"]}
        batch = next(b for b in range(len(taken) + 1) if b not in taken)
        if batch >= max_batches:
            return None
        state["running"][str(batch)] = dict(lane=lane, manifest=str(manifest_for(batch)))
        return batch


def finish_batch(inputs: Path, batch: int, session: str | None, manifest: Path, kept: int, decide=None) -> dict:
    """Move a claimed batch to the finished lists; returns the state after it. `decide(state)` (optional) runs under
    the same lock and returns a stop record for `collect.stop_rule`, or None."""
    with locked_state(inputs) as state:
        state["running"].pop(str(batch), None)
        state["sessions"].append(session)
        state["manifests"].append(str(manifest))
        state["numbers"].append(batch)
        state["kept"] += kept
        state["batches"] = len(state["manifests"])
        if decide is not None and not state.get("stop") and (stop := decide(state)):
            state["stop"] = stop
        return dict(state)


def episodes_on_disk(session: Path) -> int:
    """Episodes the exporter has finished writing: its episode index, and the last one's parquet in place.

    The exporter's `save_episode` stops the video writers, appends `meta/episodes.jsonl`, THEN writes the
    parquet (gear_sonic/data/exporter.py), so the index alone runs one step ahead of a finished save."""
    index = session / "meta" / "episodes.jsonl"
    if not index.exists():
        return 0
    eps = [json.loads(line)["episode_index"] for line in index.read_text().splitlines() if line.strip()]
    if not eps:
        return 0
    last = session / "data" / "chunk-000" / f"episode_{max(eps):06d}.parquet"
    return len(eps) if last.exists() and last.stat().st_size > 0 else len(eps) - 1


def wait_saved(session: str | None, manifest: Path, *, timeout: float = 120.0, poll: float = 0.5) -> int:
    """Block until the batch's exporter session holds every episode its manifest says was produced.

    The expert sends the save/discard key at the end of a round and holds 2 s, but the exporter saves from
    its own loop and a ~1900-row episode can take longer. Mid-batch the next round's reset covers it; after
    a batch's LAST round the next batch restarted the exporter at once and killed the save (and at the end
    of a run `stack.down()` does the same): 2 of 94 batches of box_room_grasp_200_cap120_20260925 lost their
    last episode, one of them a success, and convert then refused both sessions (2026-09-26).

    Raises rather than carry on: a session one episode short breaks convert's by-position join hours later.
    """
    if session is None:
        return 0
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    want = sum(1 for row in rows if row.get("produced"))
    deadline = time.monotonic() + timeout
    while True:
        have = episodes_on_disk(Path(session))
        if have >= want:
            return have
        if time.monotonic() > deadline:
            raise SystemExit(
                f"{session}: {have} episode(s) saved after {timeout:.0f} s, but {Path(manifest).name} produced {want}; "
                "the exporter never finished writing -- inspect its log before continuing"
            )
        time.sleep(poll)


def kept_in(manifest: Path, record: bool) -> int:
    """Episodes this batch kept: a round that succeeded, and that produced an episode when recording."""
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return sum(bool(row.get("ok") and (row.get("produced", True) if record else True)) for row in rows)


def wilson(successes: int, rounds: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of successes/rounds (95% at z 1.96); (0, 1) for no rounds."""
    if rounds <= 0:
        return 0.0, 1.0
    p = successes / rounds
    centre = p + z * z / (2 * rounds)
    half = z * math.sqrt(p * (1 - p) / rounds + z * z / (4 * rounds * rounds))
    scale = 1 + z * z / rounds
    return max(0.0, (centre - half) / scale), min(1.0, (centre + half) / scale)


def stop_verdict(successes: int, rounds: int, rule: dict) -> str | None:
    """`collect.stop_rule`: "above"/"below" once the Wilson interval of successes/rounds clears `threshold`,
    "undecided" at `max_rounds`; None = run another batch. Nothing is decided before `min_rounds` (two batches of
    five at least: one bad stretch of the stack must not decide a layout). Fixed before a run, never after."""
    if rounds < int(rule["min_rounds"]):
        return None
    low, high = wilson(successes, rounds)
    if low > float(rule["threshold"]):
        return "above"
    if high < float(rule["threshold"]):
        return "below"
    return "undecided" if rounds >= int(rule["max_rounds"]) else None
