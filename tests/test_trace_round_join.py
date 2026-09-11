"""A trace file must join to its manifest round by name.

The visible200 eval (2026-09-10) wrote 99 trace files for 100 manifest rounds and
named them from the VLA client's own call counter. Manifest round 0 died in
`[stand]`, before the VLA stage, so the counter never saw it and every trace file
from then on was named one lower than the round it described. Joining
`traces/round_NNN.json` to the manifest by NNN therefore paired each round's
gripper data with the NEXT round's geometry, and the analysis built on that join
(ledger bottle.md section 18.1) had to be re-derived.

The fix threads the caller's own round number down to the trace writer. These
tests pin both halves: the client honours `round_id`, and `run_round` passes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def trace_writers():
    """The three clients' trace-naming expressions, as source text.

    The writers sit at the end of ~400-line round closures that need a live
    deploy to run, so there is no seam to call them at. Pinning the source is
    the honest alternative: it catches a revert to `counter['n']` in the name.
    """
    root = Path(__file__).resolve().parents[1] / "src/locomani/backends/sonic"
    return {p.name: p.read_text() for p in (root / "vla_client.py", root / "vla_round.py", root / "lockstep_client.py")}


def test_trace_filename_uses_the_callers_round_id(trace_writers):
    for name, src in trace_writers.items():
        assert 'f"round_{rid:03d}.json"' in src, f"{name}: trace filename must come from the caller's round id"
        assert "f\"round_{counter['n']:03d}.json\"" not in src, (
            f"{name}: naming a trace from the client's own call count drifts from the manifest "
            "as soon as one round fails before the VLA stage"
        )
        assert 'rid = counter["n"] if round_id is None else int(round_id)' in src, (
            f"{name}: round_id must override the counter, with the counter kept as the fallback"
        )
        assert "round=rid" in src, f"{name}: the round id must also be a field, so a join need not trust the filename"


def test_round_id_reaches_the_policy_from_the_round_loop():
    """`round_loop` -> `run_round` -> `_run_round` -> RoundContext -> the policy.

    Read as source, not imported: `locomani.tasks.bottle` pulls in `gear_sonic`,
    which only exists on a deploy box, so importing it here would skip the test
    on exactly the machines that run `make check`.
    """
    task = Path(__file__).resolve().parents[1] / "src/locomani/tasks/bottle"
    loop = (task / "round_loop.py").read_text()
    gym = (task / "grasp_gym.py").read_text()
    phases = (task / "round_phases.py").read_text()

    assert "round_id=rnd" in loop, "round_loop must hand run_round the round number it writes into the manifest"
    assert gym.count("round_id: int | None = None,") == 2, "both run_round and _run_round must accept a round_id"
    assert "round_id=round_id," in gym, "run_round must forward it to _run_round and onto the RoundContext"
    assert "round_id: int | None = None" in phases, "RoundContext must carry it"
    assert "round_id=ctx.round_id," in phases, "the policy call site must forward the context's round id"


def test_a_round_that_never_reaches_the_vla_does_not_shift_the_next_trace(tmp_path):
    """The regression itself, at the only seam that can hold it.

    Two rounds reach the policy; manifest round 1 did not (it failed at
    `[stand]`). With the fix the traces are named 0 and 2 -- the numbers the
    manifest uses -- not 0 and 1.
    """
    counter = {"n": 0}

    def write_trace(trace_dir, round_id, ok, why):
        # the patched expression, verbatim from the three clients
        rid = counter["n"] if round_id is None else int(round_id)
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(trace_dir) / f"round_{rid:03d}.json", "w") as f:
            json.dump(dict(round=rid, ok=ok, why=why), f)
        counter["n"] += 1

    manifest = [(0, True, "ok"), (1, False, "[stand] did not stabilise"), (2, False, "[vla] reach: never closed")]
    for rnd, ok, why in manifest:
        if why.startswith("[stand]"):
            continue  # died before the VLA stage: the policy is never called
        write_trace(tmp_path, rnd, ok, why)

    names = sorted(p.name for p in tmp_path.glob("round_*.json"))
    assert names == ["round_000.json", "round_002.json"], names
    for p in tmp_path.glob("round_*.json"):
        rec = json.loads(p.read_text())
        expected = next(w for r, o, w in manifest if r == rec["round"])
        assert rec["why"] == expected, "a trace must describe the manifest round it is named after"
