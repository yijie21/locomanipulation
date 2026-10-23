"""TODO 33.11: the library monitor reads a sandbox's records and gives the two numbers -- the share of hypotheses that
cite the experience library and the mechanisms the Manager links to earlier iterations -- without running anything."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rsi" / "report"))

import library_monitor as M  # noqa: E402


def sandbox(tmp: Path, name: str, its: dict[int, dict], library: bool, replay_from: int | None = None) -> Path:
    sb = tmp / name
    loop = sb / "outputs" / "scene_rsi_20261016" / "grasp_atomic" / "loop"
    loop.mkdir(parents=True)
    (loop / "ledger.jsonl").write_text("")
    (loop / "meta.json").write_text(json.dumps({"replay": {"iteration": replay_from}} if replay_from else {}))
    for n, it in its.items():
        d = loop / f"iter_{n:02d}"
        d.mkdir()
        (d / "hypotheses.json").write_text(json.dumps({"hypotheses": it["hyps"], "notes": ""}))
        if "repeats" in it:
            (d / "manager.json").write_text(json.dumps({"decision": "engineer", "repeats": it["repeats"]}))
    if library:
        e = sb / "scripts" / "rsi" / "experience" / "r2_on_single"
        e.mkdir(parents=True)
        lesson = {"cue": "the gripper housing pushes long objects", "lesson": "stop the entry short of the hull"}
        (e / "entry.json").write_text(json.dumps({"experience": lesson}))
    return sb


def hyp(i: str, claim: str, evidence: str = "") -> dict:
    return {"id": i, "claim": claim, "instead": "change the entry depth", "evidence": evidence}


def test_citations_repeats_and_a_replays_copied_iterations(tmp_path):
    its = {
        3: {"hyps": [hyp("H1", "copied from the source run", "the library's r2_on_single")]},
        4: {
            "hyps": [
                hyp("H1", "the housing pushes the wolf", "the library's r2_on_single cue holds here"),
                hyp("H2", "the stance is too close to the shelf"),
            ],
            "repeats": [],
        },  # fmt: skip
        5: {
            "hyps": [hyp("H1", "the housing again"), hyp("H2", "a new walk mechanism", "Experience: none apply")],
            "repeats": [{"now": "H1", "same_as": ["04:H1"]}],
        },  # fmt: skip
    }
    sb = sandbox(tmp_path, "lib", its, library=True, replay_from=4)
    base = sandbox(tmp_path, "none", {0: {"hyps": [hyp("H1", "a different failure on the table")]}}, library=False)
    rep = M.cmd_run(type("A", (), {"sandboxes": [str(sb)], "baseline": [str(base)]})())
    run = rep["runs"]["lib"]
    assert [i["iteration"] for i in run["iterations"]] == ["iter_04", "iter_05"]  # iteration 3 is the source's
    assert run["hypotheses"] == 4
    assert run["cites"] == 0.5  # iteration 4's H1 names the entry, iteration 5's H2 names the library as such
    assert run["entries_cited"] == {"r2_on_single": 1, "library": 1}
    assert run["mechanisms_run"] == 3  # iteration 5's H1 is iteration 4's H1
    assert rep["runs"]["lib"]["read_index"] == 0  # no transcripts: nothing was read, so nothing counts as worded
    assert run["worded"] == 0.0


def test_a_run_without_the_managers_links_reports_no_mechanism_count(tmp_path):
    sb = sandbox(tmp_path, "old", {0: {"hyps": [hyp("H1", "x")]}, 1: {"hyps": [hyp("H1", "y")]}}, library=False)
    rep = M.cmd_run(type("A", (), {"sandboxes": [str(sb)], "baseline": []})())
    assert rep["runs"]["old"]["mechanisms_run"] == "n/a"
    assert rep["runs"]["old"]["cites"] is None
