"""TODO 43.2 and 44 (rule 38): several Reviewers on one iteration; a draft's declared scope, measured footprint and
repair turn."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
LOOP_NAMES = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "records", "design", "loop")


@pytest.fixture
def lp():
    saved, old_path = {k: sys.modules.pop(k, None) for k in LOOP_NAMES}, list(sys.path)
    sys.path[:0] = [str(HERE), str(ROOT / "src")]
    try:
        spec = importlib.util.spec_from_file_location("rsi_scope_loop", HERE / "loop.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rsi_scope_loop"] = mod
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path[:] = old_path
        sys.modules.pop("rsi_scope_loop", None)
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _rec(ok: bool, *steps: tuple) -> dict:
    return dict(ok=ok, t_end=float(len(steps)), wall_s=1.0,
                steps=[dict(step=n, ok=o, t0=float(i), t1=float(i) + d, cm=1.5) for i, (n, o, d) in enumerate(steps)])  # fmt: skip


WALK, GRIP, MISS = ("walk", True, 1.0), ("grip", True, 1.0), ("grip", False, 1.0)


def _rounds(d: Path, **recs) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for label, r in recs.items():
        (d / f"{label}.json").write_text(json.dumps(r))
    return d


def test_footprint_tells_rounds_the_code_reached_from_rounds_it_did_not(lp, tmp_path):
    a = _rounds(tmp_path / "a", s1=_rec(True, WALK, GRIP), s2=_rec(True, WALK, GRIP), s3=_rec(False, WALK, MISS))
    late = _rec(True, WALK, ("grip", True, 1.25))  # the same outcome, another timing: the code reached it
    wall = dict(_rec(True, WALK, GRIP), wall_s=99.0)  # wall time is not a difference
    b = _rounds(tmp_path / "b", s1=wall, s2=late, s3=_rec(False, WALK, MISS))
    fp = lp.footprint(a, b, dict(when="x", starts=["s3"]))
    assert fp["n"] == 3 and fp["changed"] == ["s2"] and fp["outside"] == ["s2"] and fp["missed"] == ["s3"]
    assert lp.footprint(a, b, dict(when="x", starts=["all"]))["outside"] == []
    assert lp.footprint(a, a, None)["changed"] == []


def test_first_difference_names_the_step_where_a_round_left_the_path(lp):
    d = lp.first_difference(_rec(True, WALK, GRIP), _rec(False, WALK, MISS))
    assert d["index"] == 1 and d["incumbent"]["ok"] is True and d["draft"]["ok"] is False
    short = lp.first_difference(_rec(True, WALK, GRIP), _rec(False, WALK))
    assert short["index"] == 1 and short["draft"] is None and short["incumbent"]["step"] == "grip"


class Draft:
    """repair_draft on real round directories, with the Engineer's turn and the diagnostic run stubbed. Each repair
    outcome is what the draft's rounds become after the Engineer's second turn."""

    def __init__(self, lp, monkeypatch, tmp_path, first: dict, repaired: dict | None, turn=None):
        self.lp, self.rows, self.calls = lp, [], []
        self.loop = tmp_path / "fam"
        self.it = self.loop / "loop" / "iter_00"
        self.it.mkdir(parents=True)
        self.w = self.it / "cand_1"
        (self.w / "skills").mkdir(parents=True)
        (self.w / "skills" / "__init__.py").write_text("X = 1\n")
        self.inc = dict(hash="inc")
        self.inc_rounds = _rounds(self.loop / "diag" / "incumbent_inc", s1=_rec(True, WALK, GRIP), s2=_rec(False, WALK, MISS),
                                  s3=_rec(False, WALK, MISS))  # fmt: skip
        hashes = lambda c: (c / "skills" / "__init__.py").read_text().strip().split()[-1]  # noqa: E731

        def fake_claude(role, prompt, schema, it_dir, tools, stub=None, stop=None, resume=None):
            self.calls.append(dict(role=role, prompt=prompt, resume=resume))
            if turn is not None:
                turn()
            (self.w / "skills" / "__init__.py").write_text("X = 2\n")
            return dict(mechanism="repaired", set={}, scope=dict(when="narrow", starts=["s2"])), dict(
                cost_usd=2.0, seconds=20, session="sess"
            )

        def fake_early(k, w, eng, inc, it_dir):
            cand = it_dir / f"candidate_{k}"
            if cand.exists():
                shutil.rmtree(cand)
            shutil.copytree(w / "skills", cand / "skills")
            rounds = {"1": first, "2": repaired}[hashes(cand)]
            if rounds:
                _rounds(self.loop / "diag" / f"candidate_{hashes(cand)}", **rounds)

        for k, v in dict(LOOP=self.loop, claude=fake_claude, early_diag=fake_early, ledger=self.rows.append,
                         candidate_hash=hashes, fill=lambda t, **kw: t + json.dumps(sorted(kw))).items():  # fmt: skip
            monkeypatch.setattr(lp, k, v)
        fake_early(1, self.w, None, self.inc, self.it)

    def run(self):
        res = (
            dict(mechanism="first", set={}, scope=dict(when="wide", starts=["s2"])),
            dict(cost_usd=5.0, seconds=100, session="sess"),
        )
        common = dict(SIM_PY="py")
        return self.lp.repair_draft(0, 1, self.w, res, self.inc, self.inc_rounds, self.it, common)


OK, BAD = _rec(True, WALK, GRIP), _rec(False, WALK, MISS)


def test_a_draft_that_broke_nothing_gets_no_extra_turn(lp, monkeypatch, tmp_path):
    d = Draft(lp, monkeypatch, tmp_path, dict(s1=OK, s2=OK, s3=BAD), None)
    eng, acct = d.run()
    assert eng["mechanism"] == "first" and acct["cost_usd"] == 5.0 and d.calls == [] and d.rows == []


def test_a_repair_that_breaks_fewer_starts_replaces_the_draft(lp, monkeypatch, tmp_path):
    d = Draft(lp, monkeypatch, tmp_path, dict(s1=BAD, s2=OK, s3=BAD), dict(s1=OK, s2=OK, s3=BAD))
    eng, acct = d.run()
    assert eng["mechanism"] == "repaired" and d.calls[0]["resume"] == "sess" and d.calls[0]["role"] == "engineer_1"
    assert acct["cost_usd"] == 7.0 and acct["seconds"] == 120
    assert (d.w / "skills" / "__init__.py").read_text() == "X = 2\n"
    assert (d.it / "candidate_1" / "skills" / "__init__.py").read_text() == "X = 2\n"
    brief = json.loads((d.w / "repair_brief.json").read_text())
    assert [b["label"] for b in brief["broken"]] == ["s1"] and brief["broken"][0]["first_difference"]["index"] == 1
    assert brief["fixed"] == ["s2"] and brief["outside"] == ["s1"]
    row = d.rows[0]
    assert (
        row["step"] == "repair"
        and row["kept"] == "repair"
        and row["before"]["broken"] == ["s1"]
        and row["after"]["broken"] == []
    )


@pytest.mark.parametrize(
    "rounds", [dict(s1=BAD, s2=OK, s3=OK), dict(s1=OK, s2=BAD, s3=BAD)], ids=["no better", "lost net starts"]
)
def test_a_repair_that_does_not_help_leaves_the_first_draft(lp, monkeypatch, tmp_path, rounds):
    d = Draft(lp, monkeypatch, tmp_path, dict(s1=BAD, s2=OK, s3=OK), rounds)
    eng, acct = d.run()
    assert eng["mechanism"] == "first" and acct["cost_usd"] == 7.0
    assert (d.w / "skills" / "__init__.py").read_text() == "X = 1\n"
    assert (d.it / "candidate_1" / "skills" / "__init__.py").read_text() == "X = 1\n"
    assert d.rows[0]["kept"] == "first"


def test_a_stopped_repair_leaves_the_first_draft(lp, monkeypatch, tmp_path):
    def late():
        (tmp_path / "fam" / "loop" / "iter_00" / "cand_1" / "skills" / "__init__.py").write_text("X = half\n")
        raise lp.RoleLate("stopped")

    d = Draft(lp, monkeypatch, tmp_path, dict(s1=BAD, s2=OK, s3=BAD), None, turn=late)
    eng, acct = d.run()
    assert eng["mechanism"] == "first" and acct["cost_usd"] == 5.0
    assert (d.w / "skills" / "__init__.py").read_text() == "X = 1\n"
    assert d.rows[0]["kept"] == "first" and d.rows[0]["late"] is True


def test_a_repair_runs_on_its_own_clock(lp, monkeypatch, tmp_path):
    """The Engineers' late stop is for a draft still missing; a repair's stop comes from REPAIR_MIN."""

    def turn():
        assert d.stop.wait(2)
        raise lp.RoleLate("stopped")

    d = Draft(lp, monkeypatch, tmp_path, dict(s1=BAD, s2=OK, s3=BAD), None, turn=turn)
    fake = lp.claude

    def spy(*a, stop=None, **k):
        d.stop = stop
        return fake(*a, stop=stop, **k)

    monkeypatch.setattr(lp, "claude", spy)
    monkeypatch.setattr(lp, "REPAIR_MIN", 0.0)
    eng, _ = d.run()
    assert eng["mechanism"] == "first" and d.rows[0]["late"] is True


def test_the_repair_prompt_fills_from_what_the_engineer_step_has(lp):
    text = (HERE / "prompts" / "repair.md").read_text()
    names = set(re.findall(r"\{([A-Z_]+)\}", text))
    assert names <= {"BRIEF", "ITER", "SKILLS", "N_DIAG", "SIM_PY", "FAMILY", "DIAG_INC"}


def _answer(session: str) -> str:
    return json.dumps(dict(total_cost_usd=0.5, session_id=session, structured_output=dict(a=1)))


def test_a_turn_in_a_finished_session_resumes_it_and_keeps_its_budget_ledger(lp, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(lp, "role_tmp", lambda: tmp_path)
    monkeypatch.setattr(lp, "transcript", lambda s: None)
    monkeypatch.setattr(lp, "git_head", lambda: "h")
    monkeypatch.setattr(lp, "undo_commits", lambda h: [])
    monkeypatch.setattr(lp, "allowed_dirs", lambda: [])
    monkeypatch.setattr(lp, "ISOLATION", "ISOLATION NOTE {TMP}")

    def fake_role(cmd, prompt, env, session, stop=None, seen=0):
        return _answer(session), "", 0, []

    def spy(cmd, prompt, env, session, stop=None, **kw):
        seen.update(cmd=cmd, prompt=prompt, env=env, session=session, **kw)
        return fake_role(cmd, prompt, env, session, stop=stop)

    monkeypatch.setattr(lp, "run_role", spy)
    lp.claude("engineer_1", "fresh", {}, tmp_path, "Agent")
    assert "--session-id" in seen["cmd"] and "ISOLATION NOTE" in seen["prompt"]
    ledger = seen["env"]["RSI_TRY_LEDGER"]
    so, acct = lp.claude("engineer_1", "again", {}, tmp_path, "Agent", resume="abc")
    assert so == dict(a=1) and acct["session"] == "abc"
    assert "--resume" in seen["cmd"] and "--session-id" not in seen["cmd"] and seen["prompt"] == "again"
    assert seen["env"]["RSI_TRY_LEDGER"] == ledger  # the role's 240 rounds an iteration, not 240 more
    assert (tmp_path / "engineer_1_repair_prompt.md").exists() and (tmp_path / "engineer_1_repair_raw.json").exists()


@pytest.mark.parametrize("seen,killed", [(0, True), (1, False)], ids=["a new answer", "only the earlier answer"])
def test_only_a_new_answer_starts_the_grace_period(lp, monkeypatch, tmp_path, seen, killed):
    t = tmp_path / "t.jsonl"
    t.write_text(lp.ANSWERED + "\n")
    monkeypatch.setattr(lp, "transcript", lambda s: t)
    cmd = ["bash", "-c", "sleep 2 & wait"]
    out = lp.run_role(cmd, "", dict(PATH="/usr/bin:/bin"), "s", grace=0.0, poll=0.05, seen=seen)
    assert bool(out[3]) is killed


class Twins:
    def __init__(self, lp, monkeypatch, tmp_path, reviewers=2):
        self.calls = []

        def fake_claude(role, prompt, schema, it_dir, tools, stub=None, stop=None, resume=None):
            self.calls.append(dict(role=role, prompt=prompt))
            hs = [dict(id=f"H{i}", claim=role) for i in range(1, 6)]
            return dict(notes=role, hypotheses=hs, infeasible=[dict(id="C1", task="t")]), dict(
                cost_usd=1.0, session=role, seconds=len(role)
            )

        monkeypatch.setattr(lp, "claude", fake_claude)
        monkeypatch.setattr(lp, "REVIEWERS", reviewers)


def test_two_reviewers_read_one_prompt_and_every_hypothesis_is_probed(lp, monkeypatch, tmp_path):
    t = Twins(lp, monkeypatch, tmp_path)
    hyps, acct = lp.twin_review("the prompt", tmp_path, None)
    assert sorted(c["role"] for c in t.calls) == ["reviewer_A", "reviewer_B"] and {c["prompt"] for c in t.calls} == {
        "the prompt"
    }
    ids = [h["id"] for h in hyps["hypotheses"]]
    assert ids == [f"{r}H{i}" for r in "AB" for i in range(1, 5)] and len(set(ids)) == lp.hypothesis_cap() == 8
    assert [c["id"] for c in hyps["infeasible"]] == [
        "AC1",
        "BC1",
    ]  # two claims must not share an id: parks are keyed by it
    assert acct["cost_usd"] == 2.0 and acct["sessions"] == ["reviewer_A", "reviewer_B"]


def test_merged_ids_are_short_whatever_a_reviewer_calls_its_hypotheses(lp, monkeypatch, tmp_path):
    def fake_claude(role, *a, **k):
        hs = [dict(id="H1_east_end_stance"), dict(id="H7")]
        return dict(notes="", hypotheses=hs), dict(cost_usd=0.0, session=role, seconds=1)

    monkeypatch.setattr(lp, "claude", fake_claude)
    monkeypatch.setattr(lp, "REVIEWERS", 2)
    hyps, _ = lp.twin_review("p", tmp_path, None)
    assert [h["id"] for h in hyps["hypotheses"]] == ["AH1", "AH2", "BH1", "BH2"]
    assert hyps["hypotheses"][0]["own_id"] == "H1_east_end_stance"


def test_the_manager_can_name_only_this_iterations_hypotheses(lp):
    sch = lp.manager_schema(True, ["AH1", "BH1"])["properties"]
    assert sch["hypothesis"]["enum"] == ["AH1", "BH1"] and sch["hypotheses"]["items"]["enum"] == ["AH1", "BH1"]
    assert "enum" not in lp.manager_schema(True)["properties"]["hypothesis"]


def test_one_reviewer_is_the_default(lp, monkeypatch):
    monkeypatch.setattr(lp, "family_spec", lambda: {})
    monkeypatch.setattr(lp, "PLAN", "plan", raising=False)
    assert lp.REVIEWERS == 1 and lp.hypothesis_cap() == lp.MAX_HYPOTHESES
    assert lp.TWIN_NOTE not in lp.prompt_vars()["CANDIDATES_NOTE"]
    monkeypatch.setattr(lp, "REVIEWERS", 2)
    assert lp.TWIN_NOTE in lp.prompt_vars()["CANDIDATES_NOTE"]


def test_a_failed_reviewer_stops_the_review(lp, monkeypatch, tmp_path):
    def fake_claude(role, *a, **k):
        if role == "reviewer_B":
            raise lp.RoleFailed("died")
        return dict(notes="", hypotheses=[]), dict(cost_usd=0.0, session=role, seconds=1)

    monkeypatch.setattr(lp, "claude", fake_claude)
    monkeypatch.setattr(lp, "REVIEWERS", 2)
    with pytest.raises(lp.RoleFailed, match="died"):
        lp.twin_review("p", tmp_path, None)


def test_reviewers_and_repair_are_settings_a_resume_keeps(lp):
    args = SimpleNamespace(reviewers=None, repair=None)
    assert (
        lp.resolve_settings(args, {}, False)["reviewers"] == 1
        and lp.resolve_settings(args, {}, False)["repair"] is False
    )
    kept = lp.resolve_settings(args, dict(reviewers=2, repair=True), True)
    assert kept["reviewers"] == 2 and kept["repair"] is True
    assert lp.resolve_settings(SimpleNamespace(reviewers=3), dict(reviewers=2), True)["reviewers"] == 3


def test_an_engineer_must_declare_its_scope(lp):
    assert "scope" in lp.ENGINEER_SCHEMA["required"]
    text = (HERE / "prompts" / "engineer.md").read_text()
    assert "`scope`" in text and "rules.md 38" in text
