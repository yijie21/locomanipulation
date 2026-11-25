"""TODO 54 phase 4: the gate and the check half run beside the next iteration (scripts/rsi/overlap.py), on the fake bed.

No simulator, no model: every role call is canned, the gate runs in a thread of this process, and a threading.Event
holds it so a test decides what finishes first. The walls are asserted as they are in a real run: the hidden tree has mode
000 while a (fake) role runs, the gate reads a private copy that is gone afterwards, no verdict file or ledger row exists
before the driver settles the call."""

from __future__ import annotations

import json
import re
import stat
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_loop_backend as FB  # noqa: E402
import test_rsi_loop_e2e as E2E  # noqa: E402

TASKS = E2E.TASKS
WAIT = 30


class OWorld(E2E.World):
    """The e2e world with the gate held by an Event, the roles locking the hidden tree, and an event log."""

    def __init__(self, tmp, monkeypatch):
        self.ev: list[tuple] = []
        self.lock = threading.Lock()
        self.hold: dict[int, threading.Event] = {}
        self.on_role = self.pre_role = None
        self.modes: list[int] = []
        self.gate_paths: dict[int, dict] = {}
        self.params: dict[str, int] = {}  # "<iter>:<k>" -> the alpha that draft writes into PARAMS
        super().__init__(tmp, monkeypatch)
        self.audit = self.lp.module("audit")
        self.priv = tmp / "priv"
        real = self.gate.main

        def main(argv):
            a = list(map(str, argv))
            n = int(Path(a[a.index("--candidate") + 1]).parent.name.split("_")[1])
            gate_dir, pool = Path(a[a.index("--gate") + 1]), Path(a[a.index("--pool") + 1])
            self.gate_paths[n] = dict(gate=gate_dir, pool=pool, exists=gate_dir.exists(),
                                      env=__import__("os").environ.get("RSI_HIDDEN_FAMILY"))  # fmt: skip
            self.log("gate_start", n)
            ev = self.hold.get(n)
            if ev is not None:
                if FB.SLOTS is not None:
                    with FB.SLOTS:  # the gate's rounds hold a simulator slot while it is held
                        ev.wait(WAIT)
                else:
                    ev.wait(WAIT)
            rc = real(argv)
            self.log("gate_end", n)
            return rc

        self.gate.main = main

    def log(self, *e):
        with self.lock:
            self.ev.append(e)

    def index(self, *e) -> int:
        return self.ev.index(e)

    def claude(self, role, prompt, schema, it_dir, tools, stub=None, stop=None, resume=None):
        it_dir = Path(it_dir)
        n = int(it_dir.name.split("_")[1])
        if self.pre_role:
            self.pre_role(role, n)
        with self.audit.hidden_locked():
            self.log("role", role, n)
            self.modes.append(stat.S_IMODE(self.hidden.stat().st_mode))
            if self.on_role:
                self.on_role(role, n)
            if role.startswith("engineer"):
                out = self.engineer(role, it_dir, n)
            else:
                out = super().claude(role, prompt, schema, it_dir, tools, stub, stop, resume)
            if role == "reviewer" and getattr(self, "settle_in_reviewer", None) == n:
                self.hold[n - 1].set()  # the call in flight finishes before this Reviewer answers
                ov = self.lp.module("state").OVERLAP
                assert ov.pending.done.wait(WAIT)
            return out

    def engineer(self, role, it_dir, n):
        k = int(role.split("_")[1])
        w = it_dir / f"cand_{k}"
        task = json.loads((w / "engineer_context.json").read_text())["hypothesis"]["tasks"][0]
        f = w / "skills" / "skill.py"
        text = f.read_text().replace(f"SOLVE_{task} = False", f"SOLVE_{task} = True")
        if f"{n}:{k}" in self.params:
            text = text.replace("PARAMS = {'alpha': 1}", f"PARAMS = {{'alpha': {self.params[f'{n}:{k}']}}}")
        f.write_text(text)
        return dict(mechanism=f"solve {task}", files_changed=["skill.py"], set={}, tests="none",
                    scope=dict(when=task, starts=[task])), dict(cost_usd=1.0, session=None, seconds=1.0)  # fmt: skip

    def release_at(self, role: str, n: int, hold_n: int):
        """Release the gate of iteration `hold_n` when `role` of iteration `n` is called."""
        self.hold.setdefault(hold_n, threading.Event())
        prev = self.on_role

        def hook(r, i):
            if (r, i) == (role, n):
                self.hold[hold_n].set()
            if prev:
                prev(r, i)

        self.on_role = hook

    def kinds(self) -> list[tuple]:
        return [(e["kind"], e.get("iter")) for e in self.rows("overlap")]


def build(tmp_path, monkeypatch, overlap=True):
    monkeypatch.setitem(FB.CAPS, "overlap", overlap)
    monkeypatch.setenv("RSI_PRIVATE_ROOT", str(tmp_path / "priv"))
    monkeypatch.delenv("RSI_HIDDEN_FAMILY", raising=False)
    monkeypatch.setattr(FB, "BEFORE_JOB", None)
    monkeypatch.setattr(FB, "SLOTS", None)
    for lst in (FB.JOBS, FB.RAN, FB.CALLS):
        lst.clear()
    w = OWorld(tmp_path, monkeypatch)
    drafts = w.lp.module("drafts")  # the driver's 2 s poll for the Engineers, on a fast clock
    monkeypatch.setattr(drafts, "time", types.SimpleNamespace(time=time.time, sleep=lambda s: time.sleep(min(s, 0.01))))
    return w


@pytest.fixture
def ow(tmp_path, monkeypatch):
    return build(tmp_path, monkeypatch)


def it(w, n):
    return w.lp.STATE / f"iter_{n:02d}"


# ---------------------------------------------------------------------------------------------------- rejection
def test_a_rejection_keeps_the_overlapped_iteration_and_shows_the_overlap(ow, monkeypatch):
    monkeypatch.setitem(FB.CAPS, "p_max", 1e-30)  # the check half sends the candidate, the gate rejects it
    ow.targets = {0: ["TaskA"], 1: ["TaskD"]}
    ow.release_at("engineer_1", 1, 0)  # gate 0 ends while iteration 1's Engineers work
    assert ow.run(2, stop_rate=0.99) == 0
    # order: iteration 0's candidate is sent, iteration 1's Reviewer and Engineer run while the gate is held, then it ends
    rows = [(e["step"], e.get("kind"), e.get("iter")) for e in ow.rows()]
    assert rows.index(("overlap", "sent", 0)) < rows.index(("review", None, 1)) < rows.index(("overlap", "settled", 0))
    r, e, z = (ow.index(*x) for x in (("role", "reviewer", 1), ("role", "engineer_1", 1), ("gate_end", 0)))
    assert r < e < z  # the gate is held until the Engineer of iteration 1 is called
    gates = ow.rows("gate")
    assert [g["accepted"] for g in gates] == [False, False] and [g["iter"] for g in gates] == [0, 1]
    assert not ow.rows("rebase") and not ow.rows("drop") and not ow.rows("rescreen")  # a rejection needs none
    assert ow.kinds() == [("seed_rounds", None), ("sent", 0), ("settled", 0), ("sent", 1), ("settled", 1)]
    s0 = ow.rows("overlap")[2]
    assert s0["accepted"] is False and s0["beside_s"] >= 0 and s0["waited_s"] >= 0
    # iteration 1 used the incumbent iteration 0 started from; both candidates are in the ledger once, with their iteration
    assert [e["iter"] for e in ow.rows("check")] == [0, 1] and all(e["overlapped"] for e in gates + ow.rows("check"))
    # the roles of iteration 1 read no verdict of iteration 0
    ctx = json.loads((it(ow, 1) / "reviewer_context.json").read_text())
    assert not [r_ for r_ in ctx["history"] if r_.get("step") in ("check", "gate") and r_.get("iter") == 0]
    assert ctx["pending"]["iter"] == 0 and "verdict" not in json.dumps(ctx["pending"]).replace("its verdict is not", "")
    assert (it(ow, 0) / "gate.json").exists() and (it(ow, 0) / "check.json").exists()  # written at the settle
    hit = json.loads((ow.lp.STATE / "hit_rate.json").read_text()) if (ow.lp.STATE / "hit_rate.json").exists() else None
    assert hit is None or hit["gate"]["calls"] in (0, 2)  # write_hit_rate is stubbed by the e2e world


def test_the_gate_of_a_run_without_the_cap_is_synchronous(tmp_path, monkeypatch):
    w = build(tmp_path, monkeypatch, overlap=False)
    w.targets = {0: ["TaskA"], 1: ["TaskB"]}
    w.run(2, stop_rate=0.99)
    assert not w.rows("overlap") and w.lp.module("state").OVERLAP is None
    assert w.index("gate_end", 0) < w.index("role", "reviewer", 1)


# ---------------------------------------------------------------------------------------------------- acceptance
def test_an_accept_before_the_engineers_moves_the_iteration_on_and_drops_stale_hypotheses(ow):
    ow.targets = {0: ["TaskA", "TaskB"], 1: ["TaskA", "TaskC"]}
    ow.hold[0] = threading.Event()
    ow.settle_in_reviewer = 1
    assert ow.run(2, stop_rate=0.99) == 0
    drops = ow.rows("drop")
    assert [(d["kind"], d["hypothesis"], d["iter"]) for d in drops] == [("hypothesis", "H1", 1)]  # TaskA is solved now
    reb = ow.rows("rebase")
    assert [(x["kind"], x["iter"]) for x in reb] == [("iteration", 1)] and reb[0]["dropped"] == ["H1"]
    assert ow.rows("assign")[1]["assigned"] == ["H2"]
    assert ow.rows("gate")[0]["accepted"]
    eng = [e for e in ow.rows("engineer") if e["iter"] == 1]
    assert [e["mechanism"] for e in eng] == ["solve TaskC"]
    # the new incumbent's diagnostics ran before any draft of iteration 1 was screened
    names = [j[2][0] for j in FB.JOBS if j[2]]
    h0 = ow.rows("gate")[0]["hash"]
    d1 = json.loads((it(ow, 1) / "candidates.json").read_text())[0]["hash"]
    assert names.index(f"incumbent_{h0}") < names.index(f"candidate_{d1}")


def test_an_accept_while_the_drafts_are_built_rebases_them_screens_again_and_drops_the_rest(ow):
    ow.targets = {0: ["TaskA"], 1: ["TaskB", "TaskA", "TaskC"]}
    ow.params = {"0:1": 3, "1:3": 5}  # iteration 0 changes PARAMS; draft 3 of iteration 1 changes the same line
    ow.release_at("engineer_1", 1, 0)
    assert ow.run(2, stop_rate=0.99) == 0
    assert ow.index("role", "reviewer", 1) < ow.index("role", "engineer_1", 1) < ow.index("gate_end", 0)
    by = {
        (r["step"], r.get("kind"), r.get("draft")): r
        for r in ow.rows()
        if r.get("iter") == 1 and r["step"] in ("drop", "rebase", "rescreen")
    }
    assert by[("drop", "draft", 2)]["reason"] == "stale"  # the new incumbent wins every start draft 2 fixed
    assert by[("drop", "draft", 3)]["reason"] == "conflict"  # it edits the PARAMS line the new incumbent changed
    clean = by[("rebase", "draft", 1)]
    assert clean["result"] == "clean" and clean["hash"] != clean["from_hash"] and ("rescreen", None, 1) in by
    # order: rebase, re-screen, then the compose row
    steps = [r["step"] for r in ow.rows() if r.get("iter") == 1]
    assert steps.index("rebase") < steps.index("rescreen") < steps.index("compose")
    comp = [r for r in ow.rows("compose") if r["iter"] == 1][0]
    assert comp["decision"] == "adopt" and comp["adopted"] == 1
    cands = json.loads((it(ow, 1) / "candidates.json").read_text())
    assert (
        [c["number"] for c in cands] == [1]
        and cands[0]["hash"] == clean["hash"]
        and cands[0]["rebased_from"] == clean["from_hash"]
    )
    # the rebased draft is the new incumbent plus its own change; the gate judged that one, against that incumbent
    text = (it(ow, 1) / "rebased_1" / "skills" / "skill.py").read_text()
    assert "SOLVE_TaskA = True" in text and "SOLVE_TaskB = True" in text and "'alpha': 3" in text
    g = [x for x in ow.rows("gate") if x["iter"] == 1][0]
    assert g["hash"] == clean["hash"] and g["accepted"]
    # new incumbent's diagnostics before the re-screen
    names = [j[2][0] for j in FB.JOBS if j[2]]
    h0 = ow.rows("gate")[0]["hash"]
    assert names.index(f"incumbent_{h0}") < names.index(f"candidate_{clean['hash']}")
    assert all(
        isinstance(r["iter"], int) for r in ow.rows() if r["step"] in ("drop", "rebase", "rescreen", "check", "gate")
    )


def test_stale_hypotheses_and_clean_rebase_are_functions_of_the_rounds(ow, tmp_path):
    OV = ow.lp.module("overlap")
    old, new = tmp_path / "old", tmp_path / "new"
    for d, ok in ((old, {"TaskA_s0": False, "TaskA_s1": False, "TaskB_s0": False, "TaskB_s1": True}),
                  (new, {"TaskA_s0": True, "TaskA_s1": True, "TaskB_s0": False, "TaskB_s1": True})):  # fmt: skip
        d.mkdir()
        for k, v in ok.items():
            (d / f"{k}.json").write_text(json.dumps(dict(label=k, ok=v)))
    keep, dropped = OV.stale_hypotheses([dict(id="H1", tasks=["TaskA"]), dict(id="H2", tasks=["TaskB"]),
                                         dict(id="H3", starts=["TaskA_s0"]), dict(id="H4", tasks=["TaskC"])], old, new)  # fmt: skip
    assert [h["id"] for h in keep] == ["H2", "H4"] and [d[0] for d in dropped] == ["H1", "H3"]


def test_merge_onto_a_new_base_is_a_three_way_merge_with_the_new_incumbents_changes_as_touched(tmp_path):
    sys.path.insert(0, str(E2E.HERE))
    import merge as MG

    base, new, draft = tmp_path / "b", tmp_path / "n", tmp_path / "d"
    for d in (base, new, draft):
        d.mkdir()
        (d / "a.py").write_text("x = 1\ny = 1\nz = 1\n")
    (new / "a.py").write_text("x = 2\ny = 1\nz = 1\n")
    (draft / "a.py").write_text("x = 1\ny = 1\nz = 9\n")
    res = MG.merge_drafts(base, {}, [(1, draft, {"k": 1})], tmp_path / "out", onto=new, onto_set={"j": 1})
    assert res["merged"] == [1] and res["set"] == {"j": 1, "k": 1}
    assert (tmp_path / "out" / "a.py").read_text() == "x = 2\ny = 1\nz = 9\n"
    (draft / "a.py").write_text("x = 3\ny = 1\nz = 1\n")
    assert MG.merge_drafts(base, {}, [(1, draft, {})], tmp_path / "out2", onto=new, onto_set={})["conflicts"]
    assert MG.merge_drafts(base, {"k": 0}, [(1, draft, {"k": 1})], tmp_path / "out3", onto=new, onto_set={"k": 2})[
        "conflicts"
    ]


# ---------------------------------------------------------------------------------------------------- the walls
def test_the_hidden_tree_is_locked_while_roles_run_and_the_gate_reads_only_its_private_copy(ow):
    ow.targets = {0: ["TaskA", "TaskB", "TaskC"], 1: ["TaskD"]}
    seen = {}

    def hook(role, n):
        if (role, n) == ("engineer_1", 1):
            ow.hold[0].set()

    def pre(role, n):  # before the role's lock: what the driver could see of the call in flight
        if (role, n) == ("engineer_1", 1):
            seen["verdicts"] = [(it(ow, 0) / f).exists() for f in ("gate.json", "check.json", "gate.log")]
            seen["rows"] = [r["step"] for r in ow.rows() if r.get("iter") == 0 and r["step"] in ("check", "gate")]
            seen["hidden_gate_ledger"] = (ow.hidden / "gate" / "ledger.jsonl").exists()
            seen["priv_alive"] = sorted(p.name for p in ow.priv.iterdir())
            seen["env"] = __import__("os").environ.get("RSI_HIDDEN_FAMILY")

    ow.hold[0] = threading.Event()
    ow.on_role, ow.pre_role = hook, pre
    assert ow.run(2, stop_rate=0.99) == 0
    assert ow.modes and set(ow.modes) == {0}  # mode 000 at every role call
    assert stat.S_IMODE(ow.hidden.stat().st_mode) != 0 and not (ow.hidden / "hidden_mode.txt").exists()
    g = ow.gate_paths[0]
    assert (
        g["gate"].parent.parent == ow.priv and g["pool"].parent == g["gate"].parent
    )  # the private copy, not the hidden tree
    assert g["gate"].parent.name.startswith(".rsi_private_") and ow.hidden not in g["gate"].parents and g["exists"]
    assert g["env"] == str(g["gate"].parent)
    # while the call was in flight: nothing of its result in the visible tree, none in the hidden one, one copy standing
    assert seen["verdicts"] == [False, False, False] and seen["rows"] == [] and not seen["hidden_gate_ledger"]
    assert len(seen["priv_alive"]) == 1 and seen["env"] == str(ow.priv / seen["priv_alive"][0])
    # afterwards: copies gone, the environment clean, the hidden tree holds the paths as its own
    assert list(ow.priv.iterdir()) == [] and "RSI_HIDDEN_FAMILY" not in __import__("os").environ
    assert not list(ow.hidden.glob("private_*.txt"))
    inc = (ow.hidden / "gate" / "incumbent.json").read_text()
    assert str(ow.priv) not in inc and str(ow.hidden) in inc
    assert all(str(ow.priv) not in json.dumps(r) for r in ow.rows())
    assert [r for r in ow.rows("gate")][0]["pool"] == str(ow.hidden / "pool_accept")


def test_the_transcript_audit_flags_a_role_that_names_the_private_copy(ow):
    AU = ow.audit

    def line(cmd):
        return (
            json.dumps(
                dict(message=dict(content=[dict(type="tool_use", name="Bash", id="1", input=dict(command=cmd))]))
            )
            + "\n"
        )

    found = lambda cmd: {k for k, pat in AU.leak_patterns().items() if re.search(pat, AU.audited(line(cmd)))}  # noqa: E731
    assert "private" in found("ls /tmp/.rsi_private_0123456789abcdef01234567/gate")
    assert "private" in found("cat ../.rsi_private_ab/gate/ledger.jsonl")
    assert "private" not in found("ls skills")


# ---------------------------------------------------------------------------------------------------- stopping
def test_a_stop_with_the_gate_in_flight_waits_records_it_and_runs_the_sealed_set(ow):
    ow.targets = {0: ["TaskA", "TaskB", "TaskC"], 1: ["TaskD"]}
    ow.release_at("engineer_1", 1, 0)
    assert ow.run(5, stop_rate=0.55) == 0  # iteration 0's accept reaches 55 %: iteration 1 is abandoned at its join
    steps = [e["step"] for e in ow.rows() if e["step"] in ("gate", "overlap", "stop", "sealed")]
    assert ("abandoned", 1) in ow.kinds() and len(ow.rows("gate")) == 1 and ow.rows("gate")[0]["accepted"]
    assert steps.index("gate") < steps.index("stop") < steps.index("sealed")
    assert "sealed set" in ow.rows("stop")[-1]["reason"] and len(ow.sealed_runs) == 1


def test_max_iters_drains_the_last_call_before_the_wrap_up(ow):
    ow.targets = {0: ["TaskA"], 1: ["TaskB"]}
    ow.hold[1] = threading.Event()  # the last call is still running when the loop is out of iterations
    threading.Timer(0.5, ow.hold[1].set).start()
    assert ow.run(2, stop_rate=0.99) == 0
    rows = [e["step"] for e in ow.rows() if e["step"] in ("gate", "stop", "sealed")]
    assert rows == ["gate", "gate", "stop", "sealed"] and len(ow.sealed_runs) == 1
    assert ow.rows("stop")[-1]["reason"].startswith("max iterations 2")


def test_the_gate_call_budget_counts_the_call_in_flight(ow, monkeypatch):
    monkeypatch.setattr(ow.lp.module("state"), "MAX_GATE_CALLS", 1)
    ow.targets = {0: ["TaskA"], 1: ["TaskB"]}
    ow.hold[0] = threading.Event()
    threading.Timer(0.3, ow.hold[0].set).start()
    assert ow.run(5, stop_rate=0.99) == 0
    assert len(ow.rows("review")) == 1 and len(ow.rows("gate")) == 1  # iteration 1 never started
    assert ow.rows("stop")[-1]["reason"].startswith("gate calls 1 >= 1") and len(ow.sealed_runs) == 1


def test_a_hung_gate_times_out_into_a_gate_error_and_leaves_no_copy(ow, monkeypatch):
    monkeypatch.setenv("RSI_OVERLAP_WAIT_H", "0.0002")
    monkeypatch.setenv("RSI_OVERLAP_KILL_WAIT_S", "0.2")
    ow.targets = {0: ["TaskA"], 1: ["TaskB"]}
    ow.hold[0] = threading.Event()  # never set during the run
    try:
        assert ow.run(2, stop_rate=0.99) == 3
        assert ow.rows("stop")[-1]["reason"].startswith("gate error at iteration 0") and ("timeout", 0) in ow.kinds()
        assert (
            list(ow.priv.iterdir()) == []
            and not ow.rows("gate")
            and "RSI_HIDDEN_FAMILY" not in __import__("os").environ
        )
    finally:
        ow.hold[0].set()  # the fake gate cannot be killed: it ends now, into a directory that is gone


def test_a_crash_leaves_no_private_copy_and_an_orphan_is_removed_at_the_next_start(ow, tmp_path):
    OV = ow.lp.module("overlap")
    tree = OV.PrivateTree("gate", None)
    assert (ow.hidden / "private_gate.txt").exists() and tree.root.exists()
    assert OV.cleanup_orphans() == ["gate"] and not tree.root.exists()
    import os

    os.environ.pop("RSI_HIDDEN_FAMILY", None)


# ---------------------------------------------------------------------------------------------------- seed rounds
def test_the_seed_incumbents_rounds_start_with_the_run_and_the_first_gate_reuses_them(ow):
    ow.targets = {0: ["TaskA"]}
    ev = threading.Event()
    started = threading.Event()

    def before(where, jobs, procs):
        if where == "gate" and jobs[0].out.name == "incumbent_seed_rounds" and not started.is_set():
            started.set()
            ow.log("warm_start")
            assert ev.wait(WAIT)  # held until the first Reviewer has been called

    FB.BEFORE_JOB = before

    def on_role(role, n):
        # The warm-up thread is started before the first role call; on a loaded machine it may reach its first job only
        # after the Reviewer has been called, so the Reviewer waits for it here instead of the test racing the scheduler.
        if role == "reviewer":
            started.wait(WAIT)
            ow.log("warm_running", started.is_set())
            ev.set()

    ow.on_role = on_role
    assert ow.run(1, stop_rate=0.99) == 0
    assert ("warm_running", True) in ow.ev and ow.index("warm_start") < ow.index("gate_start", 0)
    n_accept = len(list((ow.hidden / "pool_accept").glob("*.json")))
    seed_rounds = [x for x in FB.RAN if x[0] == "incumbent_seed_rounds"]
    assert len(seed_rounds) == n_accept == len(set(seed_rounds))  # each start once: the gate found them done
    assert ow.kinds()[0] == ("seed_rounds", None)
    assert (ow.hidden / "gate" / "incumbent_seed_rounds").is_dir()
    assert len(ow.rows("gate")) == 1 and ow.rows("gate")[0]["accepted"]


# ---------------------------------------------------------------------------------------------------- slots
@pytest.mark.parametrize("slots", [1, 3])
def test_drafts_screen_on_one_process_while_the_gate_runs_and_nothing_deadlocks(ow, monkeypatch, slots):
    monkeypatch.setattr(FB, "SLOTS", threading.Semaphore(slots))
    ow.targets = {0: ["TaskA", "TaskB"], 1: ["TaskC", "TaskD"]}
    ow.release_at("engineer_2", 1, 0)
    seen = []
    state = ow.lp.module("state")

    def before(where, jobs, procs):
        ov = state.OVERLAP
        seen.append((where, procs, bool(ov and ov.busy()), threading.current_thread().name))

    FB.BEFORE_JOB = before
    done = threading.Event()

    def go():
        ow.run(2, stop_rate=0.99)
        done.set()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    assert done.wait(60), "deadlock"
    probes = [
        x for x in seen if x[0] == "probe" and x[2] and "(job)" not in x[3]
    ]  # the drafts', not the call's own rounds
    assert probes and all(procs == 1 for _, procs, _, _ in probes)  # drafts: one slot while the call runs
    assert any(
        w == "probe" and not busy and procs == ow.lp.PROCS for w, procs, busy, _ in seen
    )  # all of them otherwise
    assert len(ow.rows("gate")) == 2 and not any(g.get("void") for g in ow.rows("gate"))
    assert all(g["accepted"] for g in ow.rows("gate"))


# ---------------------------------------------------------------------------------------------------- accounting
def test_hit_rate_counts_overlapped_iterations_by_the_iteration_each_row_belongs_to(ow):
    import hit_rate

    ow.targets = {0: ["TaskA"], 1: ["TaskB"]}
    ow.release_at("engineer_1", 1, 0)
    ow.run(2, stop_rate=0.99)
    d = hit_rate.compute(ow.lp.STATE)
    assert d["per_iter"]["0"]["gate"] == 1 and d["per_iter"]["1"]["gate"] == 1 and d["gate"]["calls"] == 2
    assert (
        d["overlap"]["calls"] == 2
        and d["overlap"]["call_s"] >= 0
        and d["overlap"]["rebased_drafts"] == 1
        and d["per_iter"]["1"]["rebase"] == 1
    )
    assert re.search(r"overlap", hit_rate.line(d))
