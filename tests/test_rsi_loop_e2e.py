"""The inner loop end to end on a fake bed (tests/fake_loop_backend.py): no simulator, no model. Every role call is a
canned answer; the loop's own code runs for the rest (diagnostic rounds, cards, drafts, the check half, the gate's
call, the pool, the stop rules, the wrap-up). Written first (TODO 54) as the net under the lean flow: it asserts the
step order of two iterations, the ledger rows, the gate verdict, and that a stop runs the sealed set."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import re
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("skfmm")  # gate.py imports the scene runtime

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
LOOP_NAMES = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "records", "design", "loop")
TASKS = ("TaskA", "TaskB", "TaskC", "TaskD")
SEED_SKILL = "PARAMS = {'alpha': 1}\n" + "".join(f"# --- {t}\nSOLVE_{t} = False\nBREAK_{t} = False\n" for t in TASKS)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_loop_backend as FB  # noqa: E402


def _isolated(name: str, path: Path):
    """`path` under its own module name (the warehouse experiment has a loop.py, gate.py and batch.py too)."""
    saved, old_path = {k: sys.modules.pop(k, None) for k in LOOP_NAMES}, list(sys.path)
    sys.path[:0] = [str(HERE), str(ROOT / "src")]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = old_path
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


class World:
    """A loop on the fake bed in a temp tree, its role calls canned, its gate and sealed set run in this process."""

    def __init__(self, tmp: Path, monkeypatch, family_extra: dict | None = None):
        self.tmp, self.calls, self.sealed_runs = tmp, [], []
        self.targets: dict[int, list[str]] = {}  # iteration -> the task each hypothesis (H1, H2, ...) is about
        self.conflict: set[int] = set()  # iterations in which drafts 1 and 2 edit the same line
        self.composer = dict(decision="adopt", adopted=1, notes="canned")
        self.breaks: dict[str, list[str]] = {}  # "<iter>:<k>" -> tasks that draft breaks
        monkeypatch.setenv("E2E_ROOT", str(tmp / "bed"))
        monkeypatch.setenv("RSI_BACKEND_PATH", "e2efake=fake_loop_backend")
        monkeypatch.setenv("RSI_NO_SEALED", "0")
        (tmp / "bed" / "skills").mkdir(parents=True)
        (tmp / "bed" / "skills" / "skill.py").write_text(SEED_SKILL)
        fam = dict(name="e2e", backend="e2efake", tasks=list(TASKS), pools={"diag": {"n": 40}, "accept": {"n": 60}},
                   caps=dict(FB.CAPS), plan=None, rules="x", prompts="x", **(family_extra or {}))  # fmt: skip
        (tmp / "families").mkdir()
        (tmp / "families" / "e2e.json").write_text(json.dumps(fam))
        self.fam = fam
        self.lp = lp = _isolated("rsi_e2e_loop", HERE / "loop.py")
        self.gate = _isolated("rsi_e2e_gate", HERE / "gate.py")
        import backend as BK

        monkeypatch.setattr(BK, "HERE", tmp)
        monkeypatch.setattr(lp, "REPO", tmp)
        monkeypatch.setattr(lp, "family_spec", lambda: self.fam)
        monkeypatch.setattr(lp, "PROCS", 2)
        lp.configure("e2e")
        self.hidden = lp.HIDDEN
        for pool, n_per, where in (("pool_diag", 10, lp.LOOP), ("pool_accept", 15, lp.HIDDEN)):
            (where / pool).mkdir(parents=True)
            for t in TASKS:
                for i in range(n_per):
                    (where / pool / f"{t}_s{i}.json").write_text(json.dumps(dict(task=t)))
        for k, v in dict(claude=self.claude, transcript_leaks=lambda s: [], changed_files=lambda: [],
                         role_tmp=lambda: tmp / "roletmp", write_hit_rate=lambda: None,
                         copy_review_notes=lambda *a: None).items():  # fmt: skip
            monkeypatch.setattr(lp, k, v)
        monkeypatch.setattr(lp, "subprocess", self.shim())
        FB.CALLS.clear()

    # -- the process seam: the gate and the sealed set run in this process
    def shim(self):
        world, real = self, subprocess

        class GateProc:
            def __init__(self, args, cwd=None, stdout=None, stderr=None, text=None, env=None):
                with contextlib.redirect_stdout(stdout or sys.stdout):
                    self.returncode = world.gate.main(list(map(str, args[2:])))

            def poll(self):
                return self.returncode

            def wait(self):
                return self.returncode

            def kill(self):
                pass

        def run(args, *a, **kw):
            if len(args) > 1 and str(args[1]).endswith("sealed.py"):
                world.sealed_runs.append(list(map(str, args)))
                return types.SimpleNamespace(returncode=0)
            return real.run(args, *a, **kw)

        shim = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real) if not k.startswith("__")})
        shim.Popen, shim.run = GateProc, run
        return shim

    # -- the roles
    def claude(self, role, prompt, schema, it_dir, tools, stub=None, stop=None, resume=None):
        it_dir = Path(it_dir)
        n = int(it_dir.name.split("_")[1]) if it_dir.name.startswith("iter_") else -1
        self.calls.append(role)
        (it_dir / f"{role}_prompt.md").write_text(prompt)
        acct = dict(cost_usd=1.0, session=None, seconds=1.0)
        tasks = self.targets.get(n, ["TaskA", "TaskB", "TaskC"])
        if role == "reviewer":
            hyps = [dict(id=f"H{i}", claim=f"{t} fails", instead=f"fix {t}", evidence="cards", delta={"set": {}},
                         predicted_fixed=[], predicted_broken=[], tasks=[t]) for i, t in enumerate(tasks, 1)]  # fmt: skip
            return dict(notes="canned", hypotheses=hyps), acct
        if role.startswith("engineer"):
            k = int(role.split("_")[1]) if "_" in role else 0
            w = it_dir / f"cand_{k}" if k else None
            skills = (w / "skills") if w else self.lp.SKILLS
            task = tasks[k - 1] if k else tasks[0]
            f = skills / "skill.py"
            text = f.read_text().replace(f"SOLVE_{task} = False", f"SOLVE_{task} = True")
            for b in self.breaks.get(f"{n}:{k}", []):
                text = text.replace(f"BREAK_{b} = False", f"BREAK_{b} = True")
            if n in self.conflict and k in (1, 2):  # drafts 1 and 2 both rewrite the PARAMS line, differently
                text = text.replace("PARAMS = {'alpha': 1}", f"PARAMS = {{'alpha': {k + 1}}}")
            f.write_text(text)
            return dict(mechanism=f"solve {task}", files_changed=["skill.py"], set={}, tests="none",
                        scope=dict(when=task, starts=[task])), acct  # fmt: skip
        if role == "composer":
            return self.composer, acct
        raise AssertionError(f"unexpected role {role}")

    def run(self, max_iters: int, **kw) -> int:
        a = Namespace(max_iters=max_iters, reprobe=None, stub=False, **kw)
        return self.lp.cmd_run(a)

    def rows(self, step=None) -> list[dict]:
        es = self.lp.entries()
        return [e for e in es if step is None or e.get("step") == step]


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def steps_of(w: World, n: int) -> list[str]:
    return [e["step"] for e in w.rows() if e.get("iter") == n]


def test_two_iterations_run_the_lean_flow_and_the_gate_accepts_both(world):
    world.targets = {0: ["TaskA", "TaskB", "TaskC"], 1: ["TaskD"]}
    assert world.run(2, stop_rate=0.99) == 0
    # no Manager, no Prober, no Composer: the program assigns, merges and the drafts go to the gate as one candidate
    assert world.calls == ["reviewer", "engineer_1", "engineer_2", "engineer_3", "reviewer", "engineer_1"] or sorted(
        world.calls[:4]
    ) == ["engineer_1", "engineer_2", "engineer_3", "reviewer"] and set(world.calls) <= {
        "reviewer", "engineer_1", "engineer_2", "engineer_3"
    }  # fmt: skip
    order = [
        e["step"]
        for e in world.rows()
        if e["step"] in ("cards", "review", "assign", "engineer", "compose", "check", "gate")
    ]
    first = order[: order.index("gate") + 1]
    assert [x for i, x in enumerate(first) if i == 0 or x != first[i - 1]] == [
        "cards", "review", "assign", "engineer", "compose", "check", "gate"
    ]  # fmt: skip
    assert world.rows("assign")[0]["assigned"] == ["H1", "H2", "H3"]
    comp = world.rows("compose")[0]
    assert comp["by"] == "program" and comp["decision"] == "merge" and "merged" not in comp
    gates = world.rows("gate")
    assert [g["accepted"] for g in gates] == [True, True]
    assert gates[0]["result"]["b"] >= 5 and gates[0]["result"]["c"] == 0
    assert all(c["sent"] for c in world.rows("check"))
    ledger_gate = [json.loads(x) for x in (world.hidden / "gate" / "ledger.jsonl").read_text().splitlines()]
    assert [x["accepted"] for x in ledger_gate] == [True, True] and ledger_gate[0]["stage1"]["b"] > 0
    inc = json.loads((world.hidden / "gate" / "incumbent.json").read_text())
    text = (Path(inc["dir"]) / "skills" / "skill.py").read_text()
    assert all(f"SOLVE_{t} = True" in text for t in TASKS)  # A, B, C merged in iteration 0, D in iteration 1


def test_conflicting_drafts_go_to_the_composer(world):
    world.targets = {0: ["TaskA", "TaskB", "TaskC"]}
    world.conflict = {0}
    world.composer = dict(decision="adopt", adopted=3, notes="canned")
    world.run(1, stop_rate=0.99)
    assert world.calls.count("composer") == 1
    comp = world.rows("compose")[0]
    assert (
        comp["mode"] == "conflict"
        and comp["conflicts"]
        and comp["merged_cleanly"] == [1, 3]
        and list(comp["conflicts"]) == ["2"]
    )
    ctx = json.loads((world.lp.STATE / "iter_00" / "composer_context.json").read_text())
    assert ctx["compose_mode"] == "conflict"


def test_no_positive_draft_calls_the_composer_which_may_keep(world):
    world.targets = {0: ["TaskA", "TaskB", "TaskC"]}
    world.breaks = {f"0:{k}": list(TASKS) for k in (1, 2, 3)}
    world.composer = dict(decision="keep", adopted=None, notes="nothing worth a gate call")
    world.run(1, stop_rate=0.99)
    assert world.rows("compose")[0]["mode"] == "none_positive" and not world.rows("gate")


def test_a_draft_that_fails_the_check_half_is_not_sent(world):
    world.targets = {0: ["TaskA"]}
    world.composer = dict(decision="adopt", adopted=1, notes="canned")
    world.breaks = {"0:1": list(TASKS)}  # the lone draft fails every start: a negative screen, the Composer adopts it
    world.run(1, stop_rate=0.99)
    chk = world.rows("check")
    assert chk and chk[0]["broken"] > chk[0]["fixed"] and not chk[0]["sent"] and not world.rows("gate")


def test_two_iterations_without_progress_end_the_run_and_the_wrap_up_runs(world):
    world.breaks = {f"{n}:{k}": list(TASKS) for n in (0, 1) for k in (1, 2, 3)}
    world.composer = dict(decision="keep", adopted=None, notes="keep")
    assert world.run(6, stop_rate=0.99) == 0
    stop = world.rows("stop")[-1]
    assert stop["reason"].startswith("no progress: iterations 0, 1") and len(world.rows("review")) == 2
    assert not world.sealed_runs  # nothing was accepted: the sealed set has no incumbent to confirm


def test_max_iters_ends_through_the_same_wrap_up_and_runs_the_sealed_set(world):
    """TODO 54 (verify.md 1): --max-iters used to return without finish(), so the sealed set never ran."""
    world.targets = {0: ["TaskA"], 1: ["TaskB"]}
    assert world.run(2, stop_rate=0.99) == 0
    assert world.rows("stop")[-1]["reason"].startswith("max iterations 2")
    assert len(world.sealed_runs) == 1 and world.rows("sealed")


def test_a_stop_rule_runs_the_sealed_set(world):
    world.targets = {0: ["TaskA", "TaskB", "TaskC"], 1: ["TaskD"]}
    assert world.run(5, stop_rate=0.55) == 0
    stops = world.rows("stop")
    assert stops and "sealed set" in stops[-1]["reason"]
    assert len(world.sealed_runs) == 1 and world.rows("sealed")


def test_a_failed_role_ends_through_the_wrap_up_too(world, monkeypatch):
    world.targets = {0: ["TaskA"]}
    world.run(1, stop_rate=0.99)  # one accepted iteration

    def boom(*a, **k):
        raise world.lp.RoleFailed("the role died")

    monkeypatch.setattr(world.lp, "claude", boom)
    assert world.run(3, stop_rate=0.99) == 3
    assert world.rows("stop")[-1]["reason"].startswith("role call failed")
    assert len(world.sealed_runs) == 2


def test_every_role_prompt_is_filled_and_names_no_missing_plan(world):
    world.targets = {0: ["TaskA", "TaskB", "TaskC"]}
    world.conflict = {0}
    world.run(1, stop_rate=0.99)
    for p in (world.lp.STATE / "iter_00").glob("*_prompt.md"):
        text = p.read_text()
        assert not re.findall(r"\{[A-Z_]{3,}\}", text), p.name
        assert "scene-rsi-plan" not in text
    rev = (world.lp.STATE / "iter_00" / "reviewer_prompt.md").read_text()
    assert "episodes per task that you see" in rev and "(30 episodes per task)" not in rev
    assert "Prober" not in rev


def test_the_hypothesis_schema_follows_the_caps(world):
    lp = world.lp
    props = lambda: lp.hypotheses_schema()["properties"]["hypotheses"]["items"]["properties"]  # noqa: E731
    p = props()
    assert not {"measure", "design", "sweep"} & set(p) and "starts" in p
    assert "infeasible" not in lp.hypotheses_schema()["properties"]
    assert "tasks" in lp.hypotheses_schema()["properties"]["hypotheses"]["items"]["required"]
    scene = dict(lp.CAPS, claims=True, design_probes=True, measure_claims=True, prober=True, manager="llm")
    lp.CAPS = scene
    p = props()
    assert {"measure", "design", "sweep"} <= set(p) and "starts" not in p
    assert "infeasible" in lp.hypotheses_schema()["properties"]
    assert (
        "tune" in lp.manager_schema(True)["properties"]
        and "reopen" in lp.manager_schema(True)["properties"]["decision"]["enum"]
    )
    lp.CAPS = dict(scene, claims=False, prober=False)
    m = lp.manager_schema(True)["properties"]
    assert "tune" not in m and "gap_proposals" not in m and "reopen" not in m["decision"]["enum"]


def test_program_assignment_is_in_reviewer_order_with_disjoint_tasks(world):
    lp = world.lp
    h = lambda i, *t, **kw: dict(id=i, tasks=list(t), **kw)  # noqa: E731
    ids, skipped = lp.assign_program([h("H1", "A"), h("H2", "A", "B"), h("H3", "B"), h("H4", "C"), h("H5", "D")], 3)
    assert ids == ["H1", "H3", "H4"] and "overlaps" in skipped["H2"] and skipped["H5"] == "no Engineer slot left"
    assert lp.assign_program([h("H1", "A"), h("H2", "A")], 3)[0] == ["H1"]  # fewer when fewer are disjoint
    assert lp.assign_program([h("H1"), h("H2", "A")], 3)[0] == ["H1"]  # no declared task holds them all
    assert lp.assign_program([h("H1", "A", status="refuted"), h("H2", "B")], 3)[0] == ["H2"]
    assert lp.assign_program([dict(id="H1", starts=["A_s1"]), h("H2", "A")], 3)[0] == ["H1"]


def test_prompt_vars_count_the_starts_and_drop_a_missing_plan(world):
    v = world.lp.prompt_vars()
    assert v["N_DIAG"] == 40 and v["N_DIAG_TASK"] == "10"  # before the check half is split off
    world.lp.split_check_half()
    v = world.lp.prompt_vars()
    assert v["N_DIAG"] == 20 and v["N_DIAG_TASK"] == "5" and v["N_ACCEPT"] == 60 and v["PLAN"] == ""
    assert v["N_DIAG_FULL"] == 40 and v["N_DIAG_FULL_TASK"] == "10"


def test_a_sentinel_drift_voids_the_call_and_does_not_blacklist_the_candidate(world, monkeypatch):
    """TODO 54 (verify.md 8): gate.py's drift branch said "void" in its reason and left `void` unset, so the hash
    counted as rejected for good and the call counted against the gate budget."""
    world.targets = {0: ["TaskA", "TaskB", "TaskC"]}
    monkeypatch.setenv("E2E_DRIFT", "1")
    world.run(1, stop_rate=0.99)
    row = [json.loads(x) for x in (world.hidden / "gate" / "ledger.jsonl").read_text().splitlines()][0]
    assert not row["accepted"] and row["void"] is True and "sentinel" in row["reason"]
    g = world.rows("gate")[0]
    assert g["void"] is True and not world.lp.run_gates(world.lp.entries(), None)  # no gate call spent
    monkeypatch.delenv("E2E_DRIFT")
    world.lp.STATE.joinpath("ledger.jsonl").write_text("")
    world.run(1, stop_rate=0.99)  # the same candidate is judged now, not "rejected before"
    gates = [json.loads(x) for x in (world.hidden / "gate" / "ledger.jsonl").read_text().splitlines()]
    assert "rejected before" not in gates[-1]["reason"]
