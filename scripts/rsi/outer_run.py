"""TODO 27.10.8 / 27.10.9 (plan docs/handoff/scene-rsi-plan-20261016.md §12.6): one outer round's families end to end,
the steps write-backs 1-7 took by hand (run_*.sh, sealed_*.sh, wb*_regress.sh, capability_write.py, re-bucketing).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    nohup setsid $PY scripts/rsi/outer_run.py <round dir> <family> [<family> ...] \\
        [--parallel 2] > <round dir>/run/driver.log 2>&1 &

First, once per round (efficiency 4: scene changes belong before a round, not inside it), scripts/scene_lint.py on
the scene and the round's discovered tasks: a scene-level error (an upright axis, a shadowed surface, a fall at rest)
stops the round. Then per family, each step recorded in `<round dir>/run/<family>.json` so a restarted driver goes on
where it stopped:

1. `sandbox`: sandbox.py at /workspace/code/rsi-scene/<family>, unless it exists;
2. `lint`: the family's tasks; one whose goal holds at every drawn start (or at the scene pose) blocks the family;
3. `loop`: loop.py run in the sandbox, to its end; a non-zero exit blocks, and so does a 0 exit whose ledger does not
   end in a normal stop (the incumbent's win, gate calls, wall clock, cost cap, a rejection streak, Manager stop):
   a role failure, a leak or a replay stop is a person's to look at (TODO 36, review 2026-10-22);
4. `sealed`: sealed.py --all in the sandbox;
5. `reflect`: reflect.py writes the family's entry into the experience library, scripts/rsi/experience (TODO 32.15:
   the library is in every sandbox by default, so later families read it once it is committed); a failure leaves
   the library without the entry and does not block;
6. `writeback`: writeback.py into the catalog (nothing when the final is the seed); the catalog's skills directory
   is copied to `<round dir>/run/<family>/skills_before_writeback` first;
7. `tests`: the scene runner's and the catalog's own tests; a family blocked here or at 8 / 9 has the catalog put
   back from that copy (recorded in its state file as `restored`), so a blocked family leaves no write-back behind;
8. `regress`: the family enters regress.py's registry; every registered family's sealed pool re-runs under the
   catalog (tag `after_<family>`); a family worse at p <= 0.05 blocks;
9. `capability`: carry_time.py (the put-down's planned carry time against the route, for the creep rule), then
   capability_write.py from the registry, then every task set re-bucketed (capability/feasibility/after_<family>_<set>).

Up to `--parallel` families run steps 1-5 at once (the simulation slots in batch.py share the machine; reflect
runs one at a time, they share the library's INDEX.md); steps 6-9 take one family at a time, in the order the loops
end. Each sandbox is `git archive HEAD` and the write-backs stay uncommitted, so every family of a round starts from
the round's HEAD, not from the catalog the family before it wrote back; the write-backs merge three-way into the
working tree (writeback.py), so the second family keeps the first's. A blocked family stops with its reason in its state file and the others go on; a person looks, fixes,
and sets its `step` back. Nothing is committed: the write-backs, library entries and tables are left in the working
tree.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import regress  # noqa: E402

from locomani import paths  # noqa: E402
from locomani.scene_discovery.predicates import parse  # noqa: E402

SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
SANDBOXES = Path(paths.WORKSPACE) / "code" / "rsi-scene"  # <LOCOMANI_WORKSPACE>/code/rsi-scene/<family>, as the runbook
SCENE = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"
FEAS = REPO / "outputs" / "scene_rsi_20261016" / "capability" / "feasibility"
OVERLAY = REPO / "outputs" / "scene_rsi_20261016" / "capability" / "stance" / "adt_apartment_mars.json"
CAPABILITY = REPO / "configs" / "scene_discovery" / "capability_mars_scalebfm.yaml"
TASK_SETS = {
    "round1": REPO / "outputs/scene_discovery/adt_apartment_mars/discovery/rsi_round1/tasks.jsonl",
    "round2": REPO / "outputs/scene_discovery/adt_apartment_mars/discovery/rsi_round2/tasks.jsonl",
    "templates_2830": FEAS / "templates_2830.jsonl",
    "templates": None,  # scene_feasibility.py's own templates over the stance overlay
    "demo1": REPO / "outputs/scene_discovery/adt_apartment/discovery/opus_20261005b/tasks.jsonl",
}
STEPS = ("sandbox", "lint", "loop", "sealed", "reflect", "writeback", "tests", "regress", "capability", "done")
CLAUDE_DIR = Path(shutil.which("claude") or "/usr/local/bin/claude").parent  # the roles' and reflect.py's claude
ENV = {"PYTHONPATH": "src", "TMPDIR": paths.TMP, "LOCOMANI_WORKSPACE": paths.WORKSPACE, "OMP_NUM_THREADS": "2",
       "NUMPY_MADVISE_HUGEPAGE": "0", "MUJOCO_GL": "egl", "PATH": f"/usr/local/bin:/usr/bin:/bin:{CLAUDE_DIR}"}  # fmt: skip
SERIAL = threading.Lock()  # steps 6-9: one family at a time
REFLECT = threading.Lock()  # reflect.py rebuilds the library's INDEX.md; two families' reflects raced (TODO 36)
SKILLS = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"  # the catalog writeback.py writes into
ROLLBACK_STEPS = ("tests", "regress", "capability")  # a block here puts the catalog back
# loop.py stop reasons that end a run normally (stop_reason, TODO 36 review 2026-10-22); any other is abnormal
NORMAL_STOP = re.compile(  # TODO 36.7: the "rejections in a row" stop is gone; "audit: ..." is abnormal
    r"the incumbent wins|gate calls \d+ >=|wall clock >=|model cost \d+ >=|manager: stop$"
)


class Blocked(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def sh(cmd: list[str], cwd: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        import os

        return subprocess.run(cmd, cwd=cwd, env={**os.environ, **ENV}, stdout=fh, stderr=subprocess.STDOUT).returncode


def last_stop(ledger: Path) -> str | None:
    """The reason of the ledger's last row when that row is a stop; None when the ledger ends otherwise."""
    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []
    return rows[-1].get("reason", "") if rows and rows[-1].get("step") == "stop" else None


def family_file(family: str) -> dict:
    return json.loads((HERE / "families" / f"{family}.json").read_text())


def kind_of(family: str) -> str:
    """capability_write.py's loop kind from the family's goals."""
    fam = family_file(family)
    rows = {
        json.loads(x)["id"]: json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()
    }
    names = {parse(g).name for t in fam["tasks"] if t in rows for g in rows[t]["goal"]}
    for name, kind in (("near", "near"), ("in", "into"), ("on", "place"), ("held", "held")):
        if name in names:
            return kind
    return "none"


class Family:
    def __init__(self, round_dir: Path, name: str):
        self.name, self.state_file = name, round_dir / "run" / f"{name}.json"
        self.sandbox = SANDBOXES / name.replace("_", "-")
        self.logs = round_dir / "run" / name
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else dict(step="sandbox")

    def save(self, **kw) -> None:
        self.state.update(kw, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=1))

    @property
    def hidden(self) -> Path:
        return self.sandbox / "outputs" / "scene_rsi_20261016_hidden" / self.name

    @property
    def loop_ledger(self) -> Path:
        return self.sandbox / "outputs" / "scene_rsi_20261016" / self.name / "loop" / "ledger.jsonl"

    def run_in_sandbox(self, script: str, *args: str, step: str) -> None:
        rc = sh([str(SIM_PY), f"scripts/rsi/{script}", *args], self.sandbox,
                self.logs / f"{step}.log")  # fmt: skip
        if rc:
            raise Blocked(f"{step}: {script} exited {rc} (see {self.logs / f'{step}.log'})")

    # -- steps ------------------------------------------------------------------------------------------------------
    def sandbox_(self) -> None:
        if not self.sandbox.exists():
            rc = sh(
                ["python3", str(HERE / "sandbox.py"), str(self.sandbox), self.name], REPO, self.logs / "sandbox.log"
            )
            if rc:
                raise Blocked(f"sandbox.py exited {rc}")

    def lint_(self) -> None:
        fam = family_file(self.name)
        rows = [json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()]
        with tempfile.TemporaryDirectory(dir=paths.TMP) as d:
            tasks = Path(d) / self.name / "tasks.jsonl"
            tasks.parent.mkdir()
            tasks.write_text("".join(json.dumps(r) + "\n" for r in rows if r["id"] in fam["tasks"]))
            rc = sh([str(SIM_PY), "scripts/scene_lint.py", "--scene", str(SCENE), "--no-rest", "--tasks", str(tasks),
                     "--out", str(self.logs / "lint")], REPO, self.logs / "lint.log")  # fmt: skip
        report = self.logs / "lint" / "lint.json"
        if rc not in (0, 1) or not report.exists():  # scene_lint.py exits 1 on errors, with its report written
            raise Blocked(f"lint: scene_lint.py exited {rc} (see {self.logs / 'lint.log'})")
        rep = json.loads(report.read_text())
        bad = rep["task_errors"]
        if bad:
            raise Blocked(f"lint: {bad}")

    def loop_(self) -> None:
        self.run_in_sandbox("loop.py", "run", self.name, step="loop")
        why = last_stop(self.loop_ledger)  # a 0 exit is not a normal end: loop.py also returns 0 after a role failure
        if why is None or not NORMAL_STOP.search(why):
            raise Blocked(f"loop: stopped abnormally ({why if why is not None else 'no stop row in the ledger'}); "
                          f"see {self.logs / 'loop.log'} and {self.loop_ledger}")  # fmt: skip
        self.save(loop_stop=why)

    def sealed_(self) -> None:
        self.run_in_sandbox("sealed.py", self.name, "--all", step="sealed")  # rc != 0 blocks
        summ = json.loads((self.hidden / "sealed" / "summary.json").read_text())
        self.save(sealed=dict(final=summ["final"], rate=summ["final_rate"], ci=summ.get("final_rate_ci"),
                              vs_seed=summ["final_vs_seed"]))  # fmt: skip

    def reflect_(self) -> None:
        with REFLECT:
            rc = sh(["python3", str(HERE / "reflect.py"), str(self.sandbox), self.name], REPO,
                    self.logs / "reflect.log")  # fmt: skip
        if rc:
            log(f"{self.name}: reflect.py exited {rc} (see {self.logs / 'reflect.log'}); no library entry")
        self.save(reflect=dict(rc=rc))

    @property
    def snapshot(self) -> Path:
        return self.logs / "skills_before_writeback"

    def restore(self) -> None:
        """The catalog's skills directory as it was before this family's write-back (TODO 36, review 2026-10-22)."""
        shutil.rmtree(SKILLS)
        shutil.copytree(self.snapshot, SKILLS)
        shutil.rmtree(self.snapshot)

    def unregister(self) -> None:
        """This family's registry row as it was before its regress step (TODO 36.9: a family blocked at regress or
        capability had its catalog put back but stayed registered, so every later family's regression re-ran it)."""
        prev = self.state.pop("registry_prev", None)
        if prev is None:
            return
        reg = regress.load()
        if prev["present"]:
            reg[self.name] = prev["row"]
        else:
            reg.pop(self.name, None)
        regress.REGISTRY.write_text(json.dumps(reg, indent=1))
        self.save()

    def writeback_(self) -> None:
        if self.snapshot.exists():  # an earlier attempt of this step: start again from before its write-back
            self.restore()
        shutil.copytree(SKILLS, self.snapshot)
        rc = sh(
            ["python3", str(HERE / "writeback.py"), str(self.sandbox), self.name], REPO, self.logs / "writeback.log"
        )
        report = self.hidden / "writeback.json"
        if rc not in (0, 1) or not report.exists():  # writeback.py exits 1 on conflicts, with its report written
            raise Blocked(f"writeback: writeback.py exited {rc} (see {self.logs / 'writeback.log'})")
        wb = json.loads(report.read_text())
        self.save(writeback={k: wb.get(k) for k in ("merged", "copied", "deleted", "folded", "unchanged", "written")})
        if rc:
            raise Blocked(f"writeback conflicts: {wb['conflicts']}")

    def tests_(self) -> None:
        rc = sh([str(SIM_PY), "-m", "pytest", "-q", "-p", "no:warnings", "tests/test_scene_task_runner.py",
                 "src/locomani/tasks/scene/skills"], REPO, self.logs / "tests.log")  # fmt: skip
        if rc:
            raise Blocked(f"tests fail under the written-back catalog (see {self.logs / 'tests.log'})")

    def regress_(self) -> None:
        reg = regress.load()
        final = self.state.get("sealed", {}).get("final", "seed")
        self.save(registry_prev=dict(present=self.name in reg, row=reg.get(self.name)))  # unregister() puts it back
        reg[self.name] = dict(sandbox=str(self.sandbox), kind=kind_of(self.name), pool=None, last=final)
        regress.REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        regress.REGISTRY.write_text(json.dumps(reg, indent=1))
        rc = sh([str(SIM_PY), str(HERE / "regress.py"), "run", f"after_{self.name}"], REPO, self.logs / "regress.log")
        report = regress.REGISTRY.parent / f"regress_after_{self.name}.json"
        if rc or not report.exists():  # regress.py exits 0 whatever it finds; non-zero is a crash
            raise Blocked(f"regress: regress.py exited {rc} (see {self.logs / 'regress.log'})")
        out = json.loads(report.read_text())
        self.save(regress=dict(total=out["total"], worse=out["worse"]))
        if out["worse"]:
            raise Blocked(f"regression: worse at p <= {regress.WORSE_P}: {out['worse']}")

    def capability_(self) -> None:
        rc = sh([str(SIM_PY), str(HERE / "carry_time.py")], REPO, self.logs / "capability.log")  # the new rounds' T_est
        if rc:
            raise Blocked(f"carry_time.py exited {rc}")
        rc = sh([str(SIM_PY), str(HERE / "capability_write.py"), *regress.capability_args()], REPO,
                self.logs / "capability.log")  # fmt: skip
        if rc:
            raise Blocked(f"capability_write.py exited {rc}")
        buckets = {}
        for name, tasks in TASK_SETS.items():
            out = FEAS / f"after_{self.name}_{name}"
            args = ["--tasks", str(tasks)] if tasks else []
            rc = sh([str(SIM_PY), "scripts/scene_feasibility.py", "--scene", str(SCENE / "scene_meta.json"),
                     "--capability", str(CAPABILITY), "--approach-edges", str(OVERLAY), *args, "--out", str(out)],
                    REPO, self.logs / "capability.log")  # fmt: skip
            if rc or not (out / "summary.json").exists():
                raise Blocked(
                    f"capability: scene_feasibility.py ({name}) exited {rc} (see {self.logs / 'capability.log'})"
                )
            buckets[name] = json.loads((out / "summary.json").read_text())["buckets_strict"]
        self.save(buckets=buckets)

    def advance(self, until: str) -> bool:
        """Run steps up to (not including) `until`; False when blocked."""
        while STEPS.index(self.state["step"]) < STEPS.index(until):
            step = self.state["step"]
            log(f"{self.name}: {step}")
            try:
                getattr(self, f"{step}_")()
            except Exception as e:  # TODO 36 (review 2026-10-22): a crash blocks this family, not the whole round
                why = str(e) if isinstance(e, Blocked) else repr(e)
                self.save(blocked=why)
                log(f"{self.name}: blocked at {step}: {why}")
                if step in ROLLBACK_STEPS and self.snapshot.exists():
                    self.restore()
                    self.unregister()
                    self.save(restored=f"catalog skills put back to before the write-back (blocked at {step}); set "
                                       "`step` back to writeback to try again")  # fmt: skip
                    log(f"{self.name}: catalog restored")
                return False
            self.save(step=STEPS[STEPS.index(step) + 1], blocked=None)
        return True


def round_lint(round_dir: Path) -> None:
    out = round_dir / "run" / "lint"
    tasks = round_dir / "discovery" / "tasks.jsonl"
    sh([str(SIM_PY), "scripts/scene_lint.py", "--scene", str(SCENE), "--tasks", str(tasks), "--out", str(out)], REPO,
       round_dir / "run" / "lint.log")  # fmt: skip
    rep = json.loads((out / "lint.json").read_text())
    if rep["scene_errors"]:
        raise SystemExit(f"scene errors before the round: {rep['scene_errors']}")
    log(f"round lint: {len(rep['task_errors'])} task errors, {len(rep['warnings'])} warnings ({out / 'lint.json'})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round_dir", type=Path)
    ap.add_argument("families", nargs="+")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--skip-round-lint", action="store_true")
    a = ap.parse_args(argv)
    if not a.skip_round_lint:
        round_lint(a.round_dir)
    fams = [Family(a.round_dir, f) for f in a.families]
    sem = threading.Semaphore(a.parallel)

    def one(f: Family) -> None:
        with sem:
            if not f.advance("writeback"):
                return
        with SERIAL:
            f.advance("done")

    threads = [threading.Thread(target=one, args=(f,), name=f.name) for f in fams]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for f in fams:
        log(f"{f.name}: {f.state['step']}" + (f" (blocked: {f.state['blocked']})" if f.state.get("blocked") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
