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
3. `loop`: loop.py run in the sandbox, to its end;
4. `sealed`: sealed.py --all in the sandbox;
5. `writeback`: writeback.py into the catalog (nothing when the final is the seed);
6. `tests`: the scene runner's and the catalog's own tests;
7. `regress`: the family enters regress.py's registry; every registered family's sealed pool re-runs under the
   catalog (tag `after_<family>`); a family worse at p <= 0.05 blocks;
8. `capability`: carry_time.py (the put-down's planned carry time against the route, for the creep rule), then
   capability_write.py from the registry, then every task set re-bucketed (capability/feasibility/after_<family>_<set>).

Up to `--parallel` families run steps 1-4 at once (the simulation slots in batch.py share the machine); steps 5-8
take one family at a time, in the order the loops end, and merge three-way, so the second family keeps the first's
write-back. A blocked family stops with its reason in its state file and the others go on; a person looks, fixes,
and sets its `step` back. Nothing is committed: the write-backs and tables are left in the working tree.
"""

from __future__ import annotations

import argparse
import json
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

from locomani.scene_discovery.predicates import parse  # noqa: E402

SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
SANDBOXES = Path("/workspace/code/rsi-scene")
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
STEPS = ("sandbox", "lint", "loop", "sealed", "writeback", "tests", "regress", "capability", "done")
ENV = {"PYTHONPATH": "src", "TMPDIR": "/workspace/tmp", "OMP_NUM_THREADS": "2", "NUMPY_MADVISE_HUGEPAGE": "0",
       "MUJOCO_GL": "egl", "PATH": "/usr/local/bin:/usr/bin:/bin"}  # fmt: skip
SERIAL = threading.Lock()  # steps 5-8: one family at a time


class Blocked(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def sh(cmd: list[str], cwd: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        import os

        return subprocess.run(cmd, cwd=cwd, env={**os.environ, **ENV}, stdout=fh, stderr=subprocess.STDOUT).returncode


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
        with tempfile.TemporaryDirectory(dir="/workspace/tmp") as d:
            tasks = Path(d) / self.name / "tasks.jsonl"
            tasks.parent.mkdir()
            tasks.write_text("".join(json.dumps(r) + "\n" for r in rows if r["id"] in fam["tasks"]))
            sh([str(SIM_PY), "scripts/scene_lint.py", "--scene", str(SCENE), "--no-rest", "--tasks", str(tasks),
                "--out", str(self.logs / "lint")], REPO, self.logs / "lint.log")  # fmt: skip
        rep = json.loads((self.logs / "lint" / "lint.json").read_text())
        bad = rep["task_errors"]
        if bad:
            raise Blocked(f"lint: {bad}")

    def loop_(self) -> None:
        self.run_in_sandbox("loop.py", "run", self.name, step="loop")

    def sealed_(self) -> None:
        self.run_in_sandbox("sealed.py", self.name, "--all", step="sealed")
        summ = json.loads((self.hidden / "sealed" / "summary.json").read_text())
        self.save(sealed=dict(final=summ["final"], rate=summ["final_rate"], ci=summ.get("final_rate_ci"),
                              vs_seed=summ["final_vs_seed"]))  # fmt: skip

    def writeback_(self) -> None:
        rc = sh(
            ["python3", str(HERE / "writeback.py"), str(self.sandbox), self.name], REPO, self.logs / "writeback.log"
        )
        wb = json.loads((self.hidden / "writeback.json").read_text())
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
        reg[self.name] = dict(sandbox=str(self.sandbox), kind=kind_of(self.name), pool=None, last=final)
        regress.REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        regress.REGISTRY.write_text(json.dumps(reg, indent=1))
        rc = sh([str(SIM_PY), str(HERE / "regress.py"), "run", f"after_{self.name}"], REPO, self.logs / "regress.log")
        out = json.loads((regress.REGISTRY.parent / f"regress_after_{self.name}.json").read_text())
        self.save(regress=dict(total=out["total"], worse=out["worse"]))
        if rc or out["worse"]:
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
            sh([str(SIM_PY), "scripts/scene_feasibility.py", "--scene", str(SCENE / "scene_meta.json"), "--capability",
                str(CAPABILITY), "--approach-edges", str(OVERLAY), *args, "--out", str(out)], REPO,
               self.logs / "capability.log")  # fmt: skip
            buckets[name] = json.loads((out / "summary.json").read_text())["buckets_strict"]
        self.save(buckets=buckets)

    def advance(self, until: str) -> bool:
        """Run steps up to (not including) `until`; False when blocked."""
        while STEPS.index(self.state["step"]) < STEPS.index(until):
            step = self.state["step"]
            log(f"{self.name}: {step}")
            try:
                getattr(self, f"{step}_")()
            except Blocked as e:
                self.save(blocked=str(e))
                log(f"{self.name}: blocked at {step}: {e}")
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
