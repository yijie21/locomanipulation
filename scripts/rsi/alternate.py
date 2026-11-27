"""Mode 3 of the scene RSI inner loop (user 2026-11-18): two families take turns, one inner-loop iteration each, and
share one knowledge library and one skills package. Modes 1 and 2 are the loop as it is (`sandbox.py --fresh`, then
`loop.py run`, then `knowledge.py build`).

    python3 scripts/rsi/alternate.py init <run dir> <family A> <family B> [--ref HEAD] [--seed DIR]
    python3 scripts/rsi/alternate.py run <run dir> [--iters 5] [--max-hours 20] [--stop-rate 0.95] [--loop-args "..."]
    python3 scripts/rsi/alternate.py status <run dir>
    python3 scripts/rsi/alternate.py publish <run dir> [--out scripts/rsi/knowledge]

`init` builds two sandboxes, `<run>/a` and `<run>/b` (sandbox.py --fresh: no earlier knowledge, no experience
library; both start from the repository's skills package at `--ref`, or from the committed skills package directory
`--seed`, e.g. scripts/rsi/seeds/gso_picked_start, the package gso-alt-a started from), the shared library `<run>/knowledge` (empty) and
the merge base `<run>/sync/base/skills`. Each sandbox's family file gets `shares_skills_with` (the other family, its
kind and skill file), which loop.py turns into a line of the Reviewer's and the Engineer's prompts.

`run` does `--iters` iterations; one iteration is a half of each family, A first. One half of family F:
1. loop: the shared library (knowledge.jsonl, sync.jsonl, INDEX.md) is copied into F's sandbox and committed there,
   then one inner-loop iteration (`loop.py run F --max-iters 1`: the diagnostic pool, Reviewer, Prober, Manager,
   Engineers, Composer, gate). Before its Reviewer the loop writes KNOWLEDGE.md, the whole library marked with F's
   incumbent's rounds (knowledge.write_hits).
2. build: `knowledge.build` on F's whole ledger so far: F's entries in the shared library are replaced, tagged by
   F's kind of task (knowledge.tag_of: single / bimanual), and F's citations of entries are recounted into
   usage.jsonl (knowledge.usage_rows), which sets each entry's standing (tested, candidate, doubtful).
3. sync: F's incumbent (its parameter overrides folded into the constants, writeback.fold) is merged three-way into
   the other family G's incumbent (`git merge-file`; the base is the version of the last accepted sync, at first the
   common start). Where both changed the same lines the sync fails (rules.md 40, version 28, user 2026-11-23: until
   then G's side was kept silently and F's work there lost): nothing is merged, the conflict is recorded in
   `sync.jsonl` for both families' Engineers to see, the base stays and the next sync tries again. Otherwise the
   merged package's own tests (skills/test_*.py, rules.md 41) run in G's sandbox; a failure is a failed sync too.
   Then the merge goes to G's gate as a candidate under the non-inferior rule (`loop.py sync G`, gate.py
   --non-inferior: kept iff it loses no more of G's acceptance starts than it wins); the verdict is a gate row of G's
   ledger (it counts as a gate call) and a line of the shared `sync.jsonl`, which KNOWLEDGE.md shows both families.
   Accepted, it is G's incumbent and the next merge base; rejected, the two packages differ there until a later merge
   carries it.
When it stops (user 2026-11-18), whichever comes first:
- `--iters` iterations are done (5: five halves of each family);
- the run has been running `--max-hours` (default 20), both families together: the time the driver ran, summed over
  its runs (the time it stood stopped does not count), checked before each phase, so a phase under way finishes;
- both families have reached the stop rate (`--stop-rate`, default 0.95 since 2026-11-24, 0.8 before: the incumbent wins that share of the
  family's acceptance pool). Each family's loop runs with `--alternating`: the stop rate is its only stop (no gate-call
  cap, no wall clock, no iteration cap, no Manager stop). A family that reached it stops -- its halves are skipped,
  nothing is merged into it, its entries stay -- and the other goes on.
A loop that stops abnormally (exit 3: a role call failed, the audit voided two iterations in a row, a crash) or a
gate error blocks the run: `run` exits and says why; fix it and run again. A build that keeps no entry or fails is
recorded and the run goes on. The state (`alternate.json`: iteration, turn, phase, hours) is saved after every phase,
so `run` resumes where it stopped.
Once the run has ended for any reason but a block, each family's sealed set runs in its sandbox, a then b
(`sealed.py <family> --all`, TODO 55.4, user 2026-11-27), with a `sealed` event each (the final version, its wins,
the exit code, `missing`) and `alternate.json`'s `sealed`, which `status` shows: the number the run is judged by
exists when it ends. `run` on an ended run fills in a family's sealed set that is not complete (sealed.py skips the
rounds already written), so `run` again on an ended run from before this (gso-alt-seed, whose set was run by hand)
records it without rerunning the rounds; `--loop-args` with `--stub` skips the sealed set.

`publish` puts the run's entries into the repository's library (replacing those of the run's two families); commit
it yourself.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE)]

import knowledge as KN  # noqa: E402
import sandbox as SB  # noqa: E402
import writeback as WB  # noqa: E402

VISIBLE = "outputs/scene_rsi_20261016"
HIDDEN = "outputs/scene_rsi_20261016_hidden"
SKILLS_REL = Path("src/locomani/tasks/scene/skills")
STATE_FILE = "alternate.json"
SIDES = ("a", "b")
SKILL_FILE = {"single": "pick.py", "bimanual": "bimanual.py"}  # user 2026-11-18: each family's own skill
KIND = {"single": "one-hand pick", "bimanual": "two-hand pick"}
SIM_PY = Path(os.environ.get("VENV_SIM_PY") or REPO / "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jdump(p: Path, obj) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def commit(sb: Path, paths: list[str], message: str) -> bool:
    """Stage `paths` in the sandbox's own git and commit when anything changed (the loop's changed_files() is `git
    status`: what the driver puts there must not look like an Engineer's stray edit). True when it committed."""
    git(sb, "add", "-A", "--", *paths)
    if git(sb, "diff", "--cached", "--quiet").returncode == 0:
        return False
    git(sb, "-c", "user.name=rsi-alternate", "-c", "user.email=rsi-alternate@localhost", "commit", "-q", "-m", message)
    return True


# -- the skills packages -------------------------------------------------------------------------------------------


def tree(root: Path) -> dict[str, Path]:
    """rel -> file under `root`, without caches."""
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def same(a: Path | None, b: Path | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return filecmp.cmp(a, b, shallow=False)


def merge_trees(base: Path, ours: Path, theirs: Path) -> dict:
    """Three-way merge of `theirs` into `ours` from `base`, file by file. Returns {texts: {rel: new text, or None to
    delete} -- the changes to make to `ours` --, merged, copied, deleted, conflicts}. A file only one side changed takes
    that side; one both changed alike stays; both changed differently: `git merge-file` line by line, and where the
    same lines differ `ours` is kept (--ours) and the hunks counted as a conflict; a file one side deleted and the other
    changed, or both added differently, keeps `ours` whole and is a conflict."""
    b, o, t = tree(base), tree(ours), tree(theirs)
    out = dict(texts={}, merged=[], copied=[], deleted=[], conflicts=[])
    for rel in sorted(set(b) | set(o) | set(t)):
        fb, fo, ft = b.get(rel), o.get(rel), t.get(rel)
        if same(ft, fb) or same(fo, ft):  # theirs did not change it, or both have it alike
            continue
        if same(fo, fb):  # only theirs changed it
            if ft is None:
                out["texts"][rel] = None
                out["deleted"].append(rel)
            else:
                out["texts"][rel] = ft.read_text()
                out["copied" if fb is None else "merged"].append(rel)
            continue
        if fb is None or fo is None or ft is None:
            what = "added differently on both sides" if fb is None else "deleted on one side, changed on the other"
            out["conflicts"].append(f"{rel}: {what}")
            continue
        r = subprocess.run(["git", "merge-file", "-p", str(fo), str(fb), str(ft)], capture_output=True, text=True)
        if r.returncode == 0:
            out["texts"][rel] = r.stdout
            out["merged"].append(rel)
            continue
        if r.returncode < 0 or r.returncode > 127:
            out["conflicts"].append(f"{rel}: git merge-file failed ({r.stderr.strip()[:120]})")
            continue
        kept = subprocess.run(["git", "merge-file", "-p", "--ours", str(fo), str(fb), str(ft)], capture_output=True,
                              text=True)  # fmt: skip
        out["texts"][rel] = kept.stdout
        out["merged"].append(rel)
        out["conflicts"].append(f"{rel}: {r.returncode} hunk(s) changed on both sides")
    return out


def fold_set(skills: Path, overrides: dict, family: str) -> tuple[dict[str, str], list[str]]:
    """A candidate's parameter overrides written into its skills' constants, as writeback.py folds a final incumbent's
    (`module.CONST` or `module.CONST.key...`; the module is the longest prefix that is a file). Returns ({rel: new
    text}, the overrides that could not be folded: a package-level name, a constant that is not a literal)."""
    texts: dict[str, str] = {}
    unfolded = []
    for key, value in sorted(overrides.items()):
        parts = key.split(".")
        if len(parts) < 2:
            unfolded.append(f"override {key}: a package-level parameter has no module")
            continue
        for i in range(len(parts) - 1, 0, -1):
            rel = "/".join(parts[:i]) + ".py"
            if rel in texts or (skills / rel).exists():
                break
        else:
            rel, i = "/".join(parts[:-1]) + ".py", len(parts) - 1
        const, entry = parts[i], parts[i + 1 :]
        src = texts.get(rel) or ((skills / rel).read_text() if (skills / rel).exists() else None)
        if src is None:
            new, found = None, False
        elif entry:
            new, found = WB.fold_entry(src, const, entry, value, family)
        else:
            new, found = WB.fold(src, const, value, family)
        if not found:
            unfolded.append(f"override {key}: no literal to fold in {rel}")
            continue
        texts[rel] = new
    return texts, unfolded


def incumbent_of(sb: Path, family: str) -> tuple[Path, dict]:
    """(skills directory, overrides) of the family's incumbent: the gate's (incumbent.json), else -- before its first
    gate call -- the sandbox's own skills package with no overrides."""
    inc = KN.jload(sb / HIDDEN / family / "gate" / "incumbent.json")
    if inc and inc.get("dir"):
        d = Path(inc["dir"])
        return d / "skills", (KN.jload(d / "candidate.json", {}) or {}).get("set") or {}
    return sb / SKILLS_REL, {}


def copy_skills(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def write_texts(root: Path, texts: dict[str, str | None]) -> None:
    for rel, text in texts.items():
        p = root / rel
        if text is None:
            p.unlink(missing_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)


def prepare_sync(run: Path, st: dict, i_from: int, i_to: int) -> dict:
    """The merge of family i_from's incumbent into family i_to's, as a gate candidate under
    <run>/sync/<NN>_<from>_to_<to>/: `theirs` (the sender's skills, overrides folded) and `cand` (the receiver's skills
    with the merge applied, the receiver's own overrides in candidate.json). `cand` is None when the merge changes
    nothing, or when it conflicts (`failed` says where: both sides changed the same lines since the last accepted
    sync, or a sender's override could not be folded)."""
    fams = st["families"]
    s_sb, r_sb = run / SIDES[i_from], run / SIDES[i_to]
    s_skills, s_set = incumbent_of(s_sb, fams[i_from])
    r_skills, r_set = incumbent_of(r_sb, fams[i_to])
    work = run / "sync" / f"{st['iter'] + 1:02d}_{fams[i_from]}_to_{fams[i_to]}"
    theirs = work / "theirs"
    copy_skills(s_skills, theirs)
    folded, unfolded = fold_set(theirs, s_set, fams[i_from])
    write_texts(theirs, folded)
    rep = merge_trees(run / "sync" / "base" / "skills", r_skills, theirs)
    rep["conflicts"] += unfolded
    if rep["conflicts"]:  # version 28: a conflict fails the sync instead of silently keeping the receiver's side
        report = {k: v for k, v in rep.items() if k != "texts"}
        return dict(work=str(work), theirs=str(theirs), cand=None, report=report,
                    failed="conflict: " + "; ".join(rep["conflicts"]))  # fmt: skip
    # a file whose merge is the receiver's own text changes nothing (every hunk a conflict kept the receiver's side):
    # without this the candidate was the receiver's incumbent and the gate refused it as such (2026-11-19 smoke run)
    r_files = tree(r_skills)
    same_as_ours = {rel for rel, t in rep["texts"].items()
                    if (t is None and rel not in r_files) or (t is not None and rel in r_files and r_files[rel].read_text() == t)}  # fmt: skip
    rep["texts"] = {rel: t for rel, t in rep["texts"].items() if rel not in same_as_ours}
    for k in ("merged", "copied", "deleted"):
        rep[k] = [rel for rel in rep[k] if rel not in same_as_ours]
    out = dict(work=str(work), theirs=str(theirs), cand=None, report={k: v for k, v in rep.items() if k != "texts"})
    if not rep["texts"]:
        return out
    cand = work / "cand"
    copy_skills(r_skills, cand / "skills")
    write_texts(cand / "skills", rep["texts"])
    (cand / "candidate.json").write_text(json.dumps({"set": r_set}) + "\n")
    out["cand"] = str(cand)
    return out


def advance_base(run: Path, theirs: Path, conflicts: list[str]) -> None:
    """After an accepted sync the merge base is the sender's version, except for a file the merge left in conflict:
    there the old base stays (since version 28 a conflict fails the sync, so an accepted one has none; kept for the
    records of older runs). The receiver kept its own lines there; with the sender's file as the base, the next sync
    the other way read the receiver's lines as its only change and overwrote the other side without a conflict (the
    2026-11-19 smoke run). With the old base the two sides stay in conflict there, each keeping its own, every sync
    recording it."""
    base = run / "sync" / "base" / "skills"
    conflicted = {c.split(":", 1)[0] for c in conflicts}
    nxt = base.parent / "skills.next"
    copy_skills(theirs, nxt)
    for rel in conflicted:
        if (base / rel).is_file():
            (nxt / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / rel, nxt / rel)
        elif (theirs / rel).is_file():
            (nxt / rel).unlink()
    shutil.rmtree(base)
    nxt.rename(base)


# -- the shared library ---------------------------------------------------------------------------------------------


def install_knowledge(run: Path, sb: Path) -> bool:
    """The shared library (knowledge.jsonl, sync.jsonl, usage.jsonl, INDEX.md; not the build logs, whose generality notes name objects)
    copied into the sandbox's scripts/rsi/knowledge and committed there; KNOWLEDGE.md is the loop's own (gitignored).
    True when it changed."""
    src, dst = run / "knowledge", sb / "scripts" / "rsi" / "knowledge"
    dst.mkdir(parents=True, exist_ok=True)
    for name in (KN.LIBRARY, KN.SYNCS, KN.USAGE, "INDEX.md"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
        else:
            (dst / name).unlink(missing_ok=True)
    return commit(sb, ["scripts/rsi/knowledge"], "alternate.py: the shared knowledge library")


def append_sync(run: Path, row: dict) -> None:
    with (run / "knowledge" / KN.SYNCS).open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# -- the processes (a test replaces this class) ---------------------------------------------------------------------


class Runner:
    def __init__(
        self, run: Path, loop_args: list[str] | None = None, stub_knowledge: bool = False, stop_rate: float = 0.95
    ):
        self.run, self.loop_args, self.stub, self.stop_rate = run, loop_args or [], stub_knowledge, stop_rate
        (run / "logs").mkdir(parents=True, exist_ok=True)

    def loop(self, sb: Path, family: str, tag: str) -> int:
        """One inner-loop iteration in the sandbox; the loop's exit code (0 normal, 3 an abnormal stop)."""
        with (self.run / "logs" / f"{tag}_loop.log").open("a") as log:
            cmd = [str(SIM_PY), "scripts/rsi/loop.py", "run", family, "--max-iters", "1", "--alternating",
                   "--stop-rate", str(self.stop_rate), *self.loop_args]  # fmt: skip
            return subprocess.run(cmd, cwd=sb, stdout=log, stderr=subprocess.STDOUT).returncode

    def build(self, sb: Path, family: str) -> dict:
        lib = self.run / "knowledge"
        return KN.build(sb, family, lib, stub=self.stub, ctx_root=self.run / "ctx" / family)

    def tests(self, sb: Path, family: str, cand: Path, tag: str) -> tuple[bool, str]:
        """The merged package's own tests (its skills/test_*.py, rules.md 41) run in the receiving sandbox before its
        gate: (passed, pytest's last line). The candidate directory is a package of its own (skills/__init__.py), so
        the tests' relative imports read the merged copy, not the sandbox's."""
        env = {**os.environ, "PYTHONPATH": str(sb / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        cmd = [
            str(SIM_PY),
            "-m",
            "pytest",
            "-q",
            "-x",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:warnings",
            str(cand / "skills"),
        ]
        r = subprocess.run(cmd, cwd=sb, env=env, capture_output=True, text=True)
        (self.run / "logs" / f"{tag}_sync_tests.log").write_text(r.stdout + r.stderr)
        lines = [x for x in r.stdout.splitlines() if x.strip()]
        return r.returncode == 0, (lines[-1] if lines else f"pytest exited {r.returncode}")[:200]

    def sync_gate(self, sb: Path, family: str, cand: Path, source: dict, tag: str) -> tuple[int, dict]:
        """`loop.py sync` in the receiving sandbox: (exit code, its verdict line)."""
        cmd = [str(SIM_PY), "scripts/rsi/loop.py", "sync", family, "--candidate", str(cand), "--source",
               json.dumps(source, ensure_ascii=False)]  # fmt: skip
        r = subprocess.run(cmd, cwd=sb, capture_output=True, text=True)
        (self.run / "logs" / f"{tag}_sync.log").write_text(r.stdout + r.stderr)
        lines = [x for x in r.stdout.splitlines() if x.startswith("{")]
        try:
            return r.returncode, json.loads(lines[-1]) if lines else {}
        except ValueError:
            return r.returncode, {}

    def sealed(self, sb: Path, family: str, tag: str) -> tuple[int, dict]:
        """`sealed.py <family> --all` in the family's sandbox (TODO 55.4): (exit code, its summary.json, or {}). The
        processes and GPUs are sealed.py's own defaults (RSI_PROCS, RSI_GPUS)."""
        with (self.run / "logs" / f"{tag}.log").open("a") as log:
            cmd = [str(SIM_PY), "scripts/rsi/sealed.py", family, "--all"]
            env = {**os.environ, "PYTHONPATH": str(sb / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
            rc = subprocess.run(cmd, cwd=sb, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        p = sb / HIDDEN / family / "sealed" / "summary.json"
        try:
            return rc, json.loads(p.read_text()) if p.exists() else {}
        except ValueError:
            return rc, {}


# -- init, the state machine ----------------------------------------------------------------------------------------


def loop_rows(sb: Path, family: str) -> list[dict]:
    return KN.jl(sb / VISIBLE / family / "loop" / "ledger.jsonl")


def share_note(fam_file: Path, other: str, own_tag: str, other_tag: str) -> None:
    fam = json.loads(fam_file.read_text())
    fam["shares_skills_with"] = dict(other=other, other_kind=KIND.get(other_tag, other_tag),
                                     other_skill=SKILL_FILE.get(other_tag, "?"), own_skill=SKILL_FILE.get(own_tag, "?"))  # fmt: skip
    fam_file.write_text(json.dumps(fam, indent=1, ensure_ascii=False) + "\n")


def init(run: Path, fam_a: str, fam_b: str, ref: str = "HEAD", seed: str | None = None) -> dict:
    """The run directory: two fresh sandboxes (their skills package the one at `ref`, or `seed`, sandbox.seed_skills),
    the empty shared library, the merge base, the state."""
    if run.exists():
        raise SystemExit(f"{run} exists")
    if fam_a == fam_b:
        raise SystemExit("two different families")
    run.mkdir(parents=True)
    sha = subprocess.run(["git", "rev-parse", ref], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    for side, fam in zip(SIDES, (fam_a, fam_b), strict=True):
        SB.build(run / side, fam, sha, fresh=True, seed=seed)
        if not (run / side / "scripts" / "rsi" / "knowledge.py").exists():
            raise SystemExit(f"{fam}: its sandbox has no scripts/rsi/knowledge.py (add it and scripts/rsi/knowledge "
                             "to the family file's sandbox.keep)")  # fmt: skip
    a_skills, b_skills = run / "a" / SKILLS_REL, run / "b" / SKILLS_REL
    if merge_trees(a_skills, a_skills, b_skills)["texts"]:
        raise SystemExit(f"{fam_a} and {fam_b} start from different skills packages (a family `sandbox.seed`?)")
    tags = [KN.tag_of(run / s, f) for s, f in zip(SIDES, (fam_a, fam_b), strict=True)]
    for i, (side, fam) in enumerate(zip(SIDES, (fam_a, fam_b), strict=True)):
        fam_file = run / side / "scripts" / "rsi" / "families" / f"{fam}.json"
        share_note(fam_file, (fam_a, fam_b)[1 - i], tags[i], tags[1 - i])
        commit(run / side, [str(fam_file.relative_to(run / side))], "alternate.py: shares_skills_with")
    (run / "knowledge").mkdir()
    copy_skills(a_skills, run / "sync" / "base" / "skills")
    st = dict(families=[fam_a, fam_b], tags=tags, ref=sha, seed=seed, created=now(), iter=0, turn=0, phase="loop",
              loop_mark=None, stopped={}, blocked=None, events=[])  # fmt: skip
    jdump(run / STATE_FILE, st)
    return st


def event(st: dict, **kw) -> None:
    st["events"].append(dict(t=now(), iter=st["iter"] + 1, **kw))


def step(run: Path, st: dict, runner) -> None:
    """One phase of the present half (module docstring), then the state saved."""
    i = st["turn"]
    fam, other = st["families"][i], st["families"][1 - i]
    sb = run / SIDES[i]
    tag = f"{st['iter'] + 1:02d}_{SIDES[i]}_{fam}"
    if st["phase"] == "loop":
        if fam in st["stopped"]:
            event(st, family=fam, step="skip", why=f"stopped: {st['stopped'][fam]}")
            st["phase"] = "next"
        else:
            install_knowledge(run, sb)
            if st["loop_mark"] is None:
                st["loop_mark"] = len(loop_rows(sb, fam))
                jdump(run / STATE_FILE, st)
            rc = runner.loop(sb, fam, tag)
            new = loop_rows(sb, fam)[st["loop_mark"] :]
            stop = next((r for r in reversed(new) if r.get("step") == "stop"), None)
            event(st, family=fam, step="loop", rc=rc, stop=(stop or {}).get("reason"))
            if rc not in (0,):
                st["blocked"] = (
                    f"{fam}'s loop exited {rc}: {(stop or {}).get('reason', 'see the log')} ({tag}_loop.log)"
                )
            else:
                if stop:
                    st["stopped"][fam] = stop["reason"]
                st["phase"] = "build"
    elif st["phase"] == "build":
        try:
            res = runner.build(sb, fam)
            event(st, family=fam, step="build", entries=res.get("entries"), cost_usd=res.get("cost_usd"),
                  tag=res.get("tag"))  # fmt: skip
        except (Exception, SystemExit) as e:  # a build that fails is recorded; the run goes on
            event(st, family=fam, step="build", error=f"{type(e).__name__}: {e}"[:300])
        st["phase"] = "sync"
    elif st["phase"] == "sync":
        if other in st["stopped"]:
            event(st, family=fam, step="sync", skipped=f"{other} stopped")
            append_sync(run, {"iter": st["iter"] + 1, "from": fam, "to": other, "skipped": f"{other} has stopped"})
            st["phase"] = "next"
        else:
            prep = prepare_sync(run, st, i, 1 - i)
            rep = prep["report"]
            row = {"iter": st["iter"] + 1, "from": fam, "to": other, **rep}
            tests_ok, tests_line = (True, "") if prep["cand"] is None else runner.tests(run / SIDES[1 - i], other,
                                                                                         Path(prep["cand"]), tag)  # fmt: skip
            if prep["cand"] is None or not tests_ok:
                row["skipped"] = prep.get("failed") or ("the merge changes nothing" if prep["cand"] is None
                                                        else f"the merged package fails its tests: {tests_line}")  # fmt: skip
                append_sync(run, row)
                event(st, family=fam, step="sync", skipped=row["skipped"])
                st["phase"] = "next"
            else:
                source = {"from": fam, **rep}
                rc, verdict = runner.sync_gate(run / SIDES[1 - i], other, Path(prep["cand"]), source, tag)
                if rc not in (0, 1) or not verdict or "error" in verdict:
                    st["blocked"] = (
                        f"{other}'s sync gate failed (exit {rc}): {verdict.get('error', 'see the log')} ({tag}_sync.log)"
                    )
                    event(st, family=fam, step="sync", error=st["blocked"])
                else:
                    row.update(accepted=bool(verdict.get("accepted")), reason=verdict.get("reason"),
                               result=verdict.get("result"), hash=verdict.get("hash"))  # fmt: skip
                    append_sync(run, row)
                    if row["accepted"]:
                        advance_base(run, Path(prep["theirs"]), rep["conflicts"])
                    event(st, family=fam, step="sync", accepted=row["accepted"], result=row["result"])
                    st["phase"] = "next"
    if st["phase"] == "next":
        st["phase"], st["loop_mark"] = "loop", None
        st["turn"] = 1 - i
        if st["turn"] == 0:
            st["iter"] += 1
    jdump(run / STATE_FILE, st)


def done(st: dict, iters: int, max_hours: float | None = None) -> str | None:
    """Why the run stops now, or None (module docstring)."""
    if st["blocked"]:
        return f"blocked: {st['blocked']}"
    if all(f in st["stopped"] for f in st["families"]):
        return "both families have stopped: " + "; ".join(f"{f}: {r}" for f, r in st["stopped"].items())
    if st["iter"] >= iters:
        return f"{st['iter']} iterations done"
    hours = st.get("active_s", 0.0) / 3600
    if max_hours is not None and hours >= max_hours:
        return f"time limit: the run has run {hours:.1f} h >= {max_hours:g} h"
    return None


def sealed_done(rec: dict | None) -> bool:
    """A family's sealed set is complete: sealed.py exited 0, named a final version and no version lacks rounds."""
    return bool(rec) and rec.get("rc") == 0 and rec.get("final") is not None and not rec.get("missing")


def collect(run: Path, st: dict, runner) -> None:
    """TODO 55.4 (user 2026-11-27, ported from main's A2): once the run has ended (not blocked), each family's sealed
    set runs in its sandbox, a then b (`runner.sealed`, sealed.py --all), and `st["sealed"][family]` with a `sealed`
    event keep the exit code, the final version with its wins and `missing`. A family whose set is complete
    (sealed_done) is not run again, so `run` on an ended run -- one from before this phase too (gso-alt-seed) --
    fills in what is missing (sealed.py skips the rounds already written). `--stub` among the loop's arguments skips
    it: a stub loop has no incumbent worth the rounds. The time it takes does not count as the run's hours."""
    recs = st.setdefault("sealed", {})
    stub = "--stub" in getattr(runner, "loop_args", ())
    for i, fam in enumerate(st["families"]):
        if sealed_done(recs.get(fam)):
            continue
        if stub:
            event(st, family=fam, step="sealed", skipped="stub")
            continue
        t0 = time.time()
        rc, summary = runner.sealed(run / SIDES[i], fam, f"sealed_{SIDES[i]}_{fam}")
        final = summary.get("final")
        v = (summary.get("versions") or {}).get(final) or {}
        recs[fam] = dict(rc=rc, final=final, ok=v.get("ok"), n=v.get("n"), missing=summary.get("missing") or {},
                         seconds=round(time.time() - t0, 1))  # fmt: skip
        event(st, family=fam, step="sealed", **recs[fam])
        jdump(run / STATE_FILE, st)


def run_all(run: Path, iters: int, runner, max_hours: float | None = None) -> str:
    st = json.loads((run / STATE_FILE).read_text())
    if st["blocked"]:  # a run again after the cause was fixed
        event(st, step="unblock", was=st["blocked"])
        st["blocked"] = None
        jdump(run / STATE_FILE, st)
    while (why := done(st, iters, max_hours)) is None:
        t0 = time.time()
        step(run, st, runner)
        st["active_s"] = st.get("active_s", 0.0) + time.time() - t0  # the hours both families ran, together
        jdump(run / STATE_FILE, st)
    if not why.startswith("blocked"):
        collect(run, st, runner)  # TODO 55.4: the sealed set of both families once the run has ended
    event(st, step="end", why=why)
    jdump(run / STATE_FILE, st)
    return why


class Lock:
    """One driver per run directory."""

    def __init__(self, run: Path):
        self.p = run / "alternate.lock"

    def __enter__(self):
        if self.p.exists():
            try:
                pid = int(self.p.read_text().strip())
                os.kill(pid, 0)
                raise SystemExit(f"another alternate.py (pid {pid}) is running on this directory")
            except (ValueError, ProcessLookupError):
                pass
        self.p.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.p.unlink(missing_ok=True)


def publish(run: Path, out: Path) -> dict:
    """The run's entries into the library at `out`, replacing those of the run's families; INDEX.md rebuilt."""
    st = json.loads((run / STATE_FILE).read_text())
    mine = KN.entries(run / "knowledge")
    fams = set(st["families"]) | {KN.source_family(e) for e in mine}
    KN.save_entries(out, [e for e in KN.entries(out) if KN.source_family(e) not in fams] + mine)
    runs = {(r.get("run"), r.get("family")) for r in KN.jl(run / "knowledge" / KN.USAGE)}  # the entries' citations
    for rn, fam in sorted(runs):
        KN.update_usage(
            out, rn, fam, [r for r in KN.jl(run / "knowledge" / KN.USAGE) if (r["run"], r["family"]) == (rn, fam)]
        )
    if (run / "knowledge" / "build").is_dir():
        (out / "build").mkdir(parents=True, exist_ok=True)
        for p in sorted((run / "knowledge" / "build").glob("*.md")):
            shutil.copy2(p, out / "build" / p.name)
    KN.index(out)
    return dict(out=str(out), entries=len(mine), families=sorted(fams))


def status(run: Path) -> str:
    st = json.loads((run / STATE_FILE).read_text())
    lines = [f"families {st['families']} (tags {st['tags']}); iteration {st['iter'] + 1}, "
             f"{st['families'][st['turn']]}'s half, phase {st['phase']}",
             f"stopped: {st['stopped'] or 'none'}; blocked: {st['blocked'] or 'no'}; "
             f"run {st.get('active_s', 0.0) / 3600:.1f} h",
             f"library: {len(KN.entries(run / 'knowledge'))} entries; syncs: {len(KN.jl(run / 'knowledge' / KN.SYNCS))}"]  # fmt: skip
    if st.get("sealed"):  # TODO 55.4
        lines.append("sealed: " + "; ".join(
            f"{f}: {r.get('final')} {r.get('ok')}/{r.get('n')} (exit {r.get('rc')}"
            + (f", missing {r['missing']}" if r.get("missing") else "") + ")" for f, r in st["sealed"].items()))  # fmt: skip
    lines += [json.dumps(e, ensure_ascii=False)[:300] for e in st["events"][-10:]]
    return "\n".join(lines)


def glue_loop_args(argv: list[str]) -> list[str]:
    """`--loop-args "--candidates 3"` as `--loop-args=--candidates 3`: argparse reads a value that starts with `-` as
    another option and refused the runbook's own example (the 2026-11-19 smoke run)."""
    out, k = [], 0
    while k < len(argv):
        if argv[k] == "--loop-args" and k + 1 < len(argv):
            out.append(f"--loop-args={argv[k + 1]}")
            k += 2
        else:
            out.append(argv[k])
            k += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("run", type=Path)
    i.add_argument("family_a")
    i.add_argument("family_b")
    i.add_argument("--ref", default="HEAD", help="the commit both sandboxes are exported from")
    i.add_argument("--seed", default=None, metavar="DIR", help="a committed skills package directory both sandboxes "
                   "start from instead of the main library (e.g. scripts/rsi/seeds/gso_picked_start, gso-alt-a's start)")  # fmt: skip
    r = sub.add_parser("run")
    r.add_argument("run", type=Path)
    r.add_argument("--iters", type=int, default=5, help="iterations, each a half of both families")
    r.add_argument(
        "--max-hours", type=float, default=20.0, help="hours both families may run together (default 20; 15 before)"
    )
    r.add_argument("--stop-rate", type=float, default=0.95, help="a family stops once its incumbent wins this share "
                   "of its acceptance pool (default 0.95; 0.8 until 2026-11-24)")  # fmt: skip
    r.add_argument("--loop-args", default="", help='more arguments for each `loop.py run`, e.g. "--candidates 3"')
    r.add_argument("--stub-knowledge", action="store_true", help="knowledge.build without the model (plumbing)")
    s = sub.add_parser("status")
    s.add_argument("run", type=Path)
    p = sub.add_parser("publish")
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=KN.OUT)
    a = ap.parse_args(glue_loop_args(sys.argv[1:] if argv is None else list(argv)))
    run = a.run.resolve()
    if a.cmd == "init":
        st = init(run, a.family_a, a.family_b, a.ref, a.seed)
        print(json.dumps({k: st[k] for k in ("families", "tags", "ref", "seed")}))
        return 0
    if not (run / STATE_FILE).exists():
        raise SystemExit(f"no {STATE_FILE} in {run}: alternate.py init first")
    if a.cmd == "status":
        print(status(run))
        return 0
    if a.cmd == "publish":
        print(json.dumps(publish(run, a.out.resolve())))
        return 0
    with Lock(run):
        runner = Runner(run, shlex.split(a.loop_args), a.stub_knowledge, a.stop_rate)
        why = run_all(run, a.iters, runner, a.max_hours)
    print(why)
    return 3 if why.startswith("blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
