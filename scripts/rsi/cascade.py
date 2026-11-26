"""From a candidate to the verdict: the incumbent, the check half, the gate call, the candidate pool, after the Engineer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "src")]
import audit as AU  # noqa: E402
import backend as BK  # noqa: E402
import batch as B  # noqa: E402
import drafts as DR  # noqa: E402
import ledger as LG  # noqa: E402
import roles as RL  # noqa: E402
import state as S  # noqa: E402
import verdict as V  # noqa: E402
from gate import GATE_RULE, candidate_hash, drop_limit, overrides  # noqa: E402


class View:
    """overlap.py: what a gate or check half running beside the roles sees instead of the hidden tree -- `root`, a private
    copy of what it reads and writes --, where its verdict files wait (`stash`), the work to do once the driver settles it
    (`post`) and its gate process (`proc`, to kill on an abort). Thread-local: the driver's own thread has none."""

    def __init__(self, root: Path, stash: Path):
        self.root, self.stash, self.post, self.proc = root, stash, None, None


_VIEW = threading.local()


def hidden() -> Path:
    """The hidden tree this thread reads: the private copy of an overlapped gate / check half, else S.HIDDEN."""
    v = getattr(_VIEW, "v", None)
    return v.root if v else S.HIDDEN


def gate_dir() -> Path:
    return hidden() / "gate"


def out_file(it_dir: Path, name: str) -> Path:
    """Where this call's verdict file `name` goes: the iteration directory, or -- overlapped -- the stash, which the
    driver copies there when it settles the call (a role of the next iteration must not read it before)."""
    v = getattr(_VIEW, "v", None)
    return (v.stash if v else it_dir) / name


POOL_KEEP = 2  # TODO 33.8 (D7, user 2026-10-22): rejected candidates kept for the Manager's `base` and the Composer


def accept_pool() -> Path:
    """The acceptance pool the gate judges on: the family's, or (scene line) the one `redraw_accept` drew for the parks."""
    return S.H.accept_pool() or hidden() / "pool_accept"


def incumbent() -> dict:
    """The gate's incumbent copied into the loop's tree (roles must not open the gate's files)."""
    inc = LG.jload(gate_dir() / "incumbent.json")
    src = Path(inc["dir"]) if inc else None
    if src is None:  # before the first gate call: the sandbox's own skills
        src = S.STATE / "seed_candidate"
        if not src.exists():
            shutil.copytree(S.SKILLS, src / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (src / "candidate.json").write_text('{"set": {}}')
    h = candidate_hash(src)
    dst = S.LOOP / "incumbents" / h
    if not dst.exists():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dict(dir=str(dst), hash=h, set=overrides(dst), since_gate_call=(inc or {}).get("since"))


def diag_incumbent(inc: dict, it_dir: Path) -> Path:
    d = S.LOOP / "diag" / f"incumbent_{inc['hash']}"
    side = S.LOOP / "diag" / f"candidate_{inc['hash']}"
    if not d.exists() and side.exists():
        side.rename(d)
    B.run_jobs([B.Job(Path(inc["dir"]) / "skills", d, inc["set"], None, True)], S.FAMILY, S.LOOP / "pool_diag", S.procs_low(),
               it_dir / "prober.log", S.GPUS)  # fmt: skip
    return d


def check_pool() -> Path:
    """check_half: the diagnostic starts the roles do not see ([check-half]), in the hidden tree."""
    return hidden() / "pool_check"


def check_rounds(kind: str, h: str) -> Path:
    """check_half: rounds on the check pool, `incumbent_<hash>` or `candidate_<hash>` as under LOOP/diag."""
    return hidden() / "check" / f"{kind}_{h}"


def check_labels_of(labels: list[str]) -> list[str]:
    """Half (rounded down) of each task's starts, ordered by a hash of the family and the label: the same split every
    time, and no start is picked for what its rounds showed."""
    by: dict[str, list[str]] = {}
    for x in labels:
        by.setdefault(LG.task_of(x), []).append(x)
    out: list[str] = []
    for xs in by.values():
        xs = sorted(xs, key=lambda x: hashlib.sha1(f"{S.FAMILY}:{x}".encode()).hexdigest())
        out += xs[: len(xs) // 2]
    return sorted(out)


def split_check_half() -> dict | None:
    """check_half, once per loop at the start of a run: half of each task's starts move from LOOP/pool_diag to the
    hidden pool_check, and every file of those starts under LOOP/diag/<dir> (rounds, _old_ rounds, traces) to
    HIDDEN/check/<dir>, so an incumbent's or a pooled candidate's check rounds are there already. Rounds of those
    starts in earlier iteration directories stay where they are: the half was visible until now. Returns the ledger
    row, or None when the family has no check half or the pool is split already."""
    if not S.cap("check_half") or check_pool().exists():
        return None
    pool = S.LOOP / "pool_diag"
    chk = check_labels_of(B.pool_labels(pool))
    dst = check_pool()
    dst.mkdir(parents=True)
    for x in chk:
        shutil.move(str(pool / f"{x}.json"), str(dst / f"{x}.json"))
    names = set(chk)
    moved = 0
    for d in sorted((S.LOOP / "diag").glob("*")):
        if not d.is_dir():
            continue
        for f in list(d.iterdir()):
            stem = f.name[5:] if f.name.startswith("_old_") else f.name
            if stem.split(".", 1)[0] in names:
                out = S.HIDDEN / "check" / d.name
                out.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(out / f.name))
                moved += 1
    row = dict(step="check_split", n_build=len(B.pool_labels(pool)), n_check=len(chk), files_moved=moved)
    LG.jdump(dst / "_split.json", dict(row, labels=chk, t=LG.now()))
    LG.ledger(row)
    print(f"check half: {len(chk)} of {len(chk) + row['n_build']} diagnostic starts kept from the roles")
    return row


def check_candidate(n: int, it_dir: Path, cand: Path) -> dict:
    """check_half: the candidate and the incumbent on the check pool, start by start. The incumbent's rounds run once
    per incumbent (an accepted candidate's are taken over); the per-start lists stay in the hidden tree, the ledger
    row and the iteration directory get counts only."""
    t_start = time.time()
    inc = incumbent()
    base = check_rounds("incumbent", inc["hash"])
    side = check_rounds("candidate", inc["hash"])
    if not base.exists() and side.exists():
        side.rename(base)
    h = candidate_hash(cand)
    d = check_rounds("candidate", h)
    log = hidden() / "check" / "run.log"
    B.run_jobs([B.Job(Path(inc["dir"]) / "skills", base, inc["set"], None, True),
                B.Job(cand / "skills", d, overrides(cand), None, True)], S.FAMILY, check_pool(), S.PROCS, log, S.GPUS)  # fmt: skip
    f = flips(base, d)
    t = V.counts(
        B.outcomes(d), B.outcomes(base)
    )  # TODO 54: verdict.py, the same function as the gate's and the sealed set's
    v = V.decide(
        t,
        "check",
        dict(
            check_min_net=int(S.cap("check_min_net", 1)),
            check_alpha=S.cap("check_alpha", None),
            alpha_task=S.cap("alpha_task_check", None),
        ),
    )
    out = dict(n=f["n"], fixed=len(f["fixed"]), broken=len(f["broken"]), net=len(f["fixed"]) - len(f["broken"]),
               ok=f["ok"], incumbent_ok=f["incumbent_ok"], sent=v["decision"], reasons=v["reasons"],
               task_regress={t_: x["net"] for t_, x in v["per_task"].items() if x["regress"]},
               seconds=round(time.time() - t_start))  # fmt: skip
    with (hidden() / "check" / "calls.jsonl").open("a") as fh:
        fh.write(
            json.dumps(dict(out, t=LG.now(), iter=n, hash=h, fixed_starts=f["fixed"], broken_starts=f["broken"])) + "\n"
        )
    return out


def stage1_passed(gate_dir: Path, h: str, t0: float) -> bool:
    """stage1_first: gate.py wrote this call's stage-1 file (its work directory is <call>_<hash>)."""
    return any(p.stat().st_mtime >= t0 - 1 for p in gate_dir.glob(f"*_{h}/stage1_passed.json"))


def put_skills(src: Path) -> None:
    """The sandbox's skills package := `src` (the incumbent's snapshot)."""
    if be := BK.of(S.FAMILY):  # a second backend refuses to write inside its seed package
        be.guard_write(S.FAMILY, S.SKILLS)
    shutil.rmtree(S.SKILLS)
    shutil.copytree(src, S.SKILLS, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def changed_files() -> list[str]:
    r = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=S.REPO, capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        f = line[3:].split(" -> ")[-1].strip('"')
        if not f.startswith(("outputs/", "__pycache__")) and "__pycache__" not in f:
            out.append(f)
    return sorted(out)


def snapshot(n: int) -> None:
    """The loop's mutable state as iteration n starts, so replay.py can restart a copy of the loop there (TODO 32.5):
    the parks, the Manager's focus, the gap files, meta, the ledgers' lengths and the gate's incumbent. In the hidden
    tree: the gate's incumbent names hidden paths, and the roles must not be led to them."""
    d = S.HIDDEN / "state_before" / f"iter_{n:02d}"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("parked.json", "focus.json", "gap_candidates.json", "gaps.jsonl", "meta.json", "interventions.jsonl"):
        if (S.STATE / f).exists():
            shutil.copy2(S.STATE / f, d / f)
    if (S.GATE_DIR / "incumbent.json").exists():
        shutil.copy2(S.GATE_DIR / "incumbent.json", d / "gate_incumbent.json")
    n_lines = lambda f: len(f.read_text().splitlines()) if f.exists() else 0  # noqa: E731
    LG.jdump(
        d / "lengths.json",
        dict(ledger=n_lines(S.STATE / "ledger.jsonl"), gate_ledger=n_lines(S.GATE_DIR / "ledger.jsonl")),
    )


def base_of(n: int, dec: dict) -> dict | None:
    """The Manager's `base` (TODO 33.8): its candidate_pool entry, or None (no base given; an unknown hash is recorded
    and the Engineers start from the incumbent)."""
    h = dec.get("base")
    if not h:
        return None
    e = pool_entry(h)
    if e is None:
        LG.ledger(dict(step="base_unknown", iter=n, base=h, pool=[x["hash"] for x in pool_load()]))
    return e


def base_brief(e: dict | None) -> dict | None:
    """What an Engineer or the Composer reads of the base: the gate still compares the result with the incumbent."""
    if e is None:
        return None
    return dict(hash=e["hash"], dir=e["dir"], set=overrides(Path(e["dir"])), gate=e.get("gate"),
                older_rule=e.get("rule", 1) != GATE_RULE)  # fmt: skip


def engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub) -> str | None:
    """The Manager's decision carried out: the chosen hypothesis (or the reopened park) to the Engineer, then the
    gate (after_engineer)."""
    if S.CANDIDATES > 1 and dec["decision"] == "engineer":
        return DR.multi_engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub)
    reopen = dec.get("reopen_claim") if dec["decision"] == "reopen" else None
    park_ = LG.jload(S.STATE / "parked.json", {}).get(reopen) if reopen else None
    if reopen and park_ is None:
        LG.ledger(dict(step="reopen_test", iter=n, claim=reopen, n=0, ok=[], why={}, error="no such park"))
        return None
    chosen = next((h for h in hyps["hypotheses"] if h["id"] == dec.get("hypothesis")), None)
    bentry = base_of(n, dec)  # TODO 33.8: the working copy starts from a pool candidate instead of the incumbent
    if bentry:
        put_skills(Path(bentry["dir"]) / "skills")
    ctx = it_dir / "engineer_context.json"
    claims_of = S.STATE / f"iter_{int(park_['iter']):02d}" / "claims.json" if reopen else None
    tried = [r for r in LG.jload(claims_of, []) if r.get("id") == reopen] if reopen else []
    LG.jdump(ctx, LG.context(inc, dict(hypothesis=None if reopen else chosen, manager=dec,
                                 probe=[] if reopen else [p for p in pr if p.get("id") == dec.get("hypothesis")],
                                 park=dict(park_, id=reopen, checked=tried) if reopen else None,
                                 base=base_brief(bentry))))  # fmt: skip
    prompt = RL.fill(RL.role_template("engineer"), CONTEXT=ctx, ITER=it_dir, **common)
    start_set = overrides(Path(bentry["dir"])) if bentry else inc["set"]
    stub_e = dict(mechanism="stub", files_changed=[], set={**start_set, **(chosen or {}).get("delta", {}).get("set", {})},
                  tests="none", scope=dict(when="stub", starts=["all"]))  # fmt: skip
    eng, acct = RL.claude(
        "engineer", prompt, RL.ENGINEER_SCHEMA, it_dir, "NotebookEdit,Agent", stub_e if stub else None
    )
    return after_engineer(n, it_dir, inc, eng, acct, base_changes, stub, reopen)


def after_engineer(
    n: int, it_dir: Path, inc: dict, eng: dict, acct: dict, base_changes: list, stub: bool, reopen: str | None = None
) -> str | None:
    rel = S.SKILLS.relative_to(S.REPO).as_posix() + "/"
    now_changed = changed_files()
    changed = [f for f in now_changed if f not in base_changes or f.startswith(rel)]
    stray = [f for f in now_changed if f not in base_changes and not f.startswith(rel)]
    leaks = AU.transcript_leaks(acct["session"])
    LG.ledger(dict(step="engineer", iter=n, changed=changed, stray=stray, mechanism=eng["mechanism"], set=eng["set"],
                scope=eng.get("scope"), void=bool(leaks), leaks=leaks, **acct))  # fmt: skip
    cand = it_dir / "candidate"
    if cand.exists():
        shutil.rmtree(cand)
    shutil.copytree(S.SKILLS, cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (cand / "candidate.json").write_text(json.dumps({"set": eng.get("set") or {}}, indent=1))
    patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(Path(inc["dir"]) / "skills"),
                            str(cand / "skills")], capture_output=True, text=True).stdout  # fmt: skip
    (it_dir / "code.patch").write_text(patch)
    for f in stray:  # back as they were
        if (
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", f], cwd=S.REPO, capture_output=True).returncode
            == 0
        ):
            subprocess.run(["git", "checkout", "--", f], cwd=S.REPO, capture_output=True)
        elif (S.REPO / f).exists():
            (S.REPO / f).unlink()
    put_skills(Path(inc["dir"]) / "skills")
    if leaks or stray:
        AU.hide_void(n, cand, it_dir / "code.patch")
    if leaks:
        AU.void_iteration(n, "engineer", leaks)
        return None
    if stray:
        LG.ledger(
            dict(
                step="gate",
                iter=n,
                accepted=False,
                void=True,
                reason=f"engineer changed files outside the skills package: {stray}",
            )
        )
        return None
    named = S.H.named_objects(cand / "skills")  # TODO 33.13 ([branch])
    if named:
        named_void(n, named, "the candidate")
        return None
    if reopen and not S.H.reopen_test(n, it_dir, cand, reopen):
        return None
    return send(n, it_dir, cand, stub, reopen=(reopen, S.H.reopen_won(it_dir, reopen)) if reopen else None)


def gate_candidate(n: int, it_dir: Path, cand: Path, stub: bool, reopen: tuple | None = None) -> str | None:
    """`cand` to the gate on the acceptance pool, its diagnostic rounds beside it (the next incumbent's, if accepted).
    `reopen` = (park id, starts won): the park lifts only if this gate call accepts (settle_reopen).
    TODO 36 (review 2026-10-22): the verdict is the ledger line gate.py wrote for THIS candidate; no new line, a line
    for another hash, or an exit code other than 0 / 1 is a gate error and stops the run (the last line used to be
    taken as the verdict, so a crashed gate recorded the previous call's).
    TODO 48.17: with check_half the candidate first runs on the hidden half of the diagnostic starts and goes to the gate
    only when it wins there by `check_min_net`; with stage1_first its diagnostic rounds start once gate.py has
    passed stage 1."""
    if S.cap("check_half") and not stub and check_pool().exists():
        chk = check_candidate(n, it_dir, cand)
        sent = chk["sent"]  # verdict.decide, kind "check": net >= check_min_net and no task's loss significant
        LG.jdump(out_file(it_dir, "check.json"), chk)
        LG.ledger(dict(step="check", iter=n, hash=candidate_hash(cand), **chk))
        if not sent:
            print(f"iteration {n}: the candidate wins {chk['fixed']} and loses {chk['broken']} of the {chk['n']} hidden "
                  f"diagnostic starts ({'; '.join(chk['reasons'])}): not sent to the gate")  # fmt: skip
            return None
    S.H.redraw_accept()
    t_gate = time.time()
    before = len(LG.gate_lines(gate_dir()))
    pool = accept_pool()
    dry = stub and not (
        BK.of(S.FAMILY) and os.environ.get("RSI_STUB_REAL_GATE") == "1"
    )  # a fake bed can run the real gate
    gate_d, env = gate_dir(), dict(os.environ, PYTHONPATH="src")
    gate_args = [str(S.SIM_PY), str(S.HERE / "gate.py"), S.FAMILY, "--gate", str(gate_d), "--pool", str(pool),
                 "--candidate", str(cand), "--gpus", S.GPUS]  # fmt: skip
    if dry:
        gate_args += ["--dry-run"]
    share = max(1, S.PROCS * 5 // 12)
    gate_args += ["--procs", str(S.PROCS - share if not stub else S.PROCS)]
    view = getattr(_VIEW, "v", None)
    if view is not None:  # overlapped: the gate reads and writes the private copy, whose seeds and practice it names
        env["RSI_HIDDEN_FAMILY"] = str(view.root)
    stop, side = threading.Event(), None
    pre = S.LOOP / "diag" / f"candidate_{candidate_hash(cand)}"

    def start_side() -> None:
        """The candidate's diagnostic rounds beside the gate (the next incumbent's, if accepted)."""
        nonlocal side
        if not stub:
            side = threading.Thread(target=B.run_jobs, args=([B.Job(cand / "skills", pre, overrides(cand), None, True)],
                                    S.FAMILY, S.LOOP / "pool_diag", share, it_dir / "prober.log", S.GPUS),
                                    kwargs=dict(stop=stop), daemon=True)  # fmt: skip
            side.start()

    first = bool(S.cap("stage1_first")) and not dry  # TODO 48.17: the side work waits for stage 1
    if not first:
        start_side()
    t0, proc = time.time(), None
    with out_file(it_dir, "gate.log").open("w") as glog:
        try:
            proc = subprocess.Popen(gate_args, cwd=S.REPO, stdout=glog, stderr=subprocess.STDOUT, text=True, env=env)
            if view is not None:
                view.proc = proc
            if first:
                while proc.poll() is None and not stage1_passed(gate_d, candidate_hash(cand), t0):
                    time.sleep(5.0)
                if proc.poll() is None:  # stage 1 passed and the gate runs on
                    start_side()
            rc = proc.wait()
        except BaseException:
            stop.set()
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            raise
    new = LG.gate_lines(gate_d)[before:]
    try:
        verdict = json.loads(new[-1]) if new else None
    except ValueError:
        verdict = None
    if rc not in (0, 1) or verdict is None or verdict.get("hash") != candidate_hash(cand):
        if side is not None:
            stop.set()
            side.join()
        LG.ledger(dict(step="gate_error", iter=n, rc=rc, log=str(it_dir / "gate.log"),
                    new_lines=len(new)))  # fmt: skip
        return f"gate error at iteration {n} (see gate.log)"
    LG.jdump(out_file(it_dir, "gate.json"), verdict)
    void = bool(dry or verdict.get("void"))  # a void verdict (incomplete pairing) is no rejection and no gate call
    accepted = bool(verdict["accepted"]) and not void
    if side is not None:
        if not accepted:
            stop.set()
        side.join()
    rounds = gate_d / f"{verdict.get('call', 0):02d}_{verdict['hash']}" / "candidate"
    LG.ledger(dict(step="gate", iter=n, accepted=accepted, reason=verdict["reason"], hash=verdict["hash"],
                result=verdict.get("result"), stage1=verdict.get("stage1"), unparked=S.H.unparked_view(rounds),
                pool=str(accept_pool()), void=void, rule=verdict.get("rule", 1),
                seconds=round(time.time() - t_gate)))  # fmt: skip

    def after() -> None:  # in the driver's thread: the pool lives in the loop's visible tree
        if not stub:
            pool_update(n, cand, verdict, accepted)
        if reopen:
            S.H.settle_reopen(n, reopen[0], reopen[1], accepted)

    if view is not None:
        view.post = after
    else:
        after()
    return None


def send(n: int, it_dir: Path, cand: Path, stub: bool, reopen: tuple | None = None) -> str | None:
    """The candidate to the check half and the gate: in this thread, or -- when the run overlaps them with the next
    iteration (caps overlap, overlap.py) -- in the background, the result written when the driver settles it."""
    ov = S.OVERLAP
    if ov is not None and reopen is None and ov.on(stub):
        return ov.send(n, it_dir, cand, stub)
    return gate_candidate(n, it_dir, cand, stub, reopen=reopen)


def revert_strays(base_changes: list) -> list[str]:
    """Files a role changed outside the skills working copies, put back as they were; returns them."""
    stray = [f for f in changed_files() if f not in base_changes]
    for f in stray:
        if (
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", f], cwd=S.REPO, capture_output=True).returncode
            == 0
        ):
            subprocess.run(["git", "checkout", "--", f], cwd=S.REPO, capture_output=True)
        elif (S.REPO / f).exists():
            (S.REPO / f).unlink()
    return stray


def flips(inc_dir: Path, cand_dir: Path) -> dict:
    """A candidate's diagnostic rounds against the incumbent's, start by start."""
    a, b = B.outcomes(inc_dir), B.outcomes(cand_dir)
    both = sorted(set(a) & set(b))
    return dict(ok=sum(bool(b[x]["ok"]) for x in both), incumbent_ok=sum(bool(a[x]["ok"]) for x in both), n=len(both),
                fixed=[x for x in both if b[x]["ok"] and not a[x]["ok"]],
                broken=[x for x in both if a[x]["ok"] and not b[x]["ok"]])  # fmt: skip


def pool_path() -> Path:
    return S.STATE / "candidates_pool.json"


def pool_load() -> list[dict]:
    return LG.jload(pool_path(), []) or []


def pool_ok(r: dict | None) -> bool:
    """A rejected full-pool tally the pool keeps: more wins than losses, falls and drops within the gate's limits."""
    if not r or "b" not in r:
        return False
    falls_ok = r["candidate_fell"] <= r["incumbent_fell"] + 1
    return r["b"] > r["c"] and falls_ok and r["candidate_dropped"] <= drop_limit(r)


def iter_of(path: str) -> int | None:
    """The iteration a candidate directory belongs to (loop/iter_NN/...)."""
    for part in Path(path).parts:
        if part.startswith("iter_") and part[5:].isdigit():
            return int(part[5:])
    return None


GATE_KEYS = ("n", "candidate_ok", "incumbent_ok", "b", "c", "p_one_sided", "candidate_fell", "incumbent_fell",
             "candidate_dropped", "incumbent_dropped")  # fmt: skip


def pool_add(src: Path, verdict: dict, n: int | None) -> dict:
    """`src` (skills + candidate.json, as the gate judged it) copied to STATE/pool/<hash>; its pool entry."""
    h = verdict["hash"]
    d = S.STATE / "pool" / h
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    r = verdict["result"]
    return dict(hash=h, iter=n, call=verdict.get("call"), rule=verdict.get("rule", 1), dir=str(d),
                gate={k: r[k] for k in GATE_KEYS if k in r}, reason=verdict.get("reason"))  # fmt: skip


def pool_rank(es: list[dict]) -> list[dict]:
    """The best POOL_KEEP by b - c, the most recent gate call first on ties."""
    return sorted(es, key=lambda e: (-(e["gate"]["b"] - e["gate"]["c"]), -(e.get("call") or 0)))[:POOL_KEEP]


def pool_update(n: int, cand: Path, verdict: dict, accepted: bool) -> None:
    """After a gate call: an accepted candidate clears the pool (the new incumbent's rejects start a new one; the old
    entries stay in the ledger and under STATE/pool); a rejected one with b > c and no fall or drop excess joins it."""
    if accepted:
        old = pool_load()
        LG.jdump(pool_path(), [])
        if old:
            LG.ledger(dict(step="pool", iter=n, cleared=[e["hash"] for e in old], reason="the incumbent changed"))
        return
    if verdict.get("void") or not pool_ok(verdict.get("result")):
        return
    kept = pool_rank([e for e in pool_load() if e["hash"] != verdict["hash"]] + [pool_add(cand, verdict, n)])
    LG.jdump(pool_path(), kept)
    LG.ledger(dict(step="pool", iter=n, added=verdict["hash"], kept=[e["hash"] for e in kept]))


def rebuild_pool() -> list[dict]:
    """The pool from the gate's ledger, for state from before the pool existed (a replay.py setup of an older run): the
    rejected calls since the present incumbent was accepted (incumbent.json `since`), copied from the gate's own
    snapshots."""
    inc = LG.jload(S.GATE_DIR / "incumbent.json") or {}
    since = int(inc.get("since") or 0)
    found = []
    for line in LG.gate_lines():
        v = json.loads(line)
        if v.get("dry_run") or v.get("void") or v.get("accepted") or int(v.get("call") or 0) <= since:
            continue
        src = S.GATE_DIR / f"{int(v['call']):02d}_{v['hash']}" / "cand"
        if pool_ok(v.get("result")) and src.exists():
            found.append(pool_add(src, v, iter_of(v.get("candidate", ""))))
    kept = pool_rank(found)
    LG.jdump(pool_path(), kept)
    if kept:
        LG.ledger(dict(step="pool", rebuilt=[e["hash"] for e in kept], since_gate_call=since))
    return kept


def pool_entry(h: str | None) -> dict | None:
    return next((e for e in pool_load() if h and e["hash"] == h), None)


def pool_view(inc: dict, diag_inc: Path | None) -> list[dict]:
    """The pool as a role reads it: per entry the gate's tally, whether an older gate rule rejected it (`older_rule`:
    it may be judged again as it stands), its directory and overrides, its diff against the incumbent, and its flips
    against the incumbent's diagnostic rounds where it ran on them."""
    out = []
    for e in pool_load():
        d = Path(e["dir"])
        patch = S.STATE / "pool" / f"{e['hash']}.patch"
        patch.write_text(subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc",
                                         str(Path(inc["dir"]) / "skills"), str(d / "skills")],
                                        capture_output=True, text=True).stdout)  # fmt: skip
        v = dict(e, older_rule=e.get("rule", 1) != GATE_RULE, set=overrides(d), patch=str(patch))
        diag = S.LOOP / "diag" / f"candidate_{e['hash']}"
        if diag_inc is not None and diag.exists():
            v["diag"] = dict(flips(diag_inc, diag), rounds=str(diag))
        out.append(v)
    return out


def named_void(n: int, named: list[str], what: str) -> None:
    LG.ledger(dict(step="gate", iter=n, accepted=False, void=True, named_objects=named,
                reason=f"{what} branches on object names ([branch]): {', '.join(named[:6])}"))  # fmt: skip
