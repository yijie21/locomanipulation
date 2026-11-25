"""TODO 54 phase 4 (review 2026-11-25, design.md "asynchronous queue" and flow.md F7, F13): the check half and the gate run
beside the next iteration, and the seed incumbent's gate rounds start with the run.

Most gate calls reject (15 of 33), and the sim idles while the Reviewer and the Engineers work, so once a candidate is
sent the next iteration starts at once, against the same incumbent. At most one candidate is in flight. The next
iteration meets the result at two places, both between role calls: after its Reviewer (`after_review`: nothing is
built yet, so an accept just means the iteration goes on from the new incumbent) and before its own candidate is
composed (`join`: it waits for the call, and an accept rebases its drafts). `stop_probe` and `drain` end the wait when
a stop rule fires or the run ends.

On accept (`rebase_drafts`): the new incumbent's diagnostic rounds run first (the candidate's own rounds ran beside the
gate and are renamed, `CS.diag_incumbent`); a draft whose target starts the new incumbent already wins is dropped; every
other draft is 3-way merged onto the new incumbent (merge.py, draft against the old incumbent), a conflict drops it, a
clean one is screened again on its scope; every rebase, drop and re-screen is a ledger row (`rebase`, `drop`, `rescreen`).

The walls hold as before. While a role runs the hidden tree has mode 000 (audit.hidden_locked), so the call reads and
writes a private copy made before (`PrivateTree`): a random dot directory outside the checkout, named by no file a role can
read, gone again once the driver has copied the call's new files back; the transcript audit voids a role that names it
(audit.PRIVATE_PREFIX). The call's ledger rows, verdict files and log wait in memory / the copy until the driver settles
it, so no role reads a verdict before the driver wrote it. Simulator work for the drafts uses one process while the call
runs (state.procs_low); the gate and the check half keep the rest."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "src")]
import audit as AU  # noqa: E402
import backend as BK  # noqa: E402
import batch as B  # noqa: E402
import cascade as CS  # noqa: E402
import drafts as DR  # noqa: E402
import gate as G  # noqa: E402
import ledger as LG  # noqa: E402
import merge as MG  # noqa: E402
import roles as RL  # noqa: E402
import state as S  # noqa: E402
import stop as ST  # noqa: E402
from gate import candidate_hash, overrides  # noqa: E402

TEXT_EXT = (".json", ".jsonl", ".txt", ".md")
WAIT_H_DEFAULT = (
    8.0  # hours a wait for the in-flight call may last before the driver kills the gate and calls it an error
)


# ----------------------------------------------------------------------------------------------- the private copy
def private_root() -> Path:
    """Where private copies are made: outside the checkout and the roles' scratch (RSI_PRIVATE_ROOT, else the system's)."""
    return Path(os.environ.get("RSI_PRIVATE_ROOT") or tempfile.gettempdir())


def rewrite_paths(root: Path, old: str, new: str) -> None:
    """Every `old` in the text files (TEXT_EXT) under `root` becomes `new`."""
    for f in root.rglob("*"):
        if f.suffix in TEXT_EXT and f.is_file() and not f.is_symlink():
            t = f.read_text(errors="surrogateescape")
            if old in t:
                f.write_text(t.replace(old, new), errors="surrogateescape")


def sig(f: Path) -> tuple[int, int]:
    st = f.stat()
    return st.st_size, st.st_mtime_ns


def tree_sigs(root: Path) -> dict[str, tuple[int, int]]:
    return {
        p.relative_to(root).as_posix(): sig(p) for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
    }


class PrivateTree:
    """What a gate or check half reads and writes, copied from the hidden tree into a dot directory with a random name:
    the gate directory's top files and the incumbent's directories, the acceptance and check pools, the check rounds of the
    present incumbent, the seeds and the practice files (RSI_HIDDEN_FAMILY). Paths inside are rewritten from the hidden tree
    to the copy; `copy_back` writes what the call added or changed back, rewritten the other way."""

    def __init__(self, kind: str, inc_hash: str | None):
        self.kind = kind
        self.root = private_root() / (AU.PRIVATE_PREFIX + secrets.token_hex(12))
        self.root.mkdir(mode=0o700, parents=True)
        self.src = S.HIDDEN
        (self.root / "_stash").mkdir()
        self.mark = S.HIDDEN / f"private_{kind}.txt"
        self.mark.write_text(str(self.root))
        for rel in self.wanted(inc_hash):
            a, b = self.src / rel, self.root / rel
            b.parent.mkdir(parents=True, exist_ok=True)
            if a.is_dir():
                shutil.copytree(a, b, symlinks=True)
            else:
                shutil.copy2(a, b)
        rewrite_paths(self.root, str(self.src), str(self.root))
        self.manifest = tree_sigs(self.root)
        os.environ["RSI_HIDDEN_FAMILY"] = str(self.root)

    def wanted(self, inc_hash: str | None) -> list[str]:
        out = ["pool_accept", "pool_check", "seeds.json", "practice", "check/calls.jsonl", "check/run.log"]
        out += [f"check/{k}_{inc_hash}" for k in ("incumbent", "candidate")] if inc_hash else []
        gate = S.HIDDEN / "gate"
        out += [f"gate/{f}" for f in ("ledger.jsonl", "incumbent.json")]
        inc = LG.jload(gate / "incumbent.json") or {}
        for key in ("dir", "rounds"):
            p = Path(inc.get(key) or "")
            if p.is_relative_to(gate):
                rel = p.relative_to(gate).parts
                if rel:  # the call's directory holds both `cand` and `candidate`; copy the whole of it
                    out.append("gate/" + rel[0])
        return [r for r in dict.fromkeys(out) if (S.HIDDEN / r).exists()]

    def copy_back(self) -> list[str]:
        """The call's new and changed files into the hidden tree (the driver, between role calls: the tree is unlocked),
        and the files it removed. Returns what was written."""
        now_, written = tree_sigs(self.root), []
        for rel, s_ in now_.items():
            if rel.startswith("_stash/") or self.manifest.get(rel) == s_:
                continue
            f, g = self.root / rel, self.src / rel
            g.parent.mkdir(parents=True, exist_ok=True)
            if f.suffix in TEXT_EXT:
                g.write_text(f.read_text(errors="surrogateescape").replace(str(self.root), str(self.src)),
                             errors="surrogateescape")  # fmt: skip
            else:
                shutil.copy2(f, g)
            written.append(rel)
        for rel in self.manifest:
            if rel not in now_ and (self.src / rel).exists():
                (self.src / rel).unlink()
        return written

    def back(self, text: str) -> str:
        return text.replace(str(self.root), str(self.src))

    def remove(self) -> None:
        os.environ.pop("RSI_HIDDEN_FAMILY", None)
        shutil.rmtree(self.root, ignore_errors=True)
        self.mark.unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------------- the in-flight call
class Pending:
    """A candidate in the check half or the gate: its thread, its private copy, its rows waiting for the ledger."""

    def __init__(self, n: int, it_dir: Path, cand: Path, stub: bool, tree: PrivateTree):
        self.n, self.it_dir, self.cand, self.stub, self.tree = n, it_dir, cand, stub, tree
        self.hash = candidate_hash(cand)
        self.view = CS.View(tree.root, tree.root / "_stash")
        self.rows: list[dict] = []
        self.done = threading.Event()
        self.thread: threading.Thread | None = None
        self.why: str | None = None
        self.error: str | None = None
        self.t0 = time.time()
        self.t_end = self.t0
        self.first_wait: float | None = None  # when the driver first stopped to wait for it
        self.timed_out = False


class Overlap:
    """The driver's side of the overlap: at most one `Pending`, and the seed's warm-up. `S.OVERLAP` holds it."""

    def __init__(self, meta: dict, stub: bool):
        self.meta, self.stub = meta, stub
        self.pending: Pending | None = None
        self.warm: tuple[PrivateTree, threading.Thread, dict] | None = None
        self.wait_h = float(os.environ.get("RSI_OVERLAP_WAIT_H") or S.cap("overlap_wait_h", WAIT_H_DEFAULT))

    # -- state
    def on(self, stub: bool) -> bool:
        """Benchmark families only (the scene line keeps its flow), K > 1 Engineers, the program as Manager and Composer;
        a stub run only with a real gate."""
        return bool(S.cap("overlap") and BK.of(S.FAMILY) and S.CANDIDATES > 1 and S.cap("manager", "llm") == "program"
                    and (not stub or os.environ.get("RSI_STUB_REAL_GATE") == "1"))  # fmt: skip

    def busy(self) -> bool:
        return self.pending is not None and not self.pending.done.is_set()

    def note(self) -> dict | None:
        """What a role's context says of the call in flight: that there is one, never what it found ([overlap])."""
        p = self.pending
        return None if p is None else dict(iter=p.n, hash=p.hash, note="judged beside this iteration; its verdict is "
                                           "not written yet. If it is accepted your draft is rebased onto the new "
                                           "incumbent by the driver, a conflicting one is dropped")  # fmt: skip

    # -- the seed incumbent's rounds on the acceptance pool, at run start
    def start_warmup(self) -> None:
        if not self.on(self.stub):
            return
        inc = CS.incumbent()  # the present incumbent: the seed on a fresh run
        tree = PrivateTree("warm", inc["hash"])
        box: dict = {}

        def job() -> None:
            t0 = time.time()
            CS._VIEW.v = CS.View(tree.root, tree.root / "_stash")
            try:
                gi = G.init_incumbent(tree.root / "gate", S.FAMILY)
                d = Path(gi["dir"])
                B.run_jobs([B.Job(d / "skills", Path(gi["rounds"]), overrides(d))], S.FAMILY, CS.accept_pool(),
                           max(1, S.PROCS // 4), tree.root / "gate" / "warm.log", S.GPUS, where="gate")  # fmt: skip
            except BaseException as e:  # noqa: BLE001 -- the first gate call runs what is missing
                box["error"] = f"{type(e).__name__}: {e}"[:300]
            box["seconds"] = round(time.time() - t0)

        th = threading.Thread(target=job, daemon=True)
        self.warm = (tree, th, box)
        th.start()

    def join_warmup(self) -> None:
        """Before the first call: wait for the warm-up (the gate needs those rounds, and two writers would repeat them),
        copy what it ran into the hidden tree, drop its copy."""
        if self.warm is None:
            return
        tree, th, box = self.warm
        self.warm = None
        th.join()
        written = tree.copy_back()
        tree.remove()
        LG.ledger(dict(step="overlap", kind="seed_rounds", files=len(written), **box))

    # -- send, poll, wait, settle
    def send(self, n: int, it_dir: Path, cand: Path, stub: bool) -> str | None:
        if self.pending is not None:  # a path that never joined (K = 1, a re-run): the call before it is settled first
            why = self.wait()
            if why:
                return why
        self.join_warmup()
        p = Pending(n, it_dir, cand, stub, PrivateTree("gate", CS.incumbent()["hash"]))
        self.pending = p

        def job() -> None:
            CS._VIEW.v, LG._SINK.rows = p.view, p.rows
            try:
                p.why = CS.gate_candidate(n, it_dir, cand, stub)
            except BaseException as e:  # noqa: BLE001 -- settled as a gate error
                p.error = f"{type(e).__name__}: {e}"[:400]
            finally:
                p.t_end = time.time()
                p.done.set()

        p.thread = threading.Thread(target=job, daemon=True)
        p.thread.start()
        LG.ledger(dict(step="overlap", kind="sent", iter=n, hash=p.hash))
        return None

    def poll(self) -> tuple[str | None, bool] | None:
        """Settle the call if it has finished (no waiting): (a reason to stop or None, whether it accepted); None when
        there is nothing to settle."""
        p = self.pending
        if p is None or not p.done.is_set():
            return None
        return self.settle(p)

    def wait(self) -> str | None:
        return self.wait_accept()[0]

    def wait_accept(self) -> tuple[str | None, bool]:
        p = self.pending
        if p is None:
            return None, False
        p.first_wait = p.first_wait or time.time()
        if not p.done.wait(self.wait_h * 3600):  # a hung gate: kill it, the thread then ends in a gate error
            LG.ledger(dict(step="overlap", kind="timeout", iter=p.n, hours=self.wait_h))
            if p.view.proc is not None and p.view.proc.poll() is None:
                p.view.proc.kill()
            p.done.wait(float(os.environ.get("RSI_OVERLAP_KILL_WAIT_S") or 60))
            if not p.done.is_set():  # it does not end: what it wrote is not trusted, nothing of it is copied back
                p.error, p.timed_out = f"timed out after {self.wait_h} h", True
                p.done.set()
        return self.settle(p)

    def settle(self, p: Pending) -> tuple[str | None, bool]:
        """The driver takes over a finished call: the private copy's new files back, its rows, verdict files and log into
        the loop's tree, the pool update; the rows of an abandoned call are kept too."""
        if p.thread is not None:
            p.thread.join(timeout=0.05 if p.timed_out else 30)
        self.pending = None
        waited = max(0.0, time.time() - p.first_wait) if p.first_wait else 0.0
        written = [] if p.timed_out else p.tree.copy_back()
        stash = p.view.stash
        rows = (
            [] if p.timed_out else [_deep(r, p.tree.back) for r in list(p.rows)]
        )  # the private copy's paths are the hidden tree's again
        for r in rows:
            r["overlapped"] = True
        LG.write_rows(rows)
        for f in [] if p.timed_out else sorted(stash.glob("*")):
            dst = p.it_dir / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(p.tree.back(f.read_text(errors="surrogateescape")), errors="surrogateescape")
        p.tree.remove()
        accepted = any(r.get("step") == "gate" and r.get("accepted") and not r.get("void") for r in rows)
        why = p.why
        if p.error:
            LG.ledger(dict(step="gate_error", iter=p.n, error=p.error, log=str(p.it_dir / "gate.log")))
            why = f"gate error at iteration {p.n}: {p.error}"
        elif p.view.post is not None:
            p.view.post()
        total = p.t_end - p.t0
        LG.ledger(dict(step="overlap", kind="settled", iter=p.n, hash=p.hash, accepted=accepted, files=len(written),
                       call_s=round(total), waited_s=round(waited), beside_s=round(max(0.0, total - waited)),
                       reason=why))  # fmt: skip
        return why, accepted

    # -- the stop rules and the end of the run
    def stop_probe(self) -> str | None:
        """ST.stop_reason as the loop asks it before an iteration, with a call in flight counted: a finished one is
        settled first; a rule that fires, or the gate-call budget one call from used up, waits for the call and asks again
        (a "no progress" count must not see an iteration whose check row is not written yet)."""
        r = self.poll()
        if r and r[0]:
            return r[0]
        reason = ST.stop_reason(self.meta)
        if self.pending is not None:
            gates = len(LG.run_gates(LG.entries(), self.meta)) + 1
            if reason or gates >= S.MAX_GATE_CALLS:
                why, _ = self.wait_accept()
                reason = why or ST.stop_reason(self.meta)
        return reason

    def drain(self) -> str | None:
        """The run ends: the call in flight finishes and is recorded, the warm-up copy is taken, before the wrap-up."""
        why = self.wait()
        self.join_warmup()
        return why

    def abort(self) -> None:
        """A crash or SIGTERM: nothing of an unfinished call is recorded (its rows were in memory); the copy goes."""
        p = self.pending
        if p is not None:
            if p.view.proc is not None and p.view.proc.poll() is None:
                p.view.proc.kill()
            p.tree.remove()
            self.pending = None
        if self.warm is not None:
            self.warm[0].remove()
            self.warm = None

    # -- the next iteration meets the result
    def after_review(self, n, it_dir, inc, base, hyps, cards, attribution, stub) -> tuple:
        """After the Reviewer, before any draft: a finished call is settled. Returns (why, inc, base, hyps, common, extra);
        why set ends the run; on an accept the iteration goes on from the new incumbent, its stale hypotheses dropped."""
        r = self.poll()
        if r is None:
            return None, inc, base, hyps, None, None
        why, accepted = r
        why = why or ST.stop_reason(self.meta)
        if why:
            LG.ledger(dict(step="overlap", kind="abandoned", iter=n, reason=why))
            return why, inc, base, hyps, None, None
        if not accepted:
            return None, inc, base, hyps, None, None
        old_base, inc = base, CS.incumbent()
        CS.put_skills(Path(inc["dir"]) / "skills")
        base = CS.diag_incumbent(inc, it_dir)  # diagnostics first
        keep, dropped = stale_hypotheses(hyps["hypotheses"], old_base, base)
        for h, why_ in dropped:
            LG.ledger(dict(step="drop", iter=n, kind="hypothesis", hypothesis=h, reason=why_, incumbent=inc["hash"]))
        LG.ledger(dict(step="rebase", iter=n, kind="iteration", incumbent=inc["hash"], hypotheses=len(keep),
                       dropped=[h for h, _ in dropped]))  # fmt: skip
        hyps = dict(hyps, hypotheses=keep)
        common, extra = RL.review_common(n, inc, base, cards, attribution)
        return None, inc, base, hyps, common, extra

    def join(self, n, it_dir, inc, src, ref, cands, labels, stub) -> tuple:
        """After the drafts were screened, before they are composed: wait for the call in flight. Returns (why, inc, src,
        ref, cands); why set ends the run (a gate error, a stop rule). On an accept every draft is rebased."""
        if self.pending is None:
            return None, inc, src, ref, cands
        why, accepted = self.wait_accept()
        why = why or ST.stop_reason(self.meta)
        if why:
            LG.ledger(dict(step="overlap", kind="abandoned", iter=n, reason=why))
            return why, inc, src, ref, cands
        if not accepted:
            return None, inc, src, ref, cands
        old, inc = inc, CS.incumbent()
        CS.put_skills(Path(inc["dir"]) / "skills")
        old_base = S.LOOP / "diag" / f"incumbent_{old['hash']}"
        new_base = CS.diag_incumbent(inc, it_dir)  # diagnostics first, then the drafts
        cands = rebase_drafts(n, it_dir, old, inc, old_base, new_base, cands, labels)
        return None, inc, Path(inc["dir"]), new_base, cands


def _deep(v, f):
    if isinstance(v, str):
        return f(v)
    if isinstance(v, list):
        return [_deep(x, f) for x in v]
    if isinstance(v, dict):
        return {k: _deep(x, f) for k, x in v.items()}
    return v


# ----------------------------------------------------------------------------------------------- rebase on accept
def stale_hypotheses(hyps: list[dict], old_base: Path, new_base: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    """Hypotheses whose target starts -- the starts it names, else those of its tasks -- the old incumbent failed and the
    new one now wins all of, are stale; a hypothesis with no failing target start stays (nothing shows it moot)."""
    old = {k: bool(v["ok"]) for k, v in B.outcomes(old_base).items()}
    new = {k: bool(v["ok"]) for k, v in B.outcomes(new_base).items()}
    keep, dropped = [], []
    for h in hyps:
        named = [x for x in h.get("starts") or [] if isinstance(x, str)]
        tasks = DR.hyp_tasks(h)
        target = [k for k in old if (k in named if named else LG.task_of(k) in tasks) and not old[k]]
        if target and all(new.get(k) for k in target):
            dropped.append((h["id"], f"the new incumbent wins all {len(target)} of its failing target starts"))
        else:
            keep.append(h)
    return keep, dropped


def rebase_drafts(n: int, it_dir: Path, old: dict, new: dict, old_base: Path, new_base: Path, cands: list[dict],
                  labels: dict) -> list[dict]:  # fmt: skip
    """Each draft (a package changed against the old incumbent) onto the new incumbent. Rows: `drop` (the new incumbent
    already wins its fixed starts; the rebase leaves nothing of it; the edits conflict), `rebase` (clean, with the new
    hash), `rescreen` (the rebased draft's scope run against the new incumbent). Returns the surviving drafts, rebased."""
    out, jobs = [], []
    new_ok = {k: bool(v["ok"]) for k, v in B.outcomes(new_base).items()}
    for c in cands:
        k = c["number"]
        fixed = CS.flips(old_base, Path(c["diag"]))["fixed"]
        if fixed and all(new_ok.get(x) for x in fixed):
            LG.ledger(dict(step="drop", iter=n, kind="draft", draft=k, hypothesis=c.get("hypothesis"), reason="stale",
                           detail=f"the new incumbent wins all {len(fixed)} starts the draft fixed"))  # fmt: skip
            continue
        w = it_dir / f"rebased_{k}"
        res = MG.merge_drafts(Path(old["dir"]) / "skills", overrides(Path(old["dir"])),
                              [(k, Path(c["dir"]) / "skills", overrides(Path(c["dir"])))], w / "skills",
                              onto=Path(new["dir"]) / "skills", onto_set=overrides(Path(new["dir"])))  # fmt: skip
        if res["conflicts"]:
            LG.ledger(dict(step="drop", iter=n, kind="draft", draft=k, hypothesis=c.get("hypothesis"), reason="conflict",
                           detail=res["conflicts"].get(k)))  # fmt: skip
            shutil.rmtree(w, ignore_errors=True)
            continue
        (w / "candidate.json").write_text(json.dumps({"set": res["set"]}, indent=1))
        h = candidate_hash(w)
        if h == new["hash"]:
            LG.ledger(dict(step="drop", iter=n, kind="draft", draft=k, hypothesis=c.get("hypothesis"),
                           reason="absorbed", detail="the new incumbent already holds the draft's edits"))  # fmt: skip
            continue
        LG.ledger(dict(step="rebase", iter=n, kind="draft", draft=k, hypothesis=c.get("hypothesis"), result="clean",
                       from_hash=c["hash"], hash=h, onto=new["hash"]))  # fmt: skip
        c2 = dict(c, dir=str(w), hash=h, diag=str(S.LOOP / "diag" / f"candidate_{h}"), rebased_from=c["hash"])
        patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(Path(new["dir"]) / "skills"),
                                str(w / "skills")], capture_output=True, text=True).stdout  # fmt: skip
        (it_dir / f"code_{k}_rebased.patch").write_text(patch)
        c2["patch"] = str(it_dir / f"code_{k}_rebased.patch")
        lab = labels.get(k)
        jobs.append(B.Job(w / "skills", Path(c2["diag"]), res["set"], tuple(lab) if lab else None, True))
        out.append(c2)
    if jobs:
        B.run_jobs(jobs, S.FAMILY, S.LOOP / "pool_diag", S.procs_low(), it_dir / "prober.log", S.GPUS)
    for c in out:
        LG.ledger(dict(step="rescreen", iter=n, draft=c["number"], hash=c["hash"], against=new["hash"],
                       starts=len(labels.get(c["number"]) or B.pool_labels(S.LOOP / "pool_diag"))))  # fmt: skip
    return out


def make(meta: dict, stub: bool) -> Overlap | None:
    """The run's overlap, or None when the family's caps do not ask for it."""
    ov = Overlap(meta, stub)
    return ov if ov.on(stub) else None


def cleanup_orphans() -> list[str]:
    """A driver that died with a private copy standing (a crash, SIGKILL): its copy goes; what the call had not written
    to the hidden tree is lost, and the candidate is simply not judged (the ledger holds no row of it)."""
    gone = []
    if not S.HIDDEN.is_dir():
        return gone
    for kind in ("gate", "warm"):
        mark = S.HIDDEN / f"private_{kind}.txt"
        if mark.exists():
            shutil.rmtree(mark.read_text().strip(), ignore_errors=True)
            mark.unlink(missing_ok=True)
            gone.append(kind)
    if gone:
        LG.ledger(dict(step="overlap", kind="orphan_removed", trees=gone))
    return gone
