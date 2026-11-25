"""The Engineers' drafts: the parallel step, the screen on the diagnostic pool, the knob sweeps, the merge or the Composer."""

from __future__ import annotations

import json
import math
import re
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
import cascade as CS  # noqa: E402
import ledger as LG  # noqa: E402
import merge as MG  # noqa: E402
import roles as RL  # noqa: E402
import segments as SG  # noqa: E402
import state as S  # noqa: E402
from gate import candidate_hash, overrides  # noqa: E402

PATCHED = re.compile(r"^\+\+\+ \S+/skills/(\S+)", re.M)  # the files a candidate patch changes


def sweep_specs(h: dict) -> list[dict]:
    """A hypothesis's `sweep` made safe: at most 2 keys, each 2-5 distinct values."""
    out: list[dict] = []
    for it in (h.get("sweep") or [])[:2]:
        if not isinstance(it, dict) or not isinstance(it.get("key"), str) or it["key"] in [o["key"] for o in out]:
            continue
        vals: list = []
        for v in it.get("values") or []:
            if v not in vals:
                vals.append(v)
        if len(vals) >= 2:
            out.append(dict(key=it["key"], values=vals[:5]))
    return out


def sweep_row(key: str, value, before: dict, after: dict) -> dict:
    """One swept value against the incumbent, over the starts both ran: fixed (incumbent failed, now ok), broken
    (incumbent ok, now failed), net = fixed - broken."""
    fixed = sum(1 for k in after if after[k] and not before.get(k))
    broken = sum(1 for k in after if before.get(k) and not after[k])
    return dict(key=key, value=value, n=len(after), fixed=fixed, broken=broken, net=fixed - broken)


def best_row(rows: list[dict]) -> int:
    """Index of the best swept value: largest net, then fewer broken, then the earlier in the list."""
    return min(range(len(rows)), key=lambda i: (-rows[i]["net"], rows[i]["broken"], i))


def subsample_starts(labels, before: dict, m: int) -> list[str]:
    """m starts, deterministically: the incumbent's failures first up to ceil(2m/3) in label order, then its successes in
    label order, more failures when the successes run out. Label order in the result."""
    labels = sorted(labels)
    if m >= len(labels):
        return labels
    lose = [x for x in labels if not before.get(x)]
    win = [x for x in labels if before.get(x)]
    f = lose[: math.ceil(2 * m / 3)]
    w = win[: m - len(f)]
    if len(f) + len(w) < m:
        f = lose[: m - len(w)]
    return sorted(f + w)


class SweepBudget:
    """Rounds an iteration's program sweeps (the probe's and the Engineers' knobs) may run in all (caps sweep_rounds)."""

    def __init__(self, total: int):
        self.total, self.spent, self._lock = int(total), 0, threading.Lock()

    def reserve(self, want: int) -> int:
        with self._lock:
            got = max(0, min(int(want), self.total - self.spent))
            self.spent += got
            return got

    @property
    def left(self) -> int:
        return max(0, self.total - self.spent)


def sweep_budget() -> SweepBudget:
    if S.SWEEP is None:
        S.SWEEP = SweepBudget(S.CAPS.get("sweep_rounds", 1200))
    return S.SWEEP


def _ok_of(d: Path) -> dict:
    return {k: bool(r["ok"]) for k, r in B.outcomes(d).items()}


def sweep_mechanism(best_set: dict, keys: list[str]) -> str:
    return "program sweep: " + ", ".join(f"{k}={best_set[k]}" for k in keys if k in best_set)


def draft_labels(pool: list[str], declared, inc_ok: dict, sentinels: int) -> tuple[tuple | None, dict]:
    """Change 4: the starts a draft's diagnostic run covers. Declared starts ["all"], none, or none of the pool's: the whole
    pool (None, as before). Otherwise every pool start of the tasks its declared starts belong to, plus up to `sentinels`
    starts of each other task -- the incumbent's successes first, then its failures, in label order. Returns the labels
    and `ran` = {n, tasks, sentinels (the labels)}."""
    pool = sorted(pool)
    tasks_all = sorted({LG.task_of(x) for x in pool})
    whole = dict(n=len(pool), tasks=tasks_all, sentinels=[])
    decl = [x for x in (declared or []) if isinstance(x, str)]
    if not decl or "all" in decl:
        return None, whole
    own = ({LG.task_of(x) for x in decl} | {x for x in decl if x in tasks_all}) & set(tasks_all)
    if not own:
        return None, whole
    sent: list[str] = []
    for t in tasks_all:
        if t in own:
            continue
        starts = [x for x in pool if LG.task_of(x) == t]
        sent += ([x for x in starts if inc_ok.get(x)] + [x for x in starts if not inc_ok.get(x)])[:sentinels]
    labels = sorted([x for x in pool if LG.task_of(x) in own] + sent)
    return tuple(labels), dict(n=len(labels), tasks=sorted(own), sentinels=sent)


def draft_run(eng: dict, inc: dict) -> tuple[tuple | None, dict]:
    """draft_labels for a draft's declared scope, on this loop's diagnostic pool and the incumbent's rounds."""
    ok = _ok_of(S.LOOP / "diag" / f"incumbent_{inc['hash']}")
    return draft_labels(B.pool_labels(S.LOOP / "pool_diag"), ((eng or {}).get("scope") or {}).get("starts"), ok,
                        int(S.CAPS.get("draft_sentinels", 3)))  # fmt: skip


def seed_rounds(src: Path, dst: Path, labels=None) -> int:
    """Round files (and their .npz) of `src` copied into `dst`, those not there yet; `labels` limits them."""
    n = 0
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        if f.name.startswith("_") or (labels is not None and f.stem not in labels) or (dst / f.name).exists():
            continue
        for g in (f, f.with_suffix(".npz")):
            if g.exists():
                shutil.copy2(g, dst / g.name)
        n += 1
    return n


def probe_labels(h: dict) -> tuple | None:
    """The diagnostic starts a hypothesis's probe runs; None = the whole pool, as for every scene family. A backend
    whose rounds cost minutes (caps["probe_scope"]) probes a hypothesis only on the tasks it is about: its `tasks`,
    else the tasks of its predicted starts; when it names none, the whole pool. The drafts built later still run on
    every start, so a regression elsewhere is seen there and at the gate."""
    if not S.CAPS.get("probe_scope"):
        return None
    labels = B.pool_labels(S.LOOP / "pool_diag")
    task_of = lambda x: x.rsplit("_s", 1)[0]  # noqa: E731
    known = sorted({task_of(x) for x in labels})
    named = {t for t in h.get("tasks") or [] if t in known}
    if not named:
        named = {task_of(x) for x in (h.get("predicted_fixed") or []) + (h.get("predicted_broken") or [])} & set(known)
    return tuple(x for x in labels if task_of(x) in named) if named else None


def hyp_tasks(h: dict) -> set[str]:
    """The tasks a hypothesis declares it acts on: its `tasks`, plus the tasks of its `starts`."""
    return {t for t in (h.get("tasks") or []) if isinstance(t, str)} | {LG.task_of(x) for x in h.get("starts") or []
                                                                         if isinstance(x, str)}  # fmt: skip


def assign_program(hyps: list[dict], k: int) -> tuple[list[str], dict[str, str]]:
    """TODO 54 (review 2026-11-25, flow F4: the Manager chose "engineer" 28 times of 31): the Reviewer's hypotheses go
    to up to `k` Engineers in the Reviewer's order, one distinct hypothesis each, with non-overlapping declared targets
    (a hypothesis that shares a task with one already chosen is skipped; a refuted one is skipped; one that declares no
    task is taken only first and then holds every task); fewer Engineers when fewer are disjoint, at least one when
    there is any hypothesis. Returns (ids, {skipped id: why})."""
    ids: list[str] = []
    skipped: dict[str, str] = {}
    taken: set[str] = set()
    everything = False
    for h in hyps:
        if not isinstance(h, dict) or "id" not in h:
            continue
        if len(ids) >= k or everything:
            skipped[h["id"]] = "no Engineer slot left" if len(ids) >= k else "an earlier hypothesis declares no task"
            continue
        t = hyp_tasks(h)
        if h.get("status") == "refuted":
            skipped[h["id"]] = "refuted"
        elif t & taken:
            skipped[h["id"]] = f"overlaps {sorted(t & taken)} with a hypothesis already assigned"
        elif not t and ids:
            skipped[h["id"]] = "declares no task and another hypothesis is assigned"
        else:
            ids.append(h["id"])
            taken |= t
            everything = not t
    return ids, skipped


def assign_ids(dec: dict, hyps: list[dict], k: int, tune=()) -> tuple[list[str], int]:
    """Change 3: the Engineers' hypothesis ids -- the Manager's `hypotheses` (else `hypothesis`) with duplicates removed,
    order kept, at most k. Fewer than k: topped up from the other hypotheses in Reviewer order that are not refuted, not
    in `tune`, and whose `tasks` do not overlap the tasks already assigned; none qualifies, fewer Engineers. No
    hypotheses at all and `tune` set: no Engineer (the tuned drafts go to the Composer alone). Returns (ids, duplicates
    dropped)."""
    known = {h["id"]: h for h in hyps if isinstance(h, dict) and "id" in h}
    tune = [t for t in dict.fromkeys(tune or []) if t in known]
    raw = [h for h in (dec.get("hypotheses") or []) if h in known] or [h for h in [dec.get("hypothesis")] if h in known]
    if not raw:
        if tune:
            return [], 0
        raw = [next(iter(known))] if known else []
    ids = list(dict.fromkeys(raw))
    dropped = len(raw) - len(ids)
    ids = ids[:k]
    taken = {t for i in ids for t in known[i].get("tasks") or []}
    for hid, h in known.items():
        if len(ids) >= k:
            break
        if hid in ids or hid in tune or h.get("status") == "refuted":
            continue
        tasks = set(h.get("tasks") or [])
        if tasks & taken:
            continue
        ids.append(hid)
        taken |= tasks
    return ids, dropped


def early_diag(k: int, w: Path, eng: dict, inc: dict, it_dir: Path) -> None:
    """TODO 37.4 (E): a draft's diagnostic rounds start the moment its Engineer answers, in its own thread, instead of
    after the slowest Engineer; the rounds go to the same hash-keyed directory the main step uses, so that later call
    finds them done. A failure here only costs the head start (the main step runs what is missing)."""
    try:
        cand = it_dir / f"candidate_{k}"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(w / "skills", cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (cand / "candidate.json").write_text(json.dumps({"set": eng.get("set") or {}}, indent=1))
        h = candidate_hash(cand)
        if h == inc["hash"] or S.H.named_objects(cand / "skills"):
            return
        labels, _ = draft_run(eng, inc)  # change 4: the draft's tasks and a few sentinels, or the whole pool
        B.run_jobs([B.Job(cand / "skills", S.LOOP / "diag" / f"candidate_{h}", overrides(cand), labels, True)], S.FAMILY,
                   S.LOOP / "pool_diag", S.procs_low(), it_dir / f"diag_{k}.log", S.GPUS)  # fmt: skip
    except Exception as e:  # noqa: BLE001 -- the main step repeats the run
        print(f"early diag of draft {k}: {type(e).__name__}: {e}", flush=True)


def round_path(r: dict) -> list:
    """What a round did, step by step. Rounds are deterministic, so two rounds of a start with different paths mean
    the code reached this one."""
    return [[s.get("step"), s.get("ok"), s.get("t0"), s.get("t1")] for s in r.get("steps") or []]


def same_round(a: dict, b: dict) -> bool:
    return (
        round_path(a) == round_path(b) and bool(a.get("ok")) == bool(b.get("ok")) and a.get("t_end") == b.get("t_end")
    )


def footprint(ref: Path, cand: Path, scope: dict | None) -> dict:
    """Rule 38: the diagnostic rounds a draft changed against the rounds it started from (`ref`), measured, beside what
    its Engineer said: `outside` changed and not declared, `missed` declared and left as they were. Only starts both ran
    count; `not_run` are the pool starts the draft did not run (change 4)."""
    a, b = B.outcomes(ref), B.outcomes(cand)
    both = sorted(set(a) & set(b))
    changed = [x for x in both if not same_round(a[x], b[x])]
    declared = list((scope or {}).get("starts") or [])
    return dict(n=len(both), changed=changed, declared=declared, not_run=sorted(set(a) - set(b)),
                outside=[] if "all" in declared else [x for x in changed if x not in declared],
                missed=[x for x in declared if x in both and x not in changed])  # fmt: skip


def segments_touched(ref: Path, cand: Path, changed: list[str], table: dict) -> dict:
    """TODO 48.17.6 ([segments]): where each round a draft changed first left the rounds it started from, by the kind
    of that step in its task (segments.touched): a change that starts in a stable step is a risk the Composer weighs."""
    a, b = B.outcomes(ref), B.outcomes(cand)
    where = {}
    for x in changed:
        if x in a and x in b:
            d = first_difference(a[x], b[x])
            s = d["incumbent"] or d["draft"]
            if s:
                where[x] = (a[x].get("task") or LG.task_of(x), s["step"])
    return SG.touched(where, table)


def first_difference(a: dict, b: dict) -> dict:
    """Where round `b` left round `a`'s path: the first step that differs, as each round has it."""
    pa, pb = round_path(a), round_path(b)
    k = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
    keys = ("step", "ok", "t0", "t1")
    return dict(index=k, incumbent=dict(zip(keys, pa[k])) if k < len(pa) else None,
                draft=dict(zip(keys, pb[k])) if k < len(pb) else None)  # fmt: skip


MIN_SWEEP_STARTS = 3  # an Engineer-declared knob is not swept on fewer starts than this


def unknown_keys(skills: Path, set_: dict, items: list[dict]) -> list[str]:
    """The tunable keys the draft's package does not accept (its loader's apply_set refuses them)."""
    if be := BK.of(S.FAMILY):
        try:
            pkg = be.load_skills(skills, "tunable_check", S.FAMILY)
        except Exception:  # noqa: BLE001 -- a package that does not load tunes nothing
            return [it["key"] for it in items]
        bad = []
        for it in items:
            try:
                pkg.apply_set(pkg, {**set_, it["key"]: it["values"][0]})
            except Exception:  # noqa: BLE001
                bad.append(it["key"])
        return bad
    return S.H.unknown_keys(skills, set_, items)  # a scene family: its own process loads the package


def tunables_of(eng: dict) -> list[dict]:
    """An Engineer's `tunable` made safe: at most 3 distinct keys, each with 2-5 distinct values (as a sweep)."""
    out: dict = {}
    for it in (eng.get("tunable") or [])[:3]:
        for spec in sweep_specs(dict(sweep=[it])) if isinstance(it, dict) else []:
            out.setdefault(spec["key"], spec)
    return list(out.values())


def tune_draft(k: int, w: Path, eng: dict, inc: dict, it_dir: Path) -> dict:
    """Change 1 (Engineer-declared knobs): after the draft's own diagnostic run, sweep its `tunable` keys one at a time on
    the draft's scope starts (sentinels excluded), each against the incumbent as in the probe; a key's best value joins the
    draft's set when it beats the draft as it stands (net, then fewer broken). Unknown keys are dropped and recorded. The
    draft's candidate dir, hash and rounds then follow the tuned set (the best variant's rounds are copied, its sentinels
    run by early_diag). Returns the draft's answer with `tuned`, `tunable_sweep` and `tunable_dropped` added."""
    items = tunables_of(eng)
    cand = it_dir / f"candidate_{k}"
    if not items or not cand.exists():
        return eng
    set0 = dict(eng.get("set") or {})
    bad = set(unknown_keys(cand / "skills", set0, items))
    kept = [it for it in items if it["key"] not in bad]
    out = dict(eng, tuned={}, tunable_sweep=[], tunable_dropped=sorted(bad))
    labels, ran = draft_run(eng, inc)
    pool = B.pool_labels(S.LOOP / "pool_diag")
    starts = sorted(set(labels if labels is not None else pool) - set(ran["sentinels"]))
    before = _ok_of(S.LOOP / "diag" / f"incumbent_{inc['hash']}")
    cur_dir = S.LOOP / "diag" / f"candidate_{candidate_hash(cand)}"
    cur_ok = _ok_of(cur_dir)
    starts = [x for x in starts if x in cur_ok and x in before]
    n_vals = sum(len(it["values"]) for it in kept)
    if not kept or not starts or n_vals == 0:
        return out
    m = min(len(starts), sweep_budget().left // n_vals)
    if m < MIN_SWEEP_STARTS:
        out["tunable_sweep"] = [dict(skipped="sweep budget spent", left=sweep_budget().left)]
        return out
    sweep_budget().reserve(n_vals * m)
    sub = subsample_starts(starts, before, m)
    cur_row = sweep_row("", None, before, {x: cur_ok[x] for x in sub})
    cur_set, best_dir, table = dict(set0), None, []
    for ki, it in enumerate(kept):
        dirs = [it_dir / f"tune_{k}" / f"{ki}_{i}" for i in range(len(it["values"]))]
        try:
            B.run_jobs([B.Job(cand / "skills", d, {**cur_set, it["key"]: v}, tuple(sub))
                        for d, v in zip(dirs, it["values"])], S.FAMILY, S.LOOP / "pool_diag", S.procs_low(),
                       it_dir / f"tune_{k}.log", S.GPUS)  # fmt: skip
        except Exception as e:  # noqa: BLE001
            table.append(dict(key=it["key"], error=f"{type(e).__name__}: {e}"[:200]))
            continue
        rows = [sweep_row(it["key"], v, before, _ok_of(d)) for d, v in zip(dirs, it["values"])]
        live = [i for i, r in enumerate(rows) if r["n"]]
        table.append(
            dict(key=it["key"], baseline=cur_row, rows=rows, starts=len(sub), subsampled=len(sub) < len(starts))
        )
        if not live:
            continue
        b = live[best_row([rows[i] for i in live])]
        if (rows[b]["net"], -rows[b]["broken"]) > (cur_row["net"], -cur_row["broken"]):
            cur_set[it["key"]] = it["values"][b]
            cur_row, best_dir = rows[b], dirs[b]
            table[-1]["adopted"] = it["values"][b]
    out["tunable_sweep"] = table
    out["tuned"] = {key: v for key, v in cur_set.items() if set0.get(key) != v}
    if out["tuned"]:
        out["set"] = cur_set
        (cand / "candidate.json").write_text(json.dumps({"set": cur_set}, indent=1))
        if best_dir is not None:
            seed_rounds(best_dir, S.LOOP / "diag" / f"candidate_{candidate_hash(cand)}", sub)
        early_diag(k, w, out, inc, it_dir)  # the draft's other starts and its sentinels, under the tuned hash
    return out


def wait_drafts(threads: dict, stops: dict, poll: float = 2.0, sessions: dict | None = None) -> list:
    """TODO 37.3 (D, user 2026-10-23): wait for the Engineers; once ceil(K/2) Engineers' SESSIONS have ended, the rest get
    ENGINEER_LATE_MIN minutes counted from that moment, then the stop events of the sessions still running are set
    (run_role kills the session, RoleLate) and the iteration goes on with the drafts it has. `sessions` ({k: time the
    Engineer's session ended}, filled by the callers' threads) keeps the draft's own diagnostic rounds and knob sweep, which
    a thread also runs after its session, out of the deadline; a thread that ends without an entry counts from its end.
    The late hypothesis counts as not built, so [scene-status] builds it first next time. A straggler is stopped rather than left
    running: the Engineers share the sandbox, and a file it changed outside its copy later could not be told from the
    Composer's ([scene-candidates]). Returns the Engineers stopped."""
    need = (len(threads) + 1) // 2
    sessions = {} if sessions is None else sessions
    late: list = []
    while True:
        for k, t in threads.items():
            if k not in sessions and not t.is_alive():
                sessions[k] = time.time()
        alive = [k for k, t in threads.items() if t.is_alive()]
        if not alive:
            break
        running = [k for k in alive if k not in sessions]
        if S.ENGINEER_LATE_MIN > 0 and need <= len(sessions) < len(threads) and running and not late:
            if time.time() - sorted(sessions.values())[need - 1] > S.ENGINEER_LATE_MIN * 60:
                for k in running:
                    stops[k].set()
                late = running
        time.sleep(poll)
    for t in threads.values():
        t.join()
    return late


def multi_engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub) -> str | None:
    """TODO 32.9 (Harness2's contrastive compose step): CANDIDATES Engineers build the Manager's hypotheses in parallel,
    each in its own copy of the incumbent's skills package (`cand_k/skills`); every valid candidate runs on the
    diagnostic pool; the Composer merges what those rounds support into one candidate, adopts one, or keeps the
    incumbent; only that one candidate reaches the gate, so the gate's test stays one test per call."""
    known = {h["id"]: h for h in hyps["hypotheses"]}
    tune = [t for t in dict.fromkeys(dec.get("tune") or []) if t in known]  # change 1: tuned drafts, no Engineer
    ids, _ = assign_ids(dec, hyps["hypotheses"], S.CANDIDATES, tune)  # change 3: no duplicate assignments
    bentry = CS.base_of(
        n, dec
    )  # TODO 33.8: every working copy starts from that pool candidate instead of the incumbent
    src = Path(bentry["dir"]) if bentry else Path(inc["dir"])
    calls = []
    for k, hid in enumerate(ids, 1):
        w = it_dir / f"cand_{k}"
        if w.exists():
            shutil.rmtree(w)
        shutil.copytree(src / "skills", w / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        ctx = w / "engineer_context.json"
        LG.jdump(ctx, LG.context(inc, dict(hypothesis=known[hid], manager=dec,
                                     probe=[p for p in pr if p.get("id") == hid],
                                     park=None, draft=dict(number=k, of=len(ids)), base=CS.base_brief(bentry))))  # fmt: skip
        prompt = RL.fill(
            RL.role_template("engineer"),
            CONTEXT=ctx,
            ITER=w,
            **dict(common, SKILLS=w / "skills", TMP=RL.role_dir(f"engineer_{k}")),
        )
        stub_e = dict(mechanism=f"stub {k}", files_changed=[], set={**overrides(src)}, tests="none",
                      scope=dict(when="stub", starts=["all"]))  # fmt: skip
        if stub:  # a stub draft must differ from the incumbent, or the plumbing after it never runs
            with (w / "skills" / S.STUB_FILE).open("a") as fh:
                fh.write(f"\n# stub draft {k}\n")
        calls.append((k, hid, w, prompt, stub_e))
    out: dict = {}

    stops = {k: threading.Event() for k, *_ in calls}
    sess_end: dict = {}  # k -> when Engineer k's session ended (wait_drafts' deadline counts from there)
    inc_rounds = S.LOOP / "diag" / f"incumbent_{inc['hash']}"
    pooled = S.LOOP / "diag" / f"candidate_{bentry['hash']}" if bentry else None
    ref = pooled if pooled is not None and pooled.is_dir() else inc_rounds  # the rounds the drafts start from

    def one(k, w, prompt, stub_e):
        try:
            out[k] = RL.claude(f"engineer_{k}", prompt, RL.ENGINEER_SCHEMA, it_dir, "NotebookEdit,Agent",
                            stub_e if stub else None, stop=stops[k])  # fmt: skip
        except RL.RoleFailed as e:
            out[k] = e
            return
        finally:
            sess_end[k] = time.time()
        if not stub:
            early_diag(k, w, out[k][0], inc, it_dir)
            try:  # change 1: the draft's declared knobs, swept on its own starts
                out[k] = (tune_draft(k, w, out[k][0], inc, it_dir), out[k][1])
            except Exception as e:  # noqa: BLE001 -- the draft stands as it was
                print(f"tunables of draft {k}: {type(e).__name__}: {e}", flush=True)

    threads = {k: threading.Thread(target=one, args=(k, w, pr_, se)) for k, _, w, pr_, se in calls}
    for t in threads.values():
        t.start()
    wait_drafts(threads, stops, sessions=sess_end)
    stray = CS.revert_strays(base_changes)
    CS.put_skills(Path(inc["dir"]) / "skills")
    draft_labels_of: dict[
        int, tuple | None
    ] = {}  # draft number -> the starts its screen ran on (a rebase runs them again)
    cands, jobs, first, leaked = [], [], {}, {}  # first: candidate hash -> the draft kept (identical drafts run once)
    for k, hid, w, _, _ in calls:
        res = out.get(k)
        if isinstance(res, Exception) or res is None:
            LG.ledger(dict(step="engineer", iter=n, candidate=k, hypothesis=hid, void=True, late=isinstance(res, RL.RoleLate),
                        error=str(res)[:300]))  # fmt: skip
            continue
        eng, acct = res
        leaks = AU.transcript_leaks(acct["session"])
        cand = it_dir / f"candidate_{k}"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(w / "skills", cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (cand / "candidate.json").write_text(json.dumps({"set": eng.get("set") or {}}, indent=1))
        patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(Path(inc["dir"]) / "skills"),
                                str(cand / "skills")], capture_output=True, text=True).stdout  # fmt: skip
        (it_dir / f"code_{k}.patch").write_text(patch)
        h = candidate_hash(cand)
        named = S.H.named_objects(cand / "skills")  # TODO 33.13 ([branch]): a draft that branches on names is void
        void = bool(leaks) or bool(stray) or h == inc["hash"] or bool(named)
        dup = first.get(h) if not void else None  # TODO 36 (review 2026-10-22, L5): the first of identical drafts stays
        labels, ran = draft_run(eng, inc)  # change 4: where the draft's diagnostic rounds ran
        draft_labels_of[k] = labels
        LG.ledger(dict(step="engineer", iter=n, candidate=k, hypothesis=hid, changed=sorted(set(PATCHED.findall(patch))),
                    stray=stray, mechanism=eng["mechanism"], set=eng["set"], scope=eng.get("scope"), void=void, leaks=leaks,
                    ran=ran, **({"tuned": eng["tuned"], "tunable_sweep": eng["tunable_sweep"],
                                 "tunable_dropped": eng["tunable_dropped"]} if "tuned" in eng else {}),
                    **({"named_objects": named} if named else {}), **({"base": bentry["hash"]} if bentry else {}),
                    **({"duplicate_of": dup["number"]} if dup else {}), **acct))  # fmt: skip
        if leaks:
            leaked[k] = leaks
        if leaks or stray:  # voided drafts leave the visible tree
            AU.hide_void(n, w, cand, it_dir / f"code_{k}.patch")
        if dup:
            dup.setdefault("duplicate_drafts", []).append(k)
        elif not void:
            cands.append(dict(number=k, kind="engineer", hypothesis=hid, mechanism=eng["mechanism"], scope=eng.get("scope"),
                              dir=str(cand), patch=str(it_dir / f"code_{k}.patch"), hash=h, ran=ran,
                              diag=str(S.LOOP / "diag" / f"candidate_{h}"), **({"tuned": eng["tuned"]} if "tuned" in eng else {})))  # fmt: skip
            first[h] = cands[-1]
            jobs.append(B.Job(cand / "skills", S.LOOP / "diag" / f"candidate_{candidate_hash(cand)}", overrides(cand),
                              labels, True))  # fmt: skip
    cands += S.H.tuned_drafts(n, it_dir, inc, known, pr, tune, src, bentry, len(calls) + 1, first, jobs)  # change 1
    if not cands:
        LG.ledger(dict(step="compose", iter=n, decision="none", reason="no valid candidate", stray=stray))
        if leaked and len(leaked) == len(calls):  # every draft voided by its Engineer's leak: the iteration is void
            AU.void_iteration(n, "engineers", sorted({x for v in leaked.values() for x in v}))
        return None
    if not stub:
        B.run_jobs(jobs, S.FAMILY, S.LOOP / "pool_diag", S.procs_low(), it_dir / "prober.log", S.GPUS)
    if (
        S.OVERLAP is not None
    ):  # [overlap]: the call in flight is waited for; an accept rebases the drafts onto its candidate
        why, inc, src, ref, cands = S.OVERLAP.join(n, it_dir, inc, src, ref, cands, draft_labels_of, stub)
        if why:
            return why
        if not cands:
            LG.ledger(dict(step="compose", iter=n, decision="none", reason="every draft was dropped by the rebase"))
            return None
    base = S.LOOP / "diag" / f"incumbent_{inc['hash']}"
    seg_table = SG.segment_table(B.outcomes(base).values())  # TODO 48.17.6 ([segments])
    for c in cands:
        c.update(CS.flips(base, Path(c["diag"])), footprint=footprint(ref, Path(c["diag"]), c["scope"]))
        c["segments_touched"] = segments_touched(ref, Path(c["diag"]), c["footprint"]["changed"], seg_table)
    LG.jdump(it_dir / "candidates.json", cands)
    # the Composition: by the program when the family says so (a 3-way merge of the drafts with a positive screen), the
    # Composer on a conflict or when no draft is positive (TODO 54); always the Composer for a family that says "llm"
    wc = it_dir / "compose" / "skills"
    if wc.parent.exists():
        shutil.rmtree(wc.parent)
    mode, comp_extra, program = None, {}, None
    if S.cap("compose", "llm") == "program":
        for c in cands:
            c["screen_net"] = len(c.get("fixed") or []) - len(c.get("broken") or [])
        pos = sorted((c for c in cands if c["screen_net"] > 0), key=lambda c: (-c["screen_net"], c["number"]))
        if not pos:
            mode = "none_positive"
        else:
            res = MG.merge_drafts(src / "skills", overrides(src), [(c["number"], Path(c["dir"]) / "skills",
                                  overrides(Path(c["dir"]))) for c in pos], wc)  # fmt: skip
            comp_extra = dict(merged_cleanly=res["merged"], conflicts=res["conflicts"], merged_set=res["set"])
            if res["conflicts"]:
                mode = "conflict"
            elif len(pos) == 1:
                program = dict(
                    decision="adopt", adopted=pos[0]["number"], notes="program: the one draft with a positive screen"
                )
            else:
                program = dict(decision="merge", set=res["set"], notes=f"program: 3-way merge of drafts {res['merged']}",
                               mechanism="program merge of drafts " + ", ".join(map(str, res["merged"])))  # fmt: skip
    if mode in (None, "none_positive") and program is None:  # the Composer works on a fresh copy of the base
        shutil.copytree(src / "skills", wc, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if program is not None:
        comp, acct, leaks = program, dict(cost_usd=0.0, session=None, seconds=0.0, by="program"), []
    else:
        ctx = it_dir / "composer_context.json"
        LG.jdump(ctx, LG.context(inc, dict(hypotheses=[known[h] for h in dict.fromkeys([*ids, *tune])], candidates=cands,
                                     incumbent_diag=str(base), manager=dec, base=CS.base_brief(bentry),
                                     candidate_pool=CS.pool_view(inc, base),
                                     **({"compose_mode": mode, **comp_extra} if mode else {}))))  # fmt: skip
        prompt = RL.fill(RL.role_template("composer"), CONTEXT=ctx, ITER=it_dir, **dict(common, SKILLS=wc))
        stub_c = dict(decision="adopt", adopted=cands[0]["number"], notes="stub")
        comp, acct = RL.claude(
            "composer", prompt, RL.COMPOSER_SCHEMA, it_dir, "NotebookEdit,Agent", stub_c if stub else None
        )
        leaks = AU.transcript_leaks(acct["session"])
        if (
            leaks and S.cap("role_rerun") and not stub
        ):  # TODO 48.17: the first leak discards the composition, one more call
            CS.revert_strays(base_changes)
            CS.put_skills(Path(inc["dir"]) / "skills")
            RL.discard_attempt(n, "composer", acct, leaks, it_dir / "compose")
            shutil.copytree(
                src / "skills", wc, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True
            )
            comp, acct = RL.claude("composer", prompt + RL.rerun_note("Composer", leaks), RL.COMPOSER_SCHEMA, it_dir,
                                "NotebookEdit,Agent")  # fmt: skip
            leaks = AU.transcript_leaks(acct["session"])
    stray = CS.revert_strays(base_changes)
    CS.put_skills(Path(inc["dir"]) / "skills")
    LG.jdump(it_dir / "composer.json", comp)
    LG.ledger(dict(step="compose", iter=n, decision=comp["decision"], adopted=comp.get("adopted"), notes=comp["notes"],
                mechanism=comp.get("mechanism", ""), set=comp.get("set"), stray=stray, void=bool(leaks or stray),
                leaks=leaks, **({"mode": mode, **comp_extra} if mode else {}), **acct))  # fmt: skip
    if leaks or stray:  # its working copy leaves the visible tree as a voided draft does
        AU.hide_void(n, it_dir / "compose")
        if leaks:
            AU.void_iteration(n, "composer", leaks)
        return None
    if comp["decision"] == "keep":
        return None
    if comp["decision"] == "adopt" and comp.get("adopted_pool"):  # TODO 33.8: a pool candidate as it stands
        pe = CS.pool_entry(comp["adopted_pool"])
        if pe is None:
            LG.ledger(dict(step="gate", iter=n, accepted=False, void=True,
                        reason=f"adopted pool candidate {comp['adopted_pool']} is not in the pool"))  # fmt: skip
            return None
        cand = Path(pe["dir"])
    elif comp["decision"] == "adopt":
        pick = next((c for c in cands if c["number"] == comp.get("adopted")), None)
        if pick is None:
            LG.ledger(
                dict(
                    step="gate",
                    iter=n,
                    accepted=False,
                    void=True,
                    reason=f"adopted candidate {comp.get('adopted')} is not a valid one",
                )
            )
            return None
        cand = Path(pick["dir"])
    else:
        cand = it_dir / "candidate"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(wc, cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (cand / "candidate.json").write_text(json.dumps({"set": comp.get("set") or {}}, indent=1))
        patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(Path(inc["dir"]) / "skills"),
                                str(cand / "skills")], capture_output=True, text=True).stdout  # fmt: skip
        (it_dir / "code.patch").write_text(patch)
        if candidate_hash(cand) == inc["hash"]:
            LG.ledger(
                dict(step="gate", iter=n, accepted=False, void=True, reason="the composed candidate is the incumbent")
            )
            return None
    named = S.H.named_objects(cand / "skills")  # TODO 33.13 ([branch])
    if named:
        CS.named_void(n, named, "the composed candidate")
        return None
    return CS.send(n, it_dir, cand, stub)
