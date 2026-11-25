"""The RSI inner loop (TODO 27.2.4, rewritten as a core plus a plug-in in TODO 54 phase 3): `run`, `status`, `redraw`.

One iteration (`iterate`): the Reviewer reads the incumbent's diagnostic rounds and names hypotheses; `program_step`
hands them to up to K Engineers (`drafts.multi_engineer_step`: each draft is screened on the diagnostic pool, the drafts
are merged by the program or by the Composer); `cascade.gate_candidate` sends the one candidate through the check half
and the gate. `stop.stop_reason` ends the run; `stop.run_sealed` confirms the final incumbent on the sealed pool. State
lives in files under outputs/scene_rsi_20261016/<family>/loop, so the driver stops and resumes at any iteration.

Modules: state (paths, caps, limits, hooks), ledger (rows, context), audit (hidden tree lock, transcript audit),
roles (role calls, prompts, schemas), drafts (Engineers' step), cascade (check half, gate, pool), stop, plugin
(the seam to scripts/rsi/scene/, loaded only for a family without a `backend`).

Runs inside a sandbox checkout (sandbox.py). The roles see the sandbox, the rules and the plan; the acceptance and sealed
pools and the gate live in the hidden tree (`..._hidden/`), and every tool call in a role's transcript is grepped for
that name and for paths outside the sandbox (audit.py). A hit voids what the role produced and the loop goes on; two
voided iterations in a row stop it. The Engineer may change only the family's skills package; anything else it touches
voids the call and is restored. Model cost has no cap (user 2026-10-19; RSI_COST_CAP_USD sets one).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    nohup setsid $PY scripts/rsi/loop.py run <family> > /dev/null 2>&1 &
    $PY scripts/rsi/loop.py status <family>
    $PY scripts/rsi/loop.py run <family> --stub --max-iters 1   # plumbing, no model calls
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import signal
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "src")]
import audit as AU  # noqa: E402
import cascade as CS  # noqa: E402
import drafts as DR  # noqa: E402
import ledger as LG  # noqa: E402
import roles as RL  # noqa: E402
import state as S  # noqa: E402
import stop as ST  # noqa: E402


def iteration_dirs() -> list[Path]:
    """The iteration directories (iter_NN), not the renamed voids."""
    return [d for d in sorted(S.STATE.glob("iter_*")) if re.fullmatch(r"iter_\d+", d.name)]


def iterate(n: int, meta: dict, stub: bool) -> str | None:
    it_dir = S.STATE / f"iter_{n:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    CS.snapshot(n)
    inc = CS.incumbent()
    CS.put_skills(Path(inc["dir"]) / "skills")
    base_changes = CS.changed_files()
    base = CS.diag_incumbent(inc, it_dir)
    attribution = S.H.before_review(
        inc, base, n, it_dir, stub
    )  # scene line: diagnostic redraw, lifted parks, attribution
    cards = RL.reviewer_cards(n, base, it_dir)
    common, extra = RL.review_common(n, inc, base, cards, attribution)
    hyps, acct, sessions, leaks = RL.run_review(n, it_dir, inc, base, common, extra, stub)
    if leaks and S.cap("role_rerun") and not stub:  # TODO 48.17: the first leak discards the review, one more call
        RL.discard_attempt(n, "reviewer", acct, leaks, it_dir / "hypotheses.json")
        hyps, acct, sessions, leaks = RL.run_review(n, it_dir, inc, base, common, extra, stub,
                                                 note=RL.rerun_note("Reviewer", leaks))  # fmt: skip
    RL.review_row(n, hyps, acct, sessions, leaks)
    if leaks:
        AU.void_iteration(n, "reviewer", leaks)
        return None
    step = (
        program_step if S.cap("manager", "llm") == "program" else S.H.after_review
    )  # the scene line keeps the Manager
    return step(n, it_dir, inc, base, base_changes, hyps, common, extra, stub)


def forward_escalations(n: int, hyps: dict) -> None:
    """The Manager used to forward the Reviewer's `LEVEL 3` and `BED` paragraphs ([levels]) to people; with the
    program as Manager they go to loop/escalations.jsonl as they are."""
    for para in str(hyps.get("notes") or "").split("\n\n"):
        head = para.strip().split(" ", 1)[0].strip(":.").upper() if para.strip() else ""
        if head in ("LEVEL", "BED") and (head == "BED" or para.strip().upper().startswith("LEVEL 3")):
            with (S.STATE / "escalations.jsonl").open("a") as fh:
                fh.write(
                    json.dumps(
                        dict(
                            t=LG.now(),
                            iter=n,
                            kind="level3" if head == "LEVEL" else "bed",
                            text=para.strip(),
                            by="reviewer",
                        )
                    )
                    + "\n"
                )


def program_step(n, it_dir, inc, base, base_changes, hyps, common, extra, stub) -> str | None:
    """The lean iteration after the Reviewer (caps manager "program", TODO 54): no Prober, no Manager. The program
    assigns the hypotheses to the Engineers (assign_program); the Engineers' drafts are screened, merged and gated as
    in multi_engineer_step. The parameter search is the program's sweep of the Engineers' `tunable` keys, in one budget
    for the whole iteration (a fresh one here)."""
    S.SWEEP = None  # one budget per iteration, shared by the drafts' tunable sweeps
    forward_escalations(n, hyps)
    ids, skipped = DR.assign_program(hyps["hypotheses"], S.CANDIDATES)
    dec = dict(decision="engineer", hypothesis=ids[0] if ids else None, hypotheses=ids, by="program",
               reason="the Reviewer's order, disjoint tasks", summary_zh="")  # fmt: skip
    LG.jdump(it_dir / "assignment.json", dict(dec, skipped=skipped))
    LG.ledger(dict(step="assign", iter=n, assigned=ids, skipped=skipped,
                tasks={i: sorted(DR.hyp_tasks(h)) for h in hyps["hypotheses"] for i in [h["id"]] if i in ids}))  # fmt: skip
    if not ids:
        return None
    return CS.engineer_step(n, it_dir, inc, hyps, [], dec, common, base_changes, stub)


def resume(meta: dict) -> None:
    """A run started again after its own stop: the hours it stood stopped do not count against MAX_WALL_H (32.12
    stood stopped for a day while the audit was fixed, TODO 32.13). The stop may be followed by bookkeeping rows --
    a standalone `loop.py redraw` (33.23), a `settings` row, a wall-clock `stop` the resume itself wrote -- so the
    pause runs from the first row of that trailing run (run C, 2026-10-23: a redraw row after the stop hid a 12 h
    pause and the resumed run stopped at once on the wall clock); a `resume` row ends the walk back, its gap is
    already counted."""
    es = LG.entries()
    tail = []
    for e in reversed(es):
        if e.get("step") not in ST.BOOKKEEPING_STEPS:
            break
        tail.append(e)
    if not any(e.get("step") == "stop" for e in tail):
        return
    t = calendar.timegm(time.strptime(tail[-1]["t"], "%Y-%m-%dT%H:%M:%SZ"))
    if t < int(meta["started_s"]):  # a stop from before this run started (a replay's copied ledger); t is whole seconds
        return
    gap = time.time() - t
    meta["paused_s"] = meta.get("paused_s", 0.0) + gap
    LG.ledger(dict(step="resume", stopped_h=round(gap / 3600, 2), wall_h=round(ST.wall_h(meta), 2)))


EFFORT_ROLE = re.compile(r"^(reviewer(_[PO])?|manager|engineer(_\d+)?|composer)$")


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def parse_effort(items: list[str]) -> dict:
    """TODO 36 (review 2026-10-22, L7): `--effort ROLE=LEVEL`; an unknown role (`enginer=medium` used to add an unused
    key silently) or level is refused with the valid ones listed."""
    out = {}
    for x in items:
        role, _, level = x.partition("=")
        if not EFFORT_ROLE.match(role):
            raise SystemExit(f"--effort: unknown role {role!r}; valid: reviewer (reviewer_P, reviewer_O), manager, "
                             f"engineer (engineer_1, engineer_2, ...), composer")  # fmt: skip
        if level not in EFFORT_LEVELS:
            raise SystemExit(f"--effort {x}: level must be one of {', '.join(EFFORT_LEVELS)}")
        out[role] = level
    return out


def settings_changes(cfg: dict, meta: dict, resuming: bool) -> dict:
    """What a resume changes in the run's settings (a `settings` ledger row). A key an older run did not record compares
    with the default it ran under (stop_rate: 0.70 before 2026-10-22)."""
    if not resuming:
        return {}
    was = {k: meta.get(k, S.STOP_RATE_DEFAULT if k == "stop_rate" else None) for k in cfg}
    return {k: dict(old=was[k], new=v) for k, v in cfg.items() if was[k] is not None and was[k] != v}


class Terminated(BaseException):
    pass


def cmd_run(a) -> int:
    """TODO 36 (review 2026-10-22, L4): exit 0 for a normal stop, 3 for an abnormal one (see is_abnormal) or a crash.
    An uncaught exception or SIGTERM writes a `stop` row first, so resume() counts the downtime as a pause."""
    S.STATE.mkdir(parents=True, exist_ok=True)
    efforts = parse_effort(getattr(a, "effort", None) or [])
    AU.unlock_hidden()
    AU.unlock_rounds()

    def on_term(*_):
        raise Terminated()

    old = None
    try:
        old = signal.signal(signal.SIGTERM, on_term)
    except ValueError:  # not the main thread
        pass
    try:
        return _run(a, efforts)
    except Terminated:
        LG.ledger(dict(step="stop", reason="terminated"))
        print("stop: terminated")
        return 3
    except (Exception, KeyboardInterrupt) as e:
        LG.ledger(dict(step="stop", reason=f"crashed: {e!r}"[:500]))
        traceback.print_exc()
        return 3
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
        AU.unlock_hidden()


def _run(a, efforts: dict) -> int:
    prior = LG.jload(S.STATE / "meta.json")
    resuming = bool(prior and "started" in prior)
    meta = prior or dict(
        started=LG.now(), started_s=time.time(), cost_cap_usd=S.COST_CAP, model=S.MODEL, family=S.FAMILY
    )
    resume(meta)
    cfg = S.resolve_settings(a, meta, resuming)
    S.CANDIDATES, S.MAX_WALL_H = cfg["candidates"], cfg["max_wall_h"]
    S.STOP_RATE, S.ENGINEER_LATE_MIN = cfg["stop_rate"], cfg["engineer_late_min"]
    S.DIAG_MIN_FAILS, S.DIAG_TARGET_FAILS = cfg["diag_min_fails"], cfg["diag_target_fails"]
    S.DIAG_DRAW_MAX = cfg["diag_draw_max"]
    if not cfg["stop_rate"] > 0:
        raise SystemExit(f"--stop-rate must be > 0 (got {cfg['stop_rate']})")
    changed = settings_changes(cfg, meta, resuming)
    if changed:
        LG.ledger(dict(step="settings", **changed))
    meta.update(cost_cap_usd=S.COST_CAP, model=S.MODEL, **cfg)  # this run's limits (roles read meta.json)
    LG.jdump(S.STATE / "meta.json", meta)
    if not (S.LOOP / "pool_diag").exists() or not CS.accept_pool().exists():
        raise SystemExit("draw the pools first (family.py draw ... diag / accept)")
    if not CS.pool_path().exists():  # TODO 33.8: state from before the candidate pool (a replay of an older run)
        CS.rebuild_pool()
    CS.split_check_half()  # TODO 48.17 check_half: once per loop
    n = len(iteration_dirs())
    iters = a.max_iters
    S.STOP_AFTER = getattr(a, "stop_after", None)
    S.EFFORT.update(efforts)

    def finish(why: str, sealed: bool = True) -> int:
        """The one wrap-up of every way a run ends (--max-iters, the wall clock, a stop rule, a failed role, a gate
        error): the stop row, the hit rates, and the sealed set when the family's caps ask for it (TODO 54, review
        verify.md 1: --max-iters used to return without it, so no run's final incumbent was ever confirmed)."""
        LG.ledger(dict(step="stop", reason=why))
        print("stop:", why)
        ST.write_hit_rate()
        if sealed:
            ST.run_sealed(why, a.stub)
        return 3 if ST.is_abnormal(why) else 0

    if getattr(a, "engineer_from", None) is not None:  # one iteration: the Engineer and the gate on a recorded decision
        try:
            why = ST.stop_reason(meta) or S.H.engineer_again(n, a.engineer_from.resolve(), a.stub)
        except RL.RoleFailed as e:
            why = f"role call failed: {e}"
        return finish(why or "replay: engineer and gate done", sealed=False)
    if a.reprobe is not None:  # the re-probed iteration counts as one of --max-iters
        try:
            why = S.H.reprobe(a.reprobe, a.stub)
        except RL.RoleFailed as e:
            why = f"role call failed: {e}"
        if why:
            return finish(why, sealed=False)
        iters -= 1
    for _ in range(iters):
        why = ST.stop_reason(meta)
        if why is None:
            try:
                why = iterate(n, meta, a.stub)
            except RL.RoleFailed as e:
                why = f"role call failed: {e}"
            n += 1
            ST.write_hit_rate()
        if why:
            return finish(why)
    return finish(f"max iterations {a.max_iters} reached")


def cmd_status(_a) -> int:
    es = LG.entries()
    meta = LG.jload(S.STATE / "meta.json", {})
    print(f"family {S.FAMILY}; started {meta.get('started')}; entries {len(es)}; "
          f"cost {sum(e.get('cost_usd', 0) for e in es):.2f} USD")  # fmt: skip
    for e in es[-12:]:
        print(json.dumps({k: v for k, v in e.items() if k not in ("usage", "results")}, ensure_ascii=False)[:300])
    parked = LG.jload(S.STATE / "parked.json", {})
    for cid, p in parked.items():
        print(f"parked {cid}: {p['task']} ({p['scope']}, {len(p['starts'])} diag starts, {p['kind']})")
    try:  # while a role runs the hidden directory has mode 000 (hidden_locked): no acceptance line, not a crash
        inc = LG.jload(S.GATE_DIR / "incumbent.json")
        v = S.H.unparked_view(Path(inc["rounds"])) if inc else None
    except PermissionError:
        print("acceptance: not readable now (a role is running and the hidden directory is locked); ask again later")
        return 0
    if v:
        print(f"acceptance: {v['all_ok']}/{v['all_n']} of all starts, {v['ok']}/{v['n']} outside {v['parked_tasks']}")
    try:  # TODO 48.19.3
        import hit_rate

        print(hit_rate.line(hit_rate.compute(S.STATE)))
    except Exception as e:
        print(f"hit rate: {e!r}"[:200])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("family")
    r.add_argument("--max-iters", type=int, default=100)
    r.add_argument("--stub", action="store_true", help="canned role outputs, gate --dry-run: plumbing only")
    r.add_argument(
        "--reprobe", type=int, default=None, help="re-probe iteration N's hypotheses first (probes lost or not run)"
    )
    r.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Engineers per iteration, merged by a Composer (TODO 32.9; default 3 since TODO 32.15; 1: no Composer; "
        "a resumed run keeps its recorded value unless given)",
    )
    r.add_argument("--stop-rate", type=float, default=None, help="stop when the incumbent wins this share of the "
                   "acceptance pool (default 0.70; a resumed run keeps its recorded value unless given)")  # fmt: skip
    r.add_argument("--engineer-late-min", type=float, default=None, help="minutes a straggler Engineer gets once half "
                   "the drafts are in, then it is stopped (default 10; 0 waits; a resumed run keeps its value; TODO 37.3)")  # fmt: skip
    for flag, key, text in (("--diag-min-fails", "diag_min_fails", "re-draw the diagnostic pool when the incumbent fails "
                             "fewer than this many of its starts (default 10; 0 never; TODO 33.23)"),
                            ("--diag-target-fails", "diag_target_fails", "... until this many fail (default 15)"),
                            ("--diag-draw-max", "diag_draw_max", "fresh starts a redraw may draw and run (default 90)")):  # fmt: skip
        r.add_argument(flag, dest=key, type=int, default=None, help=text)
    r.add_argument("--stop-after", choices=("probe",), help="replays: end the iteration after the Prober (TODO 32.6)")
    r.add_argument("--effort", action="append", default=[], metavar="ROLE=LEVEL",
                   help="claude --effort for a role (reviewer, manager, engineer, composer), e.g. engineer=medium (TODO 32.8)")  # fmt: skip
    r.add_argument("--engineer-from", type=Path, help="replays: the next iteration takes this recorded iteration's "
                   "hypotheses, probes and Manager decision and runs only the Engineer and the gate (TODO 32.8)")  # fmt: skip
    s = sub.add_parser("status")
    s.add_argument("family")
    d = sub.add_parser("redraw", help="re-draw the diagnostic pool for the present incumbent now (TODO 33.23)")
    d.add_argument("family")
    for flag, key in (("--diag-min-fails", "diag_min_fails"), ("--diag-target-fails", "diag_target_fails"),
                      ("--diag-draw-max", "diag_draw_max")):  # fmt: skip
        d.add_argument(flag, dest=key, type=int, default=None)
    a = ap.parse_args(argv)
    S.configure(a.family)
    return cmd_run(a) if a.cmd == "run" else S.H.cmd_redraw(a) if a.cmd == "redraw" else cmd_status(a)


if __name__ == "__main__":
    raise SystemExit(main())
