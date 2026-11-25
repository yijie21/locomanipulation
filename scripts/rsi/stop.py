"""When the loop stops (stop rules, no progress, wall clock) and what runs at the end (hit rates, the sealed set)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "src")]
import batch as B  # noqa: E402
import cascade as CS  # noqa: E402
import ledger as LG  # noqa: E402
import state as S  # noqa: E402


def incumbent_ok_here(es: list[dict]) -> int | None:
    """TODO 36 (review 2026-10-22, L12): the incumbent's successes on the CURRENT acceptance pool, from the latest gate
    call that was judged on it (rows carry `pool`): the candidate's count if that call accepted, the incumbent's if it
    rejected. None when no such call exists (a rule-26 redraw since, or rows from before `pool` was recorded) -- the
    caller then keeps the last accepted count."""
    pool = str(CS.accept_pool())
    for e in reversed(es):
        r = e.get("result")
        if e.get("step") == "gate" and not e.get("void") and e.get("pool") == pool and r:
            return r["candidate_ok"] if e.get("accepted") else r["incumbent_ok"]
    return None


def voided_in_a_row(es: list[dict]) -> list[int]:
    """The latest iterations, oldest first, that the audit voided (`void_iter` rows), counted back one by one from the
    newest iteration in the ledger until one that was not voided or has no rows. An iteration a re-probe (`void`) or
    re-audit (`reaudit`) row takes up again after its void_iter row is not void any more."""
    status: dict[int, bool] = {}
    for e in es:
        if "iter" not in e or not isinstance(e["iter"], int):
            continue
        n = e["iter"]
        if e.get("step") == "void_iter":
            status[n] = True
        elif e.get("step") in ("void", "reaudit"):
            status[n] = False
        else:
            status.setdefault(n, False)
    out, n = [], max(status, default=-1)
    while status.get(n):
        out.append(n)
        n -= 1
    return out[::-1]


def stop_reason(meta: dict) -> str | None:
    es = LG.entries()
    voided = voided_in_a_row(es)
    if len(voided) >= S.MAX_VOID_STREAK:  # TODO 36.7: an abnormal stop (is_abnormal)
        return f"audit: iterations {', '.join(map(str, voided))} voided in a row (roles named off-limits paths)"
    accepted = [e for e in es if e.get("step") == "gate" and e.get("accepted") and not e.get("void")]
    n_accept = len(B.pool_labels(CS.accept_pool()))
    here = incumbent_ok_here(es)
    if here is not None:
        if here >= S.STOP_RATE * n_accept:
            return f"the incumbent wins {here}/{n_accept} >= {S.STOP_RATE:.0%}: sealed set"
    elif accepted and accepted[-1]["result"]["candidate_ok"] >= S.STOP_RATE * n_accept:
        return (
            f"the incumbent wins {accepted[-1]['result']['candidate_ok']}/{n_accept} >= {S.STOP_RATE:.0%}: sealed set"
        )
    used_up = no_progress_stop(es, meta)  # TODO 54
    if used_up:
        return used_up
    gates = LG.run_gates(es, meta)
    if len(gates) >= S.MAX_GATE_CALLS:
        return f"gate calls {len(gates)} >= {S.MAX_GATE_CALLS}"
    if wall_h(meta) >= S.MAX_WALL_H:
        return f"wall clock >= {S.MAX_WALL_H} h"
    cost = LG.run_cost(es, meta)
    if S.COST_CAP is not None and cost >= S.COST_CAP:
        return f"model cost {cost:.0f} >= {S.COST_CAP:.0f} USD"
    return S.H.stop_reason(es)  # the scene line: a Manager `stop`, the starts outside the parked tasks


def progressed(es: list[dict], n: int) -> bool:
    """Iteration n produced a candidate that passed the check half; for a family without a check half, one that passed
    the gate's stage 1 (an iteration whose Composer kept the incumbent produced none)."""
    if S.cap("check_half"):
        return any(e.get("step") == "check" and e.get("iter") == n and e.get("sent") for e in es)
    return any(e.get("step") == "gate" and e.get("iter") == n and not e.get("void") and e.get("stage1")
               and not str(e.get("reason", "")).startswith("stage 1") for e in es)  # fmt: skip


def no_progress_stop(es: list[dict], meta: dict | None) -> str | None:
    """TODO 54 (review 2026-11-25, records F1 and flow F5: 45-61 % of the spend came after the last accepted candidate):
    caps `stop_no_progress` N -- the last N iterations of this run each ended without a candidate that passed the check
    half (without one that passed the gate's stage 1 where the family has no check half). An iteration whose Composer
    kept the incumbent counts; a voided one does not. The run then ends through finish(), so the sealed set runs."""
    k = int(S.cap("stop_no_progress", 0) or 0)
    t0 = (meta or {}).get("started") or ""
    ran = sorted({e["iter"] for e in es if e.get("step") == "review" and not e.get("void") and isinstance(e.get("iter"), int)
                  and str(e.get("t", "")) >= t0})  # fmt: skip
    if k and len(ran) >= k and not any(progressed(es, i) for i in ran[-k:]):
        what = "the check half" if S.cap("check_half") else "the gate's stage 1"
        return f"no progress: iterations {', '.join(map(str, ran[-k:]))} produced no candidate that passed {what} ([check-half])"
    return None


def wall_h(meta: dict) -> float:
    """Hours the run has been running: since its start, less the time it stood between a stop and a resume."""
    return (time.time() - meta["started_s"] - meta.get("paused_s", 0.0)) / 3600


BOOKKEEPING_STEPS = (
    "stop",
    "settings",
    "diag_pool",
    "sealed",
)  # rows a stopped run's ledger can gain without an iteration


def is_abnormal(why: str) -> bool:
    """A stop that is a failure, not a finish: a role call that failed, a gate error, the audit voiding
    MAX_VOID_STREAK iterations in a row (TODO 36.7; "named off-limits" is the reason older runs stopped with)."""
    return why.startswith(("role call failed", "gate error", "audit:")) or "named off-limits" in why


def write_hit_rate() -> None:
    """TODO 48.19.3: the roles' hit rates (hit_rate.py) to loop/hit_rate.json; never stops the run."""
    try:
        import hit_rate

        LG.jdump(S.STATE / "hit_rate.json", hit_rate.compute(S.STATE))
    except Exception as e:  # bookkeeping only
        print(f"hit rate not written: {e!r}"[:300])


def run_sealed(why: str, stub: bool) -> None:
    """TODO 48.19.2 (user 2026-11-23, "三条都加上"; [sealed]), TODO 54: every run that ends through finish() (a stop
    rule, --max-iters, the wall clock, a failed role or gate) runs the sealed evaluation once for its final incumbent (sealed.py: the final incumbent and the seed on the sealed pool, in the gate's environment, and
    its confirmation against the seed). Its numbers stay in the hidden tree (<hidden>/sealed/summary.json, history.jsonl);
    the ledger records only that it ran. Off for a stub run, a family without caps sealed_at_stop,
    a run with nothing accepted, and a final incumbent the sealed history already holds (a crash or SIGTERM never
    reaches finish)."""
    if stub or not S.cap("sealed_at_stop"):
        return
    inc = LG.jload(S.GATE_DIR / "incumbent.json")
    if not inc or Path(inc["dir"]).name == "incumbent_seed":
        return
    final = Path(inc["dir"]).parent.name  # <call>_<hash>
    hist = S.HIDDEN / "sealed" / "history.jsonl"
    if hist.exists() and any(json.loads(x).get("final") == final for x in hist.read_text().splitlines() if x.strip()):
        return
    (S.HIDDEN / "sealed").mkdir(parents=True, exist_ok=True)
    log = S.HIDDEN / "sealed" / f"run_{final}.log"
    print(f"sealed evaluation of {final} (log in the hidden tree)", flush=True)
    t0 = LG.now()
    with log.open("a") as fh:
        rc = subprocess.run([str(S.SIM_PY), str(S.HERE / "sealed.py"), S.FAMILY, "--procs", str(S.PROCS), "--gpus", S.GPUS],
                            cwd=S.REPO, env=dict(os.environ, PYTHONPATH="src"), stdout=fh, stderr=subprocess.STDOUT,
                            check=False).returncode  # fmt: skip
    LG.ledger(dict(step="sealed", final=final, started=t0, rc=rc))
