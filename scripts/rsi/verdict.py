"""TODO 56.1 (user 2026-11-27, from main's TODO 54 B6 / B7): the one function that says whether a candidate beat the
incumbent, used by the gate (stage 1, the full pool, the `sync` rule) and the sealed set (the final against the seed).
Before, gate.py and sealed.py each held a copy of the rule and no per-task view.

Input: a tally `t` as gate.tally writes it (`b` wins, `c` losses over the shared starts, the successes, falls and drops
of each arm, `per_task` {task: {n, b, c}}) or two arms' round records ({label: record}, `judge`); and `kind`:
    stage1   the screen on 30 starts: rejected iff net wins b - c <= 0 (the scene gate's rule since TODO 27; main
             rejects on < 0 only -- not changed here)
    gate     the full pool: b - c >= MIN_NET, c <= b / LOSS_SHARE, falls <= the incumbent's + 1, drops <= drop_limit
             (gate rule 2, TODO 33.8; `params` may set min_net / loss_share)
    sync     scripts/rsi/alternate.py's non-inferior rule: c <= b
    sealed   the final incumbent against the seed on the sealed pool: the gate's net and loss-share conditions,
             reported (sealed.py judges nothing)
Per task, every kind reports a regression test (`per_task`): the one-sided exact sign test that the candidate is WORSE
on the task (c losses of its b + c discordant pairs), Holm-corrected across the tasks at ALPHA_TASK, and `regress`,
whether that is significant. It is a report, never a reason to reject (user 2026-11-27): the gso families have 10
acceptance starts a task, so after Holm across 20 tasks only a near-total loss on one task could ever show; the
figure is there for the human and the roles to read (gate.json, the ledger's gate rows, the sealed summary).

Output (`decide`): {decision, reasons, per_task {task: {n, b, c, net, p_regress, p_holm, regress}}, b, c, n, net,
p_one_sided}. Pure functions: no files, no loop state.
"""

from __future__ import annotations

from math import comb

MIN_NET = 3  # gate rule 2 (TODO 33.8): the candidate must win at least this many more starts than it loses
LOSS_SHARE = 3  # and lose at most a third of what it wins (c <= b / LOSS_SHARE)
ALPHA_TASK = 0.05  # the per-task regression report's level (Holm across the tasks)


def sign_p(b: int, c: int) -> float:
    """One-sided exact sign (McNemar) p of b wins against c losses: P(X >= b), X ~ Binomial(b + c, 1/2)."""
    n = b + c
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(b, n + 1)) / 2**n


def holm(pvals: dict[str, float], alpha: float) -> dict[str, dict]:
    """Holm's step-down correction: {key: {p, p_holm, reject}}. The smallest p is held against alpha / m, the next
    against alpha / (m - 1), and so on; the first one that fails ends the rejections. `p_holm` is the adjusted p (the
    running maximum of (m - rank) * p, at most 1)."""
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m, out, running, rejecting = len(order), {}, 0.0, True
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * pvals[k])
        running = max(running, adj)
        rejecting = rejecting and pvals[k] <= alpha / (m - i)
        out[k] = dict(p=pvals[k], p_holm=round(running, 6), reject=rejecting)
    return out


def task_of(label: str, rnd: dict | None = None) -> str:
    """A round's task: the record's `task`, else the label without its `_s<seed>` tail."""
    return (rnd or {}).get("task") or label.rsplit("_s", 1)[0]


def rank_ok(r: dict) -> int:
    """A round's rank for the paired comparison without stages: 1 a success, 0 a failure (gate.stage_of ranks a pack
    family's rounds 0-4 instead; the gate passes it as `rank`)."""
    return 1 if r["ok"] else 0


def per_task_counts(cand: dict, inc: dict, keys: list[str], rank=rank_ok) -> dict[str, dict]:
    """{task: {n, b, c}} over `keys`: b the starts the candidate ranks higher on, c lower (`rank`: rank_ok or
    gate.stage_of)."""
    per: dict[str, dict] = {}
    for k in keys:
        v = per.setdefault(task_of(k, inc[k]), dict(n=0, b=0, c=0))
        v["n"] += 1
        v["b"] += rank(cand[k]) > rank(inc[k])
        v["c"] += rank(cand[k]) < rank(inc[k])
    return dict(sorted(per.items()))


def task_tests(per_task: dict, alpha: float = ALPHA_TASK) -> dict:
    """Per task: wins, losses, net, the one-sided exact p that the candidate is WORSE on it (c losses of b + c), the
    Holm-adjusted p across the tasks, and `regress`: significant at `alpha` (False when alpha is None)."""
    pv = {t: sign_p(v["c"], v["b"]) for t, v in per_task.items()}
    h = holm(pv, alpha if alpha is not None else 0.0) if pv else {}
    return {t: dict(n=v.get("n", v["b"] + v["c"]), b=v["b"], c=v["c"], net=v["b"] - v["c"], p_regress=round(pv[t], 6),
                    p_holm=h[t]["p_holm"], regress=bool(alpha is not None and h[t]["reject"]))
            for t, v in sorted(per_task.items())}  # fmt: skip


def counts(cand: dict, inc: dict, keys: list[str] | None = None, rank=rank_ok) -> dict:
    """The pairs of two arms' round records ({label: record}) as a tally: n, b, c, the successes, falls and drops of
    each arm, `p_one_sided` and `per_task` (task_tests). Without stages a win is a success against a failure."""
    keys = sorted(set(cand) & set(inc)) if keys is None else keys
    per = per_task_counts(cand, inc, keys, rank)
    b, c = sum(v["b"] for v in per.values()), sum(v["c"] for v in per.values())
    return dict(n=len(keys), candidate_ok=sum(bool(cand[k]["ok"]) for k in keys),
                incumbent_ok=sum(bool(inc[k]["ok"]) for k in keys), b=b, c=c, p_one_sided=round(sign_p(b, c), 5),
                candidate_fell=sum(bool(cand[k].get("fell")) for k in keys),
                incumbent_fell=sum(bool(inc[k].get("fell")) for k in keys),
                candidate_dropped=sum(bool(cand[k].get("dropped")) for k in keys),
                incumbent_dropped=sum(bool(inc[k].get("dropped")) for k in keys),
                per_task=task_tests(per))  # fmt: skip


def drop_limit(t: dict) -> int:
    """The most drops a candidate may have on the pool: the incumbent's + 1, or, against an incumbent with no success,
    half the candidate's successes when that is more (user 2026-10-18)."""
    limit = t["incumbent_dropped"] + 1
    return max(limit, t["candidate_ok"] // 2) if t["incumbent_ok"] == 0 else limit


def decide(t: dict, kind: str = "gate", params: dict | None = None) -> dict:
    """The verdict on a tally `t` of `kind` (module docstring). `reasons` is empty when it is accepted; `per_task` is
    the regression report (task_tests), which never adds a reason."""
    params = params or {}
    b, c = t["b"], t["c"]
    net = b - c
    p = t.get("p_one_sided")
    p = sign_p(b, c) if p is None else p
    per = t.get("per_task") or {}
    reported = bool(per) and "p_regress" in next(iter(per.values()))  # gate.tally already ran task_tests
    tasks = per if reported else task_tests(per, params.get("alpha_task", ALPHA_TASK))
    why: list[str] = []
    if kind == "stage1":
        if net <= 0:
            why.append(f"stage 1: net wins {net} <= 0 on {t.get('n', b + c)} pairs")
    elif kind == "sync":
        if c > b:
            why.append(f"losses {c} > wins {b}")
    else:  # gate, sealed: the scene rule
        floor, share = int(params.get("min_net", MIN_NET)), int(params.get("loss_share", LOSS_SHARE))
        if net < floor:
            why.append(f"net wins {net} < {floor}")
        if c * share > b:
            why.append(f"losses {c} > wins {b} / {share}")
        if kind == "gate":
            if t["candidate_fell"] > t["incumbent_fell"] + 1:
                why.append(f"falls {t['candidate_fell']} > {t['incumbent_fell']} + 1")
            if t["candidate_dropped"] > drop_limit(t):
                why.append(f"drops {t['candidate_dropped']} > {drop_limit(t)}")
    return dict(decision=not why, reasons=why, per_task=tasks, b=b, c=c, n=t.get("n", b + c), net=net, p_one_sided=p)


def judge(cand: dict, inc: dict, kind: str = "gate", params: dict | None = None, rank=rank_ok) -> dict:
    """`decide` on the pairs of two arms' round records."""
    return decide(counts(cand, inc, rank=rank), kind, params)
