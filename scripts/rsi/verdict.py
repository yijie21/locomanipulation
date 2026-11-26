"""TODO 54 (review 2026-11-25, flow.md F6/F9, acceptance.md F2, verify.md 6 and 8): the one function that says whether
a candidate beat the incumbent, used by the check half (loop.check_candidate), the gate (gate.py, stage 1 and the
full pool) and the sealed set (sealed.confirm). Before, the three held three copies of the rule and the per-task rule
was a fixed "a task net <= -2 rejects" that is far too loose where a task flips half its starts (the faucet) and never
binds where nothing flips (oven, door).

Input: paired outcomes by start (`counts`: records of the candidate and the incumbent, task by the record or the label)
or the tally dict the gate already writes (`b`, `c`, `per_task`, falls and drops); and the family's parameters
(`params`):
    p_max        the sign test's one-sided p on all pairs must not exceed it (the SIMPLE, LML, BiGym, Arena families);
                 None: the scene rule (net >= min_net and c <= b / loss_share)
    min_net      scene families only: with p_max set it can never bind (the smallest p <= 0.05 tally is 5:0)
    loss_share   scene families only
    alpha_task   per-task regression test: a task rejects the candidate when its one-sided exact sign test on that
                 task's discordant pairs (losses c of b + c, more than half) is significant after the Holm correction
                 across the tasks, at this level (0.05 gate and sealed, 0.10 check half); None: no per-task rule
    check_min_net  the check half's net floor (1: it must win more than it loses)
    check_alpha  the check half sends only when the one-sided sign test on its pairs is at most this (0.10 on the
                 benchmarks; None: the net floor alone). 60 starts flip 8-17 % between two runs of one code, so a net of
                 4 is noise (simple_art_r3 iteration 1: 6:2 sent, 8:11 at the gate); 4:0 (p .0625) is sent, 6:2 (p .145)
                 is not, 13:2 and 19:5 (both accepted later) are.
Output (`decide`): {decision, reasons, per_task (n, b, c, net, p_regress, p_holm, regress), b, c, n, net,
p_one_sided}. `kind`: "stage1" (rejects only on net < 0), "check", "gate", "sealed".
"""

from __future__ import annotations

from math import comb

LOSS_SHARE = 3
MIN_NET = 3


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
    return (rnd or {}).get("task") or label.rsplit("_s", 1)[0]


def counts(cand: dict, inc: dict, keys: list[str] | None = None) -> dict:
    """The pairs of two arms' round records ({label: record}) as the tally core: n, b (the candidate wins), c (the
    candidate loses), the successes, falls and drops of each arm, and `per_task` {task: {n, b, c}}."""
    keys = sorted(set(cand) & set(inc)) if keys is None else keys
    per: dict[str, dict] = {}
    for k in keys:
        v = per.setdefault(task_of(k, inc[k]), dict(n=0, b=0, c=0))
        v["n"] += 1
        v["b"] += bool(cand[k]["ok"] and not inc[k]["ok"])
        v["c"] += bool(inc[k]["ok"] and not cand[k]["ok"])
    b, c = sum(v["b"] for v in per.values()), sum(v["c"] for v in per.values())
    return dict(n=len(keys), candidate_ok=sum(bool(cand[k]["ok"]) for k in keys),
                incumbent_ok=sum(bool(inc[k]["ok"]) for k in keys), b=b, c=c, p_one_sided=round(sign_p(b, c), 5),
                candidate_fell=sum(bool(cand[k].get("fell")) for k in keys),
                incumbent_fell=sum(bool(inc[k].get("fell")) for k in keys),
                candidate_dropped=sum(bool(cand[k].get("dropped")) for k in keys),
                incumbent_dropped=sum(bool(inc[k].get("dropped")) for k in keys),
                per_task=dict(sorted(per.items())))  # fmt: skip


def task_tests(per_task: dict, alpha: float | None) -> dict:
    """Per task: wins, losses, net, the one-sided exact p that the candidate is WORSE on it (c losses of b + c), the
    Holm-adjusted p across the tasks, and `regress`: significant at `alpha` (False without one)."""
    pv = {t: sign_p(v["c"], v["b"]) for t, v in per_task.items()}
    h = holm(pv, alpha if alpha is not None else 0.0) if pv else {}
    return {t: dict(n=v.get("n", v["b"] + v["c"]), b=v["b"], c=v["c"], net=v["b"] - v["c"], p_regress=round(pv[t], 6),
                    p_holm=h[t]["p_holm"], regress=bool(alpha is not None and h[t]["reject"]))
            for t, v in sorted(per_task.items())}  # fmt: skip


def drop_limit(t: dict) -> int:
    """The most drops a candidate may have on the pool: the incumbent's + 1, or, against an incumbent with no success,
    half the candidate's successes when that is more (user 2026-10-18)."""
    limit = t["incumbent_dropped"] + 1
    return max(limit, t["candidate_ok"] // 2) if t["incumbent_ok"] == 0 else limit


def decide(t: dict, kind: str = "gate", params: dict | None = None) -> dict:
    """The verdict on a tally `t` (see `counts`; the gate's own tally has the same keys) of `kind`."""
    params = params or {}
    b, c = t["b"], t["c"]
    net = b - c
    p = t.get("p_one_sided")
    p = sign_p(b, c) if p is None else p
    tasks = task_tests(t.get("per_task") or {}, params.get("alpha_task"))
    why: list[str] = []
    if kind == "stage1":  # a screen on a third of the pool: only a net loss rejects (it was net <= 0)
        if net < 0:
            why.append(f"net wins {net} < 0")
    else:
        p_max = params.get("p_max")
        if kind == "check":
            floor = int(params.get("check_min_net", 1))
            if net < floor:
                why.append(f"net wins {net} < {floor}")
            alpha = params.get("check_alpha")
            if alpha is not None and net >= floor and p > alpha:
                why.append(f"sign test p {p} > {alpha} (wins {b}, losses {c})")
        elif p_max is None:  # the scene rule; with p_max the net floor can never bind (smallest p <= 0.05: 5:0)
            floor = int(params.get("min_net", MIN_NET))
            if net < floor:
                why.append(f"net wins {net} < {floor}")
            if c * int(params.get("loss_share", LOSS_SHARE)) > b:
                why.append(f"losses {c} > wins {b} / {params.get('loss_share', LOSS_SHARE)}")
        if kind != "check" and p_max is not None and p > p_max:
            why.append(f"sign test p {p} > {p_max} (wins {b}, losses {c})")
        if kind == "gate":
            if t["candidate_fell"] > t["incumbent_fell"] + 1:
                why.append(f"falls {t['candidate_fell']} > {t['incumbent_fell']} + 1")
            if t["candidate_dropped"] > drop_limit(t):
                why.append(f"drops {t['candidate_dropped']} > {drop_limit(t)}")
        for task, v in tasks.items():
            if v["regress"]:
                why.append(f"task {task} regresses: wins {v['b']}, losses {v['c']}, one-sided p {v['p_regress']} "
                           f"(Holm {v['p_holm']}) <= alpha {params['alpha_task']}")  # fmt: skip
    return dict(decision=not why, reasons=why, per_task=tasks, b=b, c=c, n=t.get("n", b + c), net=net, p_one_sided=p)


def judge(cand: dict, inc: dict, kind: str = "gate", params: dict | None = None) -> dict:
    """`decide` on the pairs of two arms' round records."""
    return decide(counts(cand, inc), kind, params)
