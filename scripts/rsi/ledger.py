"""The loop's ledger and what a role reads of it: ledger rows, the context file, the history cap, the iteration line."""

from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "src")]
import drafts as DR  # noqa: E402
import state as S  # noqa: E402
import stop as ST  # noqa: E402
from gate import GATE_RULE  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jload(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def jdump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def ledger(entry: dict) -> None:
    S.STATE.mkdir(parents=True, exist_ok=True)
    with (S.STATE / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps({"t": now(), **entry}, ensure_ascii=False, default=str) + "\n")


def entries() -> list[dict]:
    p = S.STATE / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def run_cost(es: list[dict], meta: dict | None) -> float:
    """TODO 36 (review 2026-10-22, L6): cost of THIS run -- ledger rows from meta['started'] on; rows a replay copied
    from its source run (older `t`) are not this run's spending."""
    t0 = (meta or {}).get("started") or ""
    return sum(e.get("cost_usd", 0.0) for e in es if str(e.get("t", "")) >= t0)


def run_gates(es: list[dict], meta: dict | None) -> list[dict]:
    """TODO 33.18.2: the gate calls of THIS run, not void -- rows from meta['started'] on, like run_cost; a replay.py
    setup copies the source run's gate rows (run C started with 9 of 12 spent and stopped after 3)."""
    t0 = (meta or {}).get("started") or ""
    return [e for e in es if e.get("step") == "gate" and not e.get("void") and str(e.get("t", "")) >= t0]


def task_of(label: str) -> str:
    return label.rsplit("_s", 1)[0]


def _cut(x, k: int) -> str:
    t = " ".join(str(x if x is not None else "").split())
    return t if len(t) <= k else t[: k - 3] + "..."


ROW_CHARS = 1500  # TODO 54 phase 2: the longest string a ledger row keeps in a role's history


def _cap_row(v, n: int = ROW_CHARS):
    """A ledger row with every string longer than `n` characters cut (the full row stays in the ledger file)."""
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + f"... [{len(v) - n} more characters]"
    if isinstance(v, list):
        return [_cap_row(x, n) for x in v]
    if isinstance(v, dict):
        return {k: _cap_row(x, n) for k, x in v.items()}
    return v


def compact_history(es: list[dict], full: int = S.HISTORY_FULL) -> list:
    """TODO 54 phase 2 (review rules_prompts.md F12: the history was 60-75 % of every context and grew by an iteration's
    line each round): the ledger rows of the last `full` iterations as they are (without `usage`, long strings cut) and
    nothing older; one line says how many iterations are left out, because `mechanisms` carries what each earlier
    hypothesis came to. The size stays flat however long the run is. Rows outside an iteration (a stop, a resume) count
    when they come after the first kept row."""
    rows = [{k: v for k, v in e.items() if k != "usage"} for e in es]
    iters = sorted({e["iter"] for e in rows if isinstance(e.get("iter"), int)})
    keep = set(iters[-full:]) if full > 0 else set()
    cut = next((i for i, e in enumerate(rows) if e.get("iter") in keep), len(rows))
    older = [x for x in iters if x not in keep]
    head = (
        [
            f"{len(older)} earlier iteration(s) ({older[0]}-{older[-1]}) are not listed: `mechanisms` holds what each hypothesis came to"
        ]
        if older
        else []
    )
    return head + [_cap_row(e) for e in rows[cut:]]


def context(inc: dict, extra: dict, compact: bool = True) -> dict:
    """The role's context file. `compact` (TODO 36.7 for the Manager, 37.2 for every role): the last HISTORY_FULL
    iterations' ledger rows in full, older iterations one line each (compact_history); `mechanisms` is the index of
    every hypothesis so far (mechanism_table)."""
    es = entries()
    meta = jload(S.STATE / "meta.json", {})
    # TODO 36 (review 2026-10-22, L6) and 33.18.2: this run's cost and gate calls only (not rows a replay copied), and
    # the wall clock the Manager used to compute from meta.json by hand
    wall = dict(wall_h=round(ST.wall_h(meta), 2), max_wall_h=S.MAX_WALL_H) if "started_s" in meta else {}
    return dict(
        family=S.FAMILY,
        incumbent=inc,
        budget=dict(
            gate_calls=len(run_gates(es, meta)),
            gate_calls_max=S.MAX_GATE_CALLS,
            cost_usd=round(run_cost(es, meta), 2),
            cost_cap_usd=S.COST_CAP,
            stop_rate=S.STOP_RATE,
            **wall,
        ),  # fmt: skip
        history=compact_history(es) if compact else [{k: v for k, v in e.items() if k != "usage"} for e in es],
        mechanisms=mechanism_table(es),
        # TODO 33.8: rejections under the present gate rule; one an older rule rejected may be judged again
        rejected_hashes=sorted(
            {
                e["hash"]
                for e in es
                if e.get("step") == "gate"
                and not e.get("accepted")
                and not e.get("void")
                and e.get("hash")
                and e.get("rule", 1) == GATE_RULE
            }
        ),
        **extra,
    )


def gate_lines() -> list[str]:
    p = S.GATE_DIR / "ledger.jsonl"
    return p.read_text().splitlines() if p.exists() else []


def hyp_key(n: int, hid: str) -> str:
    return f"{int(n):02d}:{hid}"


def norm_key(x: str) -> str | None:
    """'1:H2', '01:H2', 'iter 1:H2', '1/H2' -> '01:H2'."""
    m = re.match(r"^\s*(?:iter(?:ation)?[_ ]?)?(\d+)\s*[:/ ]\s*(\S+)\s*$", str(x))
    return hyp_key(int(m.group(1)), m.group(2)) if m else None


def mechanism_links(es: list[dict]) -> tuple[dict[str, set[str]], set[str], dict[str, str]]:
    """The Manager's `repeats` links as groups of hypothesis keys (`NN:id`), the keys an Engineer built, and each
    key's group root (keys never linked are their own group). Shared by recurring() and mechanism_table()."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    built = set()
    for e in es:
        if not isinstance(e.get("iter"), int) or e.get("void"):
            continue
        n = e["iter"]
        if e.get("step") == "manager":
            for r in e.get("repeats") or []:
                a = norm_key(f"{n}:{r.get('now', '')}") if isinstance(r, dict) else None
                for b in (r.get("same_as") or []) if a else []:
                    b = norm_key(b)
                    if b:
                        parent[find(a)] = find(b)
            if e.get("decision") == "engineer":
                built |= {hyp_key(n, h) for h in (e.get("hypotheses") or [e.get("hypothesis")]) if h}
        elif e.get("step") == "engineer" and e.get("hypothesis"):
            built.add(hyp_key(n, e["hypothesis"]))
    groups = collections.defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    return dict(groups), built, {x: find(x) for x in list(parent)}


def mechanism_table(es: list[dict]) -> list[dict]:
    """TODO 37.2 (user 2026-10-23): one row per hypothesis so far -- the index of what was tried, for every role. Run
    C's Reviewer read the whole ledger (203-255 KB of `history`, 170-210k tokens of context per turn) and paged
    through it with scripts; the Manager alone had the compacted history (36.7). Per row: the iteration and id, the
    claim and `instead`, the probe's status and counts, the Manager's `repeats` link (`same_as`) or `skip`, what was
    built from it (each draft's diagnostic flips, void, duplicate, late), the Composer's and the gate's verdicts when
    its draft went on, and `open`: nothing of its mechanism (its linked group) was ever built and the probe did not
    refute it. Hypotheses only; parks and reopens stay in `history`."""
    groups, built, root = mechanism_links(es)
    iters = sorted({e["iter"] for e in es if isinstance(e.get("iter"), int)})
    out = []
    for n in iters:
        rows = [e for e in es if e.get("iter") == n]
        hyps = (jload(S.STATE / f"iter_{n:02d}" / "hypotheses.json", {}) or {}).get("hypotheses") or []
        cands = {c.get("number"): c for c in (jload(S.STATE / f"iter_{n:02d}" / "candidates.json", []) or [])
                 if isinstance(c, dict)}  # fmt: skip
        probe = {r.get("id"): r for e in rows if e.get("step") == "probe"
                 for r in e.get("results") or [] if isinstance(r, dict)}  # fmt: skip
        man = next((e for e in reversed(rows) if e.get("step") == "manager" and not e.get("void")), None)
        single = man.get("hypothesis") if man and man.get("decision") == "engineer" else None
        eng = [e for e in rows if e.get("step") == "engineer"]
        comp = next((e for e in rows if e.get("step") == "compose"), None)
        gates = [e for e in rows if e.get("step") == "gate" and not e.get("void")]
        repeats = {r.get("now"): r.get("same_as") for r in (man or {}).get("repeats") or [] if isinstance(r, dict)}
        skip = (man or {}).get("skip") or {}
        for h in hyps:
            if not isinstance(h, dict) or "id" not in h:
                continue
            hid = h["id"]
            key = hyp_key(n, hid)
            row = dict(iter=n, id=hid, claim=_cut(h.get("claim"), 160), instead=_cut(h.get("instead"), 100),
                       status=h.get("status") or probe_status(h, probe.get(hid)))  # fmt: skip
            p = probe.get(hid)
            if p and "fixed" in p:
                row["probe"] = (f"{p.get('ok_before')}->{p.get('ok_after')}/{p.get('n')}, fixed {len(p['fixed'])} "
                                f"broken {len(p.get('broken') or [])}")  # fmt: skip
            elif p and p.get("error"):
                row["probe"] = "error: " + _cut(p["error"], 80)
            if repeats.get(hid):
                row["same_as"] = repeats[hid]
            if skip.get(hid):
                row["skip"] = skip[hid]
            mine = [e for e in eng if e.get("hypothesis") == hid or (e.get("hypothesis") is None and hid == single)]
            drafts = []
            for e in mine:
                k = e.get("candidate")
                c = cands.get(k) if k is not None else None
                if e.get("late"):
                    d = "late: stopped before it answered"
                elif e.get("void"):
                    d = "void" + (f" ({_cut(e.get('error'), 60)})" if e.get("error") else "")
                elif e.get("duplicate_of"):
                    d = f"duplicate of draft {e['duplicate_of']}"
                elif c and "fixed" in c:
                    d = f"fixed {len(c['fixed'])} broken {len(c['broken'])}"
                else:
                    d = "built"
                drafts.append((f"draft {k}: " if k is not None else "") + d)
            if drafts:
                row["built"] = "; ".join(drafts)
            ks = {e.get("candidate") for e in mine}
            went_on = bool(mine) and (comp is None or comp.get("decision") == "merge"
                                      or (comp.get("decision") == "adopt" and comp.get("adopted") in ks))  # fmt: skip
            if comp and mine:
                row["composer"] = ("merge" if comp.get("decision") == "merge" else
                                   "adopted" if went_on else "kept the incumbent" if comp.get("decision") == "keep"
                                   else "not adopted")  # fmt: skip
            gate = gates[0] if gates else None
            if gate and went_on:
                r = gate.get("result") or {}
                row["gate"] = ("accepted" if gate.get("accepted") else "rejected") + (
                    f" b {r['b']} c {r['c']}" if "b" in r and "c" in r else ""
                )
            members = groups.get(root.get(key, key), {key})
            row["open"] = not (members & built) and row["status"] != "refuted"
            out.append(row)
    return out


def probe_status(h: dict, r: dict | None) -> str:
    """TODO 33.17 (plan section 3.4, D14): a hypothesis after its probe. `verified`: the probe fixed a start and broke
    none; `refuted`: it fixed none; `mixed`: it fixed some and broke some. One with no parameter delta (its test needs
    code, `needs_code`) or whose probe failed to run stays `hypothesis`: nothing was measured."""
    if r and r.get("status") == "untested":  # `needs_code`: the probe did not run it
        return "untested"
    if not ((h.get("delta") or {}).get("set") or DR.sweep_specs(h)) or not r or "error" in r or "fixed" not in r:
        return "hypothesis"
    fixed, broken = len(r.get("fixed") or []), len(r.get("broken") or [])
    if fixed == 0:
        return "refuted"
    return "verified" if broken == 0 else "mixed"
