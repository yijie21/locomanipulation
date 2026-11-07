"""The Reviewer's digest of the incumbent's diagnostic rounds (plan docs/handoff/scene-rsi-plan-20261016.md §12, user
2026-10-19): what every Reviewer so far rebuilt by hand at the start of its iteration, computed once by the driver.

The Reviewer took 33 min and 8 USD an iteration on average (57% of the inner loops' model cost); `r2_in_single`
iteration 0 made 75 tool calls over 173 turns, and about 30 of them rebuilt the same views: an outcome overview,
per-round step dumps, the objects' and containers' sizes from the scene file, the object's distance to the holding
wrist over the carry. Every turn re-reads the whole conversation, so those turns cost the most. The digest holds:

- first, the failed rounds by failure class (TODO 33.15) and the rounds no class fits;
- per task: the goal, its objects' facts (size, mass, how flat it rests against a Dex1 pad, its support's top; a
  container's inner box, depth and rim; a target surface's top), where the failures first fail, and the step numbers
  that best separate the task's successes from its failures;
- per round: the outcome, the first failing step with its reason, the steps' scalar numbers, and from the trace the
  object's lift, any move before the grasp (knocked), and the change of its distance to the holding wrist between
  the grasp and the release (slip; positions only, a turn of the wrist about the object does not show).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/digest.py <diag dir> <family> [--out <file.md>]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import records as RC  # noqa: E402

from locomani.scene_discovery import lint  # noqa: E402
from locomani.scene_discovery.predicates import parse  # noqa: E402
from locomani.tasks.scene import past as PAST  # noqa: E402

SIDE = {"left": 0, "right": 1}
SKIP_STEPS = {"settle", "hold"}
MAX_FIELDS = 7  # scalar numbers shown per step
KNOCK_CM = 1.0
TOP_SEPARATORS = 6
RELEASE_STEPS = ("release_fit", "release", "unload", "open", "set_down", "drop")  # the hand starts letting go
PAST_MAX_LINES = 115  # TODO 39.4: the scene-graph section stays under a third of the 344-line digest of 2026-10-30
PAST_TIMELINE_MAX = 6  # flips shown per failed round before folding
PAST_TOP = 3  # first unexpected transitions listed per task
QUIET_KINDS = ("none", "walk", "turn", "sidestep", "posture", "hold")  # steps left out of a timeline unless they failed


def scalars(step: dict) -> dict:
    out = {}
    for k, v in step.items():
        if k in ("step", "ok", "why", "t0", "t1") or isinstance(v, (dict, list, str)) or v is None:
            continue
        out[k] = round(float(v), 2) if isinstance(v, float) else v
    return out


def first_fail(r: dict) -> tuple[str, str]:
    for s in r.get("steps") or []:
        if not s.get("ok"):
            return s["step"], s.get("why", "")
    return "", r.get("plan_why") or r.get("why") or ""


def shape(why: str) -> str:
    """A failure reason with its numbers out, to count alike ones together."""
    return re.sub(r"[-+]?\d+(\.\d+)?", "#", why or "")[:90]


def facts(meta: dict, goal: list[str]) -> list[str]:
    """One line per object or surface the goal names, in goal order; a container (`in(_, C)`) gets its inner box on
    its one line."""
    objs, surfs = meta["objects"], meta["surfaces"]
    parsed = [parse(g) for g in goal]
    containers = {p.args[1] for p in parsed if p.name == "in" and len(p.args) > 1}
    out = {}
    for p in parsed:
        for a in p.args:
            if not isinstance(a, str) or a == "robot" or a in out:
                continue
            if a in objs:
                o = objs[a]
                box = o.get("box") or {}
                he = np.asarray(box.get("half_extent") or (0, 0, 0), float)
                s = surfs.get(o.get("surface") or "") or {}
                line = f"- {a}: box {' x '.join(f'{200 * h:.1f}' for h in he)} cm"
                mass = o.get("mass", (o.get("semantic") or {}).get("mass_kg"))
                if mass is not None:
                    line += f", {float(mass):.2f} kg"
                if o.get("free") and he.any():
                    h = lint.resting_half_height(o)
                    line += f", rests {200 * h:.1f} cm tall (pad margin {100 * (h - lint.DEX1_PAD_HALF_M):+.2f} cm)"
                if s:
                    line += f", on {o['surface']} (top {float(s['top']):.3f} m)"
                if a in containers:
                    wall = float(box.get("wall", 0.01))
                    top = float(o["init_pos"][2]) + float((box.get("centre") or (0, 0, 0))[2]) + he[2]
                    line += (f"; container: inner {200 * (he[0] - wall):.1f} x {200 * (he[1] - wall):.1f} cm, "
                             f"depth {100 * (2 * he[2] - wall):.1f} cm, rim at {top:.3f} m")  # fmt: skip
                out[a] = line
            elif a in surfs:
                s = surfs[a]
                out[a] = (f"- {a}: surface, top {float(s['top']):.3f} m, {200 * float(s['half_extent'][0]):.0f} x "
                          f"{200 * float(s['half_extent'][1]):.0f} cm")  # fmt: skip
    return list(out.values())


def trace(npz: Path, r: dict, movables: list[str], meta: dict) -> dict:
    """lift_cm, knocked_cm (moved before the grasp), slip_cm (distance-to-wrist change from grasp to release)."""
    if not npz.exists():
        return {}
    z = np.load(npz, allow_pickle=True)
    t = z["t"]
    steps = {s["step"]: s for s in r.get("steps") or []}
    out = {}
    for x in movables:
        key = f"obj_{(meta['objects'][x].get('body') or x + '_object')}"
        if key not in z.files:
            continue
        p = z[key][:, :3].astype(float)
        i0 = int(np.searchsorted(t, 1.0))
        row = dict(lift_cm=round(100 * float(p[:, 2].max() - p[i0, 2]), 1))
        g = steps.get("grasp")
        if g is not None:
            ig = int(np.searchsorted(t, g["t0"]))
            moved = 100 * float(np.linalg.norm(p[ig, :2] - p[i0, :2]))
            if moved > KNOCK_CM:
                row["knocked_cm"] = round(moved, 1)
            side = SIDE.get((steps.get("plan_ahead") or {}).get("hand") or "", None)
            if side is not None and "wrists" in z.files and g.get("ok"):
                ia = int(np.searchsorted(t, g["t1"]))
                rel = [s["t0"] for s in r["steps"] if s["step"] in RELEASE_STEPS and s["t0"] >= g["t1"]]
                ib = min(int(np.searchsorted(t, min(rel))) if rel else len(t) - 1, len(t) - 1)
                w = z["wrists"][:, side].astype(float)
                d = np.linalg.norm(p - w, axis=1)
                row["slip_cm"] = round(100 * float(d[ib] - d[ia]), 1)
                row["slip_max_cm"] = round(100 * float(np.max(np.abs(d[ia : ib + 1] - d[ia]))), 1)
        out[x] = row
    return out


def separators(rows: list[dict]) -> list[str]:
    """Step numbers that best separate a task's successes from its failures (medians, rank-biserial)."""
    vals = collections.defaultdict(lambda: ([], []))
    for r in rows:
        seen = set()
        for s in r.get("steps") or []:
            for k, v in scalars(s).items():
                key = f"{s['step']}.{k}"
                if key in seen or isinstance(v, bool):
                    continue
                seen.add(key)
                vals[key][0 if r["ok"] else 1].append(float(v))
    out = []
    for key, (ok, bad) in vals.items():
        if len(ok) < 2 or len(bad) < 2:
            continue
        a, b = np.asarray(ok), np.asarray(bad)
        rb = float(np.mean(a[:, None] > b[None, :]) - np.mean(a[:, None] < b[None, :]))
        out.append((abs(rb), key, np.median(a), np.median(b), len(a), len(b), rb))
    out.sort(reverse=True)
    return [f"- {k}: successes {ma:.2f} (n {na}), failures {mb:.2f} (n {nb}), rank-biserial {rb:+.2f}"
            for s, k, ma, mb, na, nb, rb in out[:TOP_SEPARATORS] if s >= 0.5]  # fmt: skip


def round_line(r: dict, tr: dict) -> str:
    flags = "".join(f for f, on in (("F", r.get("fell")), ("D", r.get("dropped"))) if on)
    head = f"{r['label']} {'OK  ' if r['ok'] else 'fail'}{(' ' + flags) if flags else ''} t {r.get('t_end', 0):.0f}s"
    if not r["ok"] and r.get("failure_class"):
        ev = r.get("events") or {}
        sub = ev.get("lift_class") or ev.get("pick_class")  # a pack round's class keeps the lift's or the pick's
        head += f" [{r['failure_class']}{f' < {sub}' if sub else ''}]"
    step, why = first_fail(r) if not r["ok"] else ("", "")
    lines = [head + (f" | first fail {step}: {why}" if step else (f" | {why}" if not r["ok"] else ""))]
    if not r["ok"] and r.get("why"):
        lines.append(f"    goal: {r['why']}")
    for s in r.get("steps") or []:
        if s["step"] in SKIP_STEPS:
            continue
        sc = list(scalars(s).items())[:MAX_FIELDS]
        mark = "" if s.get("ok") else " x"
        lines.append(f"    {s['step']}{mark} [{s.get('t0', 0):.1f}-{s.get('t1', 0):.1f}] " + ", ".join(f"{k} {v}" for k, v in sc))  # fmt: skip
    for x, m in tr.items():
        lines.append(f"    trace {x}: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    return "\n".join(lines)


UNPLACED = ("other", "unknown")  # failure classes the classifier could not name


def class_lines(rounds: list[dict]) -> list[str]:
    """TODO 33.15 (plan section 3.5): the failed rounds by the driver's failure class (`failure_class`, from
    scripts/rsi/failure_class.py; a record without one is "unknown"), then the rounds no class fits, which the Reviewer
    looks at first: the classified ones have a known shape and a standing suggestion."""
    fails = [r for r in rounds if not r.get("ok")]
    if not fails:
        return []
    cls = lambda r: r.get("failure_class") or "unknown"  # noqa: E731
    counts = collections.Counter(cls(r) for r in fails)
    out = ["Failure classes: " + ", ".join(f"{k} {n}" for k, n in counts.most_common()), ""]
    sub = lambda r: (r.get("events") or {}).get("lift_class") or (r.get("events") or {}).get("pick_class")  # noqa: E731
    odd = [r for r in fails if cls(r) in UNPLACED or sub(r) in UNPLACED]  # a pack round's own class may be the odd one
    if odd:
        out += ["Rounds no class fits (look at these first):"]
        out += [f"- {r['label']}: {first_fail(r)[0] or 'goal'}: {shape(first_fail(r)[1])[:90]}" for r in odd] + [""]
    return out


def _edge_text(edge: list) -> str:
    s, r, o = edge
    return f"{s} {r} {o}" if o is not None else f"{s} {r}"


def _first_unexpected(recs: list[dict]) -> tuple | None:
    """(kind, edge, expected, observed) of the first step whose expected transition did not hold."""
    for r in recs:
        if r.get("discrepancies"):
            d = r["discrepancies"][0]
            return r["kind"], tuple(d["edge"]), d.get("to"), d.get("observed")
    return None


def _b(v) -> str:
    return "unknown" if v is None else str(v).lower()


def past_lines(diag: Path, rounds: list[dict]) -> list[str]:
    """TODO 39.4 (plan docs/handoff/past-scene-graph-plan-20261030.md section 4.8): per task, the first unexpected
    transition of the failed rounds with the successes' value at the same kind of step; per failed round, the flip
    timeline. Nothing when the directory has no `.past.jsonl` (an older sandbox)."""
    recs_by = {}
    for r in rounds:
        p = diag / f"{r['label']}.past.jsonl"
        if p.exists():
            recs = [x for x in PAST.read(p) if "record_id" in x]
            if recs:
                recs_by[r["label"]] = recs
    if not recs_by:
        return []
    out = ["## Scene-graph transitions (TODO 39: `past.py show <diag dir>/<label>` for a round, `compare <failed> "
           "<successful>` for two; `expected` is a fixed table by primitive, not a judgement)", ""]  # fmt: skip
    by_task = collections.defaultdict(list)
    for r in rounds:
        if r["label"] in recs_by:
            by_task[r["task"]].append(r)
    for task, rows in sorted(by_task.items()):
        firsts = collections.Counter()
        for r in rows:
            if not r["ok"]:
                f = _first_unexpected(recs_by[r["label"]])
                if f:
                    firsts[f] += 1
        for (kind, edge, to, obs), n in firsts.most_common(PAST_TOP):
            vals = []
            for r in rows:
                if not r["ok"]:
                    continue
                for x in recs_by[r["label"]]:
                    if x["kind"] != kind:
                        continue
                    e = next(
                        (e for e in x.get("G_after", {}).get("edges", []) if (e["s"], e["r"], e["o"]) == edge), None
                    )
                    if e is not None and e["v"] is not None:
                        vals.append(bool(e["v"]) == to)
            tail = f"; successes: {sum(vals)}/{len(vals)} {_b(to)} at the same kind" if vals else ""
            out.append(f"- {task}: {n} x {kind}: {_edge_text(list(edge))} expected {_b(to)}, observed {_b(obs)}{tail}")
    out += ["", "Failed rounds, flips from the first manipulation step (t  step.kind  edge from->to; * = unexpected):"]
    for r in rounds:
        if r["ok"] or r["label"] not in recs_by:
            continue
        lines = []
        for x in recs_by[r["label"]]:
            if x["kind"] in QUIET_KINDS and not x.get("discrepancies"):
                continue
            bad = {tuple(d["edge"]) for d in x.get("discrepancies") or []}
            for e in x.get("events") or []:
                mark = " *" if tuple(e["edge"]) in bad else ""
                lines.append(
                    f"  {e['t']:.1f}  {x['step']}.{x['kind']}  {_edge_text(e['edge'])} {_b(e['from'])}->{_b(e['to'])}{mark}"
                )
            t1 = (x.get("action") or {}).get("t1", 0.0)
            for d in x.get("discrepancies") or []:
                lines.append(f"  {t1:.1f}  {x['step']}.{x['kind']}  {_edge_text(d['edge'])} expected {_b(d['to'])}, "
                             f"observed {_b(d.get('observed'))} *")  # fmt: skip
        if len(lines) > PAST_TIMELINE_MAX:
            lines = lines[:PAST_TIMELINE_MAX] + [
                f"  ... {len(lines) - PAST_TIMELINE_MAX} more: past.py show {diag / r['label']}"
            ]
        if len(out) + len(lines) + 2 > PAST_MAX_LINES:
            out.append("- ... more failed rounds: `past.py show` each")
            break
        out += [f"- {r['label']}"] + lines
    # TODO 39.6: the record-library columns these records add, for a hypothesis's `measure`
    cols = sorted({k for recs in recs_by.values() for k in RC.graph_columns(recs)})
    if cols:
        out += ["", f"Record-library columns from these records ({len(cols)}, keyed by step kind; a hypothesis's "
                    f"`measure` can split the rounds on any of them): {', '.join(cols)}",
                "  e.g. \"measure\": {\"feature\": \"reach.beside.gap_m\", \"op\": \"<\", \"value\": 0.03, \"expect\": \"fails\"}"]  # fmt: skip
    out.append("")
    return out


def build(diag: Path, family: str) -> str:
    import family as F

    fam = F.load(family)
    meta = fam["meta"]
    rounds = sorted((json.loads(p.read_text()) for p in diag.glob("*.json") if not p.name.startswith("_")),
                    key=lambda r: r["label"])  # fmt: skip
    by_task = collections.defaultdict(list)
    for r in rounds:
        by_task[r["task"]].append(r)
    ok = sum(r["ok"] for r in rounds)
    out = [f"# Diagnostic digest: {family}, {diag.name}", "",
           f"{ok}/{len(rounds)} succeed; fell {sum(bool(r.get('fell')) for r in rounds)}, dropped "
           f"{sum(bool(r.get('dropped')) for r in rounds)}. Built by digest.py from the round files in {diag} and "
           "the scene file; open a round's .json / .npz for anything not here.", ""]  # fmt: skip
    out += class_lines(rounds)
    out += past_lines(diag, rounds)  # TODO 39.4
    for task, rows in sorted(by_task.items()):
        goal = fam["rows"][task]["goal"] if task in fam["rows"] else []
        movables = list(F.task(fam, task).objects) if task in fam["rows"] else []
        n_ok = sum(r["ok"] for r in rows)
        out += [f"## {task}: {n_ok}/{len(rows)}", "", f"Goal: {' and '.join(goal)}", ""]
        out += facts(meta, goal) + [""]
        stages = collections.Counter(
            f"{first_fail(r)[0] or 'goal'}: {shape(first_fail(r)[1])}" for r in rows if not r["ok"]
        )
        if stages:
            out += ["First failing step (numbers as #):"] + [f"- {n} x {k}" for k, n in stages.most_common()] + [""]
        sep = separators(rows)
        if sep:
            out += ["Numbers that separate successes from failures:"] + sep + [""]
        out += ["Rounds:", "```"]
        for r in rows:
            out.append(round_line(r, trace(diag / f"{r['label']}.npz", r, movables, meta)))
        out += ["```", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("diag", type=Path)
    ap.add_argument("family")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    text = build(a.diag, a.family)
    if a.out:
        a.out.write_text(text)
        print(f"{len(text)} chars -> {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
