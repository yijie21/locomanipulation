"""The Reviewer's digest of the incumbent's diagnostic rounds (plan docs/handoff/scene-rsi-plan-20261016.md §12, user
2026-10-19): what every Reviewer so far rebuilt by hand at the start of its iteration, computed once by the driver.

The Reviewer took 33 min and 8 USD an iteration on average (57% of the inner loops' model cost); `r2_in_single`
iteration 0 made 75 tool calls over 173 turns, and about 30 of them rebuilt the same views: an outcome overview,
per-round step dumps, the objects' and containers' sizes from the scene file, the object's distance to the holding
wrist over the carry. Every turn re-reads the whole conversation, so those turns cost the most. The digest holds:

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

from locomani.scene_discovery import lint  # noqa: E402
from locomani.scene_discovery.predicates import parse  # noqa: E402

SIDE = {"left": 0, "right": 1}
SKIP_STEPS = {"settle", "hold"}
MAX_FIELDS = 7  # scalar numbers shown per step
KNOCK_CM = 1.0
TOP_SEPARATORS = 6
RELEASE_STEPS = ("release_fit", "release", "unload", "open", "set_down", "drop")  # the hand starts letting go


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
    objs, surfs = meta["objects"], meta["surfaces"]
    out = []
    for g in goal:
        p = parse(g)
        for i, a in enumerate(p.args):
            if not isinstance(a, str) or a == "robot":
                continue
            if a in objs:
                o = objs[a]
                box = o.get("box") or {}
                he = np.asarray(box.get("half_extent") or (0, 0, 0), float)
                s = surfs.get(o.get("surface") or "") or {}
                line = f"- {a}: box {' x '.join(f'{200 * h:.1f}' for h in he)} cm"
                if o.get("mass") is not None:
                    line += f", {float(o['mass']):.2f} kg"
                if o.get("free") and he.any():
                    h = lint.resting_half_height(o)
                    line += f", rests {200 * h:.1f} cm tall (pad margin {100 * (h - lint.DEX1_PAD_HALF_M):+.2f} cm)"
                if s:
                    line += f", on {o['surface']} (top {float(s['top']):.3f} m)"
                if p.name == "in" and i == 1:
                    wall = float(box.get("wall", 0.01))
                    top = float(o["init_pos"][2]) + float((box.get("centre") or (0, 0, 0))[2]) + he[2]
                    line += (f"; container: inner {200 * (he[0] - wall):.1f} x {200 * (he[1] - wall):.1f} cm, "
                             f"depth {100 * (2 * he[2] - wall):.1f} cm, rim at {top:.3f} m")  # fmt: skip
                out.append(line)
            elif a in surfs:
                s = surfs[a]
                out.append(f"- {a}: surface, top {float(s['top']):.3f} m, {200 * float(s['half_extent'][0]):.0f} x "
                           f"{200 * float(s['half_extent'][1]):.0f} cm")  # fmt: skip
    return list(dict.fromkeys(out))


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
