"""Rows -> rounds and cards, the part every adapter shares (TODO 54 phase 3). What a bed record means (ok, the failure class,
the steps, the card text, which fields may be stored) is the adapter's own, in bench_simple.py / bench_lml.py / bench_seeded.py;
the round's shape, a round that never ran, the steps from a phase list, and the cards file are written once here.

A stored round keeps the raw record under the adapter's own key (`simple`, `lml`, `seeded`): that is what old ledgers hold and
what `adapter_module_of` reads back to pick the card text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bench_common as C

HERE = C.HERE


def sf(x):
    """x when it is a number (not a bool), else None."""
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def f(x, nd: int = 1) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def goal(task: str, tasks_files) -> str:
    """The task's instruction from the first of the families' tasks files that names it (a record from the bed carries its own;
    this is the fallback)."""
    for p in tasks_files:
        try:
            for x in Path(p).read_text().splitlines():
                if x.strip() and json.loads(x)["id"] == task:
                    return str(json.loads(x)["goal"])
        except (OSError, ValueError, KeyError):
            continue
    return ""


def phase_steps(phases: dict, blame, cls: str | None, facts: dict) -> list[dict]:
    """One step per phase in order, t0 / t1 from the summed seconds; the phase `blame` (None: all fine) has ok False and
    carries the failure class; `facts[phase]` adds the numbers that belong to it."""
    steps, t = [], 0.0
    for name in phases:
        dt = float(sf(phases[name]) or 0.0)
        s = dict(step=name, ok=name != blame, why="", t0=round(t, 3), t1=round(t + dt, 3))
        if name == blame:
            s["why"] = cls or ""
        s.update({k: v for k, v in facts.get(name, {}).items() if v is not None})
        steps.append(s)
        t += dt
    return steps


def blank(label: str, why: str, sh: str, setting: dict, sig: str, cls: str, key: str, controller: str) -> dict:
    """A round that never ran (the guard refused the package, or the episode is not one this call may run); the raw record
    under `key` is empty."""
    return {
        **dict(label=label, task=label.rsplit("_s", 1)[0], ok=False, why=why, fell=False, dropped=False, plan_ok=False,
               plan_why=why, error=None, steps=[], t_end=None, failure_class=cls, skills=sh, set=setting, wall_s=0.0,
               rule=C.success_rule(), controller=controller, motor=sig),
        key: {},
    }  # fmt: skip


def segment_header(rounds: list[dict]) -> list[str]:
    """The cards' segment table ([segments]): per task with losses, its steps in order with the losses blamed on each
    over the rounds that reached it, the hot step marked, the stable ones named."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import segments as sg

    body = sg.table_lines(sg.segment_table(rounds), rounds)
    if not body:
        return []
    return ["## Segments: losses blamed on each step / rounds that reached it (HOT: most of the task's losses; "
            "stable: reached often, never failed)", *body, ""]  # fmt: skip


def failed_phase_lines(r: dict) -> list[str]:
    """The `failed phase` and `phase seconds` lines of a card."""
    out = []
    bad = [s for s in r.get("steps") or [] if not s.get("ok", True)]
    if bad:
        s = bad[0]
        facts = ", ".join(f"{k} {f(v)}" for k, v in s.items() if k not in ("step", "ok", "why", "t0", "t1"))
        out.append(f"failed phase: {s['step']} from {f(s['t0'])} to {f(s['t1'])} s" + (f" ({facts})" if facts else ""))
    seq = " ".join(f"{s['step']}:{f(s['t1'] - s['t0'])}" for s in r.get("steps") or [])
    if seq:
        out.append(f"phase seconds: {seq}")
    return out


def settings_line(r: dict) -> str:
    st = r.get("set") or {}
    return "settings that differ from the default: " + (", ".join(f"{k}={st[k]}" for k in sorted(st)) or "none")


def cards(rounds: list[dict], out, card_text) -> Path:
    """Failure cards for the Reviewer: <out>/cards.md (or `out` itself when it ends in .md). A two-line header with the wins
    per task, the resolved switches of the rounds (`bench`) when they carry them, the segment table ([segments]), then one card
    per failed round; `card_text(round)` is the adapter's."""
    out = Path(out)
    path = out if out.suffix == ".md" else out / "cards.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted({r["task"] for r in rounds})
    wins = {
        t: (sum(bool(r["ok"]) for r in rounds if r["task"] == t), sum(r["task"] == t for r in rounds)) for t in tasks
    }
    fails = sorted((r for r in rounds if not r["ok"]), key=lambda r: r["label"])
    lines = [f"# Failure cards: {sum(not r['ok'] for r in rounds)} of {len(rounds)} rounds failed",
             "wins per task: " + ("; ".join(f"{t} {a}/{b}" for t, (a, b) in wins.items()) or "none"), ""]  # fmt: skip
    benches = sorted({r["bench"] for r in rounds if r.get("bench")})
    if benches:
        lines.insert(2, "bench: " + "; ".join(benches))
    lines += segment_header(rounds)
    for r in fails:
        lines += card_text(r) + [""]
    path.write_text("\n".join(lines))
    return path
