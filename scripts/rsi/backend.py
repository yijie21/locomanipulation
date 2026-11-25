"""The loop's second backend seam (plan loop_seam_plan.md, section B). A family file may carry a "backend" key
("simple" -> backend_simple.py); every place of the loop that assumes the scene backend asks `of(family)` first and runs
today's code when it gets None. Nothing here imports a backend until a family names one.

The backend module's interface (the loop codes against exactly this): load(name), draw(fam, pool, out, n, tasks),
run_jobs(jobs, family, pool, procs, log, gpus, stop, where), signature(family, where), to_round(...), cards(rounds, out),
layout(family), sandbox(dest, family).

`RSI_BACKEND_PATH` ("name=module,name2=module") adds or overrides entries of the lookup table, so a test can register a
fake backend without touching the table."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TABLE = {"simple": "backend_simple", "lml": "backend_lml", "bench": "backend_bench"}

# What a scene family has today; a backend's layout()["caps"] overrides keys of it. "claims", "design_probes" and
# "diag_redraw" switch the loop's claim check, design probes and diagnostic-pool redraw; the other three are the gate's.
SCENE_CAPS = {
    "claims": True,
    "design_probes": True,
    "diag_redraw": True,
    "sentinel_exact": True,  # a sentinel re-run must reproduce `ok` and `t_end`; False: `ok` alone
    "min_net": 3,  # gate.MIN_NET; read only where p_max is None (with p_max it can never bind: the smallest p <= 0.05 is 5:0)
    "loss_share": 3,  # scene rule (p_max None): the losses may be at most a third of the wins
    "repeat": 1,  # runs per round (the gate's repeat k > 1: see gate.py, not implemented)
    "p_max": None,  # a number: the paired sign test's one-sided p replaces the loss-share condition (gate.verdict_fails)
    "sentinel_flips": 0,  # sentinel re-runs that may differ from the stored rounds before the pairing is void
    "probe_scope": False,  # True: a hypothesis's probe runs only on the tasks it is about (loop.probe_labels)
    "sweep_rounds": 1200,  # rounds an iteration's program sweeps may run in all (probe sweeps and Engineer-declared knobs)
    "draft_sentinels": 3,  # a draft with declared starts also runs this many starts of every other task (loop.draft_labels)
    "alpha_task": None,  # per-task regression test (verdict.py): a task whose losses are significant at this level, Holm across tasks, rejects at the gate and the sealed set
    "alpha_task_check": None,  # the same at the check half (looser: 15 starts a task)
    "check_min_net": 1,  # the check half sends a candidate whose net on the hidden half reaches this
    # TODO 54 (rsi-lean): the lean iteration is the benchmark default; a scene family keeps the old one through these
    "manager": "llm",  # "llm": the Manager role chooses what is built; "program": the Reviewer's hypotheses go to the Engineers in order
    "measure_claims": True,  # False: no `measure` field in the Reviewer's schema
    "prober": True,  # True: the Prober phase and the Manager's `tune`; False: only the Engineers' `tunable` keys, swept by the program
    "compose": "llm",  # "llm": the Composer always merges; "program": a 3-way merge by the program, the Composer only on a conflict
    "stop_no_progress": 0,  # N > 0: this many iterations in a row without a candidate that passes the check half end the run
    "sealed_at_stop": False,  # True: a run that stops by a stop rule runs sealed.py once (loop.run_sealed)
}


def table() -> dict:
    t = dict(TABLE)
    for item in os.environ.get("RSI_BACKEND_PATH", "").split(","):
        if "=" in item:
            name, mod = item.split("=", 1)
            t[name.strip()] = mod.strip()
    return t


def name_of(family) -> str | None:
    """The "backend" key of a family dict, or of the family file's name under families/ (None when absent)."""
    if isinstance(family, dict):
        return family.get("backend") or None
    p = HERE / "families" / f"{family}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("backend") or None


def of(family):
    """The backend module for a family (dict or name); None when its file has no "backend" key, and every call site then
    runs today's code unchanged."""
    name = name_of(family)
    if name is None:
        return None
    mods = table()
    if name not in mods:
        raise SystemExit(f"family backend {name!r} is not known; known: {', '.join(sorted(mods))}")
    return importlib.import_module(mods[name])


def layout(family) -> dict:
    """Where the family's editable package, rules, prompts and plan are, and its caps. A family without a backend gets
    today's constants (the scene skills, rules.md, prompts/, the family file's `plan` or the scene plan), all caps on."""
    be = of(family)
    if be is None:
        spec = family if isinstance(family, dict) else _spec(family)
        return dict(
            skills=REPO / "src" / "locomani" / "tasks" / "scene" / "skills",
            rules=HERE / "rules.md",
            prompts=HERE / "prompts",
            plan=REPO / spec["plan"] if spec.get("plan") else REPO / "docs" / "handoff" / "scene-rsi-plan-20261016.md",
            leak=[],
            class_suggest={},
            caps=dict(SCENE_CAPS),
        )
    lay = dict(be.layout(family))
    lay["caps"] = {**SCENE_CAPS, **(lay.get("caps") or {})}
    lay.setdefault("leak", [])
    return lay


def caps(family) -> dict:
    return layout(family)["caps"]


def _spec(name: str) -> dict:
    p = HERE / "families" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}
