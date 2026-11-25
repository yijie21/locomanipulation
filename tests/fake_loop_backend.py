"""A fake benchmark backend for tests/test_rsi_loop_e2e.py: the whole loop (loop.py, gate.py, sealed set) runs on it
without a simulator. A start's outcome is a function of the package text and the setting, so a draft that edits
`skill.py` changes outcomes the way a real draft would, deterministically.

skill.py of a package holds lines the fake reads:
    PARAMS = {"alpha": 1}          # `alpha` >= 2 fixes every TaskD start (a knob a tunable sweep can find)
    SOLVE_TaskX = True             # every start of TaskX is solved (one line per task, so drafts merge cleanly)
    BREAK_TaskX = True             # every start of TaskX fails
The seed solves about a third of the starts by a hash of the label (BASE_RATE percent).
Registered as RSI_BACKEND_PATH=e2efake=fake_loop_backend."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

CALLS: list = []
BASE_RATE = 35
RULE = 2  # batch.SUCCESS_RULE
CAPS = {
    "claims": False, "design_probes": False, "diag_redraw": False, "sentinel_exact": False,
    "p_max": 0.05, "sentinel_flips": 1, "probe_scope": True, "sweep_rounds": 300, "check_half": True,
    "check_min_net": 2, "stage1_first": True, "role_rerun": False, "sealed_at_stop": True, "alpha_task": 0.05,
    "alpha_task_check": 0.10, "manager": "program", "prober": False, "compose": "program", "measure_claims": False,
    "stop_no_progress": 2,
}  # fmt: skip


def root() -> Path:
    return Path(os.environ["E2E_ROOT"])


def _read(skills: Path) -> dict:
    text = (Path(skills) / "skill.py").read_text()
    m = re.search(r"^PARAMS = (.*)$", text, re.M)
    return dict(
        PARAMS=ast.literal_eval(m.group(1)) if m else {},
        SOLVE_TASKS=re.findall(r"^SOLVE_(\w+) = True$", text, re.M),
        BREAK_TASKS=re.findall(r"^BREAK_(\w+) = True$", text, re.M),
    )


def task_of(label: str) -> str:
    return label.rsplit("_s", 1)[0]


def outcome(skills: Path, set_: dict, label: str) -> bool:
    pkg = _read(skills)
    t = task_of(label)
    if t in pkg["BREAK_TASKS"]:
        return False
    base = int(hashlib.sha1(label.encode()).hexdigest(), 16) % 100 < BASE_RATE
    alpha = {**pkg["PARAMS"], **(set_ or {})}.get("alpha", 1)
    return bool(base or t in pkg["SOLVE_TASKS"] or (t == "TaskD" and alpha >= 2))


def load(name):
    return dict(name=name, backend="e2efake", tasks=["TaskA", "TaskB", "TaskC", "TaskD"], rows={}, pools={})


def draw(fam, pool, out, n=None, tasks=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return dict(written=0, skipped=[], unavailable={}, next_seed=0)


def run_jobs(jobs, family, pool, procs, log, gpus, stop=None, where="probe"):
    labels_all = sorted(p.stem for p in Path(pool).glob("*.json") if not p.name.startswith("_"))
    for j in jobs:
        j.out.mkdir(parents=True, exist_ok=True)
        for x in j.labels or labels_all:
            f = j.out / f"{x}.json"
            if f.exists():
                continue
            ok = outcome(j.skills, j.overrides, x)
            if os.environ.get("E2E_DRIFT") == "1" and j.out.name == "sentinel":
                ok = not ok  # the environment moved: the incumbent's own rounds no longer reproduce
            rec = dict(label=x, task=task_of(x), ok=ok, why="" if ok else "[goal] not done", fell=False, dropped=False,
                       t_end=10.0, rule=RULE, failure_class=None if ok else "grasp", where=where)  # fmt: skip
            f.write_text(json.dumps(rec))
    CALLS.append(("run_jobs", where, sum(len(j.labels or labels_all) for j in jobs)))
    return {k: [] for k in range(len(jobs))}


def signature(family, where):
    return f"fake-{where}"


def cards(rounds, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cards.md").write_text("# fake cards\n" + "\n".join(f"- {r['label']} failed" for r in rounds if not r["ok"]))
    return out / "cards.md"


def layout(family):
    here = Path(__file__).resolve().parents[1] / "scripts" / "rsi"
    return dict(skills=root() / "skills", rules=here / "rules_notes" / "simple_art.md", prompts=here / "prompts",
                plan=None, leak=[], class_suggest={}, stub_file="skill.py", caps=dict(CAPS))  # fmt: skip


def sandbox(dest, family):
    pass


def guard_write(family, path):
    pass


def load_skills(skills, name="x", family=None):
    pkg = _read(skills)
    params = dict(pkg["PARAMS"])

    def apply_set(self, set_):
        for k in set_ or {}:
            if k not in params:
                raise KeyError(f"unknown parameter {k}")

    return SimpleNamespace(apply_set=apply_set, parameters=lambda self: dict(params))
