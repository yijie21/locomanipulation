"""The row flavor of the benchmark adapters whose episodes are seed pools, not practice files (TODO 49.7a; 2026-11-24):
BiGym 2.0 (`locomani.bench.bigym2`) and HumanoidArena (`locomani.bench.humanoidarena`).

Both adapters leave the same row (their record.py): `id` `<task>/<pool>/<n>` (pool P practice, A acceptance, S sealed), `success`
(the benchmark's check, the hands did it, nobody fell), `bench_success` (the benchmark's own verdict), `why` ("" or one of
not_success, fell, no_hand, body_push, error), `fell`, `steps` (50 Hz), `hand_steps`, `body_steps`, `body_parts`, `marks`
({name: control step} the package's own marks, in order), `flow` (the package's notes), `wall_s`, `error`, plus the run's
switches (`bench`) and the adapter's config. HumanoidArena adds `obj_move`, BiGym `termination`. What this module writes anew,
against backend_lml.py:

- to_round: `ok` = the row's `success` (the compliance check is inside it); the raw row is kept under `seeded` in its
  allow-listed form (`public`): no seed, no adapter config, no paths;
- the steps of a round are the package's marks in order (t0 / t1 from the control steps); the phase after the last mark reached
  is the one that failed;
- the failure classes: error, fall, comply (the check held and the hands did not do it), timeout (the episode ran out of steps),
  short (it ended before that);
- the leak patterns and forbidden paths: the evaluation seeds, the hidden pools' ids and the benchmarks' own sources.

Nothing here starts a simulator.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import backend_simple as B  # noqa: E402

CLASSES = ("error", "timeout", "fall", "comply", "short")
CLASS_SUGGEST = {
    "error": "the run raised: read the error text, a package that raises ends the episode",
    "timeout": "the episode ran out of steps: find the phase that took the clock and shorten what buys nothing",
    "fall": "the robot fell: speeds, accelerations, crouch height, a hand pulling or pushing against the walk",
    "comply": "the benchmark's check held but a hand did not do it: carry or push with the gripper's own links, keep wrist, forearm and body off the task body",  # noqa: E501
    "short": "the episode ended with the check not held: look at the last mark reached and at what the flow did next",
}
# What a role must not see. Text patterns for the files of the sandbox's readable copy (the adapter's files and the seed package
# copied on purpose are skipped, as in the other flavors).
LEAK = [
    r"\b[A-Za-z0-9_]+/[ASE]/\d+",  # an acceptance, sealed or evaluation episode id; the practice ids are `<task>/P/<n>`
    r"\b62\d{4}\b",  # BiGym's evaluation block [620000, 630000)
    r"EVAL_BLOCK|EVAL_GROUPS|eval_seeds|DEMO_SEEDS",
    r"third_party/(?:BiGym2|HumanoidArena)",  # the benchmarks' own source
    r"bigym/loco/eval|eval_vla_suite|script/eval_scripts|sim_eval_vla",  # their evaluation protocols and scripts
    r"scene_rsi_20261016_hidden",
    r"handoff/(?:bench-options|benchmarks|escalation)",
    r"bench49",  # the porting notes
]
FORBIDDEN_PATHS = (
    "third_party/BiGym2",
    "third_party/HumanoidArena",
    "docs/handoff/bench-options-*.md",
    "docs/handoff/benchmarks-*.md",
    "docs/handoff/escalation-*.md",
)
STANCE = B.STANCE

_SCALAR_KEYS = (
    "id task pool success bench_success why fell steps termination hand_steps body_steps obj_move wall_s error view "
    "depth_source package"
).split()
_FLOW_MAX = 2500  # characters of the package's own notes kept in a stored round


def _sf(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _text(v):
    if v is None:
        return None
    return re.sub(r"\s+", " ", str(v)).strip()[:600] or None


def public(rec: dict) -> dict:
    """The row's allow-listed form: the scalar keys, the body parts, the marks and a capped copy of the package's notes. The
    seed, the adapter's config and the run's paths are not kept."""
    out = {
        k: rec[k] for k in _SCALAR_KEYS if k in rec and (rec[k] is None or isinstance(rec[k], (str, int, float, bool)))
    }
    if isinstance(rec.get("body_parts"), list):
        out["body_parts"] = [str(x) for x in rec["body_parts"]][:12]
    if isinstance(rec.get("marks"), dict):
        out["marks"] = {str(k): v for k, v in rec["marks"].items() if _sf(v) is not None}
    flow = rec.get("flow")
    if isinstance(flow, dict) and flow:
        text = json.dumps(flow, default=str)
        out["flow"] = flow if len(text) <= _FLOW_MAX else {"_cut": text[:_FLOW_MAX]}
    return out


def failure_class(rec: dict) -> str | None:
    if rec.get("success"):
        return None
    why = rec.get("why")
    if rec.get("error") or why == "error":
        return "error"
    if rec.get("fell") or why == "fell":
        return "fall"
    if why in ("no_hand", "body_push", "not_moved", "no_record"):
        return "comply"
    if str(rec.get("termination") or "").lower() in ("timeout", "time_limit", "truncated", "time"):
        return "timeout"
    return "short"


def _steps(rec: dict, cls: str | None, ok: bool) -> list[dict]:
    """One step per mark in order (t0 / t1 from the control steps at 50 Hz); when the round failed, a last step from the
    last mark to the end of the episode carries the class."""
    marks = sorted(
        ((k, int(v)) for k, v in (rec.get("marks") or {}).items() if _sf(v) is not None), key=lambda kv: kv[1]
    )
    end = int(_sf(rec.get("steps")) or (marks[-1][1] if marks else 0))
    steps, t0 = [], 0.0
    for k, v in marks:
        t1 = round(0.02 * v, 3)
        steps.append(dict(step=k, ok=True, why="", t0=round(t0, 3), t1=t1))
        t0 = t1
    if not ok:
        last = marks[-1][0] if marks else "start"
        steps.append(
            dict(step=f"after_{last}", ok=False, why=cls or "", t0=round(t0, 3), t1=round(max(t0, 0.02 * end), 3))
        )
    return steps


def to_round(rec: dict, label: str, skills_hash: str, set_: dict, signature: str) -> dict:
    """A bed row as the loop's round dict. Pure. `ok` = the row's `success` (the benchmark's check, the compliance check
    and no fall)."""
    ok = bool(rec.get("success"))
    cls = failure_class(rec)
    pub = public(rec)
    task = rec["id"].split("/")[0] if rec.get("id") else label.rsplit("_s", 1)[0]
    text = _text(rec.get("error")) or _text(rec.get("why"))
    why = "" if ok else (f"{cls}: {text}" if text else str(cls))
    steps = _sf(rec.get("steps"))
    return dict(
        label=label,
        task=task,
        ok=ok,
        why=why,
        fell=bool(rec.get("fell")),
        dropped=False,
        plan_ok=not rec.get("error"),
        plan_why=_text(rec.get("error")) or "",
        error=_text(rec.get("error")),
        steps=_steps(pub, cls, ok),
        t_end=round(0.02 * steps, 2) if steps is not None else None,
        failure_class=cls or "",
        skills=skills_hash,
        set=dict(set_ or {}),
        wall_s=rec.get("wall_s"),
        rule=B._success_rule(),
        controller="native",
        motor=signature,
        seeded=pub,
    )


def _blank(label: str, why: str, sh: str, setting: dict, sig: str, cls: str = "guard") -> dict:
    """A round that never ran (the guard refused the package, or the id is not one this call may run)."""
    return dict(label=label, task=label.rsplit("_s", 1)[0], ok=False, why=why, fell=False, dropped=False,
                plan_ok=False, plan_why=why, error=None, steps=[], t_end=None, failure_class=cls, skills=sh,
                set=setting, wall_s=0.0, rule=B._success_rule(), controller="native", motor=sig, seeded={})  # fmt: skip


def _goal(task: str) -> str:
    for f in sorted((HERE / "families").glob("*_tasks.jsonl")):
        try:
            for x in f.read_text().splitlines():
                if x.strip() and json.loads(x)["id"] == task:
                    return str(json.loads(x)["goal"])
        except (OSError, ValueError, KeyError):
            continue
    return ""


_f = B._f


def card_text(r: dict) -> list[str]:
    """One failed round as plain text, from the allow-listed row only."""
    raw = public(r.get("seeded") or {})
    lines = [
        f"## {r['label']}  ({r.get('failure_class') or 'unclassified'})",
        f"task: {r.get('task')}",
        f"instruction: {_goal(r.get('task', ''))}",
        f"outcome: {r.get('why') or 'failed'}; t_end {_f(r.get('t_end'))} s ({raw.get('steps')} steps); fell {bool(r.get('fell'))}",
        f"check: the benchmark's {raw.get('bench_success')}, compliant {raw.get('success')}"
        + (f", ended by {raw['termination']}" if raw.get("termination") else ""),
    ]
    bad = [s for s in r.get("steps") or [] if not s.get("ok", True)]
    if bad:
        s = bad[0]
        lines.append(f"failed phase: {s['step']} from {_f(s['t0'])} to {_f(s['t1'])} s")
    seq = " ".join(f"{s['step']}:{_f(s['t1'] - s['t0'])}" for s in r.get("steps") or [])
    if seq:
        lines.append(f"phase seconds: {seq}")
    if raw.get("hand_steps") is not None:
        parts = ", ".join(raw.get("body_parts") or []) or "none"
        lines.append(f"contact: hand on a task body {raw['hand_steps']} steps; body on one {raw.get('body_steps')} steps (parts: {parts})")  # fmt: skip
    if raw.get("obj_move") is not None:
        lines.append(f"object moved {_f(raw['obj_move'], 2)} m")
    for k, v in (raw.get("flow") or {}).items():
        lines.append(f"flow {k}: {json.dumps(v, default=str)[:300]}")
    st = r.get("set") or {}
    lines.append("settings that differ from the default: " + (", ".join(f"{k}={st[k]}" for k in sorted(st)) or "none"))
    return lines[:60]


def cards(rounds: list[dict], out) -> Path:
    """Failure cards for the Reviewer: <out>/cards.md (or `out` itself when it ends in .md): the wins per task, the segment
    table (rules 48), then one card per failed round."""
    out = Path(out)
    f = out if out.suffix == ".md" else out / "cards.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted({r["task"] for r in rounds})
    wins = {
        t: (sum(bool(r["ok"]) for r in rounds if r["task"] == t), sum(r["task"] == t for r in rounds)) for t in tasks
    }
    fails = sorted((r for r in rounds if not r["ok"]), key=lambda r: r["label"])
    lines = [f"# Failure cards: {sum(not r['ok'] for r in rounds)} of {len(rounds)} rounds failed",
             "wins per task: " + ("; ".join(f"{t} {a}/{b}" for t, (a, b) in wins.items()) or "none"), ""]  # fmt: skip
    lines += B.segment_header(rounds)
    for r in fails:
        lines += card_text(r) + [""]
    f.write_text("\n".join(lines))
    return f


# ------------------------------------------------------------------------------------------ skill package
def package_parameters(root: Path) -> dict:
    """The parameters of a seeded package read from its source text: the keywords of `K = dict(...)` in skill.py."""
    out: dict = {}
    p = Path(root) / "skill.py"
    if not p.is_file():
        return out
    for node in ast.parse(p.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and any(isinstance(t, ast.Name) and t.id == "K" for t in node.targets)
        ):
            for kw in node.value.keywords:
                if kw.arg:
                    out[kw.arg] = B._literal(kw.value)
    return out


class SkillPackage(B.SkillPackage):
    """`apply_set` refuses a `stance_` key (these adapters have no stance: the package walks itself) and an unknown name."""

    def __init__(self, root: Path, bed_keys=(), fixed=()):
        super().__init__(root, bed_keys, fixed)
        self.defaults = package_parameters(self.root)

    def apply_set(self, _pkg, overrides: dict | None = None) -> dict:
        stance = sorted(k for k in (overrides or {}) if k.startswith(STANCE))
        if stance:
            raise KeyError(f"unknown parameters {stance}: this adapter has no stance, the package walks itself")
        return super().apply_set(_pkg, overrides)


def load_skills(root, name: str = "scene_skills", family=None) -> SkillPackage:
    fam = B._fam(family) if family is not None else {}
    return SkillPackage(Path(root), fam.get("bed_keys", []), B.bed_fixed(fam))
