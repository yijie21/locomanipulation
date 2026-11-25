"""The LocoManiLab backend of the RSI loop (plan lml_backend_plan.md, steps 7 to 9; 2026-11-20).

A family file with `"backend": "lml"` makes the loop run the LocoManiLab camera bed (the other worktree's
scripts/experiments/escalation_20261116, `run_lml_q.sh`) instead of the scene runner. It follows backend_simple.py (the
SIMPLE backend) and reuses what is generic there: the pools (`<Task>/Q/<line>` over practice files with `src.level`,
the first 10 lines of a level are diagnostic, the rest hidden and split by a hidden seed), the seed package copy and
its write guard, the guard call, the scratch directory, the process handling and the sandbox walk. What is written anew:

- one bed serves probes and the gate (there is no mirror): `where` only selects the practice files, the full ones for
  the gate and the family's own (blanked in a sandbox) for the rest;
- to_round: `ok` = the benchmark's check and the compliance check and the benchmark's own `genuine` flag;
- cards: built from an ALLOW-LIST of record fields. The scene is fixed, so a room coordinate learned in practice is the
  evaluation's too: no position in the room's frame (true_centre, true_grasp, dest_true, target, any *_true position,
  the spec's regions, the navigation log) is on a card or in the stored round;
- the signature (also the benchmark's git commit), the bed's environment (LML_*), the skill package's parameters (the
  keys of skill_simple.py's dicts and `EXTRA` of the package's __init__.py), the leak patterns and forbidden paths.

Nothing here starts a simulator itself.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import backend_simple as B  # noqa: E402

REPO = B.REPO
HIDDEN = B.HIDDEN
CLASSES = ("error", "timeout", "fall", "perceive", "navigate", "deliver", "grasp", "collide", "place", "comply")
# The phase a failure class is blamed on: the first of these that the record has, else the last phase it ran.
BLAME = {
    "navigate": ("walk",),
    "deliver": ("reach_grasp", "over", "reach_pregrasp"),
    "grasp": ("lift", "grip_hold", "close"),
    "collide": ("close", "reach_grasp"),
    "place": ("open", "over", "carry"),
    "perceive": ("search", "start", "walk"),
    "comply": ("open", "lift", "grip_hold"),
}
CLASS_SUGGEST = {
    "error": "the skill or the bed raised, or the check held at the start: read the error text, fix the code path first",
    "timeout": "the time limit ran out while the flow was still working: shorten the slow phases (walk, search, settle)",
    "fall": "the robot fell: slower or shorter moves, a wider stance, less arm reach while the feet move",
    "perceive": "the object or the destination was not found: look again from another place or angle, name it differently",
    "navigate": "a walking task that did not get close enough: where it aims, how far it walks, when it stops",
    "deliver": "the palm ended far from the grasp or release point: the reach target, the grasp choice, the arm's error",
    "grasp": "the fingers closed and the object did not come up or was lost: grip, force, palm position, lift speed",
    "collide": "the object or the receptacle was pushed or tipped before the grasp or release: approach path, entry speed",
    "place": "carried and released, but it did not stay at the destination: release point and height, when the hand opens",
    "comply": "the benchmark's check passed but not what the instruction says: carry it in a hand and release it there",
}
# Level 2: a stance is `stance_<key>` in the setting and reaches the runner as --stance (as in SIMPLE).
STANCE = B.STANCE

# What a role must not see. Text patterns for the files of the sandbox's readable copy (the bed files and the seed
# package copied on purpose are skipped, as in SIMPLE).
LEAK = [
    r"_scene\d+/[EPW]/-?\d+",  # an evaluation (E) or hand-tuning (P, W) episode id; the loop's own are `<Task>/Q/<n>`
    r"escalation_20261116/(?:a_l|a_s|a|b|practice_lml|practice|official)(?![\w.-])",  # the hand-tuning outputs of the bed tree (whole names: `b` is not `bed_lml_cam.py`)
    r"\ba_l/",
    r"locomanipulation-esc/outputs",  # the bed tree's outputs by absolute path: practice files, clouds, specs
    r"/specs\.json|/specs_table|/server_results|seed_pkg_lml_removed|loop_mode_notes|lml_backend_plan",
    r"best_p\.json|refs_from_cam|extra_cam\.json|grasp_refs\.json",  # hand-made parameters and reference palm poses
    r"master_seed",
    r"practice_lml",
    r"scene_rsi_20261016_hidden",
    r"third_party/LocoManiLab|LocoManiLab/(?:src|external)|humanoid_benchmark/robocasa",  # the benchmark's own source
    r"handoff/(?:escalation|benchmarks)",
    r"lml_bed2_smoke|bed2_truth_check",  # the articulation fork's hand-run smoke rounds and its truth-reading check
    r"lml_bed3_smoke|bed3_truth_check|drawer_smoke_practice",  # the same for the drawer fork (2026-11-21)
    r"(?:fridge|drawer)_facts|lml_bed\d_report|lml_bed2_video",  # hand-written fact sheets, fork reports, videos
]
# What an Engineer-visible tree must not hold (relative to the sandbox root; `*` globs)
FORBIDDEN_PATHS = (
    "outputs/escalation_20261116",  # the whole hand-tuning tree ...
    "outputs/escalation_20261116/a_l",  # ... and, named, what the notes call out
    "outputs/escalation_20261116/a_l/specs.json",
    "outputs/escalation_20261116/a_l/seed_pkg_lml_removed.md",
    "outputs/escalation_20261116/a_l/lml_backend_plan_20261120.md",
    "outputs/escalation_20261116/a_l/loop_mode_notes_20261120.md",
    "outputs/escalation_20261116/practice_lml",
    "outputs/escalation_20261116/practice_lml_old_20261120",
    "docs/handoff/escalation-*.md",
    "docs/handoff/benchmarks-*.md",
    # the original, un-stripped skill files of the bed folder (the seed package holds the stripped ones)
    "scripts/experiments/escalation_20261116/skill_lml.py",
    "scripts/experiments/escalation_20261116/skill_lml_cam.py",
    "scripts/experiments/escalation_20261116/view_lml.py",
    "scripts/experiments/escalation_20261116/skill_simple.py",
    "scripts/experiments/escalation_20261116/sweep_lml.py",
    "outputs/scene_rsi_20261016_hidden/lml_q/master_seed.txt",
    "outputs/scene_rsi_20261016_hidden/lml_fridge/master_seed.txt",
    "outputs/escalation_20261116/practice_lml_fridge",
    # the articulation fork (2026-11-21) copies the original, un-stripped skill files and holds a truth-reading check
    "scripts/experiments/lml_bed2_20261121/skill_lml.py",
    "scripts/experiments/lml_bed2_20261121/skill_lml_cam.py",
    "scripts/experiments/lml_bed2_20261121/view_lml.py",
    "scripts/experiments/lml_bed2_20261121/skill_simple.py",
    "scripts/experiments/lml_bed2_20261121/sweep_simple.py",
    "scripts/experiments/lml_bed2_20261121/bed2_truth_check.py",
    "scripts/experiments/lml_bed2_video_20261121",  # a copy of the fork with a video recorder (hand-run rounds)
    "outputs/scene_rsi_20261016_hidden/lml_drawer/master_seed.txt",
    "outputs/escalation_20261116/practice_lml_drawer",
    # the drawer fork (2026-11-21), a copy of the articulation fork: the same un-stripped files and its own truth check
    "scripts/experiments/lml_bed3_20261121/skill_lml.py",
    "scripts/experiments/lml_bed3_20261121/skill_lml_cam.py",
    "scripts/experiments/lml_bed3_20261121/view_lml.py",
    "scripts/experiments/lml_bed3_20261121/skill_simple.py",
    "scripts/experiments/lml_bed3_20261121/sweep_simple.py",
    "scripts/experiments/lml_bed3_20261121/bed3_truth_check.py",
    "third_party/LocoManiLab",  # its task definitions hold the checks' target regions in room coordinates
)

# ------------------------------------------------------------------------- the allow-list of record fields
_SCALAR_KEYS = (
    "episode_id track controller family skill prompt instruction success t_success t_check t_end limit_s end steps fell "
    "rise_cm sha wall_s found turns target_px target_score target_phrase dest_score dest_phrase dest_in_view "
    "grasp_vs_truth_cm stop_true_cm odo_end_cm stop_err_cm walk_cm palm_err_cm palm_deg finger_defl held_true "
    "release_cm foot_err_cm basket_cm genuine genuine_min_steps loss error why comply_error"
).split()
_LIST_KEYS = ("reach_ok", "stopped")
ART_KEYS = tuple(
    "action goal frac_start frac_max frac_min frac_end t_first_move t_goal hold_best_s touch t_touch hand_min_cm "
    "hand_min_side hand_min_t palm_dz_cm palm_dxy_cm palm_best_dz_cm grip_on_handle pelvis_drop_cm moved_by".split()
)
# dict-valued fields: key -> the sub-keys that may pass (None: every key, values scalar)
_DICT_KEYS: dict[str, tuple | None] = {
    "setting": None,
    "stance": None,
    "phases": None,
    "parse": ("kind", "obj", "dest", "on"),
    "prompt_parse": ("kind", "obj", "dest", "on"),
    "jitter": ("robot_cm", "object_cm", "robot_radius", "object_radius"),
    # an articulated object (the fridge family, 2026-11-21): the door's open fraction and the palms against the handle,
    # relative quantities only (no position in the room's frame)
    "art": ART_KEYS,
    "stance_used": ("d", "t", "yaw"),  # the stance relative to the object, the skill's own choice
    "cam_err": ("err_cm", "angle_deg"),  # not pos_cm: a vector
    "comply": ("ok", "why"),
    "comply_facts": ("t_check", "hand_at_check", "fell_at_check", "t_carry", "best_run_s", "at_check"),
    "nav": (
        "looks",
        "seen",
        "turns",
        "explored",
        "score",
        "phrase",
        "end_dist_cm",
        "centre_err_cm",
    ),  # not log / centre
}
_MARK_KEYS = ("t", "rise_cm", "moved_cm", "tilt_deg", "held", "basket_cm")
_VEC = re.compile(r"\[\s*-?\d+\.\d+(?:\s*,\s*-?\d+\.\d+)+\s*\]")  # a list of two or more decimals in a message


def _scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 400)


def _text(v):
    return _VEC.sub("[..]", v)[:300] if isinstance(v, str) else v


def public(rec: dict) -> dict:
    """The allow-listed copy of a bed record: only fields named in this module, scalars or short lists of scalars, and
    the sub-keys named for the dict-valued ones. Whatever is not named is dropped, so a field the bed adds later (a
    position of the room's frame, say) does not reach a card or a stored round until someone allows it here."""
    out: dict = {}
    for k in _SCALAR_KEYS:
        if k in rec and _scalar(rec[k]):
            out[k] = _text(rec[k]) if k in ("error", "why", "comply_error") else rec[k]
    for k in _LIST_KEYS:
        v = rec.get(k)
        if isinstance(v, (list, tuple)) and len(v) <= 12 and all(_scalar(x) for x in v):
            out[k] = list(v)
    for k, subs in _DICT_KEYS.items():
        v = rec.get(k)
        if not isinstance(v, dict):
            continue
        d = {}
        for a, b in v.items():
            if subs is not None and a not in subs:
                continue
            if _scalar(b):
                d[a] = _text(b)
            elif isinstance(b, (list, tuple)) and len(b) <= 12 and all(_scalar(x) for x in b):
                d[a] = list(b)
        out[k] = d
    marks = rec.get("marks")
    if isinstance(marks, dict):
        out["marks"] = {
            n: {a: m[a] for a in _MARK_KEYS if a in m and _scalar(m[a])}
            for n, m in marks.items()
            if isinstance(m, dict)
        }
    od = rec.get("obj_dest_end")  # how far the object ended from where it should be: the distance, not the vector
    if isinstance(od, dict):
        dxy = od.get("dxy_cm")
        d = {}
        if isinstance(dxy, (list, tuple)) and len(dxy) == 2 and all(isinstance(x, (int, float)) for x in dxy):
            d["dist_cm"] = round((dxy[0] ** 2 + dxy[1] ** 2) ** 0.5, 1)
        elif isinstance(od.get("dist_cm"), (int, float)):  # already filtered (a stored round)
            d["dist_cm"] = od["dist_cm"]
        if isinstance(od.get("dz_cm"), (int, float)):
            d["dz_cm"] = od["dz_cm"]
        out["obj_dest_end"] = d
    return out


# -------------------------------------------------------------------------------------------------- family
load = B.load
draw = B.draw
guard_write = B.guard_write
BaseWriteRefused = B.BaseWriteRefused
ensure_package = B.ensure_package
package_dir = B.package_dir
base_dir = B.base_dir
bed_root = B.bed_root
skills_hash = B.skills_hash
split_set = B.split_set
hidden_ids = B.hidden_ids
practice_dir = B.practice_dir
run_guard = B.run_guard  # the Engineer's guard command calls it by this module's name


# ----------------------------------------------------------------------------------------------- records
def _sf(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def to_round(rec: dict, label: str, skills_hash: str, set_: dict, signature: str) -> dict:
    """A bed record as the loop's round dict (plan section 3). Pure. `ok` = the benchmark's check, the compliance check
    and the benchmark's `genuine` flag, a missing `comply` or `genuine` counting as false. The record is kept under
    `lml` in its allow-listed form (`public`)."""
    comply = rec.get("comply")
    comply_ok = bool(comply.get("ok")) if isinstance(comply, dict) else False
    genuine = bool(rec.get("genuine"))
    success = bool(rec.get("success"))
    ok = success and comply_ok and genuine
    loss = rec.get("loss") or ("error" if rec.get("error") else None)
    if ok:
        cls = None
    elif loss:
        cls = loss
    elif success:
        cls = "comply"
    else:
        cls = "unclassified"
    text = _text(rec.get("why") or rec.get("error"))
    if not ok and not text and success:  # the check held; say which of the other two did not
        if not comply_ok:
            text = (comply or {}).get("why") if isinstance(comply, dict) else None
            text = f"compliance {text}" if text else "no compliance check in the record"
        else:
            text = "not genuine: the check held at the start state"
    why = "" if ok else (f"{cls}: {text}" if text else str(cls))
    marks = rec.get("marks") or {}
    end = marks.get("end") or marks.get("before_open") or marks.get("after_lift") or {}
    rise = _sf(end.get("rise_cm"))
    plan_text = _text(rec.get("why") or rec.get("error")) or ""
    pub = public(rec)
    task = rec["episode_id"].split("/")[0] if rec.get("episode_id") else label.rsplit("_s", 1)[0]
    t_end = rec.get("t_end")
    if t_end is None and _sf(rec.get("steps")) is not None:
        t_end = round(0.02 * rec["steps"], 2)
    return dict(
        label=label,
        task=task,
        ok=ok,
        why=why,
        fell=rec.get("fell") is not None,
        dropped=bool(not ok and rise is not None and rise < -15),
        plan_ok=not plan_text,
        plan_why=str(plan_text),
        error=_text(rec.get("error")),
        steps=_steps(pub, cls, ok),
        t_end=t_end,
        failure_class=cls or "",
        skills=skills_hash,
        set=dict(set_ or {}),
        wall_s=rec.get("wall_s"),
        rule=B._success_rule(),
        controller="scalebfm",
        motor=signature,
        lml=pub,
    )


def _steps(rec: dict, cls: str | None, ok: bool) -> list[dict]:
    """One step per phase in order, t0 / t1 from the summed seconds; the phase the class names has ok False."""
    phases = rec.get("phases") or {}
    marks = rec.get("marks") or {}
    names = list(phases)
    blame = None
    if not ok and names:
        blame = next((p for p in BLAME.get(cls or "", ()) if p in phases), names[-1])
    od = rec.get("obj_dest_end") or {}
    facts = {
        "walk": dict(stop_err_cm=_sf(rec.get("stop_err_cm")), walk_cm=_sf(rec.get("walk_cm"))),
        "reach_grasp": dict(palm_err_cm=_sf(rec.get("palm_err_cm")), reach_ok=rec.get("reach_ok")),
        "close": {k: marks.get("before_close", {}).get(k) for k in ("moved_cm", "tilt_deg")},
        "lift": {k: marks.get("after_lift", {}).get(k) for k in ("rise_cm", "held")},
        "open": {**{k: marks.get("before_open", {}).get(k) for k in ("basket_cm", "held")}, **od},
    }
    steps, t = [], 0.0
    for name in names:
        dt = float(_sf(phases[name]) or 0.0)
        s = dict(step=name, ok=name != blame, why="", t0=round(t, 3), t1=round(t + dt, 3))
        if name == blame:
            s["why"] = cls or ""
        s.update({k: v for k, v in facts.get(name, {}).items() if v is not None})
        steps.append(s)
        t += dt
    return steps


# ------------------------------------------------------------------------------------------------- cards
def _goal(task: str) -> str:
    """The task's instruction from a family's tasks file (a record from the bed carries its own; this is the fallback)."""
    for f in sorted((HERE / "families").glob("lml*_tasks.jsonl")):
        try:
            for x in f.read_text().splitlines():
                if x.strip() and json.loads(x)["id"] == task:
                    return str(json.loads(x)["goal"])
        except (OSError, ValueError, KeyError):
            continue
    return ""


_f = B._f
_MEASURES = (
    "palm_err_cm palm_deg grasp_vs_truth_cm stop_true_cm stop_err_cm odo_end_cm walk_cm finger_defl release_cm "
    "foot_err_cm"
).split()


def door_text(a: dict, thing: str = "door") -> str:
    """The door (or drawer) line of a card (an articulated object): its open fraction, the touch and the palms against
    the handle, and for the drawers how low the pelvis went and what was on the drawer while it moved."""
    fr = f"{thing}: open fraction {_f(a.get('frac_start'), 2)} -> max {_f(a.get('frac_max'), 2)}, min {_f(a.get('frac_min'), 2)}, end {_f(a.get('frac_end'), 2)}"  # fmt: skip
    fr += f" (goal {a.get('goal')}; first moved at {_f(a.get('t_first_move'))} s; in the goal at {_f(a.get('t_goal'))} s, held {_f(a.get('hold_best_s'))} s)"  # fmt: skip
    touch = ", ".join(map(str, a.get("touch") or [])) or "nothing"
    fr += f"; touched by {touch}" + (f" from {_f(a['t_touch'])} s" if a.get("t_touch") is not None else "")
    fr += f"; closest palm {a.get('hand_min_side')} {_f(a.get('hand_min_cm'))} cm from the handle at {_f(a.get('hand_min_t'))} s"
    fr += f" ({_f(a.get('palm_dz_cm'))} cm above it, {_f(a.get('palm_dxy_cm'))} cm off sideways); highest palm {_f(a.get('palm_best_dz_cm'))} cm above the handle"  # fmt: skip
    fr += f"; fingers closed on the handle {a.get('grip_on_handle')}"
    if a.get("moved_by") is not None:
        fr += f"; moved by {', '.join(map(str, a.get('moved_by') or [])) or 'nothing'}"
    if a.get("pelvis_drop_cm") is not None:
        fr += f"; pelvis at most {_f(a.get('pelvis_drop_cm'))} cm below standing"
    return fr


def card_text(r: dict) -> list[str]:
    """One failed round as plain text (about a dozen lines), from the allow-listed record only."""
    raw = public(r.get("lml") or {})  # again: a round written by another version is filtered too
    instr = raw.get("instruction") or raw.get("prompt") or _goal(r.get("task", ""))
    lines = [
        f"## {r['label']}  ({r.get('failure_class') or 'unclassified'})",
        f"task: {r.get('task')}",
        f"instruction: {instr}",
    ]
    lim = f" of the {_f(raw['limit_s'])} s limit" if raw.get("limit_s") is not None else ""
    lines.append(f"outcome: {r.get('why') or 'failed'}; t_end {_f(r.get('t_end'))} s{lim}; ended {raw.get('end')}; fell {bool(r.get('fell'))}; dropped {bool(r.get('dropped'))}")  # fmt: skip
    lines.append(f"check: the benchmark's {raw.get('success')}, genuine {raw.get('genuine')}")
    bad = [s for s in r.get("steps") or [] if not s.get("ok", True)]
    if bad:
        s = bad[0]
        facts = ", ".join(f"{k} {_f(v)}" for k, v in s.items() if k not in ("step", "ok", "why", "t0", "t1"))
        lines.append(
            f"failed phase: {s['step']} from {_f(s['t0'])} to {_f(s['t1'])} s" + (f" ({facts})" if facts else "")
        )
    seq = " ".join(f"{s['step']}:{_f(s['t1'] - s['t0'])}" for s in r.get("steps") or [])
    if seq:
        lines.append(f"phase seconds: {seq}")
    see = [f"found {raw['found']}"] if raw.get("found") is not None else []
    for k in ("target_phrase", "target_score", "dest_phrase", "dest_score", "dest_in_view", "turns"):
        if raw.get(k) is not None:
            see.append(f"{k} {_f(raw[k])}")
    if see:
        lines.append("perception: " + ", ".join(see))
    meas = [f"{k} {_f(raw[k])}" for k in _MEASURES if raw.get(k) is not None]
    meas += [f"{k} {_f(v)}" for k, v in (raw.get("obj_dest_end") or {}).items()]
    meas += [f"nav {k} {_f(v)}" for k, v in (raw.get("nav") or {}).items() if k in ("end_dist_cm", "centre_err_cm") and v is not None]  # fmt: skip
    if raw.get("reach_ok") is not None:
        meas.append(f"reach_ok {raw['reach_ok']}")
    if meas:
        lines.append("measurements: " + ", ".join(meas))
    art = raw.get("art") or {}
    if art:
        lines.append(door_text(art, "drawer" if "Drawer" in str(r.get("task")) else "door"))
    for k, m in (raw.get("marks") or {}).items():
        lines.append(f"mark {k}: " + ", ".join(f"{a} {_f(b)}" for a, b in m.items()))
    comply = raw.get("comply") or {}
    if comply.get("why") or raw.get("t_check") is not None:
        cf = raw.get("comply_facts") or {}
        keys = ("t_check", "hand_at_check", "fell_at_check", "t_carry", "best_run_s", "at_check")
        facts = ", ".join(f"{k} {_f(cf[k])}" for k in keys if cf.get(k) is not None)
        lines.append(f"comply: {comply.get('ok')} -- {comply.get('why') or 'ok'}" + (f"; contacts: {facts}" if facts else ""))  # fmt: skip
    st = r.get("set") or {}
    lines.append("settings that differ from the default: " + (", ".join(f"{k}={st[k]}" for k in sorted(st)) or "none"))
    return lines[:60]


def cards(rounds: list[dict], out) -> Path:
    """Failure cards for the Reviewer: <out>/cards.md (or `out` itself when it ends in .md). A two-line header with
    the wins per task, then one card per failed round."""
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
    lines += B.segment_header(rounds)  # rules_lml 48
    for r in fails:
        lines += card_text(r) + [""]
    f.write_text("\n".join(lines))
    return f


# ------------------------------------------------------------------------------------------ skill package
def package_parameters(root: Path) -> dict:
    """The package's parameter dict read from the source text: skill_simple.py's K_ORIG, K_CUT and PLACING keywords
    (backend_simple.package_parameters) and the keywords of `EXTRA` in the package's __init__.py (lift, nav_gap).
    The package imports the bed tree, so the loop's process never imports it."""
    out = B.package_parameters(root)
    init = Path(root) / "__init__.py"
    if init.exists():
        for node in ast.parse(init.read_text()).body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and any(isinstance(t, ast.Name) and t.id == "EXTRA" for t in node.targets)
            ):
                for kw in node.value.keywords:
                    if kw.arg:
                        try:
                            out[kw.arg] = ast.literal_eval(kw.value)
                        except ValueError:
                            out[kw.arg] = ast.unparse(kw.value)
    return out


# The stance keys of the seed's flows (`env.stance` in skill_simple.episode and skill_lml_cam.place): the pick's d, t (m),
# yaw (deg) and here (bool), the placing stance's d2, t2 (m) and stay (bool). There is no yaw2 (the placing heading is the
# bearing to the receptacle) and the walking tasks read none. Anything else under `stance_` is refused here, because the
# bed would accept it and ignore it.
STANCE_KEYS = {"d": float, "t": float, "yaw": float, "here": bool, "d2": float, "t2": float, "stay": bool}


class SkillPackage(B.SkillPackage):
    def __init__(self, root: Path, bed_keys=(), fixed=()):
        super().__init__(root, bed_keys, fixed)
        self.defaults = package_parameters(self.root)

    def apply_set(self, _pkg, overrides: dict | None = None) -> dict:
        bad = []
        for k, v in (overrides or {}).items():
            if not k.startswith(STANCE):
                continue
            kind = STANCE_KEYS.get(k[len(STANCE) :])
            ok = kind is bool and isinstance(v, bool) or kind is float and isinstance(v, (int, float)) and not isinstance(v, bool)  # fmt: skip
            if not ok:
                bad.append(k)
        if bad:
            raise KeyError(f"unknown or ill-typed stance settings {bad}; allowed: " + ", ".join(f"{STANCE}{k}" for k in STANCE_KEYS))  # fmt: skip
        return super().apply_set(_pkg, overrides)


def load_skills(root, name: str = "scene_skills", family=None) -> SkillPackage:
    fam = B._fam(family) if family is not None else {}
    return SkillPackage(Path(root), fam.get("bed_keys", []), B.bed_fixed(fam))


# ----------------------------------------------------------------------------------------------- signature
def bench_commit(fam: dict) -> str:
    """The benchmark's git commit (family key `benchmark`, a path below bed_root; the sandbox cuts its own third_party
    view, the bed tree's stays); "" when the family names none, "unknown" when it cannot be read."""
    rel = fam.get("benchmark")
    if not rel:
        return ""
    try:
        r = subprocess.run(["git", "-C", str(bed_root(fam) / rel), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)  # fmt: skip
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def signature(family, where: str = "probe") -> str:
    """sha1 over the bed files and the files only hashed (`hash_files`: server, simulator, truth skill, runner ...), the
    track, the base, the family's `bed_env`, its fixed bed switches (`bed_fixed`, [bed-fixed]) and the benchmark's commit.
    One bed serves every call: `where` does not enter."""
    fam = B._fam(family)
    h = hashlib.sha1()
    root = bed_root(fam)
    for rel in [*fam.get("bed_files", []), *fam.get("hash_files", [])]:
        p = root / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    for part in (fam.get("track", ""), str(fam.get("base", "")), bench_commit(fam)):
        h.update(b"\0" + part.encode())
    if fam.get("bed_env"):
        h.update(b"\0" + json.dumps(fam["bed_env"], sort_keys=True).encode())
    if B.bed_fixed(fam):  # absent: the hash of before the key existed; present, no round of the old bed is reused
        h.update(b"\0fixed" + json.dumps(B.bed_fixed(fam), sort_keys=True).encode())
    return h.hexdigest()[:10]


# -------------------------------------------------------------------------------------------------- layout
def layout(family) -> dict:
    fam = B._fam(family)
    plan = fam.get("plan")
    return dict(
        skills=ensure_package(fam),
        rules=B._abs(fam["rules"]),
        prompts=B._abs(fam["prompts"]),
        plan=B._abs(plan) if plan else None,
        leak=list(LEAK),
        stub_file="pick.py",  # __init__.py is frozen for the guard, so a stub draft appends its comment to an editable file
        class_suggest=dict(CLASS_SUGGEST),
        caps={**B.FALLBACK_CAPS, **fam.get("caps", {})},
    )


# ------------------------------------------------------------------------------------------------ run_jobs
def bed_env(fam: dict, package: Path, run_dir: Path, full: bool = False) -> dict:
    """The environment of a runner call: ours, without EXAM_OPEN, plus the loop's variables (LML_SKILLS the package,
    LML_PRACTICE the practice files this call is given, LML_RUN_DIR the call's scratch and cache root, LML_LOOP, and
    LML_SLOTS the simulator slots this machine's loop may take)."""
    env = {k: v for k, v in os.environ.items() if k != "EXAM_OPEN"}
    slots = int(fam.get("official_slots", 2))
    env.update(
        NUMPY_MADVISE_HUGEPAGE="0",
        OMP_NUM_THREADS="2",
        BFM_ROOT="true",
        MUJOCO_GL="egl",
        PYTHONPATH="src",
        LML_SKILLS=str(Path(package).resolve()),
        LML_LOOP="1",
        LML_PRACTICE=str(practice_dir(fam, full)),
        LML_RUN_DIR=str(run_dir),
        LML_SLOTS=" ".join(str(k) for k in range(1, slots + 1)),
    )
    env.update({k: str(v) for k, v in (fam.get("bed_env") or {}).items()})
    env.pop("EXAM_OPEN", None)
    return env


def runner_call(fam: dict, task: str, ids: list[str], setting: dict, out, env: dict, log, stop=None) -> int:
    """One call of the family's `runner` (run_lml_q.sh) for one task: `<runner> <Task> --ids F --setting J [--stance J]
    --out F`, in the bed root. The family's `bed_cmd` (a fake runner in a test) replaces the script. The runner takes a
    slot, makes the looks and grasps, runs the batch and stops what it started; a set `stop` takes its whole process
    tree down."""
    out = Path(out)
    ids_file = out.with_name(f"ids_{task}.txt")
    ids_file.write_text("\n".join(ids) + "\n")
    skill, stance = split_set(setting or {})
    args = [task, "--ids", str(ids_file), "--setting", json.dumps(skill), "--out", str(out)]
    args += ["--stance", json.dumps(stance)] if stance else []
    if fam.get("bed_cmd"):
        argv = [*fam["bed_cmd"], *args]
        return B._exec(argv, env, log, stop, None, tree=True)
    prog = 'cd "$1" && . scripts/paths.sh && export TMPDIR="${TMPDIR:-$LOCOMANI_TMP}" && shift && exec "$@"'
    argv = ["bash", "-c", prog, "bash", str(bed_root(fam)), *shlex.split(fam["runner"]), *args]
    return B._exec(argv, env, log, stop, None, tree=True)


def _cache(fam: dict, sh: str, sig: str) -> Path:
    """The look and grasp cache of one skill package under one bed signature (the look runs the package's view code, so
    a changed package must not read another's clouds). It lives in the scratch directory, outside every sandbox, and the
    six newest are kept."""
    root = B._scratch() / f"lml_{fam['name']}_cache"
    root.mkdir(parents=True, exist_ok=True)
    keep = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)[5:]
    for p in keep:
        shutil.rmtree(p, ignore_errors=True)
    d = root / f"{sh}_{sig}"
    d.mkdir(exist_ok=True)
    return d


def _blank(label: str, why: str, sh: str, setting: dict, sig: str, cls: str = "guard") -> dict:
    d = B._blank(label, why, sh, setting, sig, cls)
    d["lml"] = d.pop("simple")
    return d


def run_jobs(jobs, family, pool, procs, log, gpus, stop=None, where: str = "probe") -> dict:
    """Run the jobs' missing rounds; returns {job index: labels still missing} as batch.run_jobs does. A job has
    `skills` (the package directory), `out`, `overrides` (the setting) and `labels` (None = the whole pool). One bed
    serves probes and the gate; `where` only selects the practice files (the full ones for "gate"). Each job is guarded
    first; one that fails gets ok False rounds ("guard: <violation>") and runs nothing."""
    fam = B._fam(family)
    pool, log = Path(pool), Path(log)
    sig = signature(fam, where)
    every = sorted(p.stem for p in pool.glob("*.json") if not p.name.startswith("_"))
    log.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    base = package_dir(fam)
    tmp = B._scratch()
    slots = int(fam.get("official_slots", 2))
    fixed = B.bed_fixed(fam)
    with log.open("a") as fh:
        for k, job in enumerate(jobs):
            out, setting = Path(job.out), dict(job.overrides or {})
            ran = {**setting, **fixed}  # what the bed runs: the fixed switches over the job's set ([bed-fixed])
            out.mkdir(parents=True, exist_ok=True)
            todo = [x for x in (job.labels or every) if not B._current(out / f"{x}.json", sig)]
            for x in todo:  # a round made under another signature is kept aside, not read again
                f = out / f"{x}.json"
                if f.exists():
                    f.rename(f.with_name(f"_old_{x}.json"))
            if not todo or (stop is not None and stop.is_set()):
                result[k] = todo
                continue
            sh = skills_hash(Path(job.skills))
            bad = B.run_guard(fam, base, Path(job.skills))
            if bad is not None:
                for x in todo:
                    B._write(out / f"{x}.json", _blank(x, f"guard: {bad}", sh, setting, sig))
                result[k] = []
                continue
            by = B._ids_by_task(pool, todo)
            pdir = practice_dir(fam, where == "gate")  # the files this call is given (a sandbox: blanked rows)
            gone = [x for rows in by.values() for x, eid in rows if not B._in_practice(pdir, eid)]
            for x in gone:  # an id that is blank or missing there is not an episode this call may run
                B._write(out / f"{x}.json", _blank(x, "not a diagnostic episode", sh, setting, sig, "refused"))
            by = {t: [(x, e) for x, e in rows if x not in gone] for t, rows in by.items()}
            by = {t: rows for t, rows in by.items() if rows}
            if not by:
                result[k] = []
                continue
            tmp.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix=f"lml_{fam['name']}_{k}_", dir=tmp))
            (run_dir / "cloud").symlink_to(_cache(fam, sh, sig))  # the runner's cache is $LML_RUN_DIR/cloud
            env = bed_env(fam, Path(job.skills), run_dir, full=where == "gate")
            by_id = {eid: x for rows in by.values() for x, eid in rows}

            def unit(u, setting=ran, env=env, run_dir=run_dir):
                i, (task, items) = u
                d = (
                    run_dir / f"u{i:02d}"
                )  # rows and the ids file of a unit; the run directory is shared, so is its lock
                d.mkdir()
                return runner_call(fam, task, [e for _, e in items], setting, d / f"rows_{task}.jsonl", env, fh, stop)

            with ThreadPoolExecutor(max_workers=slots) as ex:
                list(ex.map(unit, enumerate(B._units(by, slots))))
            for f in sorted(run_dir.glob("u*/rows_*.jsonl")):
                for line in f.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    x = by_id.get(rec.get("episode_id"))
                    if x is not None:
                        B._write(out / f"{x}.json", to_round(rec, x, sh, setting, sig))
            result[k] = [x for x in todo if not B._current(out / f"{x}.json", sig)]
            if not result[k]:
                shutil.rmtree(run_dir, ignore_errors=True)  # (the cache is a link: its target stays)
    return result


# ------------------------------------------------------------------------------------------------ sandbox
def sandbox(dest, family) -> None:
    """backend_simple.sandbox with this backend's forbidden paths: under `<dest>/<sandbox.dir>` the seed package, the
    loop-mode bed files (`bed_files`, read only), and the practice files with the hidden rows blanked; the full ones go
    to the hidden tree. The hand-tuning outputs, the hand-off records, the un-stripped skill files, the master seed and
    the benchmark's own source tree are cut (the sandbox's third_party is a link view: LocoManiLab is left out of it)."""
    B.sandbox(dest, family, forbidden=FORBIDDEN_PATHS)


def audit(dest) -> list[str]:
    """What in the sandbox `dest` a role must not see: forbidden paths that exist, practice rows of hidden ids that are
    not blank, leak patterns in text files. [] when clean."""
    return B.audit(dest, forbidden=FORBIDDEN_PATHS, leak=LEAK, backend="lml")
