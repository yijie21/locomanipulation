"""The LocoManiLab adapter's rows (`locomani.bench.locomanilab`; plan lml_backend_plan.md, steps 7 to 9; 2026-11-20; moved from
backend_lml.py, TODO 54 phase 3, rows and cards unchanged).

- to_round: `ok` = the benchmark's check and the compliance check and the benchmark's own `genuine` flag; the raw record is
  kept under `lml` in its allow-listed form (`public`);
- cards: built from an ALLOW-LIST of record fields. The scene is fixed, so a room coordinate learned in practice is the
  evaluation's too: no position in the room's frame (true_centre, true_grasp, dest_true, target, any *_true position, the
  spec's regions, the navigation log) is on a card or in the stored round;
- the skill package's parameters (the keys of skill.py's dicts and `EXTRA` of the package's __init__.py), the stance keys, the
  leak patterns and forbidden paths."""

from __future__ import annotations

import re
from pathlib import Path

import bench_common as C
import bench_rows as R

ROW_KEY = "lml"
CONTROLLER = "scalebfm"
STANCE = C.STANCE
HERE = C.HERE
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
    rise = R.sf(end.get("rise_cm"))
    plan_text = _text(rec.get("why") or rec.get("error")) or ""
    pub = public(rec)
    task = rec["episode_id"].split("/")[0] if rec.get("episode_id") else label.rsplit("_s", 1)[0]
    t_end = rec.get("t_end")
    if t_end is None and R.sf(rec.get("steps")) is not None:
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
        rule=C.success_rule(),
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
        "walk": dict(stop_err_cm=R.sf(rec.get("stop_err_cm")), walk_cm=R.sf(rec.get("walk_cm"))),
        "reach_grasp": dict(palm_err_cm=R.sf(rec.get("palm_err_cm")), reach_ok=rec.get("reach_ok")),
        "close": {k: marks.get("before_close", {}).get(k) for k in ("moved_cm", "tilt_deg")},
        "lift": {k: marks.get("after_lift", {}).get(k) for k in ("rise_cm", "held")},
        "open": {**{k: marks.get("before_open", {}).get(k) for k in ("basket_cm", "held")}, **od},
    }
    return R.phase_steps(phases, blame, cls, facts)


def _goal(task: str) -> str:
    """The task's instruction from a family's tasks file (a record from the bed carries its own; this is the fallback)."""
    return R.goal(task, sorted((HERE / "families").glob("lml*_tasks.jsonl")))


_f = R.f
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
    lines += R.failed_phase_lines(r)
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
    lines.append(R.settings_line(r))
    return lines[:60]


# ------------------------------------------------------------------------------------------ skill package
def package_parameters(root: Path) -> dict:
    """The package's parameter dict read from the source text: skill.py's K_ORIG, K_CUT and PLACING keywords (and K_ART's) and
    the keywords of `EXTRA` in the package's __init__.py (lift, nav_gap). The package imports the bench, so the loop's process
    never imports it."""
    return {**C.package_parameters(root), **C.extra_parameters(root)}


# The stance keys of the seed's flows (`env.stance` in the skill's episode and place): the pick's d, t (m), yaw (deg) and here
# (bool), the placing stance's d2, t2 (m) and stay (bool). There is no yaw2 (the placing heading is the bearing to the
# receptacle) and the walking tasks read none. Anything else under `stance_` is refused here, because the bed would accept it
# and ignore it.
STANCE_KEYS = {"d": float, "t": float, "yaw": float, "here": bool, "d2": float, "t2": float, "stay": bool}


class SkillPackage(C.SkillPackage):
    def __init__(self, root: Path, bed_keys=(), fixed=()):
        super().__init__(root, bed_keys, fixed, params=package_parameters)

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
    fam = family or {}
    return SkillPackage(Path(root), fam.get("bed_keys", []), C.bed_fixed(fam))
