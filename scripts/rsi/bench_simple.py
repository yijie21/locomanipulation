"""The SIMPLE adapter's rows (`locomani.bench.simple`): what a record means, how a failed round reads on a card, what a role
must not see (TODO 54 phase 3; moved from backend_simple.py, the rows and cards are unchanged).

A SIMPLE record has `success` (the environment's check), `comply` (the compliance check), `loss` (the failure class), `phases`
(seconds per phase), `marks`, and for the articulated tasks `art` / `art_touch`. `ok` = success and compliance. The raw record is
kept under `simple`. The failure classes and the phase each is blamed on (BLAME; BLAME_ART for the articulated tasks) are this
adapter's; the leak patterns and forbidden paths name the released evaluation episodes and the hand-tuning outputs."""

from __future__ import annotations

from pathlib import Path

import bench_common as C
import bench_rows as R

ROW_KEY = "simple"
CONTROLLER = "scalebfm"
HERE = C.HERE
# the phase a failure class is blamed on (plan table C); a class not listed, or a phase the record lacks: the last one
BLAME = {"deliver": "reach_grasp", "grasp": "lift", "place": "release", "collide": "close", "perceive": "look"}
# the articulated tasks (family simple_art): the record has `art`; their phases are walk, reach, close, move, release
BLAME_ART = {"deliver": "reach", "grasp": "close", "short": "move", "collide": "move"}
CLASS_SUGGEST = {
    "error": "the skill or the bed raised: read the error text, fix the code path before tuning anything",
    "timeout": "the check passed after the step limit: shorten the slow phases (walk, settle, search), not the grasp",
    "fall": "the robot fell: slower or shorter moves, a wider stance, less arm reach while the feet move",
    "perceive": "the target or the basket was not found: look from another place or angle before giving up",
    "deliver": "the palm ended far from the grasp: check the reach target, the grasp choice and the arm's error",
    "grasp": "the fingers closed and the object did not come up or was lost: grip width, force, palm position",
    "short": "the part moved but not far enough: keep the hand on it and push or pull through to the threshold",
    "collide": "the target was pushed or tipped before the fingers closed: approach path, clearance, entry speed",
    "place": "carried and released, but it did not stay in the basket: release point and height, hand open timing",
    "comply": "the environment's check passed but not what the instruction says: do what the sentence asks",
}
LEAK = [
    r"\bL[0-9]/e[0-9]+",  # a released evaluation episode id
    r"\bLhard\b",
    r"EXAM_OPEN",
    r"pool_(?:accept|sealed)",  # the hidden pools' directory names
    r"scene_rsi_20261016_hidden",
    r"outputs/simple_eval_20261112|simple_eval_20261112/x\b|dr-level-(?:[0-9]|hard)",  # the released episodes, not the code folder of that name
    r"escalation_20261116/official/(?:exam|rehearsal|exam_aborted|ho|q_seed|videos)",  # exam and hand-tuning outputs of the bed tree
    r"escalation_20261116/(?:a|b|practice|comply)/",
    r"locomanipulation-esc/outputs",  # the bed tree's outputs by absolute path: the full practice files, clouds of hidden episodes
    r"best_official|seed_pkg_removed|teleop_wbc",
    r"SIMPLE/data/(?:teleop_|mp_full|evals_|replay_)",  # the benchmark's evaluation-style and demonstration sets
    r"handoff/(?:escalation|benchmarks)",
]
# Level 2: a stance is given as `set` keys with this prefix and reaches the bed as its --stance argument
# (d, t, yaw for the picking stance, d2, t2, yaw2 for the placing stance, here: no walk; see skill_simple.py's header).

# What an Engineer-visible tree must not hold (relative to the sandbox root; `*` globs)
FORBIDDEN_PATHS = (
    "outputs/simple_eval_20261112",
    "outputs/escalation_20261116/official/exam",
    "outputs/escalation_20261116/official/rehearsal",
    "outputs/escalation_20261116/official/exam_aborted_0519",
    "outputs/escalation_20261116/a",
    "outputs/escalation_20261116/b",
    "outputs/escalation_20261116/practice",
    "third_party/SIMPLE/data/teleop_wbc",
    "third_party/SIMPLE/data/teleop_full",  # the benchmark's demonstration sets (100 episodes a task) and replays
    "third_party/SIMPLE/data/teleop_fix",
    "third_party/SIMPLE/data/mp_full",
    "third_party/SIMPLE/data/evals_wbc_replay",
    "third_party/SIMPLE/data/replay_decoupled_wbc",
    "docs/handoff/escalation-*.md",
    "docs/handoff/benchmarks-*.md",
)


def to_round(rec: dict, label: str, skills_hash: str, set_: dict, signature: str) -> dict:
    """A bed record as the loop's round dict (seam plan table C). Pure; the raw record is kept under `simple`."""
    comply = rec.get("comply")
    comply_ok = bool(comply.get("ok")) if isinstance(comply, dict) else False  # a missing check counts as false
    success = bool(rec.get("success"))
    ok = success and comply_ok
    loss = rec.get("loss") or ("error" if rec.get("error") else None)
    if ok:
        cls = None
    elif loss:
        cls = loss
    elif success:
        cls = "comply"
    else:
        cls = "unclassified"
    text = rec.get("why") or rec.get("error")
    if cls == "comply" and not text:
        text = (comply or {}).get("why") if isinstance(comply, dict) else None
        text = text or "no compliance check in the record"
    why = "" if ok else (f"{cls}: {text}" if text else str(cls))
    marks = rec.get("marks") or {}
    end = marks.get("end") or marks.get("before_open") or marks.get("after_lift") or {}
    rise = R.sf(end.get("rise_cm"))
    plan_text = rec.get("why") or rec.get("error") or ""
    steps = _steps(rec, marks, cls, ok)
    task = rec.get("episode_id", label).split("/")[0] if rec.get("episode_id") else label.rsplit("_s", 1)[0]
    out = dict(
        label=label,
        task=task,
        ok=ok,
        why=why,
        fell=rec.get("fell") is not None,
        dropped=bool(not ok and rise is not None and rise < -15),
        plan_ok=not plan_text,
        plan_why=str(plan_text),
        error=rec.get("error"),
        steps=steps,
        t_end=rec.get("t_end"),
        failure_class=cls or "",
        skills=skills_hash,
        set=dict(set_ or {}),
        wall_s=rec.get("wall_s"),
        rule=C.success_rule(),
        controller="scalebfm",
        motor=signature,
        simple=rec,
    )
    if isinstance(rec.get("bench"), dict) and rec["bench"].get("signature"):  # a bench adapter's row: its switches
        out["bench"] = rec["bench"]["signature"]
    return out


def _steps(rec: dict, marks: dict, cls: str | None, ok: bool) -> list[dict]:
    """One step per phase in order, t0 / t1 from the summed seconds; the phase the class names has ok False."""
    phases = rec.get("phases") or {}
    names = list(phases)
    blame = None
    if not ok and names:
        want = (BLAME_ART if isinstance(rec.get("art"), dict) else BLAME).get(cls or "")
        blame = want if want in phases else names[-1]
    facts = {
        "reach_grasp": dict(palm_err_cm=R.sf(rec.get("palm_err_cm")), reach_ok=rec.get("reach_ok")),
        "close": {k: marks.get("before_close", {}).get(k) for k in ("moved_cm", "tilt_deg")},
        "lift": {k: marks.get("after_lift", {}).get(k) for k in ("rise_cm", "held")},
        "release": {k: marks.get("before_open", {}).get(k) for k in ("basket_cm", "held")},
    }
    art, touch = rec.get("art"), rec.get("art_touch")
    if isinstance(art, dict):  # articulated record: the facts come from art / art_touch, not from the pick marks
        touch = touch if isinstance(touch, dict) else {}
        facts = {
            "reach": dict(t_hand=R.sf(touch.get("t_hand"))),
            "close": dict(t_hand=R.sf(touch.get("t_hand"))),
            "move": dict(start_deg=R.sf(art.get("start_deg")), best_deg=R.sf(art.get("best_deg")),
                         need_deg=R.sf(art.get("need_deg"))),
        }  # fmt: skip
    return R.phase_steps(phases, blame, cls, facts)


def _goal(task: str) -> str:
    return R.goal(task, [HERE / "families" / n for n in ("simple_q_tasks.jsonl", "simple_art_tasks.jsonl")])


_f = R.f


def art_lines(raw: dict) -> list[str]:
    """An articulated record's joint story in plain words (one or two lines); [] for a record without `art`."""
    art = raw.get("art")
    if not isinstance(art, dict):
        return []
    touch = raw.get("art_touch") if isinstance(raw.get("art_touch"), dict) else {}
    sign = art.get("sign")
    way = "toward success" if sign not in (1, -1) else ("up" if sign > 0 else "down")
    sd, bd, nd = art.get("start_deg"), art.get("best_deg"), art.get("need_deg")
    line = (f"joint {art.get('joint')}: started at {_f(sd)} deg, furthest {_f(bd)} deg, needed {_f(nd)} deg "
            f"(success lies {way}), ended at {_f(art.get('end_deg'))} deg")  # fmt: skip
    if art.get("t_move") is not None:
        line += f"; it first moved 3 deg at {_f(art['t_move'])} s"
    out = [line]
    th, tb = touch.get("t_hand"), touch.get("t_body")
    parts = ["a hand first touched the moving part at " + f"{_f(th)} s" if th is not None
             else "no hand ever touched the moving part"]  # fmt: skip
    if th is not None and touch.get("hand_deg") is not None:
        parts.append(f"{_f(touch['hand_deg'])} deg were made with a hand on it")
    if tb is not None or touch.get("body_links"):
        links = ", ".join(touch.get("body_links") or []) or "a link"
        parts.append(f"a non-hand link touched it ({links})" + (f" from {_f(tb)} s" if tb is not None else ""))
    out.append("; ".join(parts))
    return out


def card_text(r: dict) -> list[str]:
    """One failed round as plain text (about a dozen lines)."""
    raw = r.get("simple") or {}
    instr = raw.get("instruction") or r.get("instruction") or _goal(r.get("task", ""))
    lines = [
        f"## {r['label']}  ({r.get('failure_class') or 'unclassified'})",
        f"task: {r.get('task')}",
        f"instruction: {instr}",
    ]
    if r.get("bench"):
        lines.append(f"bench: {r['bench']}")
    lines.append(f"outcome: {r.get('why') or 'failed'}; t_end {_f(r.get('t_end'))} s; fell {bool(r.get('fell'))}; dropped {bool(r.get('dropped'))}")  # fmt: skip
    lines += R.failed_phase_lines(r)
    lines += art_lines(raw)
    if raw.get("palm_err_cm") is not None:
        lines.append(f"palm error: {_f(raw['palm_err_cm'])} cm; reach_ok {raw.get('reach_ok')}")
    gd = raw.get("grasp_desc")
    if (
        isinstance(gd, list) and len(gd) >= 6
    ):  # [scene-robot]: where on the object the chosen grasp sits, as the camera saw it
        lines.append(
            f"grasp chosen: palm {_f(100 * gd[0])} cm ahead, {_f(100 * gd[1])} cm left, {_f(100 * gd[2])} cm up of the"
            f" middle of the seen points; palm normal ({_f(gd[3], 2)}, {_f(gd[4], 2)}, {_f(gd[5], 2)}) in the same axes"
            + (f"; model score {_f(raw['score'], 2)}" if isinstance(raw.get("score"), (int, float)) else "")
        )
    for k, m in (raw.get("marks") or {}).items():
        if isinstance(m, dict):
            lines.append(f"mark {k}: " + ", ".join(f"{a} {_f(b)}" for a, b in m.items()))
    comply = raw.get("comply") if isinstance(raw.get("comply"), dict) else {}
    if comply.get("why") or raw.get("t_check") is not None:
        cf = raw.get("comply_facts") or {}
        keys = ("t_hand", "t_carry", "best_held_s", "t_right", "t_left_ok", "best_left_s", "t_container")
        facts = ", ".join(f"{k} {_f(cf[k])}" for k in keys if cf.get(k) is not None)
        first = raw.get("t_check") if raw.get("t_check") is not None else raw.get("t_success")
        lines.append(
            f"comply: {comply.get('ok')} -- {comply.get('why') or 'ok'}"
            + (f"; the environment's check first held at {_f(first)} s" if raw.get("success") else "")
            + (f"; contacts: {facts}" if facts else "")
        )
    lines.append(R.settings_line(r))
    return lines[:60]


def load_skills(root, name: str = "scene_skills", family=None) -> C.SkillPackage:
    """The package at `root`, checked for the keys a `set` may hold (the package's, the `stance_` ones, and the family's
    `bed_keys`). `name` is accepted for the scene's call shape."""
    fam = family or {}
    return C.SkillPackage(Path(root), fam.get("bed_keys", []), C.bed_fixed(fam))
