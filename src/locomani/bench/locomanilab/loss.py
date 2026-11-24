"""Why a LocoManiLab bed record (run.py) is not a success, from the record alone (2026-11-20).

    loss_of(record) -> None | one of the classes below      (None: a success; with a `comply` field, a compliant one)

error     the bed or the server raised (`error`, or end == "error"); also a success whose `genuine` is False (the
          check held at a start state: not an attempt, so a measurement fault)
timeout   the environment stopped at its limit (`end == "limit"`) and no evidence says the flow had failed: it was
          still on its way (never reached the grasp, or held the object and had not released it)
fall      the robot fell (`fell` is set), or a success with `comply.why == "fell"`
perceive  the target or the destination was not found (`found` False, or `why` "no target seen" / "no basket seen" ...)
navigate  a navigation task that did not reach
collide   the target was pushed 3 cm or tipped 20 deg before the fingers closed, or the receptacle was pushed 3 cm
          before the release (`basket_cm`, when the bed records it), or a success with `comply.why == "on_support"`
deliver   the palm was more than 2 cm from the grasp (or from the release point), a reach step failed, or the flow
          ended before any grasp
grasp     the fingers closed and the object did not come up, or was lost on the way; or a success that `comply` rejects
          as no_hand / short / not_carried
art_*     (door and drawer tasks, family articulation / navigation_articulation, fork) after error, fall, perceive:
          art_not_reached (no palm within 5 cm of the handle and no hand or arm touch), art_no_move (fraction moved 0.03
          or less), art_partial (moved, never inside the goal), art_not_held (inside the goal for less than the required
          hold), then timeout; a success rejected by comply is fall ("fell before ...") or error ("not genuine ..."), and
          (bed3, drawers) art_not_reached when the drawer was moved by a leg or by no hand ("drawer moved by a leg" /
          "drawer not moved by a hand": the hand never worked the drawer)
          Family articulation_pnp (LowDrawerPnP, bed3) after error, fall, perceive, in this order of evidence:
          the apple's pick when a grasp was tried and the apple did not come up (collide / deliver / grasp, as below);
          the drawer never opened to the stage-1 threshold (art_not_reached, art_no_move, art_partial); no grasp tried
          at all (deliver, or timeout at the limit); the apple lost, not released or not delivered (grasp / deliver /
          timeout / collide, as below); then, released: the drawer left open (art_partial), the apple not inside or not
          staying (place), inside and closed for less than the hold (art_not_held), else timeout. A success that comply
          rejects as not_carried is grasp, as hand_on is place.
place     carried to the destination, released, and it did not stay; or a success with `comply.why == "hand_on"`

The order is the evidence's strength: error, fall, perceive, then the family's own evidence; `timeout` only when the
limit was reached with nothing else wrong. Fields read: error, end, fell, found, why, success, genuine, comply,
family, phases, marks (before_close / after_lift / before_open / end: moved_cm, tilt_deg, rise_cm, held,
basket_cm), palm_err_cm, reach_ok, release_cm.

"""

import re

CLASSES = ("error", "timeout", "fall", "perceive", "navigate", "deliver", "grasp", "collide", "place", "art_not_reached",
           "art_no_move", "art_partial", "art_not_held")  # fmt: skip
ART = (
    "articulation",
    "navigation_articulation",
    "articulation_pnp",
)  # the fridge tasks (bed2), the drawer tasks (bed3)
ARM_PARTS = ("hand", "forearm", "upper_arm", "arm")  # touch names of the arm (bed2 recorded `arm`; bed3 splits it)
ART_REACH_CM, ART_MOVE = (
    5.0,
    0.03,
)  # a palm within 5 cm of the handle reached it; the door moved if its fraction changed more
COMPLY_CLASS = {
    "fell before the door check held": "fall",
    "not genuine (too few steps)": "error",
    "drawer not moved by a hand": "art_not_reached",
    "drawer moved by a leg": "art_not_reached",
} | dict(  # fmt: skip
    fell="fall", on_support="collide", no_hand="grasp", short="grasp", not_carried="grasp", hand_on="place"
)
LIFT_RISE_CM = 5.0  # the pick counts as lifted at this rise
BEFORE_CLOSE_MOVED_CM, BEFORE_CLOSE_TILT_DEG, BASKET_CM, PALM_ERR_CM = 3.0, 20.0, 3.0, 2.0
NOT_FOUND = ("no target seen", "no basket seen", "no destination seen", "target not found")


def _ladder(a: dict, goal_hold: bool = True) -> str:
    """The drawer / door evidence in order: art_not_reached, art_no_move, art_partial, art_not_held, timeout."""
    hm = a.get("hand_min_cm")
    if hm is not None and hm > ART_REACH_CM and not set(ARM_PARTS) & set(a.get("touch") or []):
        return "art_not_reached"
    f0, hi, lo = a.get("frac_start"), a.get("frac_max"), a.get("frac_min")
    if f0 is None or hi is None or lo is None or max(hi - f0, f0 - lo) <= ART_MOVE:
        return "art_no_move"
    if a.get("t_goal") is None:
        return "art_partial"
    need = re.findall(
        r"held ([0-9.]+) s", a.get("goal") or ""
    )  # (the pnp's goal has two: the last is the final stage's)
    if goal_hold and need and (a.get("hold_best_s") or 0.0) < float(need[-1]) - 1e-6:
        return "art_not_held"
    return "timeout"


def art_loss(r: dict) -> str:
    """A door or drawer task that is not a success (error, fall and perceive have been looked at): art_not_reached (no palm
    within 5 cm of the handle and no hand or arm on the door), art_no_move (the fraction never moved 0.03 from the start),
    art_partial (moved, never inside the goal), art_not_held (inside the goal, not for the required time), else timeout."""
    return _ladder(r.get("art") or {})


def pnp_loss(r: dict) -> str:
    """LowDrawerPnP that is not a success: see the module docstring for the order of evidence."""
    a = r.get("art") or {}
    m = r.get("marks") or {}
    bc, al, bo, end = m.get("before_close"), m.get("after_lift"), m.get("before_open"), m.get("end") or {}
    limit = r.get("end") == "limit"
    palm_bad = (r.get("palm_err_cm") or 0.0) > PALM_ERR_CM or not all(r.get("reach_ok") or [True])
    if r.get("why") == "no grasp":
        return "deliver"
    lifted = al is not None and al.get("held") and (al.get("rise_cm") or 0.0) >= LIFT_RISE_CM
    if bc is not None and not lifted:  # a grasp was tried and the apple did not come up
        if (bc.get("moved_cm") or 0.0) > BEFORE_CLOSE_MOVED_CM or (bc.get("tilt_deg") or 0.0) > BEFORE_CLOSE_TILT_DEG:
            return "collide"
        return "deliver" if palm_bad else "grasp"
    first = re.search(r"fraction >= ([0-9.]+)", a.get("goal") or "")
    if a.get("frac_max") is None or first is None or a["frac_max"] < float(first.group(1)) - 1e-6:  # stage 1 never done
        return _ladder(dict(a, t_goal=None), goal_hold=False)
    if bc is None:  # the drawer was opened, no grasp tried
        return "timeout" if limit and not palm_bad else "deliver"
    if bo is not None and not bo.get("held"):
        return "grasp"  # lost on the way
    if bo is None:  # carried, never reached the release
        if limit and end.get("held"):
            return "timeout"
        return "grasp" if not end.get("held") else "deliver"
    if (bo.get("basket_cm") or 0.0) > BASKET_CM:
        return "collide"
    if (r.get("release_cm") or 0.0) > PALM_ERR_CM:
        return "deliver"
    if a.get("t_goal") is None:  # released: the drawer was not closed on the apple, or the apple is not inside
        close = re.search(r"fraction <= ([0-9.]+)", a.get("goal") or "")
        if close and (a.get("frac_end") or 0.0) > float(close.group(1)) + 1e-6:
            return "art_partial"
        return "place"
    return _ladder(a)


def loss_of(r: dict) -> str | None:
    if "error" in r or r.get("end") == "error":
        return "error"
    if r.get("success"):
        if r.get("genuine") is False:
            return "error"
        c = r.get("comply")
        if c is None or c.get("ok", True):
            return None
        return COMPLY_CLASS.get(c.get("why"), "grasp")
    if r.get("fell") is not None:
        return "fall"
    if r.get("found") is False or r.get("why") in NOT_FOUND:
        return "perceive"
    fam = r.get("family")
    if fam == "articulation_pnp":
        return pnp_loss(r)
    if fam in ART:
        return art_loss(r)
    if fam == "navigation":
        return "navigate"
    limit = r.get("end") == "limit"
    m = r.get("marks") or {}
    bc, al, bo, end = m.get("before_close"), m.get("after_lift"), m.get("before_open"), m.get("end") or {}
    palm_bad = (r.get("palm_err_cm") or 0.0) > PALM_ERR_CM or not all(r.get("reach_ok") or [True])
    if r.get("why") == "no grasp":
        return "deliver"
    if bc is None:  # no fingers-closing mark: the flow never got to the grasp
        return "timeout" if limit and not palm_bad else "deliver"
    lifted = al is not None and al.get("held") and (al.get("rise_cm") or 0.0) >= LIFT_RISE_CM
    if not lifted:  # the pick itself failed
        if (bc.get("moved_cm") or 0.0) > BEFORE_CLOSE_MOVED_CM or (bc.get("tilt_deg") or 0.0) > BEFORE_CLOSE_TILT_DEG:
            return "collide"
        if palm_bad:
            return "deliver"
        return "grasp"
    if fam in ("lift", "pick"):  # lifted, and the check did not hold
        return "timeout" if limit and end.get("held") else "grasp"
    if bo is not None and not bo.get("held"):
        return "grasp"  # lost on the way
    if bo is None:  # carried, never reached the release
        if limit and end.get("held"):
            return "timeout"
        return "grasp" if not end.get("held") else "deliver"
    if (bo.get("basket_cm") or 0.0) > BASKET_CM:
        return "collide"
    if (r.get("release_cm") or 0.0) > PALM_ERR_CM:
        return "deliver"
    return "place"
