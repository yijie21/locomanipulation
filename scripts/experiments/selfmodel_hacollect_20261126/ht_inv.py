"""selfmodel_hatask_20261126 revision 2: R (lmlcmp3's `reachable`) as plain functions, so HA's runtime can run it without
importing lc3_run (which imports lr_run's LocoManiLab drivers and asserts REACHWM_FDVEL off at module level).

Copied verbatim (ht_equiv.py's text check compares them with the frozen sources):
  SHRINK_IK, SHRINK_STEPS   lc3_run's constants
  ITERS, LAMBDA             lml_tasks' constants
  clip_t                    lml_tasks.clip_t (L is the caller's lml_selfmodel: set_L(); its WIDE is checked equal)
  shrink                    lc3_run.shrink, byte for byte (the caller's `dr` has lc3_run's driver's plan(x, ref, c) ->
                            (blend, last, rec) with rec["ik_err"])
  reachable_one             the body is lc3_run.episode's block from "t, evs = gh.copy(), []" to "best = evs[bi]",
                            dedented, with one change: the hand side clip_t gets is the argument `side`, not "right".
                            Its names are the caller's: dr, rows, f, frames, c (lc3_run's command dict with "target"),
                            gh, feat(cc) -> (plan, z), MD.predict(z, ("D",)) -> {"D": (pos, std)}; method is
                            "reachable"; LT is this module's namespace of ITERS, LAMBDA and clip_t.
New (the two-hand squeeze, revision 2 section 2): reachable_two / shrink2 -- each hand updates by its own prediction
(the one-hand rule), one shrink fraction for both hands (accepted when the larger IK error <= SHRINK_IK), the candidate's
loss summed over the hands; each hand's prediction is the one-hand model's on that hand's segment of the two-hand plan.
"""

from __future__ import annotations

import types

import numpy as np

SHRINK_IK, SHRINK_STEPS = 0.02, 6  # lmlcmp2 (g): m (the plan's IK error), bisection halvings
ITERS = 6  # pose_invert.ITERS_POSE
LAMBDA = 1.0  # invert_eval.LAMBDA: predicted spread's weight

L = None  # lml_selfmodel (set_L)


def set_L(module) -> None:
    global L
    L = module


def clip_t(t: np.ndarray, side: str) -> np.ndarray:
    """wrist_invert.clip_t: the target kept inside the data's WIDE box (mirrored for the left hand)."""
    lo, hi = L.WIDE
    sd = 1.0 if side == "right" else -1.0
    y = np.clip(t[1] * sd, lo[1], hi[1]) * sd
    return np.r_[np.clip(t[0], lo[0], hi[0]), y, np.clip(t[2], lo[2], hi[2])]


LT = types.SimpleNamespace(ITERS=ITERS, LAMBDA=LAMBDA, clip_t=clip_t)


def shrink(dr, x: dict, ref: np.ndarray, c: dict, gh: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, dict]:
    """(g) candidate t's plan IK error > SHRINK_IK -> back along t - gh to the largest fraction a (bisection, SHRINK_STEPS
    halvings) whose IK error <= SHRINK_IK; a = 0 (the goal) when none passes. A plan that fails counts as not passing."""

    def ik(tt: np.ndarray) -> float:
        try:
            return float(dr.plan(x, ref, dict(c, target=tt.copy()))[2]["ik_err"])
        except (np.linalg.LinAlgError, ValueError):
            return float("inf")

    e1 = ik(t)
    sh = dict(ik0_cm=100 * e1 if np.isfinite(e1) else None, corr0_cm=100 * float(np.linalg.norm(t - gh)), alpha=1.0, shrunk=False, n_plans=1)
    if e1 <= SHRINK_IK:
        return t, sh
    lo, hi, n = 0.0, 1.0, 1
    for _ in range(SHRINK_STEPS):
        mid = 0.5 * (lo + hi)
        n += 1
        if ik(gh + mid * (t - gh)) <= SHRINK_IK:
            lo = mid
        else:
            hi = mid
    t2 = gh + lo * (t - gh)
    sh.update(alpha=lo, shrunk=True, n_plans=n, corr_cm=100 * float(np.linalg.norm(t2 - gh)))
    return t2, sh


def reachable_one(dr, rows: dict, f: int, frames: np.ndarray, c: dict, gh: np.ndarray, feat, MD, side: str):
    """lc3_run's `reachable` block for one hand (module docstring). Returns (best, bi, evs, shr)."""
    method = "reachable"
    # -- verbatim from lc3_run.episode (the side passed to clip_t is the argument) --
    t, evs = gh.copy(), []
    shr = [] if method == "reachable" else None
    for it in range(LT.ITERS + 1):
        if shr is not None and it > 0:  # lmlcmp2 (g): the candidate shrunk back to IK-reachable first
            t, sh = shrink(dr, rows[f], frames[f], c, gh, t)
            shr.append(dict(sh, it=it))
        cc = dict(c, target=t.copy())
        try:
            pl_, zz = feat(cc)
        except (np.linalg.LinAlgError, ValueError) as e:  # this candidate cannot be planned
            evs.append(dict(t=t.copy(), l=np.inf, err=f"{type(e).__name__}: {str(e)[:120]}"))
            break
        pos, std = MD.predict(zz, ("D",))["D"]
        evs.append(dict(t=t.copy(), pos=pos, std=std, l=float(((pos - gh) ** 2).sum() + LT.LAMBDA * (std**2).sum()), plan=pl_, z=zz))
        t = LT.clip_t(t + (gh - pos), side)
    bi = min(range(len(evs)), key=lambda i: evs[i]["l"])  # hg_run's min(evs, key=l): the first least
    best = evs[bi]
    # -- end of the verbatim block --
    return best, bi, evs, shr


# -- the two-hand squeeze (new) -------------------------------------------------------------------------------------
def shrink2(dr2, x: dict, ref: np.ndarray, c2: dict, gh2: dict, t: dict) -> tuple[dict, dict]:
    """shrink for both hands at once: one fraction a for both corrections, the larger of the two IK errors <= SHRINK_IK.
    dr2.plan(x, ref, {side: c}) -> (blend, last, rec) with rec["ik_err"] = {side: error}."""

    def ik(tt: dict) -> float:
        try:
            e = dr2.plan(x, ref, {s: dict(c2[s], target=tt[s].copy()) for s in tt})[2]["ik_err"]
            return float(max(e.values()))
        except (np.linalg.LinAlgError, ValueError):
            return float("inf")

    def corr(tt: dict) -> float:
        return 100 * max(float(np.linalg.norm(tt[s] - gh2[s])) for s in tt)

    e1 = ik(t)
    sh = dict(ik0_cm=100 * e1 if np.isfinite(e1) else None, corr0_cm=corr(t), alpha=1.0, shrunk=False, n_plans=1)
    if e1 <= SHRINK_IK:
        return t, sh
    lo, hi, n = 0.0, 1.0, 1
    for _ in range(SHRINK_STEPS):
        mid = 0.5 * (lo + hi)
        n += 1
        if ik({s: gh2[s] + mid * (t[s] - gh2[s]) for s in t}) <= SHRINK_IK:
            lo = mid
        else:
            hi = mid
    t2 = {s: gh2[s] + lo * (t[s] - gh2[s]) for s in t}
    sh.update(alpha=lo, shrunk=True, n_plans=n, corr_cm=corr(t2))
    return t2, sh


def reachable_two(dr2, rows: dict, f: int, frames: np.ndarray, c2: dict, gh2: dict, feat2, MD2):
    """reachable_one's fixed point for a two-hand command. feat2({side: cc}) -> (plan, z with one row per hand);
    MD2.predict2(z) -> {side: (pos, std)} (each row predicted alone, as a one-hand segment). Returns (best, bi, evs, shr)."""
    sides = tuple(gh2)
    t, evs, shr = {s: gh2[s].copy() for s in sides}, [], []
    for it in range(LT.ITERS + 1):
        if it > 0:
            t, sh = shrink2(dr2, rows[f], frames[f], c2, gh2, t)
            shr.append(dict(sh, it=it))
        cc = {s: dict(c2[s], target=t[s].copy()) for s in sides}
        try:
            pl_, zz = feat2(cc)
        except (np.linalg.LinAlgError, ValueError) as e:
            evs.append(dict(t={s: t[s].copy() for s in sides}, l=np.inf, err=f"{type(e).__name__}: {str(e)[:120]}"))
            break
        pr = MD2.predict2(zz)
        loss = sum(float(((pr[s][0] - gh2[s]) ** 2).sum() + LT.LAMBDA * (pr[s][1] ** 2).sum()) for s in sides)
        evs.append(dict(t={s: t[s].copy() for s in sides}, pos={s: pr[s][0] for s in sides}, std={s: pr[s][1] for s in sides},
                        l=loss, plan=pl_, z=zz))  # fmt: skip
        t = {s: LT.clip_t(t[s] + (gh2[s] - pr[s][0]), s) for s in sides}
    bi = min(range(len(evs)), key=lambda i: evs[i]["l"])
    best = evs[bi]
    return best, bi, evs, shr
