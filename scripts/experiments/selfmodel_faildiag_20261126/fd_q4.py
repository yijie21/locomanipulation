"""selfmodel_faildiag_20261126, question 4 (offline only, no simulation): where the bowl's plan after the look fails,
and why the salt shaker's look is not recognized. Reads lmlcmp3's cached looks (hg/looks, ep 30..119) and its episode
records; imports hg_look / hg_run / lml_selfmodel from lmlcmp3's frozen copy (code/), read only.

Bowl (and every task whose plan failed after the look): hg_run.plan_from_look's filter chain recomputed grasp by grasp
from the cached look and the episode's recorded stance --
  form:  |yaw of the wrist x axis - stance yaw| <= 60 deg, pitch in [-5, 90] deg, |roll| <= ROLL_MAX (40 deg)
  reach: grasp hand point and pre-grasp hand point (8 cm back along the palm normal) inside L.WIDE (stance heading frame)
  clear: <= 3 look furniture points (furn_pts) in the hand box at the grasp and at the pre-grasp
then the highest GraspGen-X score. The recomputed counts must equal the recorded plan.counts for every episode and every
controller (checked). Per grasp also: each sub-filter alone, leave-one-out, and `resid`: the least rotation angle
between the grasp's wrist frame and any command-form frame rot_z(yaw') rot_y(pitch') with yaw' within the stance
yaw +-60 deg and pitch' in [-5, 90] deg (what the command can express at best; hg_look.command_form's roll is measured
about the grasp's own x axis, which is ill-conditioned when x is near vertical). The registration's list (approach
within 60 deg of straight down, furniture <= 3, opening <= 80 mm, score >= 0.5, IK) is BiGym's filter
(hardgrasp revision 2 point 3); its terms are reported where the LML look has them (no opening: the Dex3 grasps carry
none; IK is not part of the LML plan after the look, it comes after the walk).

Salt: the look's own SAM 3 candidates (sam_instances: score, pixels, valid non-own points) -> no candidate at the
server's 0.1 floor / candidates but all with < 15 points / chosen but < 15 points after erosion and the 0.3 m cut. The
object's expected pixel: the median of the recognized episodes' cloud centres (the object is jittered +-3 cm, so this
is +-3 cm, about +-5 px at 2.4 m), projected into each episode's head camera (cam_p_w / cam_R_w, MuJoCo camera: -z
forward, +y up), with the rendered depth there against the expected distance (occluded or not).

    python fd_q4.py --out /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126/q4
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LC3 = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126")
CODE = LC3 / "code" / "scripts" / "experiments"
sys.path[:0] = [str(CODE / "hardgrasp_20261125"), str(CODE / "locomanilab_20261117")]
import hg_look as HL  # noqa: E402
import hg_run as HG  # noqa: E402
import lml_selfmodel as L  # noqa: E402

LOOKS = LC3 / "hg" / "looks"
TASKS = (
    "IThorBathroomLiftCandleFromFloor",
    "IThorKitchenApproachTableAndLiftApple",
    "IThorKitchenApproachTableAndLiftBowl",
    "IThorKitchenApproachTableAndLiftSaltShaker",
    "IThorKitchenTakeStatueFromCabinet",
    "IThorLivingRoomApproachTableAndLiftDumbbell",
    "IThorLivingRoomApproachTableAndLiftWateringCan",
)
SHORT = dict(zip(TASKS, ("candle", "apple", "bowl", "salt", "statue_cab", "dumbbell", "wcan")))
BACKENDS = ("scalebfm", "sonic", "heft")
EPS = range(30, 120)


def rot_zy(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """rot_z(yaw) @ rot_y(pitch), broadcast over the inputs -> (..., 3, 3)."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.zeros(np.broadcast(yaw, pitch).shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = cy * cp, -sy, cy * sp
    R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = sy * cp, cy, sy * sp
    R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = -sp, 0.0, cp
    return R


def least_resid_deg(Rs: np.ndarray, yaw_s: float) -> np.ndarray:
    """Least angle (deg) between each wrist rotation and a command-form rotation with yaw' in yaw_s +-60 deg, pitch' in
    [-5, 90] deg (1 deg grid, then a 0.1 deg refinement around the best grid point)."""
    yg = yaw_s + np.radians(np.arange(-60, 61, 1.0))
    pg = np.radians(np.arange(-5, 91, 1.0))
    C = rot_zy(yg[:, None], pg[None, :]).reshape(-1, 3, 3)  # (K, 3, 3)
    out = np.empty(len(Rs))
    for i, R in enumerate(Rs):
        tr = np.einsum("kij,ij->k", C, R)  # trace(C^T R)
        k = int(np.argmax(tr))
        y0, p0 = yg[k // len(pg)], pg[k % len(pg)]
        yf = np.clip(y0 + np.radians(np.arange(-1, 1.01, 0.1)), yaw_s - math.radians(60), yaw_s + math.radians(60))
        pf = np.clip(p0 + np.radians(np.arange(-1, 1.01, 0.1)), math.radians(-5), math.radians(90))
        Cf = rot_zy(yf[:, None], pf[None, :]).reshape(-1, 3, 3)
        t = max(tr[k], float(np.max(np.einsum("kij,ij->k", Cf, R))))
        out[i] = math.degrees(math.acos(float(np.clip((t - 1) / 2, -1, 1))))
    return out


def box_hits(Tw: np.ndarray, P: np.ndarray) -> np.ndarray:
    """HL.hand_box_hits, same test and same counts, only restricted to the points within the hand box's corner radius of
    each wrist (a point farther away cannot be inside the box): KD-tree instead of every point for every grasp."""
    from scipy.spatial import cKDTree

    if len(P) == 0:
        return np.zeros(len(Tw), int)
    lo, hi = HL.HAND_BOX
    r = float(np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi)))) + 1e-6
    tree = cKDTree(P)
    out = np.empty(len(Tw), int)
    for i, T in enumerate(Tw):
        k = tree.query_ball_point(T[:3, 3], r)
        if not k:
            out[i] = 0
            continue
        q = (P[k] - T[:3, 3]) @ T[:3, :3]
        out[i] = int(np.all((q > lo) & (q < hi), axis=1).sum())
    return out


HITS: dict = {}  # (task, ep) -> (hits at the grasp, hits at the pre-grasp): a function of the look alone, not the stance


def filters(lk: dict, stance: dict, key=None) -> dict:
    """plan_from_look's per-grasp tests (same code path, split into its terms)."""
    Ts = np.asarray(lk["grasp_T"], float)
    G = np.asarray(lk["grasp_hand"], float)
    furn = np.asarray(lk["furn_pts"], float)
    n_pal = Ts[:, :3, 1]
    cf = HL.command_form(Ts)
    syaw, sxy = float(stance["yaw"]), np.asarray(stance["xy"], float)
    dyaw = np.array([HG.wrap(a - syaw) for a in cf["yaw"]])
    pitch, roll = cf["pitch"], cf["roll"]
    Rh = L.rot_z(syaw)
    th = (G[:, :2] - sxy) @ Rh[:2, :2]
    pre = G - HG.PRE_D * n_pal
    tp = (pre[:, :2] - sxy) @ Rh[:2, :2]
    lo, hi = L.WIDE
    t = {}
    t["yaw"] = np.abs(dyaw) <= math.radians(60)
    t["pitch"] = (pitch >= math.radians(-5)) & (pitch <= math.radians(90))
    t["roll"] = np.abs(roll) <= HG.ROLL_MAX
    for nm, (xy, z) in (("g", (th, G[:, 2])), ("p", (tp, pre[:, 2]))):
        t[f"box_{nm}_x"] = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0])
        t[f"box_{nm}_y"] = (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
        t[f"box_{nm}_z"] = (z >= lo[2]) & (z <= hi[2])
    Tp = Ts.copy()
    Tp[:, :3, 3] -= HG.PRE_D * n_pal
    if key is not None and key in HITS:
        hits, hits_p = HITS[key]
    else:
        hits, hits_p = box_hits(Ts, furn), box_hits(Tp, furn)
        if key is not None:
            HITS[key] = (hits, hits_p)
    t["clear_g"] = hits <= HG.HAND_CLEAR
    t["clear_p"] = hits_p <= HG.HAND_CLEAR
    form = t["yaw"] & t["pitch"] & t["roll"]
    reach = form & t["box_g_x"] & t["box_g_y"] & t["box_g_z"] & t["box_p_x"] & t["box_p_y"] & t["box_p_z"]
    clear = reach & t["clear_g"] & t["clear_p"]
    down = np.degrees(np.arccos(np.clip(-n_pal[:, 2], -1, 1)))  # palm normal (GraspGen-X's approach) vs straight down
    return dict(t=t, form=form, reach=reach, clear=clear, dyaw=np.degrees(dyaw), pitch=np.degrees(pitch),
                roll=np.degrees(roll), hits=hits, hits_p=hits_p, score=np.asarray(lk["grasp_score"], float), down=down,
                th=th, tp=tp, gz=G[:, 2], pz=pre[:, 2], Rs=Ts[:, :3, :3], syaw=syaw)  # fmt: skip


def rec(backend: str, task: str, ep: int, method: str = "alone") -> dict | None:
    f = LC3 / backend / "episodes" / task / method / f"ep{ep:+05d}.json"
    return json.loads(f.read_text()) if f.exists() else None


def q(a, p) -> str:
    return "nan" if len(a) == 0 else f"{np.percentile(a, p):.1f}"


def _validate_task(task: str):
    summary, mism, hits = {}, [], {}
    for ep in EPS:
        for be in BACKENDS:
            r = rec(be, task, ep)
            if r is None or not r.get("plan") or r["plan"].get("counts") is None:
                continue
            lk = HL.load(LOOKS / task / f"ep{ep:+05d}")
            F = filters(lk, r["plan"]["stance"], key=(task, ep))
            got = dict(form=int(F["form"].sum()), reach=int(F["reach"].sum()), clear=int(F["clear"].sum()))
            if got != r["plan"]["counts"]:
                mism.append((be, SHORT[task], ep, got, r["plan"]["counts"]))
            if r["plan"].get("fail") is None and got["clear"]:  # the chosen grasp too
                i = int(np.flatnonzero(F["clear"])[np.argmax(F["score"][F["clear"]])])
                if i != r["plan"]["idx"]:
                    mism.append((be, SHORT[task], ep, "idx", i, r["plan"]["idx"]))
            summary[(be, task, ep)] = (r["plan"].get("fail"), got)
    hits = {k: v for k, v in HITS.items() if k[0] == task}
    print(f"  validated {SHORT[task]}: {len(summary)} plans, mismatches {len(mism)}", flush=True)
    return summary, mism, hits


def bowl_like(out: Path, log) -> dict:
    """The filter chain for every episode whose plan failed after the look (all tasks), validated against the record."""
    from multiprocessing import Pool

    summary, mism = {}, []
    with Pool(len(TASKS)) as pool:
        for s_, m_, h_ in pool.map(_validate_task, TASKS):
            summary.update(s_)
            mism += m_
            HITS.update(h_)
    log(f"validation: recomputed form / reach / clear counts (and the chosen grasp) vs every record's plan: "
        f"{len(summary)} episode plans, mismatches {len(mism)} {mism[:5]}")  # fmt: skip
    return summary


def bowl_detail(out: Path, log) -> None:
    task = "IThorKitchenApproachTableAndLiftBowl"
    keys = ("yaw", "pitch", "roll", "box_g_x", "box_g_y", "box_g_z", "box_p_x", "box_p_y", "box_p_z", "clear_g", "clear_p")
    rows = []
    for be in BACKENDS:
        fails, firstfail, loo_ok, alone_ok = Counter(), Counter(), Counter(), Counter()
        n_fail = 0
        agg = dict(dyaw=[], pitch=[], roll=[], resid=[], down=[], score=[])
        per_ep = []
        for ep in EPS:
            r = rec(be, task, ep)
            lk = HL.load(LOOKS / task / f"ep{ep:+05d}")
            F = filters(lk, r["plan"]["stance"], key=(task, ep))
            failed = r["plan"].get("fail") == "reach"
            resid = least_resid_deg(F["Rs"], F["syaw"])
            n = len(F["score"])
            # each grasp's first failing test in the code's order (form terms, then the box, then clear)
            ff = Counter()
            for i in range(n):
                for k in keys:
                    if not F["t"][k][i]:
                        ff[k] += 1
                        break
                else:
                    ff["passes"] += 1
            # leave-one-group-out: grasps passing everything but that group
            grp = dict(form=("yaw", "pitch", "roll"), box=keys[3:9], clear=("clear_g", "clear_p"), roll=("roll",), yaw=("yaw",))
            loo = {}
            for gname, gk in grp.items():
                ok = np.ones(n, bool)
                for k in keys:
                    if k not in gk:
                        ok &= F["t"][k]
                loo[gname] = int(ok.sum())
            resid_ok = resid <= math.degrees(HG.ROLL_MAX)
            box = np.ones(n, bool)
            for k in keys[3:9]:
                box &= F["t"][k]
            clr = F["t"]["clear_g"] & F["t"]["clear_p"]
            alt = int((resid_ok & box & clr).sum())  # form replaced by "least command-form residual <= 40 deg"
            e = dict(ep=ep, failed=failed, n=n, form=int(F["form"].sum()), reach=int(F["reach"].sum()), clear=int(F["clear"].sum()),
                     alone={k: int(F["t"][k].sum()) for k in keys}, first_fail=dict(ff), loo=loo,
                     resid_le40=int(resid_ok.sum()), alt_clear=alt, box=int(box.sum()), clear_only=int(clr.sum()), box_clear=int((box & clr).sum()),
                     score_ge05=int((F["score"] >= 0.5).sum()), down_le60=int((F["down"] <= 60).sum()),
                     down_le60_clear=int(((F["down"] <= 60) & clr).sum()))  # fmt: skip
            per_ep.append(e)
            if failed:
                n_fail += 1
                for k in keys:
                    alone_ok[k] += int(F["t"][k].sum() > 0)
                for g, v in loo.items():
                    loo_ok[g] += int(v > 0)
                firstfail.update(ff)
                for k in ("dyaw", "pitch", "roll", "down", "score"):
                    agg[k].append(F[k])
                agg["resid"].append(resid)
        rows.append((be, n_fail, alone_ok, loo_ok, firstfail, agg, per_ep))
        (out / f"bowl_{be}.json").write_text(json.dumps(per_ep, indent=0))
    for be, n_fail, alone_ok, loo_ok, firstfail, agg, per_ep in rows:
        log(f"== bowl, {be}: plan failed after the look {n_fail}/90")
        tot = sum(firstfail.values())
        log(f"  grasps over the failed episodes {tot}; each grasp's first failing test (code order): "
            + ", ".join(f"{k} {v} ({100 * v / tot:.1f} %)" for k, v in firstfail.most_common()))
        log("  episodes (of the failed) with >= 1 grasp passing this test alone: " + ", ".join(f"{k} {v}" for k, v in alone_ok.items()))
        log("  episodes (of the failed) with >= 1 grasp passing every test except this group: " + ", ".join(f"{k} {v}" for k, v in loo_ok.items()))
        A = {k: np.concatenate(v) if v else np.zeros(0) for k, v in agg.items()}
        if len(A["pitch"]):
            log(f"  failed episodes' grasps: pitch deg p10/50/90 {q(A['pitch'], 10)}/{q(A['pitch'], 50)}/{q(A['pitch'], 90)}; "
                f"pitch >= 75 deg {100 * np.mean(A['pitch'] >= 75):.1f} %; |dyaw| deg p50 {q(np.abs(A['dyaw']), 50)}, <= 60 {100 * np.mean(np.abs(A['dyaw']) <= 60):.1f} %; "
                f"|roll| deg p50 {q(np.abs(A['roll']), 50)}, <= 40 {100 * np.mean(np.abs(A['roll']) <= 40):.1f} %")  # fmt: skip
            log(f"  least command-form residual (yaw' within +-60, pitch' -5..90) deg p10/50/90 {q(A['resid'], 10)}/{q(A['resid'], 50)}/{q(A['resid'], 90)}; "
                f"<= 40 deg {100 * np.mean(A['resid'] <= 40):.1f} % of grasps; palm normal within 60 deg of straight down {100 * np.mean(A['down'] <= 60):.1f} %; "
                f"score >= 0.5 {100 * np.mean(A['score'] >= 0.5):.1f} %")  # fmt: skip
            hi = A["pitch"] >= 75
            if hi.any():
                log(f"  grasps with pitch >= 75 deg: |dyaw| <= 60 {100 * np.mean(np.abs(A['dyaw'][hi]) <= 60):.1f} %, |roll| <= 40 {100 * np.mean(np.abs(A['roll'][hi]) <= 40):.1f} %, "
                    f"least residual <= 40 {100 * np.mean(A['resid'][hi] <= 40):.1f} % (p50 {q(A['resid'][hi], 50)} deg)")  # fmt: skip
        fe = [e for e in per_ep if e["failed"]]
        log(f"  per failed episode: form-passing grasps p50 {np.median([e['form'] for e in fe]):.0f} (max {max(e['form'] for e in fe)}); "
            f"episodes with 0 form-passing {sum(e['form'] == 0 for e in fe)}, form > 0 but reach 0 {sum(e['form'] > 0 and e['reach'] == 0 for e in fe)}, "
            f"reach > 0 but clear 0 {sum(e['reach'] > 0 and e['clear'] == 0 for e in fe)}")  # fmt: skip
        log(f"  counterfactual (descriptive only): form replaced by 'least residual <= 40 deg' -> failed episodes with >= 1 grasp left {sum(e['alt_clear'] > 0 for e in fe)}/{len(fe)}; "
            f"box and clear only (no orientation test) {sum(e['box_clear'] > 0 for e in fe)}/{len(fe)}")  # fmt: skip
        log(f"  the registration's (BiGym) terms on the failed episodes: >= 1 grasp with palm normal within 60 deg of down {sum(e['down_le60'] > 0 for e in fe)}, "
            f"of those also clear of furniture {sum(e['down_le60_clear'] > 0 for e in fe)}; >= 1 grasp with score >= 0.5 {sum(e['score_ge05'] > 0 for e in fe)}; "
            f"opening: not in the Dex3 look; IK: not part of the plan after the look")  # fmt: skip
        ok = [e for e in per_ep if not e["failed"]]
        if ok:
            log(f"  the {len(ok)} bowl episodes whose plan passed: form-passing grasps {sorted(e['form'] for e in ok)}; clear {sorted(e['clear'] for e in ok)}")


def salt(out: Path, log) -> None:
    task = "IThorKitchenApproachTableAndLiftSaltShaker"
    metas = {ep: json.loads((LOOKS / task / f"ep{ep:+05d}.json").read_text()) for ep in EPS}
    rec_ok = {ep: m for ep, m in metas.items() if m.get("status") == "ok"}
    C = np.array([m["obj_center_w"] for m in rec_ok.values()])
    c0 = np.median(C, axis=0)
    log(f"== salt: looks {len(metas)}; status {dict(Counter(m.get('status') for m in metas.values()))}; "
        f"recognized cloud centres: median {np.round(c0, 3).tolist()}, spread (max dist from median) {100 * np.max(np.linalg.norm(C - c0, axis=1)):.1f} cm")  # fmt: skip
    cat = Counter()
    per = []
    for ep, m in metas.items():
        inst = m.get("sam_instances", [])
        st = m.get("status")
        best_pts = max((i["pts"] for i in inst), default=0)
        best_sc = max((i["score"] for i in inst), default=0.0)
        if st == "ok":
            c = "recognized"
        elif not inst:
            c = "no SAM 3 candidate (score floor 0.1)"
        elif best_pts < 15:
            c = "candidates, all < 15 points"
        else:
            c = "chosen, < 15 points after erosion / 0.3 m cut"
        cat[c] += 1
        # expected pixel of the object
        Rc, pc = np.asarray(m["cam_R_w"], float), np.asarray(m["cam_p_w"], float)
        v = Rc.T @ (c0 - pc)
        dist = float(-v[2])
        f, (W, H) = float(m["f_px"]), m["size"]
        u_px = (W - 1) / 2 + f * v[0] / dist
        v_px = (H - 1) / 2 - f * v[1] / dist
        inview = dist > 0 and 0 <= u_px < W and 0 <= v_px < H
        z = np.load(LOOKS / task / f"ep{ep:+05d}.npz")
        dep = np.nan
        own = False
        if inview:
            ui, vi = int(round(u_px)), int(round(v_px))
            win = z["depth"][max(vi - 4, 0) : vi + 5, max(ui - 4, 0) : ui + 5]
            dep = float(np.nanmax(win))  # NOTE (11-26): the farthest depth picks the background; superseded by fd_saltvis.py
            own = bool(z["own"][max(vi - 4, 0) : vi + 5, max(ui - 4, 0) : ui + 5].mean() > 0.5)
        # the depth along the camera axis vs the expected one: depth images hold -z (MuJoCo depth = distance along -z)
        per.append(dict(ep=ep, cat=c, n_inst=len(inst), best_score=best_sc, best_pts=best_pts,
                        pixels=[i["pixels"] for i in inst], u=u_px, v=v_px, inview=bool(inview), exp_depth=dist,
                        depth_max_9x9=dep, own=own, mask_score=m.get("mask_score"), n_obj=m.get("n_obj")))  # fmt: skip
    (out / "salt.json").write_text(json.dumps(per, indent=0))
    log("  categories: " + ", ".join(f"{k} {v}" for k, v in cat.items()))
    for c in cat:
        P = [p for p in per if p["cat"] == c]
        iv = [p for p in P if p["inview"]]
        occ = [p for p in iv if not (abs(p["depth_max_9x9"] - p["exp_depth"]) < 0.10)]
        sc = [p["best_score"] for p in P if p["n_inst"]]
        pts = [p["best_pts"] for p in P if p["n_inst"]]
        log(f"  [{c}] n {len(P)}: expected pixel in the image {len(iv)}; of those, depth in a 9x9 window within 10 cm of the expected distance "
            f"{len(iv) - len(occ)}, not ({len(occ)}: occluded or off by the jitter), on the robot's own pixels {sum(p['own'] for p in iv)}; "
            f"expected distance m p50 {np.median([p['exp_depth'] for p in P]):.2f}; expected pixel u p50 {np.median([p['u'] for p in P]):.0f}, "
            f"v p50 {np.median([p['v'] for p in P]):.0f} (image 640 x 480)"
            + (f"; best candidate score p50 {np.median(sc):.3f} (max {max(sc):.3f}), points p50 {np.median(pts):.0f}" if sc else ""))  # fmt: skip
    ok = [p for p in per if p["cat"] == "recognized"]
    log(f"  recognized: mask score p10/50/90 {q([p['mask_score'] for p in ok], 10)}/{q([p['mask_score'] for p in ok], 50)}/{q([p['mask_score'] for p in ok], 90)} "
        f"(SAM 3 floor 0.1); object points p50 {np.median([p['n_obj'] for p in ok]):.0f}; instances per look {dict(Counter(p['n_inst'] for p in ok))}")  # fmt: skip
    ni = [p for p in per if p["cat"] != "recognized"]
    log(f"  not recognized: SAM 3 candidates per look {dict(Counter(p['n_inst'] for p in ni))}")
    # does the object's expected place differ between recognized and not? (robot jitter moves the camera)
    for nm, P in (("recognized", ok), ("not recognized", ni)):
        log(f"  {nm}: expected pixel u p10/50/90 {q([p['u'] for p in P], 10)}/{q([p['u'] for p in P], 50)}/{q([p['u'] for p in P], 90)}, "
            f"v {q([p['v'] for p in P], 10)}/{q([p['v'] for p in P], 50)}/{q([p['v'] for p in P], 90)}, distance {q([p['exp_depth'] for p in P], 50)} m")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    lk0 = HL.load(LOOKS / TASKS[2] / "ep+0030")  # the fast count equals hg_look's own on one look (every grasp)
    T0 = np.asarray(lk0["grasp_T"], float)
    F0 = np.asarray(lk0["furn_pts"], float)
    assert np.array_equal(box_hits(T0, F0), HL.hand_box_hits(T0, F0)), "box_hits != hand_box_hits"
    log(f"box_hits equals hg_look.hand_box_hits on bowl ep 30 ({len(T0)} grasps, {len(F0)} furniture points)")
    log(f"fd_q4: lmlcmp3 looks {LOOKS}, records {LC3}/<backend>/episodes/<task>/alone, episodes 30..119; hg_run {HG.__file__}")
    summ = bowl_like(out, log)
    fails = Counter((be, SHORT[t]) for (be, t, ep), (fl, _) in summ.items() if fl == "reach")
    log(f"plan failed after the look ('no grasp left'), by controller and task: {dict(fails)}")
    bowl_detail(out, log)
    salt(out, log)
    (out / "q4.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
