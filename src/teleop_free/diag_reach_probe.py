"""Probe the reach map against the real control stack.

Replays per-voxel exemplar joint solutions (the map's own evidence) on the
live sim, in two arm modes, and measures promise-vs-delivery:

    A  streamed reference: the 29-joint pose stream the grasp currently uses
       — the policy tracks it, with whatever attenuation it has;
    B  actuator pin: the arm motors' own position loop (:5582 pin_arms).

Strata: core (deep inside), edge (map boundary), chest (the both-arms
overlap at the midline — kinematically fine, empirically suspect), and
outside (just beyond the map; IK-checked first to estimate the sampling
false-negative rate).

Output: outputs/logs/reach_probe.jsonl, one record per point per mode, plus
a printed summary table.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stream_reach  # noqa: E402
from arm_ik import ArmIK  # noqa: E402
from g1_joints import ARM_MJ  # noqa: E402
from primitives import (  # noqa: E402
    LocomotionMode,
    Primitives,
    StreamMode,
)
from privileged_pub import EpisodeControlClient  # noqa: E402
from reach_map import MAP_PATH, OFFSET_DIR  # noqa: E402

OUT = "/workspace/code/locomanipulation/outputs/logs/reach_probe.jsonl"
SCENE_META = "/workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json"

N_CORE, N_EDGE, N_CHEST, N_OUT = 4, 6, 2, 3   # per arm


def erode(g: np.ndarray) -> np.ndarray:
    """6-neighbourhood binary erosion."""
    out = g.copy()
    for ax in (0, 1, 2):
        out &= np.roll(g, 1, axis=ax) & np.roll(g, -1, axis=ax)
    return out


def pick(rng, mask, n):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return []
    return [tuple(r) for r in idx[rng.choice(len(idx), min(n, len(idx)),
                                             replace=False)]]


def select_points(z, rng):
    pts = []          # (stratum, side, (i,j,k))
    vox, org = float(z["vox"]), z["origin"]
    for side in ("left", "right"):
        occ = z[side]
        e1 = erode(occ)
        e3 = erode(erode(e1))
        core, edge = e3, occ & ~e1
        js = ((np.arange(occ.shape[1]) + 0.5) * vox + org[1])
        mid = np.zeros_like(occ)
        mid[:, np.abs(js) < 0.15, :] = True
        chest = occ & z["both"] & mid
        # outside: one dilation ring
        dil = occ.copy()
        for ax in (0, 1, 2):
            dil |= np.roll(occ, 1, axis=ax) | np.roll(occ, -1, axis=ax)
        outside = dil & ~occ
        for name, mask, n in (("core", core, N_CORE), ("edge", edge, N_EDGE),
                              ("chest", chest, N_CHEST),
                              ("outside", outside, N_OUT)):
            for v in pick(rng, mask, n):
                pts.append((name, side, v))
    return pts


def main() -> None:
    z = np.load(MAP_PATH)
    rng = np.random.default_rng(7)
    pts = select_points(z, rng)
    vox, org = float(z["vox"]), z["origin"]
    ik = ArmIK()

    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()
    out = open(OUT, "w")

    def measure(side, want_pos, want_appr, n=10):
        errs, angs, pz = [], [], []
        for _ in range(n):
            s = p.obs()
            pf = p.world_to_root_full(s["palms"][side]["pos"], s)
            errs.append(float(np.linalg.norm(pf - want_pos)))
            rotp = Primitives._rot(s["palms"][side]["quat"])
            # palm rot is world-frame; express in pelvis frame like the map
            rp = Primitives._rot(s["pelvis"]["quat"]).T @ rotp
            a = rp @ OFFSET_DIR
            angs.append(float(np.degrees(np.arccos(np.clip(
                a @ want_appr / (np.linalg.norm(want_appr) + 1e-9), -1, 1)))))
            pz.append(float(s["pelvis"]["pos"][2]))
            time.sleep(0.1)
        s = p.obs()
        return (float(np.mean(errs)), float(np.mean(angs)),
                float(np.mean(pz)), bool(s["fallen"]))

    def fresh():
        p.cmd.reset_pose()
        reset.pin_arms()
        reset.reset()
        time.sleep(0.2)
        p.cmd.start_control()
        p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
        stream_reach.reanchor(p)

    fresh()
    by_side = {"left": [], "right": []}
    for st, side, v in pts:
        by_side[side].append((st, v))

    results = []
    for side, items in by_side.items():
        if not items:
            continue
        arm = stream_reach.StreamedArm(p, side)
        arm.start()
        done_in_session = 0
        for st, (i, j, k) in items:
            want_pos = z[f"ex_pos_{side}"][i, j, k]
            want_appr = z[f"ex_appr_{side}"][i, j, k]
            q7 = z[f"ex_q_{side}"][i, j, k]
            wp = float(z[f"ex_wp_{side}"][i, j, k])
            if st == "outside":
                # no exemplar exists; ask IK whether the map merely missed it
                centre = org + (np.array([i, j, k]) + 0.5) * vox
                q0 = np.zeros(29)
                qs, res, _ = ik.solve(side, centre, None, q_mj29=q0,
                                      use_waist=False, iters=60)
                rec = {"stratum": st, "side": side,
                       "voxel": [int(i), int(j), int(k)],
                       "ik_residual": float(res)}
                out.write(json.dumps(rec) + "\n")
                results.append(rec)
                continue
            q_t = arm.q.copy()
            q_t[14] = wp
            q_t[ARM_MJ[side]] = q7
            arm.move_to(q_t, speed=1.0)
            arm.hold(1.0)
            eA, aA, zA, fA = measure(side, want_pos, want_appr)
            reset.pin_arms(**{side: list(map(float, q7))})
            arm.hold(1.0)
            eB, aB, zB, fB = measure(side, want_pos, want_appr)
            reset.pin_arms()
            rec = {"stratum": st, "side": side,
                   "voxel": [int(i), int(j), int(k)], "waist": wp,
                   "A_err_cm": round(eA * 100, 1), "A_ang": round(aA, 0),
                   "B_err_cm": round(eB * 100, 1), "B_ang": round(aB, 0),
                   "pelvis_z": round(zA, 3), "fell": fA or fB}
            print(f"{st:8s} {side:5s} {np.round(want_pos, 2)}: "
                  f"A {rec['A_err_cm']} cm/{rec['A_ang']:.0f}deg  "
                  f"B {rec['B_err_cm']} cm/{rec['B_ang']:.0f}deg"
                  f"{'  FELL' if rec['fell'] else ''}")
            out.write(json.dumps(rec) + "\n")
            results.append(rec)
            done_in_session += 1
            if rec["fell"]:
                arm.stop()
                fresh()
                arm = stream_reach.StreamedArm(p, side)
                arm.start()
        arm.stop()

    out.close()
    execd = [r for r in results if "A_err_cm" in r]
    if execd:
        for m in ("A", "B"):
            e = [r[f"{m}_err_cm"] for r in execd]
            print(f"mode {m}: err mean {np.mean(e):.1f} cm  "
                  f"p90 {np.percentile(e, 90):.1f} cm  "
                  f"fails>4cm {sum(x > 4 for x in e)}/{len(e)}")
    outs = [r for r in results if r["stratum"] == "outside"]
    if outs:
        fn = sum(r["ik_residual"] < 0.01 for r in outs)
        print(f"outside voxels IK-reachable (false negatives): {fn}/{len(outs)}")


if __name__ == "__main__":
    main()
