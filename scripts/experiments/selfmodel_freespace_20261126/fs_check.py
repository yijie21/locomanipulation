"""selfmodel_freespace_20261126: the entry's consistency checks on one controller's records (read-only).

    cd $G; PYTHONPATH=$G:$W/src FS_OUT=$O $PY fs_check.py --backend scalebfm

1. A executes the goal itself: cmd_h = goal_h bit for bit, no inversion recorded; every S / R reach starts its fixed
   point at the goal and executes its chosen candidate (S: the first least loss; R: the same on the shrunk candidates).
2. Same commands -> same episodes: in every (cell, ep), groups whose executed command is identical bit for bit have the
   same episode (command row, waits, landing row, landing point, x0 / u, the three predictions, the segment's rows);
   and every group has the same steps up to the command (command row, waits, x0's start part: columns 0..CHAIN_COL).
3. R that returned to the goal (executed command = goal) is A bit for bit (a sub-case of 2, counted on its own); S and R
   with identical commands (no shrink took effect) likewise.
4. Features and predictions: x0 / u data.py builds from the segment npz = the online ones; T3 / M / D on them = online.
5. The landing rule: label_kind 0 -> settled_end, on the 10 Hz grid from the command, in [f_end + delay + 0.5 s,
   f_end + delay + 3 s]; 1 -> not settled, the first grid row at or after f_end + delay + 3 s; live_err_cm = the
   segment's jaw_err_cm; err_cm recomputed from the segment's last hand point and the goal (world, from the command
   row's pelvis). The waits: 0.5 s <= wait <= 3 s, the own rule held or 3 s ran out.
6. Targets and seeds: every record's goal_h = fs_grid.goal(cell, ep) bit for bit (so every group, controller and
   benchmark has the same list); the pitch = grid.json's; in every (cell, ep) all groups have the same first row
   (the reset's state: the per-episode env generator).
7. (fs_formal.sh: the private copy's md5 before and after the run.)
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fs_run as FR  # noqa: E402
import fs_grid as FG  # noqa: E402
import fs_load as FL  # noqa: E402

LC, D3, L, LT = FR.LC, FR.D3, FR.L, FR.LT


def main(backend: str, models: bool) -> int:
    base = FR.OUT / backend
    E, dup = FL.runs(base)
    z, A = FL.segs(base)
    grid = FR.load_grid()
    bad = 0
    n_err = sum("error" in r for r in E.values())
    print(f"{backend}: {len(E)} episode records ({n_err} worker errors, {len(dup)} duplicate jobs), "
          f"{0 if z is None else len(z['n_steps'])} segments; groups {sorted({k[2] for k in E})}")  # fmt: skip
    bad += int(bool(dup))
    seg_of = {}
    if z is not None:
        for i, (c, e, m) in enumerate(zip(z["cell"], z["ep"], z["method_id"])):
            seg_of[(int(c), int(e), LC.METHODS[int(m)])] = i
    n_steps = z["n_steps"].astype(int) if z is not None else np.zeros(0, int)
    so = np.r_[0, np.cumsum(n_steps)[:-1]] if z is not None else np.zeros(0, int)
    ok = {k: r for k, r in E.items() if "error" not in r}
    # 1. A's command, the inversion's start and choice
    b1 = []
    for k, r in ok.items():
        w = r.get("reach") or {}
        if "cmd_h" not in w:
            continue
        if k[2] == "alone":
            if w["cmd_h"] != r["goal_h"] or "inv" in w:
                b1.append(k)
        else:
            inv = w.get("inv") or []
            if not inv or inv[0]["t"] != r["goal_h"]:
                b1.append(k)
                continue
            ls = [v["l"] for v in inv]
            bi = int(np.argmin(ls))
            if "inv_best" in w and (w["inv_best"] != bi or inv[bi]["t"] != w["cmd_h"]):
                b1.append(k)
            if k[2] == "selfmodel" and "shrink" in w:
                b1.append(k)
    print(f"1. A's command = goal, inversions start at the goal and execute the first least loss: {len(b1)} violations {b1[:4]}")
    bad += int(bool(b1))
    # 2 / 3. pairing
    by_ce = defaultdict(dict)
    for (c, e, m), r in ok.items():
        by_ce[(c, e)][m] = r

    def pre_same(a: dict, b: dict) -> list[str]:
        wa, wb = a.get("reach") or {}, b.get("reach") or {}
        out = []
        if [(x["stage"], x["fire"]) for x in a["waits"]] != [(x["stage"], x["fire"]) for x in b["waits"]]:
            out.append("waits")
        if wa.get("f_cmd") != wb.get("f_cmd"):
            out.append("f_cmd")
        if "feat_x0" in wa and "feat_x0" in wb and wa["feat_x0"][: LC.CHAIN_COL + 1] != wb["feat_x0"][: LC.CHAIN_COL + 1]:
            out.append("x0 start")
        return out

    def all_same(a: dict, b: dict, ka, kb) -> list[str]:
        out = pre_same(a, b)
        wa, wb = a.get("reach") or {}, b.get("reach") or {}
        for key in ("f_label", "label_kind", "err_cm", "hand_w", "act_h", "feat_x0", "feat_u", "pred_h", "ik_err_cm", "ref_end_h"):
            if wa.get(key) != wb.get(key):
                out.append(key)
        for key in ("fell", "lost", "why", "steps"):
            if a.get(key) != b.get(key):
                out.append(key)
        ia, ib = seg_of.get(ka), seg_of.get(kb)
        if (ia is None) != (ib is None):
            out.append("segment")
        elif ia is not None:
            if n_steps[ia] != n_steps[ib] or not np.array_equal(z["S"][so[ia] : so[ia] + n_steps[ia]], z["S"][so[ib] : so[ib] + n_steps[ib]]) or \
                    not np.array_equal(z["W"][so[ia] : so[ia] + n_steps[ia]], z["W"][so[ib] : so[ib] + n_steps[ib]]):  # fmt: skip
                out.append("segment rows")
        return out

    n_pre, b_pre, n_same, b_same, n_back, b_back, n_sr, b_sr, n_diff = 0, [], 0, [], 0, [], 0, [], 0
    for (c, e), g in by_ce.items():
        ms = sorted(g)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = g[ms[i]], g[ms[j]]
                wa, wb = a.get("reach") or {}, b.get("reach") or {}
                if "cmd_h" not in wa or "cmd_h" not in wb:
                    continue
                n_pre += 1
                d = pre_same(a, b)
                if d:
                    b_pre.append(((c, e, ms[i], ms[j]), d))
                if wa["cmd_h"] == wb["cmd_h"]:
                    n_same += 1
                    d = all_same(a, b, (c, e, ms[i]), (c, e, ms[j]))
                    if d:
                        b_same.append(((c, e, ms[i], ms[j]), d))
                    pair = {ms[i], ms[j]}
                    if pair == {"alone", "reachable"}:
                        n_back += 1
                        b_back += [((c, e), d)] if d else []
                    if pair == {"selfmodel", "reachable"}:
                        n_sr += 1
                        b_sr += [((c, e), d)] if d else []
                else:
                    n_diff += 1
    print(f"2. pairing: {n_pre} group pairs with a command; steps up to the command identical except {len(b_pre)} {b_pre[:3]}; "
          f"identical commands {n_same} (different {n_diff}): whole episode identical except {len(b_same)} {b_same[:3]}")  # fmt: skip
    print(f"3. R back at the goal = A: {n_back} pairs, {len(b_back)} differ; S = R (no shrink took effect): {n_sr} pairs, {len(b_sr)} differ")
    bad += int(bool(b_pre) or bool(b_same))
    # 4. features and predictions
    if z is not None:
        dx, du, used = [], [], []
        for k, i in seg_of.items():
            w = (ok.get(k) or {}).get("reach") or {}
            if "feat_x0" not in w:
                continue
            dx.append(float(np.abs(np.asarray(w["feat_x0"], np.float32) - z["x0"][i]).max()))
            du.append(float(np.abs(np.asarray(w["feat_u"], np.float32) - z["u"][i]).max()))
            used.append((i, w))
        msg = f"4. x0 / u from the npz vs online ({len(used)} of {len(seg_of)} segments): x0 max |d| {max(dx, default=0):.3e}, u max |d| {max(du, default=0):.3e}"
        b4 = len(used) < len(seg_of) or max(dx, default=0) > 1e-5 or max(du, default=0) > 1e-6
        if models and used:
            import torch

            MD = LC.Models(backend)
            GD = MD.M.tensors(z, "cpu")
            idx = torch.tensor([i for i, _ in used])
            dp = {}
            for nm in LC.MODEL_NAMES:
                G = GD
                if nm == "T3":
                    G = dict(GD)
                    G["x0"] = GD["x0"].clone()
                    G["x0"][:, LC.CHAIN_COL] = G["x0"][:, LC.CHAIN_COL].clamp(max=LC.T3_CHAIN_MAX)
                with torch.no_grad():
                    o = MD.m[nm].predict(MD.M.sub(G, idx, False))
                pp = o["pos"].numpy().astype(float)
                dp[nm] = max(float(np.abs(pp[j] - np.asarray(w["pred_h"][nm])).max()) for j, (_, w) in enumerate(used))
            msg += "; predictions npz vs online max |d| " + ", ".join(f"{nm} {100 * v:.2e} cm" for nm, v in dp.items())
            b4 = b4 or any(v > 1e-5 for v in dp.values())
        print(msg)
        bad += int(b4)
    # 5. the landing rule and the waits
    b5, kinds, wbad = [], defaultdict(int), []
    for k, r in ok.items():
        w = r.get("reach") or {}
        for x in r["waits"]:
            if not (FR.SETTLE_F / FR.FPS - 1e-9 <= x["wait_s"] <= FR.SETTLE_MAX_F / FR.FPS + 1e-9) or not (x["how"] == "settled" or abs(x["wait_s"] - FR.SETTLE_MAX_F / FR.FPS) < 1e-9):
                wbad.append((k, x["stage"], x["wait_s"], x["how"]))
        if "f_label" not in w:
            continue
        kd, fl, fe, dl, fc = w["label_kind"], w["f_label"], w["f_end"], r["delay_f"], w["f_cmd"]
        kinds[kd] += 1
        on = (fl - fc) % FR.TICKS == 0
        lo, hi = fe + dl + FR.SETTLE_F, fe + dl + FR.SETTLE_MAX_F
        good = on and ((kd == 0 and w["settled_end"] and lo <= fl <= hi) or (kd == 1 and not w["settled_end"] and hi <= fl < hi + FR.TICKS))
        i = seg_of.get(k)
        if i is not None:
            last = so[i] + n_steps[i] - 1
            land = z["W"][last, 4:7].astype(float)
            e_rec = 100 * float(np.linalg.norm(land - np.asarray(w["goal_w"])))
            st = z["start"][i].astype(float)
            t = z["target"][i].astype(float)
            cy, sy = math.cos(st[3]), math.sin(st[3])
            tw = np.array([st[0] + cy * t[0] - sy * t[1], st[1] + sy * t[0] + cy * t[1], t[2]])
            jaw = 100 * float(np.linalg.norm(land - tw))
            good = good and abs(e_rec - r["err_cm"]) < 1e-3 and abs(float(z["jaw_err_cm"][i]) - jaw) < 1e-3 and abs(float(z["live_err_cm"][i]) - float(z["jaw_err_cm"][i])) < 1e-3
            good = good and bool(z["settled_end"][i]) == bool(w["settled_end"]) and int(z["label_kind"][i]) == kd
        else:
            good = False
        if not good:
            b5.append((k, kd, fl, fe, dl))
    print(f"5. landing rule: labels by kind {dict(kinds)}; violations {len(b5)} {b5[:4]}; waits outside the rule {len(wbad)} {wbad[:4]}")
    bad += int(bool(b5) or bool(wbad))
    # 6. targets, pitches, first rows
    b6 = []
    for (c, e, m), r in ok.items():
        if r["goal_h"] != FG.goal(FG.cell(c), e).tolist():
            b6.append(("goal", c, e, m))
        w = r.get("reach") or {}
        if "pitch_deg" in w and abs(w["pitch_deg"] - grid[c]["pitch_deg"]) > 1e-9:
            b6.append(("pitch", c, e, m))
    nf = 0
    for (c, e), g in by_ce.items():
        idx = [seg_of[(c, e, m)] for m in g if (c, e, m) in seg_of]
        for i in idx[1:]:
            nf += 1
            if not np.array_equal(z["S"][so[i]], z["S"][so[idx[0]]]):
                b6.append(("first row", c, e))
    print(f"6. goals = fs_grid.goal, pitches = grid.json, first rows equal within (cell, ep) ({nf} comparisons): {len(b6)} violations {b6[:4]}")
    bad += int(bool(b6))
    print(f"{backend}: {'CHECK PASSED' if not bad else f'CHECK FAILED ({bad} items)'}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="scalebfm")
    ap.add_argument("--no-models", action="store_true")
    a = ap.parse_args()
    sys.exit(1 if main(a.backend, not a.no_models) else 0)
