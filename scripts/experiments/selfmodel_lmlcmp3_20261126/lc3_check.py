"""selfmodel_lmlcmp3_20261126: lmlcmp2's lc2_check.py (6ebfec2f) on lc3_run's records (read-only); changes marked
"lmlcmp3:" -- 5 is the registration's consistency rule for every pair of groups, 6 adds E's and S's rules, 7 is new.

    cd $G; PYTHONPATH=$G:$W/src python lc3_check.py --backend scalebfm        (locomanilab env: torch + data.py)

lmlcmp3's 5-7 (they replace lmlcmp2's 5 and extend its 6):
5. Commands identical -> results identical, every pair of groups of an episode (records without a worker error): the
   executed commands (cmd_h) in order are compared; every reach before the first differing command must be the same
   reach (f_cmd, wait_s, cmd_h, ik_err_cm, ref_end_h, the whole x0 / u, the three predictions, the landing row and
   act_h, the segment's label / contact counts); the first differing reach must still go at the same row after the
   same wait from the same state (x0's start part, columns 0..CHAIN_COL); and when every command is the same (same
   count), the whole episode must be the same (success, reason, close_dist_cm, close_obj_cm, rise_max_cm, steps,
   n_contacts, the lift command). Plus lmlcmp2's: reachable's first pre-grasp candidate = selfmodel's.
6. lmlcmp2's reachable rules; lmlcmp3: alone and E's via point / pre-grasp execute the goal itself with no inversion;
   every inverted reach (selfmodel, reachable, E's entry) starts at the goal, chooses selfmodel's way (the first least
   loss) and executes the chosen candidate; selfmodel and E record no shrink / guard.
7. lmlcmp3: the reference's end -- |ref_end_h - cmd_h| = ik_err_cm and ref_end_w = the FK of the segment's last
   waypoint (arm.npz RP / RQ, as 1), every executed reach.

lmlcmp2's docstring follows (its 5 and 6 are replaced as above).

selfmodel_lmlcmp2_20261126: lmlcmp's lc_check.py (d72efcb3) on lc2_run's records, 4 groups (read-only); changes marked
"lmlcmp2:" -- 5 compares every group with alone; 6 is new.

1. Layout (lr_check's 1 and 3): the end error recomputed from the last row's hand point W and the target taken back to
   the world = jaw_err_cm; rows / waypoints / arm arrays aligned; ik_err_cm from the plan's last waypoint (robot_fk);
   data.py's err_cm = jaw_err_cm; KP / HQ against W / J; end_pos against data.py's pos_f; every episode's segments = its
   json's d3 records (count, reach_no, f_cmd order).
2. The landing rule on every record: label_kind 0 -> settled_end, on the 10 Hz grid from the command, within
   [f_end + delay + 0.5 s, f_end + delay + 3 s]; 1 -> not settled, the first grid row at or after f_end + delay + 3 s;
   2 -> not settled, the last grid row at or before the next command (or the close); live_err_cm (episode, world) =
   jaw_err_cm on every label.
3. Features and predictions: the x0 / u data.py builds from the npz against the ones built online at the command (stored
   in the episode json), every executed reach; T3 / M / D on the npz's features against their online predictions.
4. The settle waits: every reach / close / lift went through one wait (wait_s in [0, 3]); the own rule held when it went
   or the wait ran 3 s; the segment's pre_wait_s / prop_settled = the episode's wait log.
5. Pairing: in each (task, ep) every group has alone's steps up to the pre-grasp command -- the via point's command,
   timing and features, the pre-grasp's command row, wait and the start part of its x0 (columns 0..CHAIN_COL; the rest
   is the planned reference, which follows the command) -- since nothing differs before it; and guard / reachable's
   pre-grasp fixed point starts as selfmodel's (guard: every candidate = selfmodel's; reachable: its first candidate).
6. lmlcmp2's rules on every inverted reach. guard: the candidate before the guards = selfmodel's choice rule on its own
   candidates (least loss, first); executed |command - goal| <= 10 cm; capped -> exactly 10 cm along the chosen
   correction; would fall back (D's predicted miss after the cap > 3 cm) -> the goal is executed; else D's recorded
   prediction for the executed command is within 3 cm of the goal and = the guard's. reachable: a shrunk candidate is
   alpha * its correction (alpha < 1); the executed command's IK error <= 2 cm unless it is the goal itself.
--models: each model rebuilt on lc_run's template (lmlreal's records) and on data3's data: equal predictions on
   lmlreal's records (the normalisation is in the state, not taken from the template).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc3_run as LC  # noqa: E402  (lmlcmp3)

D3, L, LT, HG = LC.D3, LC.L, LC.LT, LC.HG


def load_d(path: Path) -> dict:
    DT, _ = LT.reachwm()
    z = dict(np.load(path))
    z.setdefault("walk", np.zeros((len(z["n_steps"]), 3), np.float32))
    z.setdefault("target_b", z["target"])
    z["files"] = [str(path)]
    DT.derive(z)
    return z


def episodes(backend: str) -> dict:
    out = {}
    for t_i, task in enumerate(LC.TASKS):
        for m_i, m in enumerate(LC.METHODS):
            for f in sorted((LC.OUT / backend / "episodes" / task / m).glob("ep*.json")):
                r = json.loads(f.read_text())
                out[(t_i, m_i, r["ep"])] = r
    return out


def main(backend: str) -> None:
    import torch

    base = LC.OUT / backend
    p = base / "data" / "all.npz"
    z, A = dict(np.load(p)), dict(np.load(base / "data" / "arm.npz"))
    DT, _ = LT.reachwm()
    bad = 0
    n = z["n_steps"].astype(int)
    last = np.cumsum(n) - 1
    N = len(n)
    E = episodes(backend)
    print(f"{backend}: {len(E)} episode records, {N} segments, {int(n.sum())} rows; method {dict(zip(*[v.tolist() for v in np.unique(z['method_id'], return_counts=True)]))}; "
          f"reach_no {dict(zip(*[v.tolist() for v in np.unique(z['reach_no'], return_counts=True)]))}; fell/lost {int(z['fell'].sum())}; "
          f"settled at the label {int(z['settled_end'].sum())}")  # fmt: skip
    # 1. layout
    st = z["start"].astype(float)
    c, s = np.cos(st[:, 3]), np.sin(st[:, 3])
    t = z["target"].astype(float)
    tw = np.c_[st[:, 0] + c * t[:, 0] - s * t[:, 1], st[:, 1] + s * t[:, 0] + c * t[:, 1], t[:, 2]]
    jaw = z["W"][last, 4:7].astype(float)
    assert (z["side"] > 0).all(), "right hand only in these tasks"
    d1 = np.abs(100 * np.linalg.norm(jaw - tw, axis=1) - z["jaw_err_cm"])
    nr = z["n_ref"].astype(int)
    ro = np.r_[0, np.cumsum(nr)[:-1]]
    al = (len(A["X"]) == len(z["S"]) == n.sum() and len(A["XR"]) == len(z["RT"]) == nr.sum() and (A["n_steps"] == n).all()
          and (A["n_ref"] == nr).all())  # fmt: skip
    pl = L.Planner()
    e3, hp_last = [], []
    for i in range(N):
        j = ro[i] + nr[i] - 1
        hp = pl.fk_world(A["RP"][j, :3].astype(float), A["RP"][j, 3:7].astype(float), A["RQ"][j].astype(float))
        e3.append(abs(100 * np.linalg.norm(hp["right"][0] - tw[i]) - z["ik_err_cm"][i]))
        hp_last.append(hp["right"][0])  # lmlcmp3: for 7
    D = load_d(p)
    d5 = np.abs(D["err_cm"] - D["jaw_err_cm"])
    dk = np.linalg.norm(z["KP"][:, 0, 0].astype(float) - z["W"][:, 4:7].astype(float), axis=1)
    q = z["HQ"][:, 0].astype(float)
    w, x, y, zz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    c0 = np.stack([1 - 2 * (y * y + zz * zz), 2 * (x * y + w * zz), 2 * (x * zz - w * y)], -1)
    c1 = np.stack([2 * (x * y - w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz + w * x)], -1)
    dj = np.abs(np.c_[c0, c1] - z["J"].astype(float)).max(1)
    pe = np.abs(z["end_pos"][:, 0].astype(float) - D["pos_f"].astype(float)).max(1)
    # per episode: segments = the json's d3 records
    key = list(zip(z["task_id"].tolist(), z["method_id"].tolist(), z["ep"].tolist()))
    seg_of: dict = {}
    for i, k in enumerate(key):
        seg_of.setdefault(k, []).append(i)
    mism = []
    for k, r in E.items():
        si = seg_of.get(k, [])
        d3 = r.get("d3", [])
        if len(si) != r.get("n_segments", 0) or [int(z["reach_no"][i]) for i in si] != [d["reach_no"] for d in d3[: len(si)]] \
                or any(int(z["chain"][i]) != d3[j]["chain"] for j, i in enumerate(si)):  # fmt: skip
            mism.append(k)
    print(f"1. end error from the last row vs jaw_err_cm: max |d| {d1.max():.5f} cm; arrays aligned {al}; ik_err_cm from the last "
          f"waypoint max |d| {max(e3) if e3 else 0:.5f} cm; data.py err_cm vs jaw_err_cm max |d| {d5.max():.5f} cm; KP vs W max "
          f"{1000 * dk.max():.4f} mm; HQ vs J max |d| {dj.max():.2e}; end_pos vs pos_f max {1000 * pe.max():.4f} mm; episodes whose "
          f"segments differ from their d3 list: {len(mism)} {mism[:5]}")  # fmt: skip
    bad += int(bool(d1.max() > 0.01 or not al or (len(e3) > 0 and max(e3) > 0.05) or d5.max() > 0.01 or dk.max() > 1e-5
                    or dj.max() > 1e-4 or pe.max() > 1e-5 or len(mism) > 0))  # fmt: skip
    # 2. the landing rule
    viol, kinds = [], {0: 0, 1: 0, 2: 0}
    for k, r in E.items():
        dl = r.get("delay_f")
        cmds = [w_["f_cmd"] for w_ in r.get("reaches", []) if "f_cmd" in w_]
        close_f = next((w_["fire"] for w_ in r.get("waits", []) if w_.get("reach_no") == -1), None)
        for j, d in enumerate(r.get("d3", [])):
            if d["f_label"] is None:
                continue
            kd, fl, fc, fe = d["label_kind"], d["f_label"], d["f_cmd"], d["f_end"]
            kinds[kd] = kinds.get(kd, 0) + 1
            nxt = cmds[j + 1] if j + 1 < len(cmds) else close_f
            ok = (fl - fc) % LC.TICKS == 0
            if kd == 0:
                ok &= bool(d["settled_end"]) and fe + dl + LC.SETTLE_F <= fl <= fe + dl + LC.SETTLE_MAX_F + LC.TICKS
            elif kd == 1:
                ok &= (not d["settled_end"]) and fe + dl + LC.SETTLE_MAX_F <= fl < fe + dl + LC.SETTLE_MAX_F + LC.TICKS
            elif kd == 2:
                ok &= (not d["settled_end"]) and nxt is not None and fl <= nxt < fl + LC.TICKS
            else:
                ok = False
            if not ok:
                viol.append((k, j, kd, fl, fe, dl, nxt))
    lv = z["live_err_cm"]
    m2 = lv >= 0
    d2 = np.abs(lv[m2] - z["jaw_err_cm"][m2])
    print(f"2. landing rule: labels by kind (0 settled, 1 3 s timeout, 2 cut) {kinds}; violations {len(viol)} {viol[:4]}; live_err_cm "
          f"vs jaw_err_cm on {int(m2.sum())} labels max |d| {d2.max() if len(d2) else 0:.5f} cm")  # fmt: skip
    bad += int(bool(viol) or bool(len(d2) and d2.max() > 0.01))
    # 3. features and predictions
    dx, du, dp, used = [], [], {nm: [] for nm in LC.MODEL_NAMES}, []
    for i, k in enumerate(key):
        r = E.get(k)
        rn = int(z["reach_no"][i])
        w_ = next((w_ for w_ in (r or {}).get("reaches", []) if w_.get("reach_no") == rn and "feat_x0" in w_), None)
        if w_ is None:
            continue
        dx.append(np.abs(np.asarray(w_["feat_x0"], np.float32) - D["x0"][i]).max())
        du.append(np.abs(np.asarray(w_["feat_u"], np.float32) - D["u"][i]).max())
        used.append((i, w_))
    MD = LC.Models(backend)
    GD = MD.M.tensors(D, "cpu")
    idx = torch.tensor([i for i, _ in used])
    for nm in LC.MODEL_NAMES:
        G = GD
        if nm == "T3":
            G = dict(GD)
            G["x0"] = GD["x0"].clone()
            G["x0"][:, LC.CHAIN_COL] = G["x0"][:, LC.CHAIN_COL].clamp(max=LC.T3_CHAIN_MAX)
        with torch.no_grad():
            o = MD.m[nm].predict(MD.M.sub(G, idx, False))
        pp = o["pos"].numpy().astype(float)
        dp[nm] = [float(np.abs(pp[j] - np.asarray(w_["pred_h"][nm])).max()) for j, (_, w_) in enumerate(used)]
    dx, du = np.array(dx), np.array(du)
    print(f"3. x0 / u from the npz vs online ({len(used)} of {N} segments): x0 max |d| {dx.max():.3e}, u max |d| {du.max():.3e}; "
          f"predictions npz vs online max |d| " + ", ".join(f"{nm} {100 * max(v):.2e} cm" for nm, v in dp.items()))  # fmt: skip
    bad += int(len(used) < N or dx.max() > 1e-5 or du.max() > 1e-6 or any(max(v) > 1e-5 for v in dp.values()))
    # 4. the waits
    wbad, nw, wseg = [], 0, []
    for k, r in E.items():
        if "error" in r:
            continue
        for w_ in r.get("waits", []):
            nw += 1
            if not (0 <= w_["wait_s"] <= LC.SETTLE_MAX_F / LC.FPS + 1e-9) or not (w_["prop_settled"] or abs(w_["wait_s"] - LC.SETTLE_MAX_F / LC.FPS) < 1e-9):
                wbad.append((k, w_["key"], w_["wait_s"], w_["prop_settled"]))
        ws = [w_ for w_ in r.get("waits", []) if w_.get("reach_no", -9) >= 0]
        for j, i in enumerate(seg_of.get(k, [])):
            wl = ws[j] if j < len(ws) else None
            if wl is None or abs(float(z["pre_wait_s"][i]) - wl["wait_s"]) > 1e-6 or bool(z["prop_settled"][i]) != wl["prop_settled"]:
                wseg.append((k, j))
    print(f"4. waits: {nw}; outside [0, 3 s] or gone unsettled before 3 s: {len(wbad)} {wbad[:4]}; segments whose pre_wait_s / "
          f"prop_settled differ from the wait log: {len(wseg)} {wseg[:4]}")  # fmt: skip
    bad += int(bool(wbad) or bool(wseg))
    # 5. lmlcmp3: commands identical -> results identical (every pair of groups of an episode)
    def reach_same(a_: dict, b_: dict, full: bool) -> list[str]:
        out = []
        keys = ("f_cmd", "wait_s") + (("cmd_h", "ik_err_cm", "ref_end_h", "feat_x0", "feat_u", "pred_h", "f_label", "act_h") if full else ())
        for kk in keys:
            if a_.get(kk) != b_.get(kk):
                out.append(kk)
        if not full and np.abs(np.asarray(a_["feat_x0"][: LC.CHAIN_COL + 1]) - np.asarray(b_["feat_x0"][: LC.CHAIN_COL + 1])).max() > 0:
            out.append("x0 start")
        return out

    def seg_same(ra: dict, rb: dict, rn: int) -> bool:
        da = next((d for d in ra.get("d3", []) if d["reach_no"] == rn), None)
        db = next((d for d in rb.get("d3", []) if d["reach_no"] == rn), None)
        ks = ("f_cmd", "f_end", "f_label", "label_kind", "settled_end", "furn_rows", "obj_rows", "win_rows", "jaw_err_cm", "live_err_cm")
        return (da is None) == (db is None) and (da is None or all(da.get(kk) == db.get(kk) for kk in ks))

    n_pairs, n_all_same, diff5, n_split = 0, 0, [], Counter()
    by_ep: dict = {}
    for (t_i, m_i, e), r in E.items():
        if "error" not in r:
            by_ep.setdefault((t_i, e), {})[m_i] = r
    for (t_i, e), grp in by_ep.items():
        ms = sorted(grp)
        for ia in range(len(ms)):
            for ib in range(ia + 1, len(ms)):
                ra, rb = grp[ms[ia]], grp[ms[ib]]
                wa = [w_ for w_ in ra.get("reaches", []) if "cmd_h" in w_]
                wb = [w_ for w_ in rb.get("reaches", []) if "cmd_h" in w_]
                n_pairs += 1
                dd = []
                j = 0
                while j < min(len(wa), len(wb)) and wa[j]["reach_no"] == wb[j]["reach_no"] and wa[j]["cmd_h"] == wb[j]["cmd_h"]:
                    dd += [f"reach {wa[j]['reach_no']} {kk}" for kk in reach_same(wa[j], wb[j], True)]
                    if not seg_same(ra, rb, wa[j]["reach_no"]):
                        dd.append(f"reach {wa[j]['reach_no']} segment")
                    j += 1
                tag = (LC.METHODS[ms[ia]], LC.METHODS[ms[ib]], LC.TASKS[t_i][5:20], e)
                if j < min(len(wa), len(wb)):  # the first differing command: same row, same wait, same state
                    n_split[wa[j]["reach_no"]] += 1
                    if wa[j]["reach_no"] != wb[j]["reach_no"]:
                        dd.append("reach kind at the split")
                    else:
                        dd += [f"split reach {wa[j]['reach_no']} {kk}" for kk in reach_same(wa[j], wb[j], False)]
                elif len(wa) == len(wb):  # every command the same: the whole episode the same
                    n_split["none"] += 1
                    for kk in ("success", "reason", "close_dist_cm", "close_obj_cm", "rise_max_cm", "steps", "n_contacts", "lift_cmd", "why"):
                        if ra.get(kk) != rb.get(kk):
                            dd.append(f"episode {kk}")
                    n_all_same += int(not dd)
                else:  # one has no further executed command: it must have stopped at that command's row (its own,
                    # differing, command failed to plan) or before it
                    n_split["count"] += 1
                    nxt = (wa if len(wa) > len(wb) else wb)[j]
                    short = rb if len(wa) > len(wb) else ra
                    if short.get("steps", 0) > nxt["f_cmd"] + 1:
                        dd.append("one group has a further command the other never reached")
                if ms[ia] == 1 and LC.METHODS[ms[ib]] == "reachable":  # lmlcmp2: reachable's first pre-grasp candidate = selfmodel's
                    ws_ = next((w_ for w_ in ra.get("reaches", []) if w_.get("reach_no") == 1 and "inv" in w_), None)
                    wr_ = next((w_ for w_ in rb.get("reaches", []) if w_.get("reach_no") == 1 and "inv" in w_), None)
                    if (ws_ is None) != (wr_ is None):
                        dd.append("pre-grasp inversion present")
                    elif ws_ is not None and (ws_["inv"][0]["t"] != wr_["inv"][0]["t"] or ws_["inv"][0]["pos"] != wr_["inv"][0]["pos"]):
                        dd.append("pre-grasp first candidate != selfmodel's")
                if dd:
                    diff5.append((tag, dd[:4]))
    print(f"5. pairs of groups: {n_pairs}; first differing command by reach_no {dict(n_split)} ('none': every command the same); "
          f"all commands the same and the whole episode the same: {n_all_same} of {n_split['none']}; violations {len(diff5)} {diff5[:6]}")  # fmt: skip
    bad += int(bool(diff5))
    # 6. the groups' rules (lmlcmp2's reachable; lmlcmp3: alone, E, selfmodel)
    r_n, r_sh, e_n, s_n, viol6 = 0, 0, 0, 0, []
    for (t_i, m_i, e), r in E.items():
        meth = LC.METHODS[m_i]
        for w_ in r.get("reaches", []):
            if "cmd_h" not in w_:
                continue
            gh, cmd = np.asarray(w_["goal_h"]), np.asarray(w_["cmd_h"])
            corr = float(np.linalg.norm(cmd - gh))
            tag = (meth, LC.TASKS[t_i][5:20], e, w_["reach_no"])
            inverted = (meth in LC.INVERT and w_["k"] >= 0) or (meth in LC.ENTRY_ONLY and w_["k"] == 1)
            if not inverted:
                if "inv" in w_ or corr != 0.0:
                    viol6.append(tag + ("a direct reach inverted or not the goal",))
                continue
            if "inv" not in w_:
                viol6.append(tag + ("no inversion record",))
                continue
            inv = w_["inv"]
            ls = [c_["l"] for c_ in inv]
            bi = min(range(len(ls)), key=lambda i: ls[i])
            if np.asarray(inv[0]["t"]).tolist() != gh.tolist():
                viol6.append(tag + ("first candidate not the goal",))
            if w_.get("model_fallback"):
                continue
            if bi != w_["inv_best"] or np.asarray(inv[bi]["t"]).tolist() != cmd.tolist():
                viol6.append(tag + ("choice rule / executed candidate",))
            if meth in ("selfmodel", "entryinv"):
                s_n += int(meth == "selfmodel")
                e_n += int(meth == "entryinv")
                if "shrink" in w_ or "guard" in w_ or len(inv) != LT.ITERS + 1 and not any(c_.get("err") for c_ in inv):
                    viol6.append(tag + ("shrink / guard record or candidate count",))
            elif meth == "reachable":
                r_n += 1
                shs = w_.get("shrink", [])
                r_sh += int(any(s_["shrunk"] for s_ in shs))
                for s_ in shs:
                    if s_["shrunk"] and not (s_["alpha"] < 1 and abs(s_["corr_cm"] - s_["alpha"] * s_["corr0_cm"]) < 1e-6):
                        viol6.append(tag + (f"shrink step {s_['it']}",))
                if corr > 0 and w_["ik_err_cm"] > 100 * LC.SHRINK_IK + 1e-6:
                    viol6.append(tag + (f"executed IK error {w_['ik_err_cm']:.2f} cm",))
    print(f"6. rules: selfmodel {s_n} inverted reaches, E {e_n} inverted entries, reachable {r_n} inverted reaches ({r_sh} with a "
          f"shrunk candidate); direct reaches = the goal, no inversion; violations {len(viol6)} {viol6[:6]}")  # fmt: skip
    bad += int(bool(viol6))
    # 7. lmlcmp3: the reference's end
    d7a, d7b, miss7 = [], [], 0
    for i, k in enumerate(key):
        r = E.get(k)
        rn = int(z["reach_no"][i])
        w_ = next((w_ for w_ in (r or {}).get("reaches", []) if w_.get("reach_no") == rn and "cmd_h" in w_), None)
        if w_ is None or "ref_end_h" not in w_:
            miss7 += 1
            continue
        d7a.append(abs(100 * np.linalg.norm(np.asarray(w_["ref_end_h"]) - np.asarray(w_["cmd_h"])) - w_["ik_err_cm"]))
        d7b.append(100 * float(np.linalg.norm(np.asarray(w_["ref_end_w"]) - hp_last[i])))
    print(f"7. reference end ({len(d7a)} of {N} segments; missing {miss7}): | |ref_end_h - cmd_h| - ik_err_cm | max {max(d7a) if d7a else 0:.2e} cm; "
          f"ref_end_w vs the FK of the segment's last waypoint max {max(d7b) if d7b else 0:.2e} cm")  # fmt: skip
    bad += int(miss7 > 0 or (d7a and max(d7a) > 1e-3) or (d7b and max(d7b) > 1e-3))
    print("CHECK", "PASSED" if bad == 0 else f"FAILED ({bad})")


def check_models() -> None:
    """Each model rebuilt on two different templates -- lc_run's (lmlreal's records of the controller) and data3's data
    of the controller -- gives the same predictions on lmlreal's records: the normalisation comes from the state, not
    from the template (lc_eval's val check then ties the models to their trained val numbers)."""
    import torch

    DT, M = LT.reachwm()
    import train as TR

    bad = 0
    for be in LC.BACKENDS:
        Dl = DT.load(str(LC.LMLREAL / be / "data" / "all.npz"))
        GD = M.tensors(Dl, "cpu")
        idx = torch.arange(len(Dl["n_steps"]))
        MD = LC.Models(be)
        Dt = DT.load(str(LC.ROOT / "selfmodel_data3_20261125" / be / "data" / "all.npz"))
        for nm in LC.MODEL_NAMES:
            m_b = TR.load_model("E3_ensemble", 1.0, Dt, "cpu", root=LC.model_roots(be)[nm])
            a_ = M.predict_batched(MD.m[nm], GD, idx)["pos"]
            b_ = M.predict_batched(m_b, GD, idx)["pos"]
            d = float((a_ - b_).abs().max())
            print(f"{be} {nm}: rebuilt on lmlreal's records vs on data3's data ({len(Dt['n_steps'])} segments): max |d pos| {100 * d:.2e} cm "
                  f"on {len(idx)} records")  # fmt: skip
            bad += int(d > 1e-6)
        del Dt
    print("MODELS", "PASSED" if bad == 0 else f"FAILED ({bad})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=LC.BACKENDS, default="scalebfm")
    ap.add_argument("--models", action="store_true")
    a = ap.parse_args()
    check_models() if a.models else main(a.backend)
