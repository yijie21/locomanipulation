"""selfmodel_lmlcmp_20261126: checks of lc_run's records (read-only).

    cd $G; PYTHONPATH=$G:$W/src python lc_check.py --backend scalebfm         (locomanilab env: torch + data.py)
    python lc_check.py --models                                               (the model loading, every backend)

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
5. Pairing: in each (task, ep) both groups have the same steps up to the pre-grasp command -- the via point's command,
   timing and features, the pre-grasp's command row, wait and the start part of its x0 (columns 0..CHAIN_COL; the rest
   is the planned reference, which follows the command) -- since nothing differs before it.
--models: each model rebuilt on lc_run's template (lmlreal's records) and on data3's data: equal predictions on
   lmlreal's records (the normalisation is in the state, not taken from the template).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_run as LC  # noqa: E402

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
    e3 = []
    for i in range(N):
        j = ro[i] + nr[i] - 1
        hp = pl.fk_world(A["RP"][j, :3].astype(float), A["RP"][j, 3:7].astype(float), A["RQ"][j].astype(float))
        e3.append(abs(100 * np.linalg.norm(hp["right"][0] - tw[i]) - z["ik_err_cm"][i]))
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
    # 5. pairing up to the pre-grasp command
    same, diff = 0, []
    for (t_i, m_i, e), ra in E.items():
        if m_i != 0 or (t_i, 1, e) not in E:
            continue
        rb = E[(t_i, 1, e)]
        dd = []
        wa = {w_.get("reach_no"): w_ for w_ in ra.get("reaches", [])}
        wb = {w_.get("reach_no"): w_ for w_ in rb.get("reaches", [])}
        for rn in (0, 1):
            if (rn in wa) != (rn in wb):
                dd.append(f"reach {rn} present")
                continue
            if rn not in wa:
                continue
            a_, b_ = wa[rn], wb[rn]
            for kk in ("f_cmd", "wait_s"):
                if a_.get(kk) != b_.get(kk):
                    dd.append(f"reach {rn} {kk}")
            # x0's start part (S at k0, k0-1, k0-2, jaw0, rot0, walked, chain: columns 0..CHAIN_COL); the rest of x0 is the
            # planned reference (GOAL_REF), which follows the command and differs at the pre-grasp by design
            nx = len(a_.get("feat_x0", [])) if rn == 0 else LC.CHAIN_COL + 1
            if "feat_x0" in a_ and "feat_x0" in b_ and np.abs(np.asarray(a_["feat_x0"][:nx]) - np.asarray(b_["feat_x0"][:nx])).max() > 0:
                dd.append(f"reach {rn} x0")
            if rn == 0 and np.abs(np.asarray(a_.get("cmd_h", 0)) - np.asarray(b_.get("cmd_h", 0))).max() > 0:
                dd.append("via cmd")
        if ra.get("n_walks") != rb.get("n_walks"):
            dd.append("n_walks")
        if dd:
            diff.append(((LC.TASKS[t_i][5:20], e), dd[:4]))
        else:
            same += 1
    print(f"5. pairs identical up to the pre-grasp command: {same} of {same + len(diff)}; differing {diff[:6]}")
    bad += int(bool(diff))
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
