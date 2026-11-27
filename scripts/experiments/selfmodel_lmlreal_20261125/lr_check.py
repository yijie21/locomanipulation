"""selfmodel_lmlreal_20261125: checks of lr_run's records (read-only).

    cd $G; PYTHONPATH=$G:$W/src python lr_check.py --backend scalebfm [--hg]      (locomanilab env: torch + data.py)

1. Layout (d3_collect.check's 1, 2, 3, 5 on these files): the end error recomputed from the last row's acting hand point
   W and the target taken back to the world = jaw_err_cm; rows / waypoints / arm arrays aligned; ik_err_cm recomputed
   from the plan's last waypoint (robot_fk); data.py's derived err_cm = jaw_err_cm on every segment, and its own settle
   flag on our settled labels.
2. Direct error three ways, settled labels: recorded jaw_err_cm, data.py's err_cm read back from the npz (heading frame
   of the command row), and live_err_cm computed in the episode at the label row from the simulator's wrist_yaw_link
   (world, the planned world target).
3. Hand point and attitude: KP's hand point (13 cm ahead of wrist_yaw_link) = W's right hand point on every row; HQ's
   right wrist quaternion = J's rotation columns on every row; the label row's hand point (hand point of the last row)
   in end_pos = data.py's pos_f.
4. Features at the command row: the x0 / u data.py builds from the npz against the ones lml_tasks.Predictor.features
   built online at the command (hardgrasp's path: lml_selfmodel.segments2 + data.derive on the live rows), stored in
   the episode json; the online path had walked = 0 and chain = min(chain, 1), so both are set so here. Then the
   hostdata model (hardgrasp's wm_s0.pt) on the npz's features against its online prediction pred_exec_h.
5. (--hg, ScaleBFM) each episode against hardgrasp's archived alone episode: outcome and every reach's command, timing,
   IK error, hand error and hostdata prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lr_run as LR  # noqa: E402

D3, L, LT, HG = LR.D3, LR.L, LR.LT, LR.HG


def load_d(path: Path, walked0: bool, chain_max: int | None) -> dict:
    """data.py's load of one file, with walked / chain overridden before derive (for the comparison with the online
    features)."""
    DT, _ = LT.reachwm()
    z = dict(np.load(path))
    if walked0:
        z["walked"] = np.zeros_like(z["walked"])
    if chain_max is not None:
        z["chain"] = np.minimum(z["chain"], chain_max)
    z.setdefault("walk", np.zeros((len(z["n_steps"]), 3), np.float32))
    z.setdefault("target_b", z["target"])
    z["files"] = [str(path)]
    DT.derive(z)
    return z


def main(backend: str, hg: bool) -> None:
    base = LR.OUT / backend
    p = base / "data" / "all.npz"
    z, A = dict(np.load(p)), dict(np.load(base / "data" / "arm.npz"))
    DT, _ = LT.reachwm()
    bad = 0
    n = z["n_steps"].astype(int)
    so, last = np.r_[0, np.cumsum(n)[:-1]], np.cumsum(n) - 1
    N = len(n)
    print(f"{backend}: {N} segments, {int(n.sum())} rows; reach_no {dict(zip(*[v.tolist() for v in np.unique(z['reach_no'], return_counts=True)]))}; "
          f"fell/lost {int(z['fell'].sum())}; settled at the label {int(z['settled_end'].sum())}")  # fmt: skip
    # 1. layout
    st = z["start"].astype(float)
    c, s = np.cos(st[:, 3]), np.sin(st[:, 3])
    t = z["target"].astype(float)
    tw = np.c_[st[:, 0] + c * t[:, 0] - s * t[:, 1], st[:, 1] + s * t[:, 0] + c * t[:, 1], t[:, 2]]
    jaw = z["W"][last, 4:7].astype(float)
    assert (z["side"] > 0).all(), "right hand only in these tasks"
    d1 = np.abs(100 * np.linalg.norm(jaw - tw, axis=1) - z["jaw_err_cm"])
    print(f"1. end error recomputed from the last row vs jaw_err_cm: max |d| {d1.max():.5f} cm")
    bad += int(d1.max() > 0.01)
    nr = z["n_ref"].astype(int)
    al = (len(A["X"]) == len(z["S"]) == n.sum() and len(A["XR"]) == len(z["RT"]) == nr.sum() and (A["n_steps"] == n).all()
          and (A["n_ref"] == nr).all())  # fmt: skip
    ro = np.r_[0, np.cumsum(nr)[:-1]]
    print(f"   arrays aligned: {al}; RT starts at 0: {bool(np.all(z['RT'][ro] == 0))}; k0 {sorted(set(z['k0'].tolist()))}")
    bad += int(not al)
    pl = L.Planner()
    e3 = []
    for i in range(N):
        j = ro[i] + nr[i] - 1
        hp = pl.fk_world(A["RP"][j, :3].astype(float), A["RP"][j, 3:7].astype(float), A["RQ"][j].astype(float))
        e3.append(abs(100 * np.linalg.norm(hp["right"][0] - tw[i]) - z["ik_err_cm"][i]))
    print(f"   ik_err_cm recomputed from the last waypoint ({N} segments): max |d| {max(e3):.5f} cm")
    bad += int(max(e3) > 0.05)
    D = DT.load(str(p))
    d5 = np.abs(D["err_cm"] - D["jaw_err_cm"])
    ok = (~z["fell"].astype(bool)) & z["settled_end"].astype(bool)
    print(f"   data.py err_cm vs jaw_err_cm, all {N}: max |d| {d5.max():.5f} cm; data.py's settled flag on our settled labels "
          f"{100 * float(np.mean(D['settled'][ok])):.1f} % ({int(ok.sum())})")  # fmt: skip
    bad += int(d5.max() > 0.01)
    # 2. direct error three ways
    lv = z["live_err_cm"]
    m2 = ok & (lv >= 0)
    d2a, d2b = np.abs(lv[m2] - z["jaw_err_cm"][m2]), np.abs(lv[m2] - D["err_cm"][m2])
    print(f"2. direct error at the label, {int(m2.sum())} settled labels: live (episode, world) vs recorded jaw_err_cm max |d| "
          f"{d2a.max():.5f} cm; live vs data.py err_cm (npz, heading frame) max |d| {d2b.max():.5f} cm; medians "
          f"{np.median(lv[m2]):.3f} / {np.median(z['jaw_err_cm'][m2]):.3f} / {np.median(D['err_cm'][m2]):.3f} cm")  # fmt: skip
    bad += int(d2a.max() > 0.01 or d2b.max() > 0.01)
    # 3. hand point and attitude
    dk = np.linalg.norm(z["KP"][:, 0, 0].astype(float) - z["W"][:, 4:7].astype(float), axis=1)
    q = z["HQ"][:, 0].astype(float)
    w, x, y, zz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    c0 = np.stack([1 - 2 * (y * y + zz * zz), 2 * (x * y + w * zz), 2 * (x * zz - w * y)], -1)
    c1 = np.stack([2 * (x * y - w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz + w * x)], -1)
    dj = np.abs(np.c_[c0, c1] - z["J"].astype(float)).max(1)
    pe = np.abs(z["end_pos"][:, 0].astype(float) - D["pos_f"].astype(float)).max(1)
    print(f"3. KP hand point vs W right hand point, every row: max {1000 * dk.max():.4f} mm; HQ right quaternion vs J columns: "
          f"max |d| {dj.max():.2e}; end_pos (right) vs data.py pos_f: max {1000 * pe.max():.4f} mm")  # fmt: skip
    bad += int(dk.max() > 1e-5 or dj.max() > 1e-4 or pe.max() > 1e-5)
    # 4. features at the command row against the online ones
    Do = load_d(p, walked0=True, chain_max=1)
    eps = {}
    for t_i, task in enumerate(LR.TASKS):
        for f in sorted((base / "episodes" / task).glob("ep*.json")):
            r = json.loads(f.read_text())
            eps[(t_i, r["ep"])] = r
    dx, du, used, miss = [], [], [], 0
    wi = 65 * 3 + 9  # x0 index of `walked` (S x3, jaw0, rot0)
    for i in range(N):
        r = eps.get((int(z["task_id"][i]), int(z["ep"][i])))
        k = int(z["reach_no"][i]) - 1
        w_ = next((w for w in (r or {}).get("reaches", []) if w["k"] == k and "feat_x0" in w), None)
        if w_ is None:
            miss += 1
            continue
        dx.append(np.abs(np.asarray(w_["feat_x0"], np.float32) - Do["x0"][i]).max())
        du.append(np.abs(np.asarray(w_["feat_u"], np.float32) - Do["u"][i]).max())
        used.append((i, w_))
    dx, du = np.array(dx), np.array(du)
    D1 = DT.load(str(p))
    wdiff = int((D1["x0"][:, wi] != Do["x0"][:, wi]).sum())
    print(f"4. x0 / u from the npz vs the online features ({len(used)} segments, {miss} without): x0 max |d| {dx.max():.3e}, u max |d| "
          f"{du.max():.3e}; segments with any x0 difference {int((dx > 0).sum())}; (the walked flag differs online/npz on "
          f"{wdiff} segments: data3's meaning in the npz, 0 online)")  # fmt: skip
    bad += int(dx.max() > 1e-5 or du.max() > 1e-6)
    P = LT.Predictor(HG.WM)
    GD = P.M.tensors(Do, "cpu", step=False)
    import torch

    with torch.no_grad():
        o = P.model.predict(P.M.sub(GD, torch.tensor([i for i, _ in used]), False))
    pp = o["pos"].numpy().astype(float)
    dp = np.array([np.abs(pp[j] - np.asarray(w_["pred_exec_h"])).max() for j, (_, w_) in enumerate(used)])
    print(f"   hostdata model on the npz's features vs its online prediction: max |d| {100 * dp.max():.2e} cm")
    bad += int(dp.max() > 1e-5)
    # 5. hardgrasp's archive
    if hg:
        same, diff, worst = 0, [], {}
        for (t_i, e), r in sorted(eps.items()):
            ap = HG.OUT / "episodes" / LR.TASKS[t_i] / "alone" / f"ep{e:+05d}.json"
            if not ap.exists():
                diff.append((LR.TASKS[t_i], e, "no archive"))
                continue
            a = json.loads(ap.read_text())
            keys = ("success", "reason", "why", "steps", "n_walks", "fell")
            dd = [k for k in keys if a.get(k) != r.get(k)]
            for k in ("close_dist_cm", "rise_max_cm"):
                if (k in a) != (k in r) or (k in a and abs(a[k] - r[k]) > 1e-9):
                    dd.append(k)
                if k in a and k in r:
                    worst[k] = max(worst.get(k, 0.0), abs(a[k] - r[k]))
            ra, rr = a.get("reaches", []), r.get("reaches", [])
            if len(ra) != len(rr):
                dd.append("n_reaches")
            for wa, wr in zip(ra, rr):
                for k in ("k", "f_cmd", "f_end"):
                    if wa.get(k) != wr.get(k):
                        dd.append(f"reach {wa.get('k')} {k}")
                for k in ("cmd_h", "ik_err_cm", "err_cm", "pred_exec_h", "act_h"):
                    if k in wa or k in wr:
                        v = float(np.abs(np.asarray(wa.get(k, np.nan)) - np.asarray(wr.get(k, np.nan))).max()) if (k in wa and k in wr) else np.inf
                        worst[k] = max(worst.get(k, 0.0), v)
                        if v > 1e-9:
                            dd.append(f"reach {wa.get('k')} {k}")
            if dd:
                diff.append((LR.TASKS[t_i], e, dd[:6]))
            else:
                same += 1
        print(f"5. against hardgrasp's archived alone episodes: {same} identical of {len(eps)}; largest differences "
              f"{ {k: float(f'{v:.3g}') for k, v in worst.items()} }")  # fmt: skip
        for d_ in diff[:12]:
            print(f"   differs: {d_}")
        bad += int(len(diff) > 0)
    print("CHECK", "PASSED" if bad == 0 else f"FAILED ({bad})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=LR.BACKENDS, default="scalebfm")
    ap.add_argument("--hg", action="store_true", help="also compare with hardgrasp's archived alone episodes")
    a = ap.parse_args()
    main(a.backend, a.hg)
