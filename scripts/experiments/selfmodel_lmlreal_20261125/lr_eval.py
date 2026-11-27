"""selfmodel_lmlreal_20261125's metrics (the entry's "度量"): one end model (train3: reachwm_20261030 train.py
E3_ensemble, frac 1.0, seed 0, trained on one controller's selfmodel_data3) on lr_run's recorded LocoManiLab task
reaches of the same controller; read-only (records, training data, model).

    cd $W && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $VPY lr_eval.py --backend scalebfm \
        [--records DIR/data/all.npz] [--data3 DIR] [--json OUT.json]

The model is rebuilt as train3's d3_eval.py does (train.load_model: normalisation from data3's frac-1.0 training split,
weights from models_s0/E3_ensemble_f1). Sanity line first: its val median on data3's kept val segments (fell 0,
settled_end 1), against the registered val error.
Records: data.py's derive on the npz, with chain clipped to the model's training maximum (data3: 0 / 1; the entry after a
via point is chain 2 in the record). Kept for the error metrics: not fell / lost, label settled (settled_end), as
d3_eval. Per reach (第一下 = reach_no 1 pre-grasp, 第二下 = reach_no 2 entry; the via point reach_no 0 on one line):
  model cm   predicted final hand point to the settled one (heading frame of the command row), median
  x val      model median / the registered val error (ScaleBFM 3.08, HEFT 3.99, SONIC 4.25 cm)
  direct cm  the hand left at the command: target to the settled hand point (data.py err_cm), median
  in range   all 29 joints at the command row (S's q at k0) inside data3's training 0.5-99.5 % band (frac-1.0 training
             split, every segment's command row)
  spread     the ensemble's member spread |pos_std| (cm) above its val 90th percentile
  settled    pre_settled: data3's settle rule true at the command row (pelvis and both hand points < 2 cm/s for 0.5 s)
Rows: all kept, kept with a settled start, in range vs out of range (and out / in), per task. 95 % intervals: 1000
resamples of episodes (task, ep: all of an episode's reaches together).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / "reachwm_20261030"), str(EXP / "selfwm_20261029")]
for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 (train3's flags) must be set before data.py is imported"
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M
NB = 1000
VAL_REG = {"scalebfm": 3.08, "heft": 3.99, "sonic": 4.25}  # the entry's registered val errors (train3)
D3_ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125")
LR_ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlreal_20261125")
TASKS = ("IThorBathroomLiftCandleFromFloor", "IThorKitchenApproachTableAndLiftApple", "IThorKitchenApproachTableAndLiftBowl",
         "IThorKitchenApproachTableAndLiftSaltShaker", "IThorKitchenTakeStatueFromCabinet",
         "IThorLivingRoomApproachTableAndLiftDumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan")  # lr_run.TASKS
SHORT = ("candle", "apple", "bowl", "salt", "statue_cab", "dumbbell", "wcan")
JOINTS = ("left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll", "right_hip_roll", "waist_roll", "left_hip_yaw",
          "right_hip_yaw", "waist_pitch", "left_knee", "right_knee", "left_shoulder_pitch", "right_shoulder_pitch",
          "left_ankle_pitch", "right_ankle_pitch", "left_shoulder_roll", "right_shoulder_roll", "left_ankle_roll",
          "right_ankle_roll", "left_shoulder_yaw", "right_shoulder_yaw", "left_elbow", "right_elbow", "left_wrist_roll",
          "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch", "left_wrist_yaw", "right_wrist_yaw")  # S order (L.SCALEBFM_ORDER)


def load_records(path: Path, chain_max: int | None) -> dict:
    z = dict(np.load(path))
    z["chain_rec"] = z["chain"].copy()
    if chain_max is not None:
        z["chain"] = np.minimum(z["chain"], chain_max)
    z.setdefault("walk", np.zeros((len(z["n_steps"]), 3), np.float32))
    z.setdefault("target_b", z["target"])
    z["files"] = [str(path)]
    DT.derive(z)
    return z


def predict(model, D: dict, idx: np.ndarray) -> dict:
    GD = M.tensors(D, "cpu")
    p = M.predict_batched(model, GD, torch.tensor(idx))
    return dict(err=(100 * (p["pos"] - GD["pos_f"][idx]).norm(dim=-1)).numpy().astype(float),
                spread=(100 * p["pos_std"].norm(dim=-1)).numpy().astype(float), pos=p["pos"].numpy())  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=list(VAL_REG))
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument("--data3", type=Path, default=None, help="the model's training data root (default data3's of the backend)")
    ap.add_argument("--models", type=Path, default=None, help="models root holding E3_ensemble_f1 (default <data3>/models_s0)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    be = a.backend
    d3 = a.data3 or D3_ROOT / be
    rec_p = a.records or LR_ROOT / be / "data" / "all.npz"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)
    # -- the model and its training data -----------------------------------------------------------------------------
    Dt = DT.load(str(d3 / "data" / "all.npz"))
    mroot = a.models or d3 / "models_s0"
    model = TR.load_model("E3_ensemble", 1.0, Dt, "cpu", root=mroot)
    sp = DT.splits(Dt, 1.0)
    va = sp["val"][(Dt["fell"][sp["val"]] == 0) & (Dt["settled_end"][sp["val"]] > 0)]
    pv = predict(model, Dt, va)
    sp90 = float(np.percentile(pv["spread"], 90))
    tr = sp["train"]
    qtr = Dt["q_0"][tr].astype(float)
    lo, hi = np.percentile(qtr, 0.5, axis=0), np.percentile(qtr, 99.5, axis=0)
    cmax = int(Dt["chain"][tr].max())
    val = VAL_REG[be]
    print(f"{be}: model {mroot / 'E3_ensemble_f1'}; val check: {len(va)} kept val segments, model median "
          f"{np.median(pv['err']):.3f} cm (registered {val:.2f}); val spread |pos_std| p90 {sp90:.3f} cm; training band: "
          f"{len(tr)} training segments' command rows; training chain max {cmax}, walked {sorted(set(Dt['walked'][tr].astype(int).tolist()))}")  # fmt: skip
    # -- the records -------------------------------------------------------------------------------------------------
    D = load_records(rec_p, cmax)
    N = len(D["n_steps"])
    allidx = np.arange(N)
    pr = predict(model, D, allidx)
    q0 = D["q_0"].astype(float)
    out_j = (q0 < lo) | (q0 > hi)
    inr = ~out_j.any(1)
    fell = D["fell"].astype(bool)
    setl = D["settled_end"].astype(bool)
    kept = ~fell & setl
    rn, tid, ep = D["reach_no"].astype(int), D["task_id"].astype(int), D["ep"].astype(int)
    pre = D["pre_settled"].astype(bool)
    big = pr["spread"] > sp90
    err, dire = pr["err"], D["err_cm"].astype(float)
    epk = tid * 1000 + ep
    print(f"records {rec_p}: {N} segments from {len(np.unique(epk))} episodes; reach_no {dict(zip(*[v.tolist() for v in np.unique(rn, return_counts=True)]))}; "
          f"fell / lost {int(fell.sum())}; label not settled {int((~fell & ~setl).sum())}; kept {int(kept.sum())}; chain in the "
          f"record {dict(zip(*[v.tolist() for v in np.unique(D['chain_rec'], return_counts=True)]))} -> clipped to {cmax}; walked "
          f"{int(D['walked'].sum())}")  # fmt: skip

    def groups(sel):
        k = epk[sel]
        o = np.argsort(k, kind="stable")
        return np.split(np.flatnonzero(sel)[o], np.flatnonzero(np.diff(k[o])) + 1)

    def boot(sel, stat):
        G = groups(sel)
        n = len(G)
        if n < 2:
            return float("nan"), float("nan")
        v = [stat(np.concatenate([G[j] for j in rng.integers(0, n, n)])) for _ in range(NB)]
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    med = lambda v, s: float(np.median(v[s])) if len(s) else float("nan")  # noqa: E731
    f2 = lambda v, ci, p=2: f"{v:.{p}f} [{ci[0]:.{p}f}, {ci[1]:.{p}f}]"  # noqa: E731
    res = dict(backend=be, val_check_cm=float(np.median(pv["err"])), val_registered_cm=val, val_spread_p90_cm=sp90, rows=[])

    def row(label, sel, indent="    "):
        n = int(sel.sum())
        if n == 0:
            print(f"{indent}{label:30s} n    0")
            return None
        ix = np.flatnonzero(sel)
        mci = boot(sel, lambda s: np.median(err[s]))
        r = dict(label=label, n=n, n_eps=len(np.unique(epk[sel])),
                 model=med(err, ix), model_ci=mci, x_val=med(err, ix) / val, x_val_ci=tuple(c / val for c in mci),
                 direct=med(dire, ix), direct_ci=boot(sel, lambda s: np.median(dire[s])),
                 ratio=med(err, ix) / med(dire, ix), in_range=float(inr[ix].mean()), in_range_ci=boot(sel, lambda s: inr[s].mean()),
                 spread_big=float(big[ix].mean()), pre_settled=float(pre[ix].mean()))  # fmt: skip
        print(f"{indent}{label:30s} n {n:4d} eps {r['n_eps']:3d} | model {f2(r['model'], r['model_ci'])} cm  x val {f2(r['x_val'], r['x_val_ci'])} | "
              f"direct {f2(r['direct'], r['direct_ci'])} cm  model/direct {r['ratio']:.2f} | in range {100 * r['in_range']:5.1f} % "
              f"[{100 * r['in_range_ci'][0]:.1f}, {100 * r['in_range_ci'][1]:.1f}] | spread > val p90 {100 * r['spread_big']:5.1f} % | "
              f"settled start {100 * r['pre_settled']:5.1f} %")  # fmt: skip
        res["rows"].append(r)
        return r

    for k, name in ((1, "reach 1 (pre-grasp, 第一下)"), (2, "reach 2 (entry, 第二下)"), (0, "via-point reach (info)")):
        m = rn == k
        if not m.any():
            continue
        print(f"== {name}: {int(m.sum())} recorded ({int(m.sum() - (m & fell).sum())} not fell / lost, {int((m & kept).sum())} kept); "
              f"settled start (all recorded) {100 * pre[m].mean():.1f} % ({int(pre[m].sum())}/{int(m.sum())}); in range (all recorded) "
              f"{100 * inr[m].mean():.1f} % ({int(inr[m].sum())}/{int(m.sum())}); label settled (not fell) "
              f"{100 * setl[m & ~fell].mean():.1f} %")  # fmt: skip
        mk = m & kept
        row("all kept", mk)
        row("kept, settled start", mk & pre)
        row("kept, start not settled", mk & ~pre)
        ri = row("kept, in range", mk & inr)
        ro = row("kept, out of range", mk & ~inr)
        if ri and ro:
            ci = boot(mk, lambda s: np.median(err[s][~inr[s]]) / np.median(err[s][inr[s]]) if inr[s].any() and (~inr[s]).any() else np.nan)
            print(f"    out / in range (model median): {ro['model'] / ri['model']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
            res["rows"].append(dict(label=f"reach {k} out/in", value=ro["model"] / ri["model"], ci=ci))
        row("kept, spread > val p90", mk & big)
        row("kept, spread <= val p90", mk & ~big)
        row("all labels incl. unsettled (not fell)", m & ~fell)
        if "furn_rows" in D:  # post hoc (not in the registration): reaches that touched nothing (the model's empty-scene data)
            free = (D["furn_rows"] == 0) & (D["obj_rows"] == 0)
            print(f"    contact during the reach (kept): furniture {int((mk & (D['furn_rows'] > 0)).sum())}, target object "
                  f"{int((mk & (D['obj_rows'] > 0)).sum())}, neither {int((mk & free).sum())}  [post hoc]")  # fmt: skip
            row("kept, no contact [post hoc]", mk & free)
            row("kept, no contact, in range [post hoc]", mk & free & inr)
            row("kept, furniture contact [post hoc]", mk & (D["furn_rows"] > 0))
        cnt = out_j[mk].sum(0)
        top = np.argsort(-cnt)[:6]
        print(f"    joints out of the band (kept segments, count): " + ", ".join(f"{JOINTS[j]} {int(cnt[j])}" for j in top if cnt[j] > 0)
              + f"; joints out per segment median {np.median(out_j[mk].sum(1)) if mk.any() else float('nan'):.0f}, max {int(out_j[mk].sum(1).max()) if mk.any() else 0}")  # fmt: skip
        print("    per task:")
        for t in range(len(TASKS)):
            if (mk & (tid == t)).any():
                row(f"{SHORT[t]}", mk & (tid == t), indent="      ")
    # sensitivity: the record's chain unclipped
    Du = load_records(rec_p, None)
    pu = predict(model, Du, allidx)
    c2 = D["chain_rec"] >= 2
    if c2.any():
        mm = c2 & kept
        print(f"sensitivity, chain as recorded (unclipped) on the {int(c2.sum())} segments with chain 2 ({int(mm.sum())} kept): model median "
              f"{np.median(pu['err'][mm]) if mm.any() else float('nan'):.2f} cm vs clipped {np.median(err[mm]) if mm.any() else float('nan'):.2f} cm")  # fmt: skip
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
