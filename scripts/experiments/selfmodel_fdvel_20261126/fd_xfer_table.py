"""selfmodel_fdvel_20261126's xfer summary, read-only: T3 and F side by side on every xfer record, from the frozen
sx_eval.py's json outputs (fd_run.sh: <out>/xfer/<record>/<T3|F_s0|F_s1|F_s2>_<settled|ended>.json), plus the inputs
outside the training distribution.

    cd $A/repo && SX_REPO=$A/repo REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $VPY .../fd_xfer_table.py \
        --out <out> --data3 <data3> --xfer <xfer> --records scalebfm_simple ... [--json F]

Per record and keep rule (settled: sx_eval's primary, d3_eval's segments; ended: the supplement), per model: prediction
error / LocoManiLab val (the model's own val) for all / first reach / chained with sx_eval's 95 % interval (xfer resets
resampled), model / direct on the xfer segments, ensemble disagreement / val; F's mean over seeds; F / T3.
Inputs outside the training distribution: |(input - mu) / sd| > 6 in the network's own normalisation (the model's
EndNet nx / nu buffers: training segments' mean and std, std floored at 1e-3; the five members share them), share of
the kept segments with at least one such input, over all inputs (x0 and u) and by block: velocity (x0's 96 velocity
columns: joint 87, pelvis angular 9), the other 182 x0 columns, u (12); T3 sees x0 as recorded, F with data.FDVEL on.
The same on LocoManiLab val for reference, and F's most frequent out-of-distribution columns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.cwd()
assert os.environ.get("REACHWM_FDVEL", "0") == "0", "run with REACHWM_FDVEL unset: T3 and F features are loaded here"
assert os.environ.get("SX_REPO") == str(REPO), "SX_REPO must be this repo (sx_eval.kept)"
for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 is train3's switch"
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029"),
                str(REPO / "scripts/experiments/selfmodel_xfer_20261125"), str(REPO / "scripts/experiments/locomanilab_20261117")]  # fmt: skip
import data as DT  # noqa: E402
import sx_eval as SE  # noqa: E402
from lml_selfmodel import SCALEBFM_ORDER  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

TAGS = ("T3", "F_s0", "F_s1", "F_s2")
SPL = ("all", "first reach", "chained")
SD_OUT = 6.0


def x0_names() -> list[str]:
    s_names = [f"q {j}" for j in SCALEBFM_ORDER] + [f"dq {j}" for j in SCALEBFM_ORDER] + ["grav x", "grav y", "grav z"] + \
        ["angvel x", "angvel y", "angvel z", "height"]  # fmt: skip
    names = [f"{n} @k0{'' if s == 0 else f'-{s}'}" for s in range(3) for n in s_names]
    names += ["jaw0 x", "jaw0 y", "jaw0 z"] + [f"rot0 {i}" for i in range(6)] + ["walked", "chain"]
    if DT.X0_REF:
        names += [f"ref jaw {c}" for c in "xyz"] + [f"ref - jaw {c}" for c in "xyz"]
    if DT.GOAL_REF:
        names += [f"plan q {j}" for j in MUJOCO_JOINT_NAMES] + [f"plan q - q0 {j}" for j in MUJOCO_JOINT_NAMES]
        names += [f"plan pelvis {c}" for c in "xyz"] + [f"plan rot c0 {c}" for c in "xyz"] + [f"plan rot c1 {c}" for c in "xyz"]
        names += ["plan duration"]
    if DT.APPROACH:
        names += ["walk fwd", "walk left", "walk sin", "walk cos-1"]
    assert len(names) == DT.X0_DIM, (len(names), DT.X0_DIM)
    return names


def norms(model_pt: Path) -> dict:
    st = torch.load(model_pt, map_location="cpu")["state"]
    out = {}
    for nm in ("nx", "nu"):
        mu, sd = st[f"members.0.{nm}.mu"].numpy(), st[f"members.0.{nm}.sd"].numpy()
        for i in range(1, 5):
            assert np.array_equal(st[f"members.{i}.{nm}.mu"].numpy(), mu) and np.array_equal(st[f"members.{i}.{nm}.sd"].numpy(), sd)
        out[nm] = (mu.astype(np.float64), sd.astype(np.float64))
    return out


def ood(D: dict, idx: np.ndarray, nz: dict) -> dict:
    zx = np.abs((np.asarray(D["x0"])[idx].astype(np.float64) - nz["nx"][0]) / nz["nx"][1]) > SD_OUT
    zu = np.abs((np.asarray(D["u"])[idx].astype(np.float64) - nz["nu"][0]) / nz["nu"][1]) > SD_OUT
    vel = np.r_[DT.X0_DQ_COLS, DT.X0_ANGVEL_COLS]
    oth = np.setdiff1d(np.arange(DT.X0_DIM), vel)
    sh = lambda m: float(m.any(1).mean()) if len(m) else float("nan")  # noqa: E731
    return dict(any=sh(np.c_[zx, zu]), velocity=sh(zx[:, vel]), joint_vel=sh(zx[:, DT.X0_DQ_COLS]),
                angvel=sh(zx[:, DT.X0_ANGVEL_COLS]), other_x0=sh(zx[:, oth]), u=sh(zu), col_share=zx.mean(0) if len(zx) else None)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data3", type=Path, required=True)
    ap.add_argument("--xfer", type=Path, required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    names = x0_names()
    mdirs = lambda be: {"T3": a.data3 / be / "models_s0", **{f"F_s{k}": a.out / be / f"models_s{k}" for k in range(3)}}  # noqa: E731
    cache: dict = {}

    def lml(be):
        if be not in cache:
            f = str(a.data3 / be / "data" / "all.npz")
            DT.FDVEL = False
            D = DT.load(f)
            DT.FDVEL = True
            DF = DT.load(f)
            DT.FDVEL = False
            vi = SE.kept(D, DT.splits(D)["val"])
            nz = {t: norms(p / "E3_ensemble_f1" / "model.pt") for t, p in mdirs(be).items()}
            cache[be] = dict(D=D, DF=DF, vi=vi, nz=nz, val_ood={t: ood(DF if t != "T3" else D, vi, nz[t]) for t in TAGS})
        return cache[be]

    res, summary = {}, []
    for rec in a.records:
        be = rec.split("_")[0]
        L = lml(be)
        f = str(a.xfer / rec / "data" / "all.npz")
        DT.FDVEL = False
        DX = DT.load(f)
        DT.FDVEL = True
        DXF = DT.load(f)
        DT.FDVEL = False
        res[rec] = {}
        print(f"\n{rec} (controller {be}; LocoManiLab {a.data3 / be}):")
        for keep in ("settled", "ended"):
            xi = SE.kept(DX, ended=keep == "ended")
            J = {t: json.loads((a.out / "xfer" / rec / f"{t}_{keep}.json").read_text()) for t in TAGS}
            assert all(j["n_xfer_kept"] == len(xi) for j in J.values()), "segment counts differ from sx_eval's"
            O = {t: ood(DXF if t != "T3" else DX, xi, L["nz"][t]) for t in TAGS}
            res[rec][keep] = dict(sx_eval=J, ood={t: {k: v for k, v in o.items() if k != "col_share"} for t, o in O.items()})
            lab = "primary (settled, d3_eval's segments)" if keep == "settled" else "supplement (ended: not fallen, settled or not)"
            print(f"  {lab}: {len(xi)} segments; prediction error / own LML val [95 %, xfer resets]; model / direct; "
                  f"disagreement / val; inputs beyond {SD_OUT:g} sd (share of segments: any | velocity (joint, angular) | other x0 | u)")  # fmt: skip
            for t in TAGS:
                s = J[t]["splits"]
                parts = []
                for sp in SPL:
                    o = s.get(sp, {})
                    parts.append(f"{sp} {o['ratio']:.3f}x [{o['ratio_ci'][0]:.3f}, {o['ratio_ci'][1]:.3f}] (n {o['n']})" if o.get("n")
                                 else f"{sp} -")  # fmt: skip
                md = " / ".join(f"{s[sp]['model_over_direct_xfer']:.3f}" if s.get(sp, {}).get("n") else "-" for sp in SPL)
                o = O[t]
                print(f"    {t:5s}: {'; '.join(parts)}; model/direct {md}; disagreement {J[t]['disagreement']['ratio']:.2f}x; "
                      f"OOD {100 * o['any']:.1f} % | {100 * o['velocity']:.1f} % ({100 * o['joint_vel']:.1f}, {100 * o['angvel']:.1f}) | "
                      f"{100 * o['other_x0']:.1f} % | {100 * o['u']:.1f} %")  # fmt: skip
            fr = {sp: [J[t]["splits"][sp]["ratio"] for t in TAGS[1:] if J[t]["splits"].get(sp, {}).get("n")] for sp in SPL}
            t3r = {sp: J["T3"]["splits"][sp]["ratio"] for sp in SPL if J["T3"]["splits"].get(sp, {}).get("n")}
            line = []
            for sp in SPL:
                if fr[sp] and sp in t3r:
                    m = float(np.mean(fr[sp]))
                    line.append(f"{sp} F mean {m:.3f}x (seeds {', '.join(f'{v:.3f}' for v in fr[sp])}; spread "
                                f"{100 * (max(fr[sp]) / min(fr[sp]) - 1):.1f} %), F / T3 {m / t3r[sp]:.3f}")  # fmt: skip
            print("    F: " + "; ".join(line))
            mdF = np.mean([J[t]["splits"]["all"]["model_over_direct_xfer"] for t in TAGS[1:]])
            dgF = np.mean([J[t]["disagreement"]["ratio"] for t in TAGS[1:]])
            oF = np.mean([O[t]["any"] for t in TAGS[1:]])
            cs = np.mean([O[t]["col_share"] for t in TAGS[1:]], 0)
            top = np.argsort(-cs)[:6]
            print(f"    F (mean of seeds): model/direct all {mdF:.3f}; disagreement {dgF:.2f}x; OOD any {100 * oF:.1f} %; "
                  f"most often beyond {SD_OUT:g} sd: {[(names[j], round(100 * float(cs[j]), 1)) for j in top if cs[j] > 0]}")  # fmt: skip
            if keep == "settled":
                summary.append((rec, J, O, fr, t3r))
        vo = L["val_ood"]
        print(f"  LocoManiLab val ({len(L['vi'])} kept): OOD any " + ", ".join(f"{t} {100 * vo[t]['any']:.2f} %" for t in TAGS)
              + " | velocity " + ", ".join(f"{t} {100 * vo[t]['velocity']:.2f} %" for t in TAGS))  # fmt: skip
        res[rec]["val_ood"] = {t: {k: v for k, v in o.items() if k != "col_share"} for t, o in vo.items()}
    print("\nsummary, primary (settled), all segments: prediction error / own LML val")
    print(f"  {'record':24s} | {'T3':>7s} | {'F s0':>7s} {'F s1':>7s} {'F s2':>7s} {'F mean':>7s} | F/T3  | model/direct T3, F | "
          f"disagreement T3, F | OOD any T3, F")  # fmt: skip
    for rec, J, O, fr, t3r in summary:
        f3 = [J[t]["splits"]["all"]["ratio"] for t in TAGS[1:]]
        mF = float(np.mean(f3))
        print(f"  {rec:24s} | {t3r['all']:7.3f} | {' '.join(f'{v:7.3f}' for v in f3)} {mF:7.3f} | {mF / t3r['all']:.3f} | "
              f"{J['T3']['splits']['all']['model_over_direct_xfer']:.3f}, "
              f"{np.mean([J[t]['splits']['all']['model_over_direct_xfer'] for t in TAGS[1:]]):.3f} | "
              f"{J['T3']['disagreement']['ratio']:.2f}, {np.mean([J[t]['disagreement']['ratio'] for t in TAGS[1:]]):.2f} | "
              f"{100 * O['T3']['any']:.1f} %, {100 * np.mean([O[t]['any'] for t in TAGS[1:]]):.1f} %")  # fmt: skip
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
