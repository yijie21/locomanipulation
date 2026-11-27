"""selfmodel_train3b_20261125, lmlreal revision 1 (coordinator, after 3cfb0ee6): the train3b models on
selfmodel_lmlreal_20261125's recorded LocoManiLab episodes, one controller; read-only (data and models).

t3b_eval.py's metrics, filter and resampling (its functions copied here; the frozen final-table script is not changed):
position (cm) and attitude (deg), median and p90, direct = the hand left at the command; ratios of medians M/T3, D/M,
MX/M (seed k with seed k, with each seed's ratio), MXa/M0, learning curve 1.0/0.3 and 0.6/0.3; 95 % intervals from 1000
resamples of the cell's episodes (reset ids), every statistic of a cell on the same resamples. Segments: every one of
lmlreal's data/all.npz that did not fall or get lost and whose end label is settled.
Changes for lmlreal:
  cells by reach_no (lr_run: 1 = pre-grasp = 第一下, 2 = entry = 第二下; via-point reaches (0) only in "all"), each in
  the groups all / first (chain 0) / chained (chain >= 1); plus chain = 2 (the entry after a via point) on its own.
  T3 (train3: data3, chain 0-1) reads chain clipped to 1 (x0 column 205); M, D, MX, MXa read the recorded chain (data3b:
  0-3).
  MX reads its rebuilt C[k0] on chained reaches (t3b_lmlmx.py); first reaches (chain 0) cannot be rebuilt -- lmlreal
  records neither d_this nor d_prev -- so MX's dump is NaN there and MX is left empty in every cell that holds a chain-0
  reach (the note says why). MXa (seed 0, stored C[k0], every reach) is the descriptive arm.
MX's and MXa's start features are longer (X0_REF): their errors come from runs with REACHWM_X0_REF=1 --dump <npz>
--arm MX|MXa (MX on mx_lmlreal.npz as --real, MXa on the stored data), read by the main run with --mx <npz>,<npz>
(same segments in the same order, checked).

    cd $REPO && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_X0_REF=1 $VPY t3b_lmleval.py --backend heft \
        --real <mx_lmlreal.npz> --dump <mx.npz> --arm MX
    cd $REPO && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $VPY t3b_lmleval.py --backend heft \
        --mx <mx.npz>,<mxa.npz> [--json <file>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M
NB = 1000
ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
SEEDS = (0, 1, 2)
RATIOS = ("M/T3", "D/M", "MX/M", "MXa/M0", "curve_1/0.3", "curve_0.6/0.3")
CHAIN_COL = 205  # data.py x0: S[k0], S[k0-1], S[k0-2], jaw0, rot0, walked, chain, ...
T3_CHAIN_MAX = 1  # data3's chains
MX_EMPTY = "MX 留空：chain 0 的 C[k0] 无法重建（lmlreal 记录里没有每一下的 d_this 和 d_prev）"
A3 = ("all", "first", "chained")


def model_list(root: Path, t3: Path, arm: str | None) -> list[tuple[str, Path, float]]:
    if arm == "MX":
        ms = [(f"MX{k}", root / f"models_MX_s{k}", 1.0) for k in SEEDS]
    elif arm == "MXa":
        ms = [("MXa", root / "models_MXa_s0", 1.0)]
    else:
        ms = [("T3", t3, 1.0)]
        ms += [(f"M{k}", root / f"models_M_s{k}", 1.0) for k in SEEDS]
        ms += [("M_f0.3", root / "models_M_s0", 0.3), ("M_f0.6", root / "models_M_s0", 0.6)]
        ms += [(f"D{k}", root / f"models_D_s{k}", 1.0) for k in SEEDS]
    return [(n, r, f) for n, r, f in ms if (TR.model_dir("E3_ensemble", f, r) / "model.pt").exists()]


def predict(models, D: dict, idx: np.ndarray, dev) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    assert np.array_equal(D["x0"][:, CHAIN_COL], np.asarray(D["chain"], np.float32)), "x0's chain column moved"
    GD = M.tensors(D, dev)
    ti = torch.tensor(idx, device=dev)
    rot_f = M.gs6(GD["rot_f"][ti])
    pos_d = np.asarray(D["err_cm"])[idx].astype(float)
    att_d = M.rot_angle_deg(M.gs6(GD["rot6"][ti]), rot_f).cpu().numpy()
    pos, att = {}, {}
    for name, root, frac in models:
        G = GD
        if name == "T3":  # train3's data has chains 0-1: the entry after a via point (2) is read as 1
            G = dict(GD)
            G["x0"] = GD["x0"].clone()
            G["x0"][:, CHAIN_COL] = G["x0"][:, CHAIN_COL].clamp(max=T3_CHAIN_MAX)
        m = TR.load_model("E3_ensemble", frac, D, dev, root=root)
        p = M.predict_batched(m, G, ti)
        pos[name] = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()
        att[name] = M.rot_angle_deg(M.gs6(p["rot6"]), rot_f).cpu().numpy()
        del m
    return pos, att, pos_d, att_d


def stats(pos: dict, att: dict, pos_d, att_d, s: np.ndarray) -> dict:
    """Every statistic of a cell on the segment indices s; a model with NaN on s (MX on chain 0) is left out."""
    o = {}
    have = [k for k in pos if not np.isnan(pos[k][s]).any()]
    for k, v, a in [("direct", pos_d, att_d)] + [(k, pos[k], att[k]) for k in have]:
        o[k], o["p90_" + k] = float(np.median(v[s])), float(np.percentile(v[s], 90))
        o["att_" + k], o["att_p90_" + k] = float(np.median(a[s])), float(np.percentile(a[s], 90))
    for pre in ("", "att_"):
        ms = [k for k in (f"M{j}" for j in SEEDS) if k in have]
        if "T3" in have and ms:
            for k in ms:
                o[f"{pre}{k}/T3"] = o[pre + k] / o[pre + "T3"]
            o[f"{pre}M/T3"] = float(np.mean([o[f"{pre}{k}/T3"] for k in ms]))
        for arm in ("D", "MX"):
            pairs = [(f"{arm}{j}", f"M{j}") for j in SEEDS if f"{arm}{j}" in have and f"M{j}" in have]
            for d, m in pairs:
                o[f"{pre}{d}/{m}"] = o[pre + d] / o[pre + m]
            if pairs:
                o[f"{pre}{arm}/M"] = float(np.mean([o[f"{pre}{d}/{m}"] for d, m in pairs]))
        if "MXa" in have and "M0" in have:
            o[f"{pre}MXa/M0"] = o[pre + "MXa"] / o[pre + "M0"]
        if "M0" in have and "M_f0.3" in have:
            o[f"{pre}curve_1/0.3"] = o[pre + "M0"] / o[pre + "M_f0.3"]
        if "M_f0.6" in have and "M_f0.3" in have:
            o[f"{pre}curve_0.6/0.3"] = o[pre + "M_f0.6"] / o[pre + "M_f0.3"]
    return o


def cell(pos, att, pos_d, att_d, rs: np.ndarray, sel: np.ndarray, rng) -> dict:
    idx = np.flatnonzero(sel)
    r = rs[idx]
    o_ = np.argsort(r, kind="stable")
    groups = np.split(idx[o_], np.flatnonzero(np.diff(r[o_])) + 1)
    point = stats(pos, att, pos_d, att_d, idx)
    boots = []
    for _ in range(NB):
        pick = rng.integers(0, len(groups), len(groups))
        boots.append(stats(pos, att, pos_d, att_d, np.concatenate([groups[j] for j in pick])))
    ci = {k: (float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))) for k in point}
    return dict(n=int(len(idx)), n_resets=len(groups), point=point, ci=ci)


def print_cell(c: dict, names: list[str]) -> None:
    p, ci = c["point"], c["ci"]
    f = lambda k, d=2: f"{p[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]" if k in p else "—"  # noqa: E731
    allm = ["direct"] + names
    print(f"  {c['group']} | {c['dim']}: {c['cell']}  n {c['n']} ({c['n_resets']} episodes)")
    if c.get("mx_empty"):
        print(f"    {MX_EMPTY}")
    print("    position cm, median: " + " | ".join(f"{k} " + f(k) for k in allm))
    print("    position cm, p90:    " + " | ".join(f"{k} " + f("p90_" + k) for k in allm))
    print("    attitude deg, median: " + " | ".join(f"{k} " + f("att_" + k) for k in allm))
    print("    attitude deg, p90:    " + " | ".join(f"{k} " + f("att_p90_" + k) for k in allm))
    rat = [k for k in RATIOS if k in p]
    if rat:
        print("    position ratios: " + " | ".join(f"{k} " + f(k, 3) for k in rat))
        seeds = [k for k in p if re.fullmatch(r"M\d/T3|D\d/M\d|MX\d/M\d", k)]
        if seeds:
            print("      per seed: " + ", ".join(f"{k} " + f(k, 3) for k in seeds))
            for arm in ("D", "MX"):
                v = [p[k] for k in seeds if re.fullmatch(arm + r"\d/M\d", k)]
                if v:
                    print(f"      {arm}/M seed range {min(v):.3f}..{max(v):.3f}; all seeds < 1.0: {all(x < 1.0 for x in v)}")
        print("    attitude ratios: " + " | ".join(f"{k} " + f("att_" + k, 3) for k in rat if "att_" + k in p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "heft", "sonic"])
    ap.add_argument("--root", type=Path, default=None, help="train3b's output for this controller")
    ap.add_argument("--t3", type=Path, default=None, help="train3's models (frac 1.0, seed 0)")
    ap.add_argument("--real", type=Path, default=None, help="lmlreal's data/all.npz (MX dump: mx_lmlreal.npz)")
    ap.add_argument("--dump", type=Path, default=None, help="REACHWM_X0_REF=1: write --arm's per-segment errors here")
    ap.add_argument("--arm", default="MX", choices=["MX", "MXa"])
    ap.add_argument("--mx", default=None, help="per-segment errors from --dump runs, comma-separated")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    be = a.backend
    assert DT.X0_REF == (a.dump is not None), "--dump runs with REACHWM_X0_REF=1, the main run without"
    root = a.root or ROOT / "selfmodel_train3b_20261125" / be
    t3 = a.t3 or ROOT / "selfmodel_data3_20261125" / be / "models_s0"
    real = a.real or ROOT / "selfmodel_lmlreal_20261125" / be / "data" / "all.npz"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    models = model_list(root, t3, a.arm if a.dump is not None else None)
    D = DT.load(str(real))
    se = np.asarray(D["settled_end"]) > 0
    idx = np.flatnonzero((np.asarray(D["fell"]) == 0) & se)
    pos, att, pos_d, att_d = predict(models, D, idx, dev)
    rs = np.asarray(D["reset"])[idx]
    ch, rn = np.asarray(D["chain"])[idx], np.asarray(D["reach_no"])[idx]
    ident = np.c_[rs, ch, np.asarray(D["k0"])[idx], np.asarray(D["n_steps"])[idx], rn].astype(np.int64)
    if a.dump is not None:
        if a.arm == "MX":  # chain 0: C[k0] not rebuilt (no d_this / d_prev recorded)
            for k in pos:
                pos[k][ch == 0], att[k][ch == 0] = np.nan, np.nan
        dump = {"ident": ident}
        for k in pos:
            dump[f"pos|{k}"], dump[f"att|{k}"] = pos[k], att[k]
        tmp = a.dump.with_suffix(".tmp.npz")
        np.savez(tmp, **dump)
        os.replace(tmp, a.dump)
        print(f"{be} lmlreal ({real}): {len(idx)} segments, dumped {list(pos)}" + (" (NaN on chain 0)" if a.arm == "MX" else ""))
        return
    for p in a.mx.split(",") if a.mx else []:
        mxd = dict(np.load(p))
        assert np.array_equal(mxd["ident"], ident), f"{p}: a dump's segments differ"
        for k in sorted({key.split("|")[1] for key in mxd if key.startswith("pos|")}):
            pos[k], att[k] = mxd[f"pos|{k}"], mxd[f"att|{k}"]
    names = list(pos)
    note = f"{real}; every segment; kept: not fell / lost, settled end; T3 reads chain clipped to {T3_CHAIN_MAX}"
    print(f"{be} lmlreal: models {names}\n== real episodes: {len(idx)} segments from {len(np.unique(rs))} episodes ({note})")
    print(f"   kept by (reach_no, chain): " + ", ".join(f"({r}, {c}) {int(((rn == r) & (ch == c)).sum())}" for r, c in sorted(set(zip(rn.tolist(), ch.tolist())))))
    cells = [
        ("all", "all (via points included)", np.ones(len(idx), bool), A3),
        ("reach", "第一下 (reach_no 1, pre-grasp)", rn == 1, A3),
        ("reach", "第二下 (reach_no 2, entry)", rn == 2, A3),
        ("chain", "chain = 2 (entry after a via point; T3 reads 1)", ch == 2, ("chained",)),
    ]
    grp = {"all": np.ones(len(idx), bool), "first": ch == 0, "chained": ch >= 1}
    out = dict(controller=be, real=str(real), mx=str(a.mx), n=int(len(idx)), note=note, models=names, cells=[])
    for g in A3:
        for dim, lab, sel, gs in cells:
            if g not in gs:
                continue
            s = sel & grp[g]
            if s.sum() == 0:
                print(f"  {g} | {dim}: {lab}: no segments")
                continue
            c = cell(pos, att, pos_d, att_d, rs, s, rng)
            c.update(group=g, dim=dim, cell=lab, mx_empty=bool(any(k.startswith("MX") and k != "MXa" for k in names) and (ch[s] == 0).any()))
            out["cells"].append(c)
            print_cell(c, names)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
