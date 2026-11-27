"""selfmodel_train3b_20261125's metrics for one controller (from selfmodel_train3_20261125's d3_eval.py, with revision
0's additions): models T3 (train3's frac 1.0, seed 0), M (data3 + data3b, seeds 0-2), M_f0.3 / M_f0.6 (seed 0),
D (M's data + mid starts, seeds 0-2) and MX (M's data with REACHWM_X0_REF=1, seeds 0-2) on one split of data3, of
data3b and, when given, on recorded real episodes; read-only (data and models).

MX's and MXa's start features are longer (X0_REF), so their predictions come from runs of this script with
REACHWM_X0_REF=1 and --dump <npz> --arm MX|MXa (that arm's models only, per-segment errors on the same segments; MX on
its own data files, t3b_mxdata.py, given as --data3 / --data3b; MXa on the stored data); the main run (X0_REF off)
reads them with --mx <npz>,<npz>, checks the segments are the same ones in the same order, and treats them like every
other model (paired resamples). MXa (seed 0, the stored C[k0]) is the descriptive arm: MXa/M0, no criterion.

    cd $REPO && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_X0_REF=1 CUDA_VISIBLE_DEVICES=1 \
        $VPY t3b_eval.py --backend heft --split test_iid --dump <mx.npz> [--real <spec>]
    cd $REPO && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 CUDA_VISIBLE_DEVICES=1 \
        $VPY t3b_eval.py --backend heft --split test_iid --mx <mx.npz> [--real <spec>] [--json <file>]

Segments: data.py's split (val, test_iid or edge; crc32 of the reset id) of data3's and of data3b's data/all.npz (one-hand
segments; data3b's detours and two-hand reaches are elsewhere), kept when they did not fall or get lost and the end label
is settled (settled_end, the collector's settle rule). Real episodes (--real): every segment, same filter, no split.
Per cell, over its segments:
  position (cm)  a model: the predicted final hand point to the real one; direct: the hand left at the command
                 (data.py err_cm, B0_command). Median and p90.
  attitude (deg) the geodesic angle between the predicted final wrist attitude (the model's rot6, Gram-Schmidt) and
                 the real one (data.py rot_f); direct: the commanded rot6. Median and p90.
  ratios of medians (position; attitude in the json): M/T3 = mean over M's seeds of median(M_k) / median(T3);
                 D/M, MX/M = mean over seeds of median(D_k) / median(M_k) (seed k with seed k), with each seed's ratio;
                 learning curve 1.0/0.3 = median(M seed 0) / median(M_f0.3), and 0.6/0.3.
  95 % intervals from 1000 resamples of the cell's resets (all of a reset's segments together), every statistic of a
  cell on the same resamples (so ratios are paired).
Every section is reported three times: all reaches, first reaches (chain 0: from the standing hold, or settled after
the walk) and chained reaches (chain >= 1: from where the previous reach left the robot; data3's second reaches,
data3b's every reach after the first). In place / walk first is compared on first reaches only.
Cells: data3 as train3 (hand, target height, |yaw|, |roll|, pitch, standing / crouched, reach duration); data3b by kind:
entry, pre-grasp, relative reach by the start distance d_start_cm < 15 / 15-30 / >= 30, and the entry cell 8-17 cm from
the grasp with a wrist turn att_change_deg < 20; real: the same three groups (first reach = chain 0, second = chain 1).
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
KIND = {"pre": 1, "entry": 2, "rel": 3}
RATIOS = ("M/T3", "D/M", "MX/M", "MXa/M0", "curve_1/0.3", "curve_0.6/0.3")


def model_list(root: Path, t3: Path, arm: str | None) -> list[tuple[str, Path, float]]:
    if arm == "MX":
        ms = [(f"MX{k}", root / f"models_MX_s{k}", 1.0) for k in SEEDS]
    elif arm == "MXa":  # the descriptive arm: the stored C[k0], seed 0 only
        ms = [("MXa", root / "models_MXa_s0", 1.0)]
    else:
        ms = [("T3", t3, 1.0)]
        ms += [(f"M{k}", root / f"models_M_s{k}", 1.0) for k in SEEDS]
        ms += [("M_f0.3", root / "models_M_s0", 0.3), ("M_f0.6", root / "models_M_s0", 0.6)]
        ms += [(f"D{k}", root / f"models_D_s{k}", 1.0) for k in SEEDS]
    return [(n, r, f) for n, r, f in ms if (TR.model_dir("E3_ensemble", f, r) / "model.pt").exists()]


def predict(models, D: dict, idx: np.ndarray, dev) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    GD = M.tensors(D, dev)
    ti = torch.tensor(idx, device=dev)
    rot_f = M.gs6(GD["rot_f"][ti])
    pos_d = np.asarray(D["err_cm"])[idx].astype(float)
    att_d = M.rot_angle_deg(M.gs6(GD["rot6"][ti]), rot_f).cpu().numpy()
    pos, att = {}, {}
    for name, root, frac in models:
        m = TR.load_model("E3_ensemble", frac, D, dev, root=root)
        p = M.predict_batched(m, GD, ti)
        pos[name] = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()
        att[name] = M.rot_angle_deg(M.gs6(p["rot6"]), rot_f).cpu().numpy()
        del m
    return pos, att, pos_d, att_d


def stats(pos: dict, att: dict, pos_d, att_d, s: np.ndarray) -> dict:
    """Every statistic of a cell on the segment indices s."""
    o = {}
    for k, v, a in [("direct", pos_d, att_d)] + [(k, pos[k], att[k]) for k in pos]:
        o[k], o["p90_" + k] = float(np.median(v[s])), float(np.percentile(v[s], 90))
        o["att_" + k], o["att_p90_" + k] = float(np.median(a[s])), float(np.percentile(a[s], 90))
    for pre in ("", "att_"):
        ms = [k for k in (f"M{j}" for j in SEEDS) if k in pos]
        if "T3" in pos and ms:
            for k in ms:
                o[f"{pre}{k}/T3"] = o[pre + k] / o[pre + "T3"]
            o[f"{pre}M/T3"] = float(np.mean([o[f"{pre}{k}/T3"] for k in ms]))
        for arm in ("D", "MX"):
            pairs = [(f"{arm}{j}", f"M{j}") for j in SEEDS if f"{arm}{j}" in pos and f"M{j}" in pos]
            for d, m in pairs:
                o[f"{pre}{d}/{m}"] = o[pre + d] / o[pre + m]
            if pairs:
                o[f"{pre}{arm}/M"] = float(np.mean([o[f"{pre}{d}/{m}"] for d, m in pairs]))
        if "MXa" in pos and "M0" in pos:
            o[f"{pre}MXa/M0"] = o[pre + "MXa"] / o[pre + "M0"]
        if "M0" in pos and "M_f0.3" in pos:
            o[f"{pre}curve_1/0.3"] = o[pre + "M0"] / o[pre + "M_f0.3"]
        if "M_f0.6" in pos and "M_f0.3" in pos:
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
    f = lambda k, d=2: f"{p[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
    allm = ["direct"] + names
    print(f"  {c['group']} | {c['dim']}: {c['cell']}  n {c['n']} ({c['n_resets']} resets)")
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


def section(title, D, idx, cells, models, dev, rng, note, mx=None, dump=None) -> dict:
    """cells: (dim, label, mask over idx, groups) with groups a subset of ("all", "first", "chained")."""
    pos, att, pos_d, att_d = predict(models, D, idx, dev)
    rs = np.asarray(D["reset"])[idx]
    ident = np.c_[rs, np.asarray(D["chain"])[idx], np.asarray(D["k0"])[idx], np.asarray(D["n_steps"])[idx]].astype(np.int64)
    if dump is not None:
        dump[f"{title}|ident"] = ident
        for k in pos:
            dump[f"{title}|pos|{k}"], dump[f"{title}|att|{k}"] = pos[k], att[k]
        print(f"{title}: {len(idx)} segments, dumped {list(pos)}")
        return {}
    for mxd in mx or []:
        if f"{title}|ident" not in mxd:
            continue
        assert np.array_equal(mxd[f"{title}|ident"], ident), f"{title}: a dump's segments differ"
        for k in sorted({key.split("|")[2] for key in mxd if key.startswith(f"{title}|pos|")}):
            pos[k], att[k] = mxd[f"{title}|pos|{k}"], mxd[f"{title}|att|{k}"]
    out = dict(title=title, n=int(len(idx)), n_resets=int(len(np.unique(rs))), note=note, models=list(pos), cells=[])
    print(f"\n== {title}: {len(idx)} segments from {out['n_resets']} resets ({note})")
    names = list(pos)
    ch = np.asarray(D["chain"])[idx]
    grp = {"all": np.ones(len(idx), bool), "first": ch == 0, "chained": ch >= 1}
    for g in ("all", "first", "chained"):
        for dim, lab, sel, gs in cells:
            if g not in gs:
                continue
            s = sel & grp[g]
            if s.sum() == 0:
                print(f"  {g} | {dim}: {lab}: no segments")
                continue
            c = cell(pos, att, pos_d, att_d, rs, s, rng)
            c.update(group=g, dim=dim, cell=lab)
            out["cells"].append(c)
            print_cell(c, names)
    return out


def kept(D: dict, sp: np.ndarray) -> np.ndarray:
    se = np.asarray(D["settled_end"]) > 0 if "settled_end" in D else np.ones(len(D["n_steps"]), bool)
    return sp[(np.asarray(D["fell"])[sp] == 0) & se[sp]]


A3 = ("all", "first", "chained")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "heft", "sonic"])
    ap.add_argument("--split", default="test_iid", choices=["val", "test_iid", "edge"])
    ap.add_argument("--root", type=Path, default=None, help="train3b's output for this controller")
    ap.add_argument("--t3", type=Path, default=None, help="train3's models (frac 1.0, seed 0)")
    ap.add_argument("--data3", type=Path, default=None)
    ap.add_argument("--data3b", type=Path, default=None)
    ap.add_argument("--real", default=None, help="data.load spec of recorded real episodes (every segment, no split)")
    ap.add_argument("--only-real", action="store_true")
    ap.add_argument("--dump", type=Path, default=None, help="REACHWM_X0_REF=1: write --arm's per-segment errors here")
    ap.add_argument("--arm", default="MX", choices=["MX", "MXa"], help="--dump: which X0_REF arm (MX: run on MX's data files)")
    ap.add_argument("--mx", default=None, help="per-segment errors from --dump runs, comma-separated")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    be = a.backend
    assert DT.X0_REF == (a.dump is not None), "--dump runs with REACHWM_X0_REF=1, the main run without"
    root = a.root or ROOT / "selfmodel_train3b_20261125" / be
    t3 = a.t3 or ROOT / "selfmodel_data3_20261125" / be / "models_s0"
    d3 = a.data3 or ROOT / "selfmodel_data3_20261125" / be / "data" / "all.npz"
    d3b = a.data3b or ROOT / "selfmodel_data3b_20261125" / be / "data" / "all.npz"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    models = model_list(root, t3, a.arm if a.dump is not None else None)
    mx = [dict(np.load(p)) for p in a.mx.split(",")] if a.mx else None
    dump = {} if a.dump else None
    print(f"{be} {a.split}: models {[n for n, _, _ in models]}" + (f" + MX from {a.mx}" if mx else ""))
    out = dict(controller=be, split=a.split, models=[(n, str(r), f) for n, r, f in models], mx=str(a.mx), sections=[])
    kw = dict(models=models, dev=dev, rng=rng, mx=mx, dump=dump)
    if not a.only_real:
        # data3, cells as train3; in place / walk first on first reaches only
        D = DT.load(str(d3))
        idx = kept(D, DT.splits(D)[a.split])
        g = lambda k: np.asarray(D[k])[idx]  # noqa: E731
        t, pz = g("target"), g("pelvis_z")
        dyaw, pitch, roll = np.abs(np.degrees(g("dyaw"))), np.degrees(g("pitch")), np.abs(np.degrees(g("roll")))
        cells = [
            ("all", "all", np.ones(len(idx), bool), A3),
            ("start", "in place", g("walked") == 0, ("first",)), ("start", "walk first", g("walked") == 1, ("first",)),
            ("hand", "right", g("side") > 0, A3), ("hand", "left", g("side") < 0, A3),
            ("height", "z < 0.3", t[:, 2] < 0.3, A3), ("height", "0.3-0.6", (t[:, 2] >= 0.3) & (t[:, 2] < 0.6), A3),
            ("height", "0.6-0.9", (t[:, 2] >= 0.6) & (t[:, 2] < 0.9), A3),
            ("height", "0.9-1.2", (t[:, 2] >= 0.9) & (t[:, 2] < 1.2), A3), ("height", "z >= 1.2", t[:, 2] >= 1.2, A3),
            ("yaw", "|yaw| <= 60", dyaw <= 60, A3), ("yaw", "|yaw| > 60", dyaw > 60, A3),
            ("roll", "|roll| <= 40", roll <= 40, A3), ("roll", "|roll| > 40", roll > 40, A3),
            ("pitch", "pitch < 0", pitch < 0, A3), ("pitch", "pitch >= 0", pitch >= 0, A3),
            ("pelvis", "standing", pz < 0, A3), ("pelvis", "crouched", pz >= 0, A3),
            ("duration", "reach <= 2.5 s", g("reach_s") <= 2.5, A3), ("duration", "reach > 2.5 s", g("reach_s") > 2.5, A3),
        ]  # fmt: skip
        out["sections"].append(section(f"data3 {a.split}", D, idx, cells, note=f"{d3}; kept: not fell / lost, settled end", **kw))
        del D
        # data3b by kind
        D = DT.load(str(d3b))
        idx = kept(D, DT.splits(D)[a.split])
        g = lambda k: np.asarray(D[k])[idx]  # noqa: E731
        kind, d0, att = g("kind"), g("d_start_cm"), g("att_change_deg")
        rel = kind == KIND["rel"]
        cells = [
            ("all", "all", np.ones(len(idx), bool), A3),
            ("start", "in place", g("walked") == 0, ("first",)), ("start", "walk first", g("walked") == 1, ("first",)),
            ("kind", "entry", kind == KIND["entry"], A3), ("kind", "pre-grasp", kind == KIND["pre"], A3),
            ("kind", "rel < 15 cm", rel & (d0 < 15), A3), ("kind", "rel 15-30 cm", rel & (d0 >= 15) & (d0 < 30), A3),
            ("kind", "rel >= 30 cm", rel & (d0 >= 30), A3),
            ("entry", "8-17 cm, wrist < 20 deg", (kind == KIND["entry"]) & (d0 >= 8) & (d0 <= 17) & (att < 20), A3),
        ]  # fmt: skip
        out["sections"].append(section(f"data3b {a.split}", D, idx, cells, note=f"{d3b}; kept: not fell / lost, settled end", **kw))
        del D
    if a.real:
        D = DT.load(a.real)
        idx = kept(D, np.arange(len(D["n_steps"])))
        ch = np.asarray(D["chain"])[idx]
        cells = [("all", "all", np.ones(len(idx), bool), A3), ("reach", "second (chain 1)", ch == 1, ("chained",))]
        out["sections"].append(section("real episodes", D, idx, cells, note=f"{a.real}; every segment; kept: not fell / lost, settled end", **kw))
    if dump is not None:
        tmp = a.dump.with_suffix(".tmp.npz")
        np.savez(tmp, **dump)
        os.replace(tmp, a.dump)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
