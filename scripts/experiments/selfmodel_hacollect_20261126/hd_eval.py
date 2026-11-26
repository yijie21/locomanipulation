"""selfmodel_hadata_20261126 revision 2b, evaluation 1 (section 3) and gate (b)'s two-hand row: the ten HA models of one
controller on HA's own held-out data (data.py's splits of the touch-filtered d3 / d3b), read-only on data and models.

    cd <repo_b>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo_b> \
        $PY hd_eval.py --root <collection root>/train/<controller> --split val|test_iid [--json F]

Self-contained: kept / predict / stats / cell / section are copied from train3b's frozen t3b_eval.py (asrun, 2f8d0c8b) and
simpledata's frozen sd_eval.py (asrun/repo, 500927fc), with the ensemble disagreement and section 3's ratios added;
neither file is imported. The reader and the models are repo_b's (LOCO_REPO): data.py
174071cb, train.py 8acdaea1, models.py 0ac27f6b, checked at start-up.

Inputs (hd_train.sh's layout, <root> = <collection root>/train/<controller>):
  data    <root>/data/d3.npz, d3b.npz (one-hand, touch-filtered by hd_touch.py), both_d3.npz, both_d3b.npz (two-hand,
          touch-filtered, evaluation only)
  models  <root>/models_<S|T>_s<seed>/E3_ensemble_f<frac>: S_f1_s0..2, T_f1_s0..2 (frac 1.0), S_f0.6_s0, S_f0.3_s0,
          T_f0.6_s0, T_f0.3_s0 (seed 0); all ten must be there
  LML     the denominators, fixed numbers (not resampled): train3b's D (seeds 0-2) on LocoManiLab data3 val, cell all / all
          / all, from selfmodel_train3b_20261125/<controller>/final_val.json (sections[0] "data3 val", points D0, D1, D2;
          their mean); fdvel's F (seed 0, the T arm's start) on the same cell, selfmodel_fdvel_20261126/<controller>/
          eval_lml_val.json (cell all / all, f[0]; the three fdvel seeds' mean beside it), printed beside
Segments: the split (val or test_iid) of each file, kept as t3b_eval.kept: not fallen / lost (data.py fell) and the end
label settled (settled_end). Groups all / first (chain 0) / chained (chain >= 1). Cells are sd_eval's: data3 all; on first
reaches in place / walk first (task walk) / walk first (d3 walk); data3b all and by kind (entry, pre-grasp, relative).
Per cell and model: position error (cm, the predicted final hand point to the real one; direct: data.py err_cm) median
and p90, attitude (deg) median, ensemble disagreement median (100 |pos_std|, the std over the 5 members' predicted
position, cm). Ratios (section 3), in every cell:
  S_f1, T_f1           the mean over seeds 0-2 of the seed's median
  S_f1/D_lml, T_f1/D_lml   that mean over the LML D denominator ("HA val is how many times LML's D val")
  1-T/S_sK, 1-T/S      seed K paired (S_f1_sK with T_f1_sK), and the mean of the three
  S 1-f1/f0.6, S 1-f1/f0.3, T ... the learning curve of each arm at seed 0
  T_f0.3/S_f1_s0       the catch-up criterion's ratio (<= 1.0 counts as caught up)
95 % intervals: 1000 resamples of the cell's resets (all of a reset's segments together), every statistic of a cell on
the same resamples (so the ratios are paired; the LML denominators stay fixed).
The registration's "HA val": data3 val, all reaches, the position median (the headline, printed last with test_iid's run
of this script beside it).
Gate (b)'s two-hand row: the two-hand segments of both_d3 and both_d3b (same split, kept), each hand's segment as read,
predicted by S_f1_s0 and T_f1_s0, beside the one-hand segments of d3 and d3b of the same split; medians and p90 of each
set (and per file), two-hand / one-hand ratios, 1000 resamples of the resets of the union (paired: a reset brings its
one-hand and two-hand segments together).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ["LOCO_REPO"])
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL"):
    assert os.environ.get(k) == "1", f"{k}=1 is the HA models' switch"
assert os.environ.get("REACHWM_X0_REF", "0") == "0", "REACHWM_X0_REF stays off"
M = TR.M
PINS = {"data.py": (DT, "174071cb"), "train.py": (TR, "8acdaea1"), "models.py": (M, "0ac27f6b")}
for fn, (mod, md) in PINS.items():
    p = Path(mod.__file__).resolve()
    assert p.name == fn and hashlib.md5(p.read_bytes()).hexdigest().startswith(md), f"{p}: not repo_b's {fn} ({md})"
NB = 1000
SEEDS = (0, 1, 2)
KIND = {"pre": 1, "entry": 2, "rel": 3}
A3 = ("all", "first", "chained")
LO = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
MODELS = [(f"{a}_f1_s{s}", a, s, 1.0) for a in ("S", "T") for s in SEEDS] + [
    ("S_f0.6_s0", "S", 0, 0.6), ("S_f0.3_s0", "S", 0, 0.3), ("T_f0.6_s0", "T", 0, 0.6), ("T_f0.3_s0", "T", 0, 0.3)]  # fmt: skip
TWO = ("S_f1_s0", "T_f1_s0")
REF = {}  # the fixed LML denominators, set in main()


def kept(D: dict, sp: np.ndarray) -> np.ndarray:  # t3b_eval.kept
    se = np.asarray(D["settled_end"]) > 0 if "settled_end" in D else np.ones(len(D["n_steps"]), bool)
    return sp[(np.asarray(D["fell"])[sp] == 0) & se[sp]]


def predict(models, D: dict, idx: np.ndarray, dev) -> tuple[dict, dict, dict]:
    """t3b_eval.predict (direct as "direct", sd_eval's), plus the ensemble disagreement (100 |pos_std|, cm)."""
    GD = M.tensors(D, dev)
    ti = torch.tensor(idx, device=dev)
    rot_f = M.gs6(GD["rot_f"][ti])
    pos = {"direct": np.asarray(D["err_cm"])[idx].astype(float)}
    att = {"direct": M.rot_angle_deg(M.gs6(GD["rot6"][ti]), rot_f).cpu().numpy()}
    sd = {}
    for name, root, frac in models:
        m = TR.load_model("E3_ensemble", frac, D, dev, root=root)
        p = M.predict_batched(m, GD, ti)
        pos[name] = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()
        att[name] = M.rot_angle_deg(M.gs6(p["rot6"]), rot_f).cpu().numpy()
        sd[name] = (100 * p["pos_std"].norm(dim=-1)).cpu().numpy()
        del m
    return pos, att, sd


def ratios(o: dict) -> dict:
    """Section 3's numbers from one cell's medians (o: model name -> position median)."""
    r = {}
    S, T = [o[f"S_f1_s{j}"] for j in SEEDS], [o[f"T_f1_s{j}"] for j in SEEDS]
    r["S_f1"], r["T_f1"] = float(np.mean(S)), float(np.mean(T))
    r["S_f1/D_lml"], r["T_f1/D_lml"] = r["S_f1"] / REF["D"], r["T_f1"] / REF["D"]
    for j in SEEDS:
        r[f"1-T/S_s{j}"] = 1 - T[j] / S[j]
    r["1-T/S"] = float(np.mean([r[f"1-T/S_s{j}"] for j in SEEDS]))
    for a in ("S", "T"):
        r[f"{a} 1-f1/f0.6"] = 1 - o[f"{a}_f1_s0"] / o[f"{a}_f0.6_s0"]
        r[f"{a} 1-f1/f0.3"] = 1 - o[f"{a}_f1_s0"] / o[f"{a}_f0.3_s0"]
    r["T_f0.3/S_f1_s0"] = o["T_f0.3_s0"] / o["S_f1_s0"]
    return r


RATIOS = ("S_f1", "T_f1", "S_f1/D_lml", "T_f1/D_lml", "1-T/S", "1-T/S_s0", "1-T/S_s1", "1-T/S_s2", "S 1-f1/f0.6", "S 1-f1/f0.3",
          "T 1-f1/f0.6", "T 1-f1/f0.3", "T_f0.3/S_f1_s0")  # fmt: skip


def stats(pos: dict, att: dict, sd: dict, s: np.ndarray) -> dict:  # sd_eval.stats + disagreement + section 3's ratios
    o = {}
    for k in pos:
        o[k], o["p90_" + k], o["att_" + k] = float(np.median(pos[k][s])), float(np.percentile(pos[k][s], 90)), float(np.median(att[k][s]))
        if k in sd:
            o["sd_" + k] = float(np.median(sd[k][s]))
    o.update(ratios(o))
    return o


def cell(pos, att, sd, rs: np.ndarray, sel: np.ndarray, rng) -> dict:  # sd_eval.cell
    idx = np.flatnonzero(sel)
    r = rs[idx]
    o_ = np.argsort(r, kind="stable")
    groups = np.split(idx[o_], np.flatnonzero(np.diff(r[o_])) + 1)
    point = stats(pos, att, sd, idx)
    boots = [stats(pos, att, sd, np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(NB)]
    ci = {k: (float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))) for k in point}
    return dict(n=int(len(idx)), n_resets=len(groups), point=point, ci=ci)


def section(title, D, idx, cells, models, dev, rng) -> tuple[dict, dict]:  # sd_eval.section (+ disagreement, ratios)
    pos, att, sd = predict(models, D, idx, dev)
    rs = np.asarray(D["reset"])[idx]
    ch = np.asarray(D["chain"])[idx]
    grp = {"all": np.ones(len(idx), bool), "first": ch == 0, "chained": ch >= 1}
    out = dict(title=title, n=int(len(idx)), n_resets=int(len(np.unique(rs))), cells=[])
    print(f"\n== {title}: {len(idx)} kept segments from {out['n_resets']} resets")
    names = list(pos)
    for g in A3:
        for dim, lab, sel, gs in cells:
            if g not in gs:
                continue
            s = sel & grp[g]
            if s.sum() == 0:
                print(f"  {g} | {dim}: {lab}: no segments")
                continue
            c = cell(pos, att, sd, rs, s, rng)
            c.update(group=g, dim=dim, cell=lab)
            out["cells"].append(c)
            p, ci = c["point"], c["ci"]
            f = lambda k, d=2: f"{p[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
            print(f"  {g} | {dim}: {lab}  n {c['n']} ({c['n_resets']} resets)")
            print("    position cm, median: " + " | ".join(f"{k} " + f(k) for k in names))
            print("    position cm, p90:    " + " | ".join(f"{k} " + f("p90_" + k) for k in names))
            print("    attitude deg, median: " + " | ".join(f"{k} " + f("att_" + k) for k in names))
            print("    disagreement cm, median: " + " | ".join(f"{k} " + f("sd_" + k) for k in names if k in sd))
            print("    ratios: " + " | ".join(f"{k} " + f(k, 3) for k in RATIOS))
    return out, dict(pos=pos, rs=rs)


def med(v: np.ndarray) -> float:
    return float(np.median(v)) if len(v) else float("nan")


def p90(v: np.ndarray) -> float:
    return float(np.percentile(v, 90)) if len(v) else float("nan")


def two_hand(one: dict, two: dict, rng) -> dict:
    """one / two: file -> dict(pos, rs) (the kept segments of the split). Resampled over the union's resets."""
    sets = [("one", "d3", one["d3"]), ("one", "d3b", one["d3b"]), ("two", "d3", two["d3"]), ("two", "d3b", two["d3b"])]
    lab = np.concatenate([np.full(len(x["rs"]), i) for i, (_, _, x) in enumerate(sets)])
    rs = np.concatenate([x["rs"] for _, _, x in sets])
    val = {k: np.concatenate([x["pos"][k] for _, _, x in sets]) for k in ("direct",) + TWO}
    masks = {"one": lab <= 1, "two": lab >= 2, "one d3": lab == 0, "one d3b": lab == 1, "two d3": lab == 2, "two d3b": lab == 3}

    def st(s: np.ndarray) -> dict:
        o = {}
        for k, v in val.items():
            for nm, mk in masks.items():
                x = v[s][mk[s]]
                o[f"{k} {nm}"], o[f"p90 {k} {nm}"] = med(x), p90(x)
            o[f"{k} two/one"] = o[f"{k} two"] / o[f"{k} one"]
            for f in ("d3", "d3b"):
                o[f"{k} two/one {f}"] = o[f"{k} two {f}"] / o[f"{k} one {f}"]
        return o

    o_ = np.argsort(rs, kind="stable")
    G = np.split(o_, np.flatnonzero(np.diff(rs[o_])) + 1)
    point = st(np.arange(len(rs)))
    boots = [st(np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])) for _ in range(NB)]
    ci = {k: (float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5))) for k in point}
    n = {nm: int(mk.sum()) for nm, mk in masks.items()}
    nr = {nm: int(len(np.unique(rs[mk]))) for nm, mk in masks.items()}
    return dict(n=n, n_resets=nr, n_resets_union=len(G), point=point, ci=ci)


def lml_refs(ctrl: str) -> dict:
    fv = json.loads((LO / "selfmodel_train3b_20261125" / ctrl / "final_val.json").read_text())
    sec = fv["sections"][0]
    assert fv["split"] == "val" and sec["title"] == "data3 val", (fv["split"], sec["title"])
    c = next(c for c in sec["cells"] if (c["group"], c["dim"], c["cell"]) == ("all", "all", "all"))
    d = [float(c["point"][f"D{j}"]) for j in SEEDS]
    fd = json.loads((LO / "selfmodel_fdvel_20261126" / ctrl / "eval_lml_val.json").read_text())
    assert fd["split"] == "val" and fd["f"][0].rstrip("/").endswith(f"selfmodel_fdvel_20261126/{ctrl}/models_s0"), fd["f"][0]
    cf = next(c for c in fd["cells"] if (c["dim"], c["cell"]) == ("all", "all"))
    return dict(D=float(np.mean(d)), D_seeds=d, D_n=int(c["n"]), F=float(cf["f"][0]), F_seeds=[float(x) for x in cf["f"]],
                F_mean=float(np.mean(cf["f"])), F_n=int(cf["n"]), src_D=str(LO / "selfmodel_train3b_20261125" / ctrl / "final_val.json"),
                src_F=str(LO / "selfmodel_fdvel_20261126" / ctrl / "eval_lml_val.json"))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="<collection root>/train/<controller>")
    ap.add_argument("--split", required=True, choices=["val", "test_iid"])
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    ctrl = a.root.resolve().name
    assert ctrl in ("scalebfm", "heft", "sonic"), ctrl
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)
    REF.update(lml_refs(ctrl))
    models = []
    for name, arm, s, frac in MODELS:
        r = a.root / f"models_{arm}_s{s}"
        assert (TR.model_dir("E3_ensemble", frac, r) / "model.pt").exists(), f"{name}: no {TR.model_dir('E3_ensemble', frac, r)}/model.pt"
        models.append((name, r, frac))
    print(f"{ctrl} {a.split}; device {dev}; FDVEL {DT.FDVEL}; reader {DT.__file__}")
    print("models " + ", ".join(f"{n} {TR.model_dir('E3_ensemble', f, r)}" for n, r, f in models))
    print(f"LML denominators (fixed): D seeds 0-2 mean on LML data3 val all/all/all from {REF['src_D']} (n {REF['D_n']}); "
          f"F (fdvel seed 0) from {REF['src_F']} (n {REF['F_n']})")  # fmt: skip
    out = dict(root=str(a.root), controller=ctrl, split=a.split, models=[(n, str(r), f) for n, r, f in models], ref=dict(REF),
               sections=[], headline={})  # fmt: skip
    one = {}
    for name, fn in (("data3", "d3"), ("data3b", "d3b")):
        D = DT.load(str(a.root / "data" / f"{fn}.npz"))
        sp = DT.splits(D)
        idx = kept(D, sp[a.split])
        print(f"\n{name} ({fn}.npz): {len(D['n_steps'])} segments, {a.split} {len(sp[a.split])}, kept {len(idx)}")
        assert len(idx), f"{name} {a.split}: no kept segments"
        g = lambda k: np.asarray(D[k])[idx]  # noqa: E731
        cells = [("all", "all", np.ones(len(idx), bool), A3)]
        if name == "data3":
            wt = g("walk_task") if "walk_task" in D else np.zeros(len(idx))
            cells += [("start", "in place", g("walked") == 0, ("first",)),
                      ("start", "walk first, task walk", (g("walked") == 1) & (wt == 1), ("first",)),
                      ("start", "walk first, d3 walk", (g("walked") == 1) & (wt == 0), ("first",))]  # fmt: skip
        else:
            kind = g("kind")
            cells += [("kind", "entry", kind == KIND["entry"], A3), ("kind", "pre-grasp", kind == KIND["pre"], A3),
                      ("kind", "relative", kind == KIND["rel"], A3)]  # fmt: skip
        sec, pr = section(f"{name} {a.split}", D, idx, cells, models, dev, rng)
        out["sections"].append(sec)
        one[fn] = pr
        del D
    # gate (b)'s two-hand row
    two = {}
    for fn in ("d3", "d3b"):
        B = DT.load(str(a.root / "data" / f"both_{fn}.npz"))
        assert "hands" in B and bool((np.asarray(B["hands"]) == 2).all()), f"both_{fn}: not all two-hand segments"
        sp = DT.splits(B)
        idx = kept(B, sp[a.split])
        print(f"\ntwo-hand both_{fn}.npz: {len(B['n_steps'])} segments, {a.split} {len(sp[a.split])}, kept {len(idx)}")
        pos, _, _ = predict([m for m in models if m[0] in TWO], B, idx, dev)
        two[fn] = dict(pos=pos, rs=np.asarray(B["reset"])[idx])
        del B
    th = two_hand(one, two, rng)
    out["two_hand"] = th
    p, ci = th["point"], th["ci"]
    f = lambda k, d=2: f"{p[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
    print(f"\n== gate (b) two-hand row, {a.split}, kept: segments {th['n']} (resets {th['n_resets']}; union {th['n_resets_union']})")
    for k in ("direct",) + TWO:
        print(f"  {k}: median one-hand {f(k + ' one')} | two-hand {f(k + ' two')} | two/one {f(k + ' two/one', 3)}")
        print(f"     p90 one-hand {f('p90 ' + k + ' one')} | two-hand {f('p90 ' + k + ' two')}")
        print("     per file: " + " | ".join(f"{fl}: one {f(k + ' one ' + fl)} two {f(k + ' two ' + fl)} two/one {f(k + ' two/one ' + fl, 3)}"
                                         for fl in ("d3", "d3b")))  # fmt: skip
    # the headline
    c = next(c for c in out["sections"][0]["cells"] if (c["group"], c["dim"], c["cell"]) == ("all", "all", "all"))
    p, ci = c["point"], c["ci"]
    out["headline"] = dict(cell="data3 all/all/all", n=c["n"], n_resets=c["n_resets"], point={k: p[k] for k in p if not k.startswith(("p90_", "att_"))},
                           ci={k: ci[k] for k in ci if not k.startswith(("p90_", "att_"))})  # fmt: skip
    print(f"\n== HA {a.split} (data3 {a.split}, all reaches, kept; n {c['n']}, {c['n_resets']} resets): position median [95 %]")
    print("  " + " | ".join(f"{n} " + f(n) for n, _, _, _ in MODELS))
    print("  " + " | ".join(f"{k} " + f(k, 3) for k in RATIOS))
    print(f"  LML denominators (fixed): D (train3b, seeds 0-2 mean) {REF['D']:.3f} cm (seeds {np.round(REF['D_seeds'], 3).tolist()}); "
          f"F (fdvel seed 0) {REF['F']:.3f} cm (fdvel seeds {np.round(REF['F_seeds'], 3).tolist()}, mean {REF['F_mean']:.3f})")  # fmt: skip
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
