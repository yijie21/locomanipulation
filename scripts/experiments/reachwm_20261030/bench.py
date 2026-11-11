"""Bench of the trained reach world models (train.py) on test_iid, the four bands and the edge set (non-fallen segments,
except the fall AUROC). Writes outputs/reachwm_20261030/bench/{results.json, results.md}.

Per model and test set: final position error in cm (median, p90, p99, mean), per-direction median absolute error in the
goal wrist frame (depth = wrist x, closing = wrist y, third = wrist z), fractions under 0.75 / 1 / 2 cm, rotation error
(median, p90), all on settled-only segments too; for models with pos_std the Spearman correlation and the AUROC (error
> 1 cm) of |std| and its mean per set; for p_fell / p_settled the AUROC; test_iid vs training fraction; ms per 1024.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/bench.py
    ... bench.py --models-root /tmp/x --out /tmp/x/bench     # tiny smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402
import models as M  # noqa: E402
import train as TR  # noqa: E402

SETS = ["test_iid", "band_x", "band_z", "band_yaw", "band_pel", "edge"]
BENCH = DT.OUT / "bench"


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summary(err, loc, rot) -> dict:
    if len(err) == 0:
        return dict(n=0)
    q = lambda x, p: float(np.percentile(x, p))  # noqa: E731
    return dict(n=int(len(err)), median=q(err, 50), p90=q(err, 90), p99=q(err, 99), mean=float(err.mean()),
                depth=float(np.median(np.abs(loc[:, 0]))), closing=float(np.median(np.abs(loc[:, 1]))),
                third=float(np.median(np.abs(loc[:, 2]))), u075=float((err < 0.75).mean()), u1=float((err < 1).mean()),
                u2=float((err < 2).mean()), rot_median=q(rot, 50), rot_p90=q(rot, 90))  # fmt: skip


def evaluate(model, D, GD, idx_np) -> dict:
    dev = GD["u"].device
    out = {}
    idx = torch.tensor(idx_np, device=dev)
    if len(idx) == 0:
        return dict(all=dict(n=0), settled=dict(n=0))
    p = M.predict_batched(model, GD, idx)
    fell = D["fell"][idx_np].astype(bool)
    if "p_fell" in p:
        out["auroc_fell"] = auroc(p["p_fell"].cpu().numpy(), fell)
    keep = torch.tensor(np.where(~fell)[0], device=dev)
    if len(keep) == 0:
        return dict(all=dict(n=0), settled=dict(n=0), **out)
    g = {k: v[idx][keep] for k, v in GD.items() if k in ("pos_f", "rot_f", "rot6", "target", "settled")}
    e = p["pos"][keep] - g["pos_f"]
    err = (100 * e.norm(dim=-1)).cpu().numpy()
    loc = (100 * torch.einsum("bji,bj->bi", M.gs6(g["rot6"]), e)).cpu().numpy()
    rot = M.rot_angle_deg(M.gs6(p["rot6"][keep]), M.gs6(g["rot_f"])).cpu().numpy()
    st = g["settled"].cpu().numpy() > 0
    out["all"], out["settled"] = summary(err, loc, rot), summary(err[st], loc[st], rot[st])
    if "pos_std" in p:
        s = (100 * p["pos_std"][keep].norm(dim=-1)).cpu().numpy()
        out["unc"] = dict(mean_std_cm=float(s.mean()), spearman=float(spearmanr(s, err)[0]) if len(s) > 2 else float("nan"),
                          auroc_1cm=auroc(s, err > 1.0))  # fmt: skip
    if "p_settled" in p:
        out["auroc_settled"] = auroc(p["p_settled"][keep].cpu().numpy(), st)
    return out


def speed_ms(model, GD, idx_np) -> float:
    idx = torch.tensor(idx_np[:1024], device=GD["u"].device)
    b = M.sub(GD, idx, model.step)
    with torch.no_grad():
        model.predict(b)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            model.predict(b)
        torch.cuda.synchronize()
    return (time.time() - t) / 3 * 1000 * 1024 / max(len(idx), 1)


def f(x, d=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def table(res, sname, kind, names) -> list[str]:
    cols = ["median", "p90", "p99", "mean", "depth", "closing", "third", "u075", "u1", "u2", "rot_median", "rot_p90"]
    hd = ["median", "p90", "p99", "mean", "depth", "closing", "third", "<0.75", "<1", "<2", "rot med", "rot p90"]
    L = ["| model | n | " + " | ".join(hd) + " |", "|---|---|" + "---|" * len(hd)]
    for n in names:
        r = res[n][sname][kind]
        if r.get("n", 0) == 0:
            L.append(f"| {n} | 0 |" + " - |" * len(hd))
            continue
        L.append(
            f"| {n} | {r['n']} | " + " | ".join(f(r[c], 3 if c in ("u075", "u1", "u2") else 2) for c in cols) + " |"
        )
    return L


def markdown(res, eff, speed, info) -> str:
    names = list(res)
    L = ["# Reach world model bench", "",
         f"Data: {info['n_segments']} segments ({info['n_files']} files). Test sets (segments): "
         + ", ".join(f"{k} {v}" for k, v in info["sets"].items()) + ". Fallen segments excluded except the fall AUROC. "
         "Errors in cm of the delivered jaw centre in the heading frame; depth / closing / third are median absolute errors "
         "in the goal wrist frame (x / y / z); rot in degrees.", ""]  # fmt: skip
    for s in SETS:
        L += [f"## {s}", "", "All non-fallen segments", ""] + table(res, s, "all", names)
        L += ["", "Settled only", ""] + table(res, s, "settled", names) + [""]
    L += [
        "## Uncertainty (models with pos_std)",
        "",
        "| model | " + " | ".join(f"{s}: mean std / Spearman / AUROC" for s in SETS) + " |",
        "|---|" + "---|" * len(SETS),
    ]
    for n in names:
        if "unc" in res[n]["test_iid"]:
            L.append(f"| {n} | " + " | ".join(
                f"{f(res[n][s]['unc']['mean_std_cm'])} / {f(res[n][s]['unc']['spearman'])} / {f(res[n][s]['unc']['auroc_1cm'])}"
                if "unc" in res[n][s] else "-" for s in SETS) + " |")  # fmt: skip
    L += [
        "",
        "## Fall and settled AUROC",
        "",
        "| model | " + " | ".join(f"{s} fell / settled" for s in SETS) + " |",
        "|---|" + "---|" * len(SETS),
    ]
    for n in names:
        if "auroc_fell" in res[n]["test_iid"] or "auroc_settled" in res[n]["test_iid"]:
            L.append(
                f"| {n} | "
                + " | ".join(f"{f(res[n][s].get('auroc_fell'))} / {f(res[n][s].get('auroc_settled'))}" for s in SETS)
                + " |"
            )
    L += [
        "",
        "## Data efficiency (test_iid median / p90 in cm)",
        "",
        "| model | frac | median | p90 |",
        "|---|---|---|---|",
    ]
    for n, fr, m, p in eff:
        L.append(f"| {n} | {fr:g} | {f(m)} | {f(p)} |")
    L += ["", "## Speed (ms per 1024 predictions, GPU)", "", "| model | ms |", "|---|---|"]
    L += [f"| {n} | {f(v, 2)} |" for n, v in speed.items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--models-root", default=str(TR.MODELS))
    ap.add_argument("--out", default=str(BENCH))
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root, out = Path(a.models_root), Path(a.out)
    D = DT.load(a.data)
    GD = M.tensors(D, dev, step=True)
    sp = DT.splits(D)
    found = []
    for d in sorted(root.glob("*_f*")):
        mm = re.fullmatch(r"(.+)_f([0-9.]+)", d.name)
        if mm and (d / "model.pt").exists():
            found.append((mm.group(1), float(mm.group(2))))
    res, eff, speed = {}, [], {}
    for name, frac in found:
        model = TR.load_model(name, frac, D, dev, root)
        r = {s: evaluate(model, D, GD, sp[s]) for s in SETS}
        if frac == 1.0:
            res[name] = r
            if torch.cuda.is_available():
                speed[name] = speed_ms(model, GD, sp["test_iid"])
        a_ = r["test_iid"]["all"]
        eff.append((name, frac, a_.get("median"), a_.get("p90")))
        print(name, frac, "test_iid median", f(a_.get("median")), flush=True)
    order = {n: i for i, n in enumerate(M.NAMES)}
    res = {n: res[n] for n in sorted(res, key=order.get)}
    eff.sort(key=lambda t: (order[t[0]], t[1]))
    info = dict(n_segments=int(len(D["n_steps"])), n_files=len(D["files"]), sets={k: int(len(sp[k])) for k in SETS})
    out.mkdir(parents=True, exist_ok=True)
    json.dump(
        dict(info=info, results=res, efficiency=eff, speed_ms_per_1024=speed), open(out / "results.json", "w"), indent=1
    )
    (out / "results.md").write_text(markdown(res, eff, speed, info))
    print(f"wrote {out / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
