"""Canonical starts (2026-10-31, improvement 1 of the bake-off): does a world model trained only on the reset's standing
start beat the all-start model there, and does gen_reach.py's REACH_HOME step turn every later start (after a walk, after
an earlier reach) back into that standing start?

Models: the all-start E3_ensemble (models/), and E2_residual / E3_ensemble trained with REACHWM_SUBSET=first_still
(models_first_still/). Sets: the main data's test_iid split by start kind, its bands and edge set restricted to the
reset start, and the homed data (data_home/, its bands and edge removed) by chain index and walk. Per set and model: final
position error median, p90 and fractions under 0.75 / 1 cm; the controller's own miss (command vs landing); the arm
joints' RMS distance from the reset pose at the command; the ensemble's mean std.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/canon_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402
import models as M  # noqa: E402
import train as TR  # noqa: E402

OUT = DT.OUT / "canon"
ARMS = slice(15, 29)


def load_as(name: str, root: Path, subset: str, D: dict, dev):
    DT.SUBSET = subset  # normalization comes from the training split the model saw
    try:
        return TR.load_model(name, 1.0, D, dev, root)
    finally:
        DT.SUBSET = ""


def stats(err, std=None) -> dict:
    out = dict(n=int(len(err)), median=float(np.median(err)), p90=float(np.percentile(err, 90)),
               lt075=float(np.mean(err < 0.75)), lt1=float(np.mean(err < 1.0)))  # fmt: skip
    if std is not None:
        out["std_mean"] = float(std.mean())
    return out


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = DT.load(None)
    H = DT.load(str(DT.OUT / "data_home" / "w[0-9]*_[0-9]*.npz"))
    models = {
        "E3_all": load_as("E3_ensemble", DT.OUT / "models", "", D, dev),
        "E2_first": load_as("E2_residual", DT.OUT / "models_first_still", "first_still", D, dev),
        "E3_first": load_as("E3_ensemble", DT.OUT / "models_first_still", "first_still", D, dev),
    }
    q_reset = D["q_0"][(D["chain"] == 0) & (D["walked"] == 0)].mean(0)

    sp = DT.splits(D)
    first = (D["chain"] == 0) & (D["walked"] == 0)
    ok = D["fell"] == 0
    sets = {}
    t = sp["test_iid"]
    for nm, m in (("reset start", first[t]), ("first after walk", (D["chain"][t] == 0) & (D["walked"][t] == 1)),
                  ("second, no walk", (D["chain"][t] == 1) & (D["walked"][t] == 0)),
                  ("second after walk", (D["chain"][t] == 1) & (D["walked"][t] == 1))):  # fmt: skip
        sets[f"main test_iid / {nm}"] = ("D", t[m & ok[t]])
    for k in ("band_x", "band_z", "band_yaw", "band_pel", "edge"):
        i = sp[k]
        sets[f"main {k} / reset start"] = ("D", i[first[i] & ok[i]])
    held = H["band_x"] | H["band_z"] | H["band_yaw"] | H["band_pel"] | H["edge"]
    hk = (~held) & (H["fell"] == 0)
    for c in range(int(H["chain"].max()) + 1):
        for w in (0, 1):
            m = hk & (H["chain"] == c) & (H["walked"] == w)
            sets[f"homed / reach {c + 1}, {'after walk' if w else 'no walk'}"] = ("H", np.where(m)[0])
    sets["homed / all later starts"] = ("H", np.where(hk & ~((H["chain"] == 0) & (H["walked"] == 0)))[0])

    G = {"D": M.tensors(D, dev), "H": M.tensors(H, dev)}
    src = {"D": D, "H": H}
    # start-blind use of the first-still models: every start fed as the reset start (bit-identical across resets, so its
    # features have ~zero spread in that training split and any real deviation is divided by ~0)
    # (one per hand: x0 carries the acting jaw)
    x0_r, x0_l = (G["D"]["x0"][int(np.where(first & (D["side"] == sd))[0][0])] for sd in (1, -1))
    blind = {k: dict(g, x0=torch.where(g["side"][:, None] > 0, x0_r, x0_l).contiguous()) for k, g in G.items()}
    models["E3_first_blind"] = models["E3_first"]
    rows = []
    for sname, (s, idx) in sets.items():
        X = src[s]
        row = dict(set=sname, n=int(len(idx)))
        row["controller_miss_median"] = float(np.median(X["err_cm"][idx]))
        row["arm_rms_rad"] = float(np.median(np.sqrt(((X["q_0"][idx][:, ARMS] - q_reset[ARMS]) ** 2).mean(1))))
        ti = torch.tensor(idx, device=dev)
        for mn, model in models.items():
            p = M.predict_batched(model, (blind if mn.endswith("_blind") else G)[s], ti)
            err = 100 * (p["pos"] - G[s]["pos_f"][ti]).norm(dim=-1).cpu().numpy()
            std = 100 * p["pos_std"].norm(dim=-1).cpu().numpy() if "pos_std" in p else None
            row[mn] = stats(err, std)
            if mn == "E3_first_blind" and s == "H":  # error against the start's distance from the reset start
                rms = np.sqrt(((X["q_0"][idx][:, ARMS] - q_reset[ARMS]) ** 2).mean(1))
                legs = np.sqrt(((X["q_0"][idx][:, :12] - q_reset[:12]) ** 2).mean(1))
                row["blind_by_arm_rms"] = [dict(lo=a, hi=b, **stats(err[(rms >= a) & (rms < b)]))
                                           for a, b in ((0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 9))
                                           if ((rms >= a) & (rms < b)).sum() >= 20]  # fmt: skip
                row["leg_rms_rad"] = float(np.median(legs))
        rows.append(row)
        print(json.dumps(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows, n_home_segments=int(len(H["n_steps"])), home_fell=int(H["fell"].sum())),
              open(OUT / "results.json", "w"), indent=1)  # fmt: skip
    lines = ["| set | n | controller miss | arm RMS rad | " + " | ".join(f"{m} med / p90 / <0.75 / <1" for m in models) + " |",
             "|" + "---|" * (4 + len(models))]  # fmt: skip
    for r in rows:
        cells = [f"{r[m]['median']:.2f} / {r[m]['p90']:.2f} / {r[m]['lt075']:.2f} / {r[m]['lt1']:.2f}" for m in models]
        lines.append(
            f"| {r['set']} | {r['n']} | {r['controller_miss_median']:.2f} | {r['arm_rms_rad']:.3f} | "
            + " | ".join(cells)
            + " |"
        )
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
