"""selfmodel_via_20261120: where along the arm a model is wrong, and what a compensated command did to each part.

    $PY via_parts.py <model dir> <dataset dir> ... [--detour]

--detour: a replay's detour routes only (routes.json; every kind but FreshUnder), as via_comp_eval.py's headline.
Per dataset (all.npz, arm.npz, table.npz) and per group of spheres (shoulder, elbow, wrist, palm, fingers), medians over
the paths, cm, of the group's largest swept-region error (via_model.sweep_err):
  model    the recorded ticks against the sweep <model dir> predicts for the command that was run -- for a compensated
           run this is the model judged on the command it sent, by any model, not only the one that compensated
  plan     the run's own reference plan against the wanted sweep (comp_ref0 of a compensated run; 0 for a direct run):
           how far the sent command's plan was moved
  actual   the recorded ticks against the wanted sweep
Last line per dataset: the model's error over all 24 spheres (what via_comp_eval.py calls the model's own error).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402

GROUPS = {
    "shoulder": lambda n: "shoulder" in n,
    "elbow": lambda n: "elbow" in n,
    "wrist": lambda n: "wrist" in n,
    "palm": lambda n: "palm" in n,
    "fingers": lambda n: n.startswith("hand_") and "palm" not in n,
}


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    detour = "--detour" in a
    a = [x for x in a if x != "--detour"]
    M = torch.load(Path(a[0]) / "via_point.pt", map_location="cpu", weights_only=False)
    for d in a[1:]:
        D = Path(d)
        T, z, X = VP.load([D]), dict(np.load(D / "all.npz")), dict(np.load(D / "arm.npz"))
        link = [str(x) for x in X["link"]]
        rows = np.arange(len(T["seg"]))
        if detour:
            names = json.load(open(D / "routes.json"))["names"]
            rows = rows[[not names[r].startswith("FreshUnder") for r in T["route"]]]
        off = VP.offsets(M, T, rows)["direction"]
        em, er, ea = [], [], []
        for n, j in enumerate(rows):
            ticks, ref = X["X"][T["a"][j] : T["b"][j]].astype(float), T["ref"][j].astype(float)
            want = z["comp_ref0"][T["seg"][j]].astype(float) if "comp_ref0" in z else ref
            full = lambda r: np.concatenate([r, r[-1:]])  # noqa: E731
            em.append(VM.sweep_err(ticks, full(ref) + off[n]))
            er.append(VM.sweep_err(ref, full(want)))
            ea.append(VM.sweep_err(ticks, full(want)))
        em, er, ea = np.array(em), np.array(er), np.array(ea)
        print(f"{D.parent.name}/{D.name}: {len(rows)} paths")
        for g, f in GROUPS.items():
            i = [k for k, n in enumerate(link) if f(n)]
            print(f"  {g:9s} model {100 * np.median(em[:, i].max(1)):5.2f}  plan {100 * np.median(er[:, i].max(1)):5.2f}  "
                  f"actual {100 * np.median(ea[:, i].max(1)):5.2f}")  # fmt: skip
        print(f"  all 24    model {100 * np.median(em.max(1)):5.2f} (90 % {100 * np.percentile(em.max(1), 90):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
