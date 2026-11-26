"""selfmodel_simplesonicdata_20261126: the collection's stop criteria and its registered descriptions; read only.

    $PY ssd_datacheck.py <data3 dir> <data3b dir> [--json F]

Per style (data/all.npz, the one-hand segments; data/w*_log.json, the resets): usable one-hand segments (not fallen, end
label settled) in all and per reset; their share that settled only under the relaxed rule (settled_2cm false); the jitter
(jit_pp_cm, jit_rms_cm) median on them; the resets that fell or got lost; the wall time per reset. Exit 1 (stop, do not
train) when the usable segments of both styles together are < 15 000 or the fallen + lost resets of either style > 40 %."""
import glob
import json
import sys

import numpy as np


def style(d: str) -> dict:
    z = np.load(f"{d}/data/all.npz")
    ok = (z["fell"] == 0) & (z["settled_end"] == 1)
    R = [r for f in sorted(glob.glob(f"{d}/data/w*_log.json")) for r in json.load(open(f))]
    lost = sum(r.get("lost") is not None for r in R)
    fell = sum(r.get("fell") is not None and r.get("lost") is None for r in R)
    wall = [r["wall_s"] for r in R if r.get("wall_s") is not None]
    return dict(resets=len(R), segments=int(len(ok)), usable=int(ok.sum()), per_reset=float(ok.sum() / max(len(R), 1)),
                only_5cm=float(1 - z["settled_2cm"][ok].mean()), jit_pp_med=float(np.nanmedian(z["jit_pp_cm"][ok])),
                jit_rms_med=float(np.nanmedian(z["jit_rms_cm"][ok])), fell=fell, lost=lost, fell_lost=(fell + lost) / max(len(R), 1),
                wall_med=float(np.median(wall)), wall_mean=float(np.mean(wall)))  # fmt: skip


def main() -> None:
    out = {"data3": style(sys.argv[1]), "data3b": style(sys.argv[2])}
    for k, o in out.items():
        print(f"{k}: {o['resets']} resets, {o['segments']} one-hand segments, usable {o['usable']} ({o['per_reset']:.2f} a reset); "
              f"settled only under the relaxed rule {100 * o['only_5cm']:.1f} %; jit_pp median {o['jit_pp_med']:.2f} cm, rms "
              f"{o['jit_rms_med']:.2f} cm; resets fell {o['fell']}, lost {o['lost']} ({100 * o['fell_lost']:.1f} %); wall "
              f"{o['wall_med']:.1f} s / reset median, {o['wall_mean']:.1f} mean")  # fmt: skip
    n = out["data3"]["usable"] + out["data3b"]["usable"]
    bad = n < 15000 or max(o["fell_lost"] for o in out.values()) > 0.40
    print(f"usable one-hand segments {n}; " + ("DATA STOP (< 15 000 usable, or fell + lost > 40 %)" if bad else "DATA OK"))
    if "--json" in sys.argv:
        open(sys.argv[sys.argv.index("--json") + 1], "w").write(json.dumps(out, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
