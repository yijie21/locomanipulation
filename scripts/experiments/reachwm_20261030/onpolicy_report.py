"""onpolicy.py's rounds on the 345 core goals (2026-11-03): wrist_invert.py runs with the retrained models (WRIST_OUT
invert/wrist_<tag>, WRIST_SERVOS=open, WRIST_METHODS=pos,pose3g,w20) against wrist_invert2 (the report's model), eyes
closed, paired by goal: same stances, same goals, only the model differs.

    python3 scripts/experiments/reachwm_20261030/onpolicy_report.py onp1 onp1x5
Writes outputs/reachwm_20261030/invert/onpolicy.md.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

INV = Path(__file__).resolve().parents[3] / "outputs" / "reachwm_20261030" / "invert"
METHODS = ("pos", "pose3g", "w20")
NAMES = {"pos": "position only", "pose3g": "wrist first", "w20": "1 cm = 20 deg"}


def rows(d: Path, pattern: str) -> dict:
    out = {}
    for f in sorted(d.glob(pattern)):
        for r in json.load(open(f)):
            if r["servo"] == "open" and r["method"] in METHODS:
                out[(r["gid"], r["method"])] = r
    return out


def main(tags: list[str]) -> None:
    runs = {"base": rows(INV / "wrist_invert2", "h_w*_pos_pose3g_w20_cap15.json")}
    for t in tags:
        runs[t] = rows(INV / f"wrist_{t}", "h_w*_pos_pose3g_w20.json")
    gids = sorted({g for g, _ in runs["base"]})
    L = [f"{len(gids)} core goals, eyes closed (open loop); base = wrist_invert2's model; paired by goal", "",
         "| model | inversion | goals | position median cm | <1 cm | >2 cm (n) | p90 cm | wrist median deg | <10 deg | >30 deg (n) | predicted position median | predicted wrist median |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    out = {}
    for m in METHODS:
        for name, R in runs.items():
            ok = [g for g in gids if (g, m) in R and not R[(g, m)]["fell"]]
            e = np.array([R[(g, m)]["err"] for g in ok])
            a = np.array([R[(g, m)]["att_deg"] for g in ok])
            pe = np.array([R[(g, m)]["pred_err"] for g in ok])
            pa = np.array([R[(g, m)]["pred_att"] for g in ok])
            out[f"{name}_{m}"] = dict(
                n=len(ok),
                med=float(np.median(e)),
                gt2=int((e > 2).sum()),
                att=float(np.median(a)),
                gt30=int((a > 30).sum()),
            )
            L.append(f"| {name} | {NAMES[m]} | {len(ok)} | {np.median(e):.2f} | {(e < 1).mean():.2f} | {int((e > 2).sum())} | "
                     f"{np.percentile(e, 90):.2f} | {np.median(a):.1f} | {(a < 10).mean():.2f} | {int((a > 30).sum())} | {np.median(pe):.2f} | {np.median(pa):.1f} |")  # fmt: skip
    L += [
        "",
        "| paired against base | inversion | position better / worse | wrist better / worse (by > 1 deg) | >2 cm: fixed / new |",
        "|---|---|---|---|---|",
    ]
    for t in tags:
        for m in METHODS:
            B, N = runs["base"], runs[t]
            ok = [g for g in gids if (g, m) in B and (g, m) in N and not B[(g, m)]["fell"] and not N[(g, m)]["fell"]]
            eb, en = np.array([B[(g, m)]["err"] for g in ok]), np.array([N[(g, m)]["err"] for g in ok])
            ab, an = np.array([B[(g, m)]["att_deg"] for g in ok]), np.array([N[(g, m)]["att_deg"] for g in ok])
            L.append(f"| {t} | {NAMES[m]} | {int((en < eb).sum())} / {int((en > eb).sum())} | {int((an < ab - 1).sum())} / {int((an > ab + 1).sum())} | "
                     f"{int(((eb > 2) & (en <= 2)).sum())} / {int(((eb <= 2) & (en > 2)).sum())} |")  # fmt: skip
    (INV / "onpolicy.md").write_text("\n".join(L) + "\n")
    json.dump(out, open(INV / "onpolicy.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:] or ["onp1"])
