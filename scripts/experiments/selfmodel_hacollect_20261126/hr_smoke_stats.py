"""selfmodel_hadata_20261126 (dev 2026-11-26): hr_real smoke tables -- per directory (one controller x task x via
rule): episodes, reaches, furniture contact (sx_ha's per-segment touch_n / touch_z, hc_ha revision 1's rule: > 1 N at
>= 0.2 m), which body touched, contact before the first reach, the command targets' forward distance (pelvis heading
frame) and whether they are inside data3's standing reach table, settledness, and the hand error to the command.

    python hr_smoke_stats.py DIR [DIR ...]      # each DIR holds data/ and log/ (hr_real --out)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

TOUCH_N, TOUCH_Z = 1.0, 0.2
LIGHT_N, LOW_MIN, DXY_DD, DXY_BOX_PAD = 15.0, 0.01, 0.12, 0.05  # the sensitivity group (revision 2a)


def light_target(L: dict, chain: int, side: str) -> bool | None:
    """Revision 2a's sensitivity group, for a reach the main rule excludes: every body over 1 N at >= 0.2 m is a link of
    the reaching hand or wrist, the peak force is under 15 N, at every contact frame the link's lowest point is >= 1 cm
    over the perceived support and its origin within the perceived object's reach (PPBox: sqrt(2) half_a + 5 cm of the
    box centre; DoubleDesk: 12 cm of the grasp point). OpenDoor: never (the handle sits on the door). None: no record."""
    cs = L.get("contacts")
    if cs is None or chain >= len(cs):
        return None
    c = cs[chain]
    if L["task"] == "opendoor" or not c["bodies"]:
        return False
    if L["task"] == "ppbox":
        r = 2**0.5 * float(L["perceived"]["half_a"]) + DXY_BOX_PAD
    else:
        r = DXY_DD
    for nm, b in c["bodies"].items():
        if not (nm.startswith(f"{side}_") and ("_hand" in nm or "_wrist" in nm)):
            return False
        if b["n"] >= LIGHT_N or b["low_over_support"] is None or b["low_over_support"] < LOW_MIN:
            return False
        if b["dxy_obj"] is None or b["dxy_obj"] > r:
            return False
    return True


def part(body: str, side: str) -> str:
    if "hand" in body or "wrist" in body:
        return "own hand" if body.startswith(side) else "other hand"
    return "body"


def one(d: Path) -> dict:
    eps = sorted(glob.glob(str(d / "log" / "*.json")))
    rows, stance, ends = [], 0, []
    for f in eps:
        L = json.loads(Path(f).read_text())
        ends.append(L.get("end"))
        t = (L.get("info") or {}).get("touch") or {}
        st = t.get("stance")
        if st is not None and st["n"] > TOUCH_N and st["pos"][2] >= TOUCH_Z:
            stance += 1
        z = d / "data" / (Path(f).stem + ".npz")
        if not z.exists():
            continue
        Z = np.load(z)
        for i, r in enumerate(L.get("reaches", [])):
            c = (t.get("reach") or [None] * (i + 1))[i]
            n, zz = float(Z["touch_n"][i]), float(Z["touch_z"][i])
            hit = n > TOUCH_N and zz >= TOUCH_Z
            rows.append(dict(ep=Path(f).stem, no=r["reach_no"], side=r["side"], hit=hit, n=n,
                             light=light_target(L, i, r["side"]) if hit else False,
                             part=part(c["body"], r["side"]) if (hit and c) else "",
                             x=r["tgt_h"][0] if "tgt_h" in r else float(Z["target"][i][0]),
                             in_table=r.get("in_table"), in_range=all(r["in_range"].values()),
                             settled=bool(Z["settled_end"][i]), fell=bool(Z["fell"][i]) if "fell" in Z.files else False,
                             jaw=float(Z["jaw_err_cm"][i]), ik=r["ik_err_cm"]))  # fmt: skip
    return dict(n_ep=len(eps), ends=ends, stance=stance, rows=rows)


def main() -> None:
    for a in sys.argv[1:]:
        d = Path(a)
        s = one(d)
        R = s["rows"]
        n = len(R)
        hit = [r for r in R if r["hit"]]
        parts = {p: sum(r["part"] == p for r in hit) for p in ("own hand", "other hand", "body")}
        clean = [r for r in R if not r["hit"]]
        xs = np.array([r["x"] for r in R]) if R else np.zeros(0)
        tab = [r["in_table"] for r in R if r["in_table"] is not None]
        print(f"{d.parent.name}/{d.name}: episodes {s['n_ep']} (ends {dict((e, s['ends'].count(e)) for e in set(s['ends']))}); "
              f"contact before the first reach {s['stance']}/{s['n_ep']}")  # fmt: skip
        lt = [r["light"] for r in hit]
        print(f"  sensitivity group (light touch on the target only): {sum(v is True for v in lt)} of the {len(hit)} excluded"
              + (f" ({sum(v is None for v in lt)} without a contact record)" if any(v is None for v in lt) else ""))
        print(f"  reaches {n}: furniture contact {len(hit)} ({parts}), clean {len(clean)}; "
              f"by reach_no: " + ", ".join(f"{k}: {sum(r['hit'] for r in R if r['no'] == k)}/{sum(r['no'] == k for r in R)} hit"
                                         for k in sorted({r['no'] for r in R})))  # fmt: skip
        if n:
            print(f"  target x (m, pelvis heading frame) min/p50/max {xs.min():.2f}/{np.median(xs):.2f}/{xs.max():.2f}; "
                  f"in data3's standing table {sum(tab)}/{len(tab)}; d3 angle/z ranges {sum(r['in_range'] for r in R)}/{n}; "
                  f"settled {sum(r['settled'] for r in R)}/{n}; fell {sum(r['fell'] for r in R)}")  # fmt: skip
            pre = [r for r in R if r["no"] == 1]
            if pre:
                print(f"  pre-grasp reaches: hand error to the command median {np.median([r['jaw'] for r in pre]):.1f} cm "
                      f"(clean ones {np.median([r['jaw'] for r in pre if not r['hit']]) if any(not r['hit'] for r in pre) else float('nan'):.1f}); "
                      f"ik_err max {max(r['ik'] for r in pre):.2f} cm")  # fmt: skip


if __name__ == "__main__":
    main()
