"""Selfwm item 9: the ADT rows of the head camera's self-calibration, paired by start (adt_compare.p_worse), with how
far the task object's camera estimates were off (every read from a fresh sighting) and what the self-calibration left
(perception_stats ... head_selfcal).

    python3 selfcal_adt_summary.py <out.md>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adt_compare import p_worse  # noqa: E402

A = HERE.parents[2] / "outputs" / "selfwm_20261029" / "adt"
L0 = {"view": Path("/workspace/code/locomanipulation-vision-rsi/outputs/vision_rsi_20261023/l0_newmap2/view"),
      "rel": A / "l0_v3", "rel4": A / "l0_v3"}  # fmt: skip
ROWS = [  # (package, row name, dir, the row it is compared with in the same family)
    ("view", "head, exact (earlier run)", "l2_view/l2h", None),
    ("view", "head, 2 deg + 2 %", "l2sc_view/l2h_cal2", None),
    ("view", "head, 2 deg + 2 %, self-calibrated", "l2sc_view/l2h_cal2_sc", "l2sc_view/l2h_cal2"),
    ("view", "head, exact, self-calibrated", "l2sc_view/l2h_sc", "l2_view/l2h"),
    ("rel", "head, exact (earlier run)", "l2_v3/l2h", None),
    ("rel", "head, 2 deg + 2 %", "l2sc_rel/l2h_cal2", None),
    ("rel", "head, 2 deg + 2 %, self-calibrated", "l2sc_rel/l2h_cal2_sc", "l2sc_rel/l2h_cal2"),
    ("rel", "head, exact, self-calibrated", "l2sc_rel/l2h_sc", "l2_v3/l2h"),
    ("view", "three cameras, exact (earlier run)", "l2all_view/l2", None),
    ("view", "three cameras, 2 deg + 2 %", "l2allsc_view/l2_cal2", None),
    ("view", "three cameras, 2 deg + 2 %, self-calibrated", "l2allsc_view/l2_cal2_sc", "l2allsc_view/l2_cal2"),
    ("rel4", "three cameras, exact (earlier run)", "l2all_rel4/l2", None),
    ("rel4", "three cameras, 2 deg + 2 %", "l2allsc_rel4/l2_cal2", None),
    ("rel4", "three cameras, 2 deg + 2 %, self-calibrated", "l2allsc_rel4/l2_cal2_sc", "l2allsc_rel4/l2_cal2"),
]


def load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("pick_*.json")):
        r = json.loads(f.read_text())
        key = r["task"].replace("pick_", "") if r.get("task") else None
        ps = (r.get("perception_stats") or {}).get(key) or {}
        seen = [x for x in ps.get("log", []) if x.get("src") == "seen"]
        out[r["label"]] = dict(ok=bool(r["ok"]), err=[x["err_cm"] for x in seen], rot=[x["rot_deg"] for x in seen],
                               sc=ps.get("head_selfcal"))  # fmt: skip
    return out


def pair(a: dict, b: dict) -> str:
    keys = sorted(set(a) & set(b))
    won = sum(b[k]["ok"] and not a[k]["ok"] for k in keys)
    lost = sum(a[k]["ok"] and not b[k]["ok"] for k in keys)
    return f"+{won} / -{lost}, p worse {p_worse(lost, won):.3f}, p better {p_worse(won, lost):.3f}"


def main(out: Path) -> None:
    lines = ["| package | row | success | vs L0 | vs the row without it | object read, median cm / deg | self-calibration left, median / max deg |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for pkg, name, d, ref in ROWS:
        if not (A / d).is_dir():
            continue
        cur, l0 = load(A / d), load(L0[pkg])
        if len(cur) < 50:
            continue
        err = [e for v in cur.values() for e in v["err"]]
        rot = [e for v in cur.values() for e in v["rot"]]
        left = [
            v["sc"].get("angle_left_deg") for v in cur.values() if v["sc"] and v["sc"].get("angle_left_deg") is not None
        ]
        lines.append(f"| {pkg} | {name} | {sum(v['ok'] for v in cur.values())}/{len(cur)} (L0 {sum(v['ok'] for v in l0.values())}) "
                     f"| {pair(l0, cur)} | {pair(load(A / ref), cur) if ref else ''} | {np.median(err):.2f} / {np.median(rot):.1f} "
                     f"| {f'{np.median(left):.3f} / {np.max(left):.3f}' if left else ''} |")  # fmt: skip
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
