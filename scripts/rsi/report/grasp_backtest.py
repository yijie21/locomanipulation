"""TODO 33.15 / 33.18 backtest: the failure classifier (failure_class.py) and the jaw fit check (graspability.py) against
the records we already have, before the loop relies on them. Read-only; loads the scene's physics model once.

Data:
- 32.12 iteration 1: the 90 rounds of its sealed (50) and acceptance (40) pools (records only);
- the A800 500: iteration 1 on 500 fresh starts (records only; the traces stayed on the A800);
- the local reruns with traces: the sealed pool under iterations 1, 6 and 9 (grasp_atomic page v3).

It reports the classes against the hand analysis of 2026-10-22 (the same rules over the step records), and how well
"the jaw does not fit at the approach used" predicts the rounds that reached their grasp and still failed.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/rsi/report/grasp_backtest.py
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts" / "rsi"), str(REPO / "src")]

import failure_class as FC  # noqa: E402

from locomani.tasks.scene import graspability as G  # noqa: E402

OUT = REPO / "outputs" / "scene_rsi_20261016" / "grasp_backtest"
OLD = Path("/workspace/code/rsi-scene/ga-confirm-025-20261021/outputs/scene_rsi_20261016_hidden/grasp_atomic")
NEW = Path("/workspace/code/rsi-scene/ga-cont-b-20261022/outputs/scene_rsi_20261016_hidden/grasp_atomic")
A800 = REPO / "outputs" / "scene_rsi_20261016" / "a800_20261022"
V3 = REPO / "outputs" / "scene_rsi_20261016" / "grasp_atomic_media" / "v3" / "rerun"
DATA = [  # (name, records, pool, traces)
    ("32.12 it1 sealed", OLD / "sealed" / "01_971864dc7013", OLD / "pool_sealed", False),
    ("32.12 it1 accept", OLD / "gate" / "01_971864dc7013" / "candidate", OLD / "pool_accept", False),
    ("A800 it1 500", A800 / "rounds_bulk", A800 / "pool_bulk", False),
    ("rerun it1 (traces)", V3 / "01_971864dc7013", NEW / "pool_sealed", True),
    ("rerun it6 (traces)", V3 / "06_49632436cd0b", NEW / "pool_sealed", True),
    ("rerun it9 (traces)", V3 / "09_011262a9020f", NEW / "pool_sealed", True),
]
WIDE_OBJECTS = ("Mug_2", "DishSoap_1")  # the hand analysis' "wider than the jaw" objects


def hand(rec: dict) -> str:
    """The 2026-10-22 hand analysis' rules, over the step records."""
    if rec["ok"]:
        return "ok"
    s = {x["step"]: x for x in rec.get("steps") or []}
    if rec.get("fell"):
        return "fell"
    if (s.get("walk", {}).get("stop_err_cm") or 0) > 5:
        return "walk_off"
    t = rec["task"].removeprefix("pick_")
    pre = s.get("pregrasp", {}).get("jaw_err_cm") or 0
    if t in WIDE_OBJECTS and pre <= 5:
        return "too_wide"
    if s.get("stance", {}).get("flat"):
        return "pinch_reach"
    if pre > 5:
        return "side_reach"
    if s.get("grip", {}).get("ok"):
        return "slip"
    if "grip" in s:
        return "miss_near"
    return "other"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = G.StaticScene(REPO / "outputs" / "scene_discovery" / "adt_apartment_mars")
    scene.fk = FC.FK(scene.m)
    rows = []
    for name, rdir, pool, traces in DATA:
        for f in sorted(glob.glob(f"{rdir}/*.json")):
            if os.path.basename(f).startswith("_"):
                continue
            rec = json.load(open(f))
            lab = rec["label"]
            key = lab.removeprefix("pick_").rsplit("_s", 1)[0]
            start = json.loads((pool / f"{lab}.json").read_text())["start"]
            scene.place(start["placements"])
            z = np.load(f[:-5] + ".npz", allow_pickle=True) if traces and os.path.exists(f[:-5] + ".npz") else None
            cls, ev = FC.classify(rec, z, scene, key)
            fit = ev.get("fit") or {}
            arrived = (not rec.get("fell") and (ev.get("stop_err_cm") or 0) <= FC.WALK_OFF_CM
                       and ev.get("reach_err_cm") is not None and ev["reach_err_cm"] <= FC.REACH_MISS_CM)  # fmt: skip
            rows.append(dict(data=name, label=lab, obj=key, ok=rec["ok"], hand=hand(rec), auto=cls, verdict=fit.get("verdict"),
                             width_m=fit.get("width_m"), margin_m=fit.get("margin_m"), arrived=arrived,
                             pinch=bool(ev.get("pinch")), empty=(ev.get("opening_at_grip_cm") or 9) < FC.EMPTY_CM,
                             source=ev.get("source")))  # fmt: skip
    (OUT / "rows.json").write_text(json.dumps(rows, indent=0))
    lines = ["# Failure classifier and jaw-fit backtest (TODO 33.15, 33.18; 2026-10-22)", "",
             "Read-only over existing records; `scripts/rsi/report/grasp_backtest.py`, rows in `rows.json`.", ""]  # fmt: skip
    groups = [("32.12 iteration 1, 90 rounds (sealed + acceptance; records only)", ("32.12 it1 sealed", "32.12 it1 accept")),
              ("A800, iteration 1 on 500 fresh starts (records only)", ("A800 it1 500",)),
              ("Local reruns of the sealed pool with traces (iterations 1, 6, 9; 150 rounds)",
               ("rerun it1 (traces)", "rerun it6 (traces)", "rerun it9 (traces)"))]  # fmt: skip
    norm = lambda c: "fell" if c in ("fell", "fell_walk") else c  # noqa: E731
    for title, names in groups:
        R = [r for r in rows if r["data"] in names]
        F = [r for r in R if not r["ok"]]
        lines += [f"## {title}", "", f"{len(R)} rounds, {len(F)} failed.", ""]
        hc, ac = collections.Counter(r["hand"] for r in F), collections.Counter(norm(r["auto"]) for r in F)
        cats = sorted(set(hc) | set(ac), key=lambda c: -(hc[c] + ac[c]))
        lines += ["| class | hand rules | classifier | same round |", "|---|---|---|---|"]
        for c in cats:
            same = sum(1 for r in F if r["hand"] == c and norm(r["auto"]) == c)
            lines.append(f"| {c} | {hc[c]} | {ac[c]} | {same} |")
        agree = sum(1 for r in F if r["hand"] == norm(r["auto"]))
        lines += ["", f"Same class on {agree} of {len(F)} failures.", ""]
        conf = collections.Counter((r["hand"], norm(r["auto"])) for r in F if r["hand"] != norm(r["auto"]))
        if conf:
            lines.append(
                "Disagreements (hand -> classifier): "
                + ", ".join(f"{a} -> {b} {n}" for (a, b), n in conf.most_common())
            )
            lines.append("")
        side = [r for r in R if r["arrived"] and not r["pinch"] and r["verdict"] in ("fits", "tight", "too_wide")]
        lines += [f"Jaw fit at the approach used, side grasps that reached their grasp (walk and reach within 5 cm): "
                  f"{len(side)} rounds.", ""]  # fmt: skip
        for label, bad in (
            ("not 'fits' (tight or too wide)", ("tight", "too_wide")),
            ("'too_wide' only", ("too_wide",)),
        ):
            tp = sum(1 for r in side if r["verdict"] in bad and not r["ok"])
            fp = sum(1 for r in side if r["verdict"] in bad and r["ok"])
            fn = sum(1 for r in side if r["verdict"] not in bad and not r["ok"])
            tn = sum(1 for r in side if r["verdict"] not in bad and r["ok"])
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec_ = tp / (tp + fn) if tp + fn else float("nan")
            lines.append(f"- predicting failure from {label}: precision {tp}/{tp + fp} = {prec:.2f}, recall {tp}/{tp + fn} = "
                         f"{rec_:.2f} (fits and failed {fn}, misfit and held {fp}, fits and held {tn})")  # fmt: skip
        lines += ["", "| object | misfit, failed | misfit, held | fits, failed | fits, held |", "|---|---|---|---|---|"]
        for o in sorted({r["obj"] for r in side}):
            S = [r for r in side if r["obj"] == o]
            c = lambda bad, ok: sum(1 for r in S if (r["verdict"] in ("tight", "too_wide")) == bad and r["ok"] == ok)  # noqa: E731
            lines.append(f"| {o} | {c(True, False)} | {c(True, True)} | {c(False, False)} | {c(False, True)} |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
