"""The stance runs side by side on the 345 core goals, eyes closed (2026-11-03): the report's model and the retrained ones
staying (wrist_invert.py runs), and with the wrist-aware stance (stance_wrist.py runs), paired by goal. Subsets: the goals
whose stance turned, and whose stance only stepped, in the first stance run with turns (stance_wrist_turn).

    python3 scripts/experiments/reachwm_20261030/stance_report.py
Writes outputs/reachwm_20261030/invert/stance_compare.md. Runs whose directory is missing are left out. The "position guard"
rows are computed from the runs themselves (guarded): a move counts only if the model predicted the position after it no
worse than staying's by GUARD_CM, judged after the move. The "guard in search" run (stance_wrist.py with STANCE_GUARD_CM)
applies it before moving, to every candidate, and takes the best that passes. The "look again after the move" rows
(look_again) are a rule a robot can run: the unguarded move, then the wrist-first pick unless its prediction from where the
robot stands fails the guard, else the position-only pick there (the stance run executed every pick from the same move).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("REACHWM_GOAL_REF", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import stance_wrist as SW  # noqa: E402

INV = SW.IE.OUT
NAMES = {"pos": "position only", "pose3g": "wrist first", "w20": "1 cm = 20 deg"}
# (label, kind, dir, prior run of the same model, the same model's no-move run)
RUNS = [
    ("report model, stay", "stay", "wrist_invert2", "", "h_w*_pos_pose3g_w20_cap15.json"),
    ("+ post-walk data, stay", "stay", "wrist_turn", "", "h_w*_pos_pose3g_w20.json"),
    ("+ post-walk data, stance with turns", "stance", "stance_wrist_turn", "", "wrist_turn/h_w*_pos_pose3g_w20.json"),
    (
        "+ post-walk data, stance steps only",
        "stance",
        "stance_wrist_steps",
        "stance_wrist_turn",
        "wrist_turn/h_w*_pos_pose3g_w20.json",
    ),
    (
        "+ post-walk data, stance with turns, guard in search",
        "stance",
        "stance_wrist_guard",
        "stance_wrist_turn",
        "wrist_turn/h_w*_pos_pose3g_w20.json",
    ),
    ("+ stance data, stay", "stay", "wrist_onp2", "", "h_w*_pos_pose3g_w20.json"),
    ("+ stance data, stance with turns", "stance", "stance_wrist_onp2", "", "wrist_onp2/h_w*_pos_pose3g_w20.json"),
]


GUARD_CM = 0.2


def guarded(stay: dict, stance: dict, eps: float = GUARD_CM) -> dict:
    """The stance run's move kept only where the model predicted the wrist-first reach's position after it no worse than
    staying's by eps cm, else the goal stays (the no-move run's rows). The rule reads predictions only. A planner that
    takes the best candidate passing it, instead of staying, is not run here."""
    out = {}
    for (g, m), r in stay.items():
        mv, st = stance.get((g, "pose3g"), {}), stay.get((g, "pose3g"))
        ok = mv.get("moved") and (g, m) in stance and mv["pred_err"] <= st["pred_err"] + eps
        out[(g, m)] = stance[(g, m)] if ok else dict(r, moved=False)
    return out


def look_again(stay: dict, stance: dict, eps: float = GUARD_CM, fallback: str = "pos") -> dict:
    """The stance run with the wrist-first pick replaced, on a goal that moved, by the fallback pick at the same stance where
    the wrist-first pick's predicted position error after the move is above staying's (before it) by more than eps."""
    out = {}
    for (g, m), r in stay.items():
        mv = stance.get((g, "pose3g"), {})
        fail = m == "pose3g" and mv.get("moved") and mv["pred_err"] > stay[(g, "pose3g")]["pred_err"] + eps
        out[(g, m)] = stance[(g, fallback)] if fail else stance.get((g, m), r)
    return out


def load() -> dict:
    out = {}
    for label, kind, d, prior, base in RUNS:
        if not (INV / d).is_dir():
            continue
        if kind == "stay":
            out[label] = SW.base_rows(str(INV / d / base))
        else:
            out[label] = SW.run_rows(INV / d, str(INV / prior) if prior else "", SW.base_rows(str(INV / base)))[0]
    for model in ("+ post-walk data", "+ stance data"):
        if f"{model}, stance with turns" in out:
            out[f"{model}, stance with turns, guard after the fact"] = guarded(
                out[f"{model}, stay"], out[f"{model}, stance with turns"]
            )
            out[f"{model}, stance with turns, look again after the move"] = look_again(
                out[f"{model}, stay"], out[f"{model}, stance with turns"]
            )
    return out


def main() -> None:
    R = load()
    turn = R["+ post-walk data, stance with turns"]
    gids = sorted({g for g, _ in R["report model, stay"]})
    turned = sorted({g for (g, m), r in turn.items() if r.get("moved") and r.get("turn_deg", 0.0) != 0.0})
    stepped = sorted({g for (g, m), r in turn.items() if r.get("moved") and r.get("turn_deg", 0.0) == 0.0})
    L = [f"{len(gids)} core goals, eyes closed, paired by goal; subsets from the first stance run with turns: "
         f"{len(turned)} turned, {len(stepped)} stepped only", ""]  # fmt: skip
    for sub, sel in (
        ("all goals", gids),
        ("goals that turned there", turned),
        ("goals that only stepped there", stepped),
    ):
        L += [f"## {sub}", "", "| run | inversion | n | moved | position median cm | >2 cm (n) | p90 cm | wrist median deg | <10 deg | >30 deg (n) | fell |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for m in ("pos", "pose3g", "w20"):
            for label, rows in R.items():
                rs = [rows[(g, m)] for g in sel if (g, m) in rows]
                ok = [r for r in rs if not r["fell"]]
                e, a = np.array([r["err"] for r in ok]), np.array([r["att_deg"] for r in ok])
                mv = sum(1 for r in rs if r.get("moved"))
                L.append(f"| {label} | {NAMES[m]} | {len(rs)} | {mv} | {np.median(e):.2f} | {int((e > 2).sum())} | {np.percentile(e, 90):.2f} | "
                         f"{np.median(a):.1f} | {(a < 10).mean():.2f} | {int((a > 30).sum())} | {len(rs) - len(ok)} |")  # fmt: skip
        L.append("")
    (INV / "stance_compare.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
