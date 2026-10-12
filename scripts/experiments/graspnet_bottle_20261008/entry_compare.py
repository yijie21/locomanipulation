"""TODO 23.4.4: round 0 of the entry-lever batches side by side -- the entry gate (ENTRY_GATE, tags _gateA/_gateB)
and the height-first candidate order (PREFER_HIGH, _highA/_highB) -- outcome classes, the gate's hold seconds, the
chosen grasp's height above the table, and for the height-first batches how many objects changed grasp against the
gate batches (whose grasp choice is the default order's). Writes outputs/graspnet_bottle_20261008/cache/entry_compare.txt.

    python3 scripts/experiments/graspnet_bottle_20261008/entry_compare.py [_gateA _gateB _highA _highB]
"""

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "outputs/graspnet_bottle_20261008"
objects = json.loads((ROOT / "objects.json").read_text())["objects"]
TAGS = sys.argv[1:] or ["_gateA", "_gateB", "_highA", "_highB"]


def classify(rec):
    if rec is None:
        return "crash"
    if rec.get("ok"):
        return "lifted"
    why = rec.get("why", "")
    for key, name in (
        ("knocked", "knocked"), ("no bilateral", "no_bilateral"), ("timed out", "entry_timeout"), ("no path", "no_path"),
        ("no cached grasp", "no_candidate"), ("fell", "fell"),
    ):  # fmt: skip
        if key in why:
            return name
    return "lost_in_lift" if rec.get("rise_m") is not None else "other"


recs = {}
for tag in TAGS:
    for o in objects:
        f = REPO / "outputs/runs" / f"gnb_obj_{o['id']:03d}{tag}" / "grasp_planned/round_00_ik/record.json"
        if f.exists():
            recs[tag, o["id"]] = json.loads(f.read_text())
lines = [
    "| tag | n | lifted | classes | gate hold s med/p75 | grasp height cm med | entry err cm med |",
    "|---|---|---|---|---|---|---|",
]
for tag in TAGS:
    rows = [r for (t, _), r in recs.items() if t == tag]
    if not rows:
        continue
    kinds = Counter(classify(r) for r in rows)
    held = [r["entry"]["gate_held_s"] for r in rows if r.get("entry") and "gate_held_s" in r["entry"]]
    hs = [100 * (r["grasp"]["tool"][2] - r["table_top"]) for r in rows if r.get("grasp")]
    errs = [100 * r["entry"]["error_m"] for r in rows if r.get("entry")]
    h = f"{st.median(held):.2f}/{sorted(held)[int(0.75 * (len(held) - 1))]:.2f}" if held else "-"
    lines.append(
        f"| {tag} | {len(rows)} | {kinds.get('lifted', 0)} | {dict(kinds)} | {h} | {st.median(hs):.1f} | {st.median(errs):.1f} |"
        if hs and errs
        else f"| {tag} | {len(rows)} | {kinds.get('lifted', 0)} | {dict(kinds)} | {h} | - | - |"
    )
base = [t for t in TAGS if "gate" in t]
for tag in [t for t in TAGS if "high" in t]:
    for b in base:
        pairs = [
            (recs[b, o["id"]], recs[tag, o["id"]]) for o in objects if (b, o["id"]) in recs and (tag, o["id"]) in recs
        ]
        both = [(x, y) for x, y in pairs if x.get("grasp") and y.get("grasp")]
        changed = [(x, y) for x, y in both if x["grasp"].get("row") != y["grasp"].get("row")]
        same = [(x, y) for x, y in both if x["grasp"].get("row") == y["grasp"].get("row")]
        dh = [100 * (y["grasp"]["tool"][2] - x["grasp"]["tool"][2]) for x, y in changed]
        lines.append(
            f"- {tag} vs {b}: {len(pairs)} objects, grasp changed {len(changed)}/{len(both)} (height delta median "
            f"{st.median(dh) if dh else 0:+.1f} cm); lifted on changed {sum(y.get('ok', False) for _, y in changed)} vs "
            f"{sum(x.get('ok', False) for x, _ in changed)} in {b}; on unchanged {sum(y.get('ok', False) for _, y in same)} vs "
            f"{sum(x.get('ok', False) for x, _ in same)}"
        )
out = "\n".join(lines) + "\n"
(ROOT / "cache" / "entry_compare.txt").write_text(out)
print(out)
