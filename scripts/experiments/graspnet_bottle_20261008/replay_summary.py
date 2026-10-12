"""TODO 23.2.6: aggregate replay_plan.py's per-object JSON -> planning seconds per mode (off / on = direct presort / staged presort), whether
the plan still succeeds, how often the chosen grasp changes, and the lift rate of the recorded rounds whose chosen
grasp the presort keeps. Prints a Markdown table; writes outputs/graspnet_bottle_20261008/cache/replay/summary.md.

    python3 scripts/experiments/graspnet_bottle_20261008/replay_summary.py
"""

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "outputs" / "graspnet_bottle_20261008" / "cache" / "replay"
rows = [r for f in sorted(glob.glob(str(ROOT / "obj_*.json"))) for r in json.loads(Path(f).read_text())]
lines = [
    f"# IK presort replay ({len(rows)} rounds)",
    "",
    "| mode | plan s median | p90 | max | plan ok | chosen = recorded | rejected median | tier 0 chosen |",
    "|---|---|---|---|---|---|---|---|",
]
for mode in ("off", "on", "staged", "feasfirst"):
    secs = [r[mode]["seconds"] for r in rows]
    ok = sum(1 for r in rows if r[mode]["ok"])
    same = sum(1 for r in rows if r[mode]["row"] == r["recorded"]["row"] and r[mode]["ok"])
    rej = st.median(r[mode]["rejected"] for r in rows)
    t0 = sum(1 for r in rows if r[mode].get("tier") == 0)
    lines.append(
        f"| {mode} | {st.median(secs):.1f} | {sorted(secs)[int(0.9 * len(secs))]:.1f} | {max(secs):.1f} | {ok}/{len(rows)} | {same} | {rej} | {t0} |"
    )
rec = [r["recorded"]["seconds"] for r in rows if r["recorded"]["seconds"]]
lines.append(
    f"| recorded | {st.median(rec):.1f} | {sorted(rec)[int(0.9 * len(rec))]:.1f} | {max(rec):.1f} | | | {st.median(r['recorded']['rejected'] for r in rows)} | |"
)
pres = [r["on"]["presort"]["seconds"] for r in rows if r["on"].get("presort")]
feas = [r["on"]["presort"]["feasible"] / max(1, r["candidates"]) for r in rows if r["on"].get("presort")]
lines += [
    "",
    f"- presort cost median {st.median(pres):.2f} s; feasible share median {st.median(feas):.2f}",
    f"- replay `off` reproduces the recorded choice in {sum(1 for r in rows if r['off']['row'] == r['recorded']['row'])}/{len(rows)} rounds (pelvis roll/pitch taken as zero)",
]
for mode in ("on", "staged", "feasfirst"):
    kept = [r for r in rows if r[mode]["row"] == r["recorded"]["row"]]
    changed = [r for r in rows if r[mode]["row"] != r["recorded"]["row"]]
    lines.append(
        f"- `{mode}`: keeps the recorded grasp in {len(kept)} rounds (recorded lift {sum(1 for r in kept if r['recorded']['lifted'])}/{len(kept)}), changes it in {len(changed)} (those rounds lifted {sum(1 for r in changed if r['recorded']['lifted'])}/{len(changed)} with the recorded grasp)"
    )
(ROOT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
