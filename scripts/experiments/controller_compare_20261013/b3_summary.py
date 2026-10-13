"""B3 summary (TODO 25.2.2): deliverable cells per controller and attitude from <dir>/{sonic,scalebfm}/b3.json.

python3 scripts/experiments/controller_compare_20261013/b3_summary.py outputs/controller_compare_20261013/bottle/b3
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main(root: Path) -> int:
    runs = {}
    for c in ("sonic", "scalebfm"):
        f = root / c / "b3.json"
        if f.exists():
            runs[c] = json.loads(f.read_text())
    if not runs:
        print("no b3.json under", root)
        return 1
    atts = next((r["attitudes"] for r in runs.values() if "attitudes" in r), None)
    if atts is None:  # an older SONIC record without the list: rebuild it from the cells
        seen = {}
        for r in runs.values():
            for x in r["cells"]:
                seen.setdefault(
                    x["attitude"], dict(id=x["attitude"], name=f"elev {x['elev']} yaw {x['yaw_dev']} roll {x['roll']}")
                )
        atts = [seen[k] for k in sorted(seen)]
    lines = [
        "# B3 可达包络：各自走进去的站位，27 位置 × 4 姿态（TODO 25.2.2）",
        "",
        "可交付 = 位置误差 ≤ 5 cm 且姿态误差 ≤ 15°（24.6 的伺服量程）。中位误差只算探到的格（没跳过、没摔）。",
        "",
        "| 姿态 | "
        + " | ".join(f"{c} 可交付" for c in runs)
        + " | "
        + " | ".join(f"{c} 中位 cm / °" for c in runs)
        + " |",
        "|---|" + "---|" * (2 * len(runs)),
    ]
    for att in atts:
        counts, meds = [], []
        for r in runs.values():
            rows = [x for x in r["cells"] if x.get("attitude") == att["id"]]
            counts.append(f"{sum(bool(x.get('deliverable')) for x in rows)}/{len(rows)}")
            probed = [x for x in rows if x.get("err_norm_cm") is not None]
            meds.append(
                "–"
                if not probed
                else f"{statistics.median(x['err_norm_cm'] for x in probed):.1f} / "
                f"{statistics.median(x['rot_err_deg'] for x in probed):.0f}"
            )
        lines.append(f"| {att['name']} | " + " | ".join(counts) + " | " + " | ".join(meds) + " |")
    total = [f"{sum(bool(x.get('deliverable')) for x in r['cells'])}/{len(r['cells'])}" for r in runs.values()]
    lines.append("| 合计 | " + " | ".join(total) + " | " + " | ".join("" for _ in runs) + " |")
    lines += ["", "站位："]
    for c, r in runs.items():
        gap = r.get("pelvis_gap_cm")
        extra = "" if gap is None else f"，骨盆到桌边 {gap:.1f} cm"
        lines.append(f"- {c}：{[round(v, 3) for v in r['stance_xy']]}，朝向 {r['stance_yaw_deg']:.1f}°{extra}")
    (root / "b3.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
