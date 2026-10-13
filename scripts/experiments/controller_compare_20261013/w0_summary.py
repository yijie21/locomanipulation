"""W0 summary (TODO 25.1): one table over every <out>/<test>/results.json, falls against the veto (>= 3 of 10).

python3 scripts/experiments/controller_compare_20261013/w0_summary.py outputs/controller_compare_20261013/w0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VETO_FALLS = 3
COLS = {
    "stand": ("pelvis_drift_cm", "yaw_drift_deg", "foot_move_cm", "foot_slip_cm", "pelvis_z"),
    "walk": ("stop_err_cm", "lat_cm", "yaw_deg", "pelvis_z"),
    "bend": (
        "left_err_cm",
        "right_err_cm",
        "left_err_deg",
        "right_err_deg",
        "pelvis_z_held",
        "torso_pitch_held",
        "waist_pitch_tau_p95",
        "box_moved_cm",
        "foot_move_cm",
        "foot_slip_cm",
    ),
}


def fmt(vals):
    if not vals:
        return "–"
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return f"{m:+.1f} ± {sd:.1f}" if len(vals) > 1 else f"{m:+.1f}"


def main(out: Path) -> int:
    lines = ["# W0 准入：ScaleBFM 在 mars 上（TODO 25.1）", ""]
    verdict = []
    for res in sorted(out.glob("*/results.json")):
        r = json.loads(res.read_text())
        test, rows = r["test"], r["rows"]
        groups = {}
        for row in rows:
            groups.setdefault(row.get("speed"), []).append(row)
        cols = [c for c in COLS[test] if any(c in row for row in rows)]
        lines += [
            f"## {res.parent.name}（{test}，mode {r['mode']}，{r['scene']}）",
            "",
            "| 组 | 局 | 摔 | " + " | ".join(cols) + " |",
            "|---|---|---|" + "---|" * len(cols),
        ]
        for key, rs in groups.items():
            falls = sum(1 for x in rs if x.get("fell") is not None)
            ok = [x for x in rs if x.get("fell") is None]
            cells = [fmt([x[c] for x in ok if x.get(c) is not None]) for c in cols]
            name = "–" if key is None else f"{key:.2f} m/s"
            lines.append(f"| {name} | {len(rs)} | {falls} | " + " | ".join(cells) + " |")
            verdict.append((f"{res.parent.name} {name}", len(rs), falls))
        lines.append("")
    lines += ["## 判定（任一组 10 次摔 ≥ 3 次即否决）", ""]
    for name, n, falls in verdict:
        lines.append(f"- {name}：{falls}/{n} {'**否决**' if falls >= VETO_FALLS else '过'}")
    (out / "w0.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
