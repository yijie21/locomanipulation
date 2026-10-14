"""S1 table (TODO 25.4, plan §4.4): ScaleBFM's mode switches (bfm_switch.py: the switch against its no-switch twin) and
SONIC's planner <-> streamed handovers (sonic_switch.py), +-0.5 s around the switch.

    python3 scripts/experiments/controller_compare_20261013/switch_summary.py   # -> outputs/controller_compare_20261013/switch/s1.{md,json}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[3] / "outputs" / "controller_compare_20261013" / "switch"
KEYS = ("pelvis_move_cm", "pelvis_speed_max", "wrist_move_cm", "foot_slip_cm")


def main() -> int:
    rows = []
    for f in sorted(S.glob("bfm_*/switch*.json")):
        rows += json.loads(f.read_text())
    groups = defaultdict(lambda: {"switch": [], "twin": []})
    for r in rows:
        groups[(r["mode_a"], r["mode_b"] if not r["baseline"] else None, r["walk"])][
            "twin" if r["baseline"] else "switch"
        ].append(r)
    pairs = {}
    for r in rows:
        if not r["baseline"]:
            pairs.setdefault((r["mode_a"], r["mode_b"], r["walk"]), []).append(r)
    twins = defaultdict(list)
    for r in rows:
        if r["baseline"]:
            twins[(r["mode_a"], r["walk"])].append(r)
    table = []
    for (ma, mb, walk), rs in sorted(pairs.items()):
        tw = twins[(ma, walk)]
        line = dict(pair=f"{ma}->{mb}", walk=walk, n=len(rs), falls=sum(r["fell"] is not None for r in rs))
        for k in KEYS:
            v = [r[k] for r in rs if r.get(k) is not None]
            b = [r[k] for r in tw if r.get(k) is not None]
            line[k] = round(float(np.mean(v)), 2) if v else None
            line[k + "_twin"] = round(float(np.mean(b)), 2) if b else None
        table.append(line)
    sonic = json.loads((S / "sonic.json").read_text()) if (S / "sonic.json").exists() else {}
    lines = [
        "# S1 切换瞬态（TODO 25.4）",
        "",
        "切换后 0.5 s 内的骨盆位移、骨盆最大速度、手腕位移；ScaleBFM 每格同抖动的不切换对照（twin）并列。生成：`switch_summary.py`。",
        "",
        "| 切换 | 局 | 摔 | 骨盆位移 cm（对照） | 骨盆峰速 m/s（对照） | 手腕位移 cm（对照） | 脚滑 cm（对照） |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in table:
        lines.append(
            f"| ScaleBFM mode {t['pair']}{' 行走中' if t['walk'] else ''} | {t['n']} | {t['falls']} | {t['pelvis_move_cm']}（{t['pelvis_move_cm_twin']}） | "
            f"{t['pelvis_speed_max']}（{t['pelvis_speed_max_twin']}） | {t['wrist_move_cm']}（{t['wrist_move_cm_twin']}） | {t['foot_slip_cm']}（{t['foot_slip_cm_twin']}） |"
        )
    for name, r in sonic.items():
        lines.append(
            f"| SONIC {name}（L1 实局） | {r['n']} | {r['falls']} | {r['pelvis_move_cm'][0]}（最大 {r['pelvis_move_cm'][1]}） | "
            f"{r['pelvis_speed_max'][0]} | {r['wrist_move_cm'][0]}（最大 {r['wrist_move_cm'][1]}） | {r['foot_slide_cm'][0]} |"
        )
    (S / "s1.md").write_text("\n".join(lines) + "\n")
    (S / "s1.json").write_text(
        json.dumps(
            dict(bfm=table, sonic={k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in sonic.items()}),
            indent=1,
        )
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
