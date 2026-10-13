"""wbc_compare step 6 (TODO 24.6): bfm_attitude_probe.py's shards -> ScaleBFM's attitude delivery envelope.
Merges <dir>/attitude_*.json; per grid attitude (elevation, yaw deviation, roll) the position error norm and the
palm rotation error are averaged over the probed positions (a fall or a missing position leaves the attitude
undelivered). `deliverable` = every position measured, mean position error <= ERR_CM and mean rotation error
<= ROT_DEG. ERR_CM sits at the hand program's servo reach: the lateral correction caps at 4 cm and the tracking
feedback clips at 8 cm, and the 24.2 probe's single attitude came in at 2.7 cm median / 4.5 cm max.
    python3 scripts/experiments/wbc_compare_20261012/attitude_summary.py outputs/wbc_compare_20261012/probe/scalebfm_attitude
        -> <dir>/envelope.json (probe_cells.Envelope reads it), <dir>/envelope.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_cells as C  # noqa: E402

ERR_CM = 5.0
ROT_DEG = 15.0


def main(argv=None) -> int:
    root = Path((argv or sys.argv[1:])[0])
    shards = sorted(root.glob("attitude_*.json"))
    if not shards:
        raise SystemExit(f"no attitude_*.json under {root}")
    cells, meta = [], None
    for f in shards:
        d = json.loads(f.read_text())
        meta = meta or d
        cells.extend(d["cells"])
    by = defaultdict(list)
    for c in cells:
        by[(c["elev"], c["yaw_dev"], c["roll"])].append(c)
    n_pos = len(meta["positions"])
    rows = []
    for att in C.attitudes():
        key = (att["elev"], att["yaw_dev"], att["roll"])
        got = by.get(key, [])
        meas = [c for c in got if "err_norm_cm" in c]
        row = dict(
            att,
            probed=len(got),
            measured=len(meas),
            fell=sum(c["fell_at_s"] is not None for c in got),
            err_cm=[c.get("err_norm_cm") for c in got],
            rot_deg=[c.get("rot_err_deg") for c in got],
        )
        if meas:
            row.update(
                err_mean_cm=round(float(np.mean([c["err_norm_cm"] for c in meas])), 2),
                err_max_cm=round(float(np.max([c["err_norm_cm"] for c in meas])), 2),
                rot_mean_deg=round(float(np.mean([c["rot_err_deg"] for c in meas])), 1),
                bias_heading_cm=[round(float(v), 2) for v in np.mean([c["err_heading_cm"] for c in meas], axis=0)],
                hold_max_cm=round(float(np.max([c["hold_max_cm"] or 0.0 for c in meas])), 2),
            )
        row["deliverable"] = bool(
            len(meas) == n_pos and row.get("err_mean_cm", 1e9) <= ERR_CM and row.get("rot_mean_deg", 1e9) <= ROT_DEG
        )
        rows.append(row)
    probed = [r for r in rows if r["measured"] == n_pos]
    sens = {
        f"err<={e:.0f}cm": sum(
            r["measured"] == n_pos and r["err_mean_cm"] <= e and r["rot_mean_deg"] <= ROT_DEG for r in rows
        )
        for e in (3.0, 4.0, 5.0, 6.0, 8.0)
    }
    by_elev = {
        str(e): dict(
            probed=sum(r["measured"] == n_pos for r in rows if r["elev"] == e),
            deliverable=sum(r["deliverable"] for r in rows if r["elev"] == e),
        )
        for e in C.ELEVATIONS
    }
    env = dict(
        thresholds=dict(err_cm=ERR_CM, rot_deg=ROT_DEG, grid_max_deg=C.GRID_MAX_DEG),
        positions=meta["positions"],
        elevations=list(C.ELEVATIONS),
        yaw_devs=list(C.YAW_DEVS),
        rolls=list(C.ROLLS),
        source=dict(object=meta["object"], shards=len(shards), cells=len(cells), hold_s=meta["hold_s"]),
        summary=dict(
            attitudes=len(rows),
            measured=len(probed),
            deliverable=sum(r["deliverable"] for r in rows),
            fell_cells=sum(c["fell_at_s"] is not None for c in cells),
            sensitivity=sens,
            by_elevation=by_elev,
        ),
        attitudes=rows,
    )
    (root / "envelope.json").write_text(json.dumps(env, indent=1))
    zs = ", ".join(f"{p['z']:.2f}" for p in meta["positions"])
    lines = [
        "# ScaleBFM attitude delivery envelope (TODO 24.6)",
        "",
        f"Object {meta['object']}, {len(cells)} cells over {n_pos} positions (z {zs}), "
        f"hold {meta['hold_s']} s; deliverable = mean position error <= {ERR_CM:.0f} cm and mean rotation error "
        f"<= {ROT_DEG:.0f} deg over every position.",
        "",
        f"Attitudes {len(rows)}, measured {len(probed)}, deliverable {env['summary']['deliverable']}, cells that fell "
        f"{env['summary']['fell_cells']}. Sensitivity: " + ", ".join(f"{k} {v}" for k, v in sens.items()) + ".",
        "",
        "By elevation: " + ", ".join(f"{e} deg {v['deliverable']}/{v['probed']}" for e, v in by_elev.items()) + ".",
        "",
    ]
    for roll in C.ROLLS:
        lines += [
            f"## roll {roll} ({'jaw left, normal up' if roll == 0 else 'jaw right, normal down'}): mean position error cm (rotation error deg), **bold** = deliverable",
            "",
        ]
        lines.append("| elev \\ yaw | " + " | ".join(str(y) for y in C.YAW_DEVS) + " |")
        lines.append("|---|" + "---|" * len(C.YAW_DEVS))
        for e in C.ELEVATIONS:
            cellstr = []
            for y in C.YAW_DEVS:
                r = next((r for r in rows if (r["elev"], r["yaw_dev"], r["roll"]) == (e, y, roll)), None)
                if r is None:
                    cellstr.append("=" if e == 90 else "-")
                elif r["measured"] < n_pos:
                    cellstr.append("fell" if r["fell"] else "-")
                else:
                    s = f"{r['err_mean_cm']:.1f} ({r['rot_mean_deg']:.0f})"
                    cellstr.append(f"**{s}**" if r["deliverable"] else s)
            lines.append(f"| {e} | " + " | ".join(cellstr) + " |")
        lines.append("")
    lines.append("At 90 deg elevation roll 180 (=) repeats roll 0 at yaw + 180.")
    (root / "envelope.md").write_text("\n".join(lines) + "\n")
    print(
        f"{len(rows)} attitudes, {len(probed)} measured, {env['summary']['deliverable']} deliverable "
        f"(sensitivity {sens}); wrote {root / 'envelope.json'} and envelope.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
