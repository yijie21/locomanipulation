"""wbc_compare step 2 (TODO 24.2): both controllers' probe.json -> the feed-forward bias table and bias.json.

For each controller: the mean delivered-minus-commanded tool error over the probe cells (the constant the hand
program subtracts from its commands, decision 2), its spread (per-axis standard deviation over cells), and what is
left after subtracting the constant (median / 90th percentile / max of the residual norm). Both over all cells and
over the nine CENTRAL cells (the object's lateral). SONIC's error is also split into IK residual / arm tracking /
pelvis motion (sonic_probe.py). The per-cell table pairs the controllers cell by cell.

    python3 scripts/experiments/wbc_compare_20261012/probe_summary.py outputs/wbc_compare_20261012/probe
        -> <dir>/bias.json, <dir>/probe.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AXES = ("forward", "left", "up")


def measured(rows: list[dict]) -> list[dict]:
    """Cells the hand program would execute: measured, and (SONIC) ending inside the IK tolerance."""
    return [r for r in rows if "err_heading_cm" in r and r.get("reachable", True)]


def stats(rows: list[dict]) -> dict | None:
    rows = measured(rows)
    err = np.array([r["err_heading_cm"] for r in rows])
    if len(err) == 0:
        return None
    bias = err.mean(axis=0)
    resid = np.linalg.norm(err - bias, axis=1)
    raw = np.linalg.norm(err, axis=1)
    out = dict(
        cells=int(len(err)),
        bias_cm=[round(float(v), 2) for v in bias],
        spread_sd_cm=[round(float(v), 2) for v in err.std(axis=0)],
        raw_norm_cm=dict(
            median=round(float(np.median(raw)), 2),
            p90=round(float(np.quantile(raw, 0.9)), 2),
            max=round(float(raw.max()), 2),
        ),
        residual_norm_cm=dict(
            median=round(float(np.median(resid)), 2),
            p90=round(float(np.quantile(resid, 0.9)), 2),
            max=round(float(resid.max()), 2),
        ),
        rot_err_deg=dict(
            median=round(float(np.median([r["rot_err_deg"] for r in rows if "rot_err_deg" in r])), 1),
            max=round(float(max(r["rot_err_deg"] for r in rows if "rot_err_deg" in r)), 1),
        ),
        hold_max_cm=round(float(max(r["hold_max_cm"] for r in rows if "hold_max_cm" in r)), 2),
        pelvis_err_cm_max=round(
            float(max(np.linalg.norm(r["pelvis_err_cm"]) for r in rows if "pelvis_err_cm" in r)), 2
        ),
        pelvis_yaw_err_deg_max=round(
            float(max(abs(r["pelvis_yaw_err_deg"]) for r in rows if "pelvis_yaw_err_deg" in r)), 2
        ),
    )
    for part in ("ik_residual", "tracking", "pelvis_motion"):
        key = f"{part}_heading_cm"
        if all(key in r for r in rows if "err_heading_cm" in r):
            v = np.array([r[key] for r in rows if key in r])
            out[f"{part}_mean_cm"] = [round(float(x), 2) for x in v.mean(axis=0)]
            out[f"{part}_norm_cm_max"] = round(float(np.linalg.norm(v, axis=1).max()), 2)
    return out


def fmt(v, nd=1):
    return "-" if v is None else "[" + ", ".join(f"{x:+.{nd}f}" for x in v) + "]"


def main(argv=None) -> int:
    root = Path((argv or sys.argv[1:])[0])
    probes = {}
    for name in ("sonic", "scalebfm"):
        f = root / name / "probe.json"
        if f.exists():
            probes[name] = json.loads(f.read_text())
    if not probes:
        raise SystemExit(f"no probe.json under {root}/{{sonic,scalebfm}}")
    bias = {}
    lines = ["# Open-loop right-hand probe: feed-forward bias per controller (wbc_compare step 2, TODO 24.2)", ""]
    lines.append(
        "Error = delivered tool - commanded tool in the heading frame (forward, left, up), cm; bias = mean over cells; "
        "residual = |error - bias|. The hand program (24.3) commands target - bias on each controller."
    )
    lines.append("")
    lines.append(
        "| controller | cells | bias f/l/u cm | spread sd cm | raw norm med / p90 / max | residual med / p90 / max | rot err med / max deg | pelvis max cm / deg |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, pr in probes.items():
        rows = pr["cells"]
        for label, sub in (("all", rows), ("central", [r for r in rows if r.get("central")])):
            s = stats(sub)
            if s is None:
                continue
            bias.setdefault(name, {})[label] = dict(s, bias_m=[round(v / 100, 4) for v in s["bias_cm"]])
            lines.append(
                f"| {name} ({label}) | {s['cells']} | {fmt(s['bias_cm'])} | {fmt(s['spread_sd_cm'])} | "
                f"{s['raw_norm_cm']['median']} / {s['raw_norm_cm']['p90']} / {s['raw_norm_cm']['max']} | "
                f"{s['residual_norm_cm']['median']} / {s['residual_norm_cm']['p90']} / {s['residual_norm_cm']['max']} | "
                f"{s['rot_err_deg']['median']} / {s['rot_err_deg']['max']} | {s['pelvis_err_cm_max']} / {s['pelvis_yaw_err_deg_max']} |"
            )
        skipped = [r["id"] for r in rows if "skipped" in r]
        unreachable = [r["id"] for r in rows if "skipped" not in r and not r.get("reachable", True)]
        extra = f"; fell at {pr['fell_at_s']} s" if pr.get("fell_at_s") else ""
        extra += f"; skipped cells {skipped}" if skipped else ""
        extra += f"; IK outside tolerance (excluded from the bias) {unreachable}" if unreachable else ""
        extra += f"; not ok: {pr['why']}" if pr.get("ok") is False else ""
        lines.append(
            f"| {name} notes | stance {pr.get('stance_xy')} yaw {pr.get('stance_yaw_deg')} deg{extra} | | | | | | |"
        )
    if "sonic" in bias:
        for label in bias["sonic"]:
            s = bias["sonic"][label]
            if "ik_residual_mean_cm" in s:
                lines.append("")
                lines.append(
                    f"SONIC ({label}) error split, mean f/l/u cm: IK residual {fmt(s['ik_residual_mean_cm'])} "
                    f"(max norm {s['ik_residual_norm_cm_max']}), arm tracking {fmt(s['tracking_mean_cm'])} "
                    f"(max {s['tracking_norm_cm_max']}), pelvis motion since the plan {fmt(s['pelvis_motion_mean_cm'])} "
                    f"(max {s['pelvis_motion_norm_cm_max']})."
                )
    lines.append("")
    lines.append("## Per cell")
    lines.append("")
    names = list(probes)
    head = "| id | fwd | left | z | " + " | ".join(f"{n} err f/l/u cm (norm) rot deg" for n in names) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (4 + len(names)))
    by = {n: {r["id"]: r for r in probes[n]["cells"]} for n in names}
    ids = sorted({r["id"] for n in names for r in probes[n]["cells"]})
    for i in ids:
        any_row = next(by[n][i] for n in names if i in by[n])
        cols = []
        for n in names:
            r = by[n].get(i)
            if r is None or "err_heading_cm" not in r:
                cols.append("skipped: " + (r or {}).get("skipped", "-") if r else "-")
            else:
                tol = "" if r.get("reachable", True) else " IK out of tol"
                cols.append(f"{fmt(r['err_heading_cm'])} ({r['err_norm_cm']}) {r['rot_err_deg']}{tol}")
        lines.append(
            f"| {i} | {any_row['forward']:.2f} | {any_row['left']:+.2f} | {any_row['z']:.2f} | "
            + " | ".join(cols)
            + " |"
        )
    (root / "bias.json").write_text(json.dumps(bias, indent=1))
    (root / "probe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"wrote {root / 'bias.json'} and {root / 'probe.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
