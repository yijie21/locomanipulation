"""TODO 25.9.1 delivery envelope (docs/handoff/scalebfm-warehouse-rsi-20261014.md §6.2): over the region the expert actually
uses -- the reach and descend of real rounds, `handoff` to `close` (or the end) -- how much of the commanded reference
each controller delivers, axis by axis. Made from round files already on disk; nothing new is run.

Axes: waist yaw / roll / pitch (the reference's joints, cmd_joints, against the measured joints), pelvis height
(cmd_pelvis z against the measured; ScaleBFM only -- SONIC's stream has no pelvis reference), each wrist's position
in the measured pelvis frame (cmd_wrist against the measured wrist; world positions would be dominated by where each
round stood). Per axis: ticks, commanded and delivered p10 / p50 / p90, the delivery ratio
(least-squares slope of delivered on commanded through the commanded mean), and the delivered p1 / p99 (where it
saturates). A ratio near 1 is a controller that follows; SONIC's waist at ~0.5 is §20's "waist half delivered".

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_warehouse_rsi_20261014/envelope.py \\
        --name sonic_v1_1 DIR [DIR ...] --out envelope/sonic_v1_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from bottle_measure import WAIST, mat, window  # noqa: E402
from warehouse_measure import Round, Scene  # noqa: E402

WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def axis_stats(cmd, meas) -> dict:
    cmd, meas = np.asarray(cmd, float), np.asarray(meas, float)
    ok = np.isfinite(cmd) & np.isfinite(meas)
    cmd, meas = cmd[ok], meas[ok]
    if len(cmd) < 20:
        return dict(ticks=int(len(cmd)))
    q = lambda x, p: round(float(np.percentile(x, p)), 4)
    c0 = cmd - cmd.mean()
    ratio = float(c0 @ (meas - meas.mean()) / (c0 @ c0)) if c0 @ c0 > 1e-12 else None
    return dict(
        ticks=int(len(cmd)),
        cmd_p10=q(cmd, 10), cmd_p50=q(cmd, 50), cmd_p90=q(cmd, 90),
        meas_p10=q(meas, 10), meas_p50=q(meas, 50), meas_p90=q(meas, 90),
        meas_p1=q(meas, 1), meas_p99=q(meas, 99),
        ratio=None if ratio is None else round(ratio, 3),
        err_p90=q(np.abs(meas - cmd), 90),
    )  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--name", required=True, help="controller + checkpoint (+ mode), e.g. scalebfm_m22200_mode2")
    ap.add_argument("--ok-only", action="store_true", help="successful rounds only")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sc = Scene()
    acc = {
        k: ([], [])
        for k in [f"waist_{n}" for n in WAIST]
        + ["pelvis_z"]
        + [f"{s}_wrist_{c}" for s in ("left", "right") for c in "xyz"]
    }
    rounds = 0
    for d in a.dirs:
        for npz in sorted(d.glob("*.npz")):
            rd = Round(npz, sc)
            if "handoff" not in rd.ev or (a.ok_only and not rd.f.get("ok")):
                continue
            w = window(rd.t, rd.ev["handoff"], rd.ev.get("close", rd.ev.get("end", rd.t[-1])))
            if len(w) < 2:
                continue
            rounds += 1
            if "cmd_joints" in rd.z:
                for n, j in WAIST.items():
                    acc[f"waist_{n}"][0].append(rd.z["cmd_joints"][w, j])
                    acc[f"waist_{n}"][1].append(rd.z["joints"][w, j])
            acc["pelvis_z"][0].append(rd.z["cmd_pelvis"][w, 2])
            acc["pelvis_z"][1].append(rd.z["pelvis"][w, 2])
            pel = rd.z["pelvis"][w]
            local = lambda x: np.einsum("tji,tj->ti", np.stack([mat(q) for q in pel[:, 3:]]), x - pel[:, :3])
            for k, s in enumerate(("left", "right")):
                cmd, meas = local(rd.z["cmd_wrist"][w, k, :3]), local(rd.body(WRISTS[k])[w, :3])
                for c, axis in enumerate("xyz"):
                    acc[f"{s}_wrist_{axis}"][0].append(cmd[:, c])
                    acc[f"{s}_wrist_{axis}"][1].append(meas[:, c])
    out = dict(
        name=a.name,
        sources=[str(d) for d in a.dirs],
        ok_only=a.ok_only,
        rounds=rounds,
        window="handoff..close|end",
        axes={},
    )
    for k, (c, m) in acc.items():
        out["axes"][k] = axis_stats(np.concatenate(c) if c else [], np.concatenate(m) if m else [])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for k, v in out["axes"].items():
        print(
            f"{a.name} {k}: ticks {v['ticks']} ratio {v.get('ratio')} cmd p50 {v.get('cmd_p50')} meas p50 {v.get('meas_p50')} err p90 {v.get('err_p90')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
