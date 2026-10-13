"""B3, ScaleBFM side (TODO 25.2.2): b3_cells.py's 108 cells from ScaleBFM's own walked-in bottle stance.

The stance is where a `walk` round of bfm_bottle.py ended (its last pelvis pose: ScaleBFM's walk scatter is
+-1-2 cm, so resetting there is the walked-in stance), on the stock bottle scene with the SONIC line's hand law; the
bottle is parked behind the robot as in 24.2. The probe machinery is 24.6's (bfm_attitude_probe.Probe: HOME at travel
height, turn the palm in place, over, down, hold, back; a fall resets and the next cell starts from HOME), mode 2.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/controller_compare_20261013/bfm_b3_probe.py \\
        --walk outputs/controller_compare_20261013/bottle/bfm_walk_6cm/bfm_walk_6cm_v0.25_r00a0.npz \\
        --out outputs/controller_compare_20261013/bottle/b3/scalebfm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "experiments" / "wbc_compare_20261012")]

import b3_cells  # noqa: E402
import bfm_attitude_probe as A  # noqa: E402
import probe_cells as C  # noqa: E402
from bfm_bottle import HAND_FORCE, HAND_KD, HAND_KP  # noqa: E402
from bfm_probe import HOLD_S, MEASURE_S, SPEED  # noqa: E402
from bottle_states import SCENE  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402


class BottleProbe(A.Probe):
    """24.6's probe on the bottle scene at a given stance (A.Probe builds its sim and stance from a 65-object scene)."""

    def __init__(self, pol, xy, yaw, hold_s, speed):
        self.pol, self.hold_s, self.speed = pol, hold_s, speed
        cfg = ScaleBfmBackendConfig(
            scene_dir=str(SCENE), object_name="water_bottle", grip_kp=HAND_KP, grip_kd=HAND_KD, grip_clip=HAND_FORCE
        )
        self.sim = R.Sim(pol.meta, backend=cfg)
        self.xy, self.yaw, self.edge = np.asarray(xy, float), float(yaw), None
        self.home = C.home_tool(self.xy, self.yaw, self.sim.table_top)
        self.R0 = C.palm_rotation(self.yaw)
        self.writer = None
        self.k, self.sim_s, self.resets = 0, 0.0, 0
        self.start()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walk", type=Path, required=True, help="a bfm_bottle.py walk round file (.npz)")
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.walk)
    pel = z["pelvis"][-1]
    xy, yaw = pel[:2], R.yaw_of(pel[3:])
    facts = json.loads(a.walk.with_suffix(".json").read_text())
    pol = R.Policy(device=a.device, backend=ScaleBfmBackendConfig(scene_dir=str(SCENE)))
    t_wall = time.time()
    probe = BottleProbe(pol, xy, yaw, a.hold, a.speed)
    edge = facts["edge"]
    rows = []
    for i, (pos, att) in enumerate(b3_cells.cells()):
        row = probe.cell(pos, att)
        row["deliverable"] = b3_cells.deliverable(row)
        rows.append(row)
        print(
            f"[{i + 1}/108] att {att['id']} pos {pos['id']:2d}: |{row.get('err_norm_cm', '-')}| cm "
            f"rot {row.get('rot_err_deg', '-')} deg fell {row['fell_at_s']} ok {row['deliverable']}",
            flush=True,
        )
    (a.out / "b3.json").write_text(
        json.dumps(
            dict(
                controller="scalebfm",
                walk_round=str(a.walk),
                stance_xy=[float(v) for v in xy],
                stance_yaw_deg=float(np.degrees(yaw)),
                pelvis_gap_cm=None
                if edge.get("axis") is None
                else 100 * float(edge["line"] - np.asarray(edge["axis"]) @ xy),
                hold_s=a.hold,
                measure_s=MEASURE_S,
                attitudes=list(b3_cells.ATTITUDES),
                resets=probe.resets,
                wall_s=round(time.time() - t_wall, 1),
                cells=rows,
            ),
            indent=1,
        )
    )
    print(f"{sum(r['deliverable'] for r in rows)}/108 deliverable, {probe.resets} resets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
