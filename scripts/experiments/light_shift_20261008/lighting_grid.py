"""TODO 21.3.1 pre-check, step 1: what the policy's three cameras see under candidate lighting shifts.

One collected state of a run (an episode's state trace at a fraction of its length) rendered by the Unreal view
under each (scale, kelvin) setting; one row per setting (ego, left wrist, right wrist), labelled, into one PNG.
Needs the editor (scripts/start_unreal.sh) with the run's level built; no stack.

    PYTHONPATH=src "$BRIDGE_PY" scripts/experiments/light_shift_20261008/lighting_grid.py \
        outputs/runs/warehouse_mars_50 visual_crate_s7000_08 outputs/light_shift_20261008
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from locomani.pipeline import run_context
from locomani.sim.unreal.view import UnrealView

SETTINGS = ((1.0, None), (0.5, None), (0.3, None), (2.0, None), (1.0, 3200.0), (0.4, 3200.0), (1.0, 9000.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("label", help="an episode label with a state trace in <run>/trace/")
    ap.add_argument("out", type=Path)
    ap.add_argument("--at", type=float, nargs="+", default=[0.35, 0.6], help="fractions of the episode's length")
    args = ap.parse_args()

    cfg = run_context.load_run_config(args.run)
    unreal = run_context.unreal_scene(cfg)
    scene_xml = next(args.run.glob("*/scene_locomani.xml"))
    qpos = np.asarray(np.load(args.run / "trace" / f"{args.label}_state_trace.npz")["qpos"])
    rows = [int(f * (len(qpos) - 1)) for f in args.at]
    args.out.mkdir(parents=True, exist_ok=True)

    view = UnrealView(str(scene_xml), unreal["level_path"], unreal["repeats"])
    record = []
    try:
        view.warm_up(qpos[rows[0]])
        for row in rows:
            grid = []
            for scale, kelvin in SETTINGS:
                changed = view.set_lighting(scale, kelvin)
                view.render(qpos[row])  # one throwaway request: Lumen settles on the new lights
                images = view.render(qpos[row])
                strip = np.concatenate([images[k] for k in ("ego_view", "left_wrist", "right_wrist")], axis=1)
                strip = np.ascontiguousarray(strip[..., ::-1])  # RGB -> BGR for cv2
                text = f"x{scale:g}  {int(kelvin) if kelvin else 'as built'} K"
                cv2.putText(strip, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(strip, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                grid.append(strip)
                means = {k: round(float(v.mean()), 1) for k, v in images.items()}
                record.append({"row": row, "scale": scale, "kelvin": kelvin, "mean": means, **changed})
                print(json.dumps(record[-1] | {"lights": len(changed["lights"]), "panels": len(changed["panels"])}))
            full = np.concatenate(grid, axis=0)
            dest = args.out / f"{args.label}_row{row}.png"
            cv2.imwrite(str(dest), cv2.resize(full, (full.shape[1] // 2, full.shape[0] // 2), cv2.INTER_AREA))
            print(f"wrote {dest}")
        view.set_lighting(1.0, None)
    finally:
        view.close()
    (args.out / f"{args.label}_grid.json").write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
