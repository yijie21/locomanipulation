#!/usr/bin/env python3
"""Drop test of a built scene file, and its MuJoCo indices written back (TODO 19.6.5). Runs in the sim venv.

    $VENV_SIM_PY scripts/scene_drop_test.py --scene outputs/scene_discovery/adt_apartment/scene_meta.json

Loads `<scene dir>/env_only.xml` (no robot) with the sim's MuJoCo, lets every free object settle for
`--seconds`, and records per object `semantic.drop_test = {moved_m, tilt_deg, final_z, penetration_m, ok}`
(mass and friction keep their `claude` provenance: resting still does not measure them). Then compiles
`scene_locomani.xml` and writes the legacy indices existing readers use (`body_id`, `qpos_adr`, `nbody`,
`ngeom`, `nq`). The report is `<scene dir>/drop_test.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from locomani.scene_discovery import drop_test, scene_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()
    path = Path(args.scene)
    meta = json.loads(path.read_text())
    t0 = time.time()
    env = mujoco.MjModel.from_xml_path(str(path.parent / "env_only.xml"))
    print(
        f"[drop] env_only.xml compiled in {time.time() - t0:.1f} s: {env.nbody} bodies, {env.ngeom} geoms", flush=True
    )
    t0 = time.time()
    result = drop_test.settle(env, args.seconds)
    print(f"[drop] {len(result)} free bodies settled for {args.seconds} s in {time.time() - t0:.1f} s", flush=True)
    bad = []
    for key, o in meta["objects"].items():
        r = result.get(f"{key}_object")
        if r is None:
            o["semantic"].pop("drop_test", None)
            continue
        o["semantic"]["drop_test"] = r
        o["provenance"]["drop_test"] = "measured"
        if not r["ok"]:
            bad.append((key, r))
    model = mujoco.MjModel.from_xml_path(meta["scene_xml"])
    for key, o in meta["objects"].items():
        if o.get("body"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o["body"])
            o["body_id"] = int(bid)
            o["qpos_adr"] = int(model.jnt_qposadr[model.body_jntadr[bid]])
    meta.update(nbody=int(model.nbody), ngeom=int(model.ngeom), nq=int(model.nq))
    problems = scene_file.validate(meta)
    path.write_text(json.dumps(meta, indent=1))
    (path.parent / "drop_test.json").write_text(json.dumps({"seconds": args.seconds, "result": result}, indent=1))
    print(
        f"[drop] {len(result) - len(bad)}/{len(result)} stay put; robot scene nq {model.nq}; {len(problems)} validation problem(s)"
    )
    for key, r in sorted(bad, key=lambda kv: -kv[1]["moved_m"])[:30]:
        print(
            f"    {key}: moved {r['moved_m']} m, tilt {r['tilt_deg']} deg, z {r['final_z']}, initial penetration {r['penetration_m']} m"
        )


if __name__ == "__main__":
    main()
