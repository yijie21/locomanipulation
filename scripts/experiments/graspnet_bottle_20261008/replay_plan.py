"""TODO 23.2.6: replay the IK planning of an object's recorded cache rounds offline, with and without the candidate
presort (grasp_planned.PRESORT), to measure the planning time and whether the chosen grasp changes. No simulation:
the round's record gives the start joints, the pelvis position and heading (align.yaw; roll and pitch taken as zero,
so this is a timing replay, not evidence about execution), the object pose and the table; the candidates are rebuilt
from the scene's cache exactly as the round did. One object per process (locomani copies the scene path at import).

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/replay_plan.py 0 [--tag _cache]
    -> outputs/graspnet_bottle_20261008/cache/replay/obj_000.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO / "outputs" / "graspnet_bottle_20261008"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", type=int)
    ap.add_argument("--tag", default="_cache")
    args = ap.parse_args()
    oid = args.object
    sys.path[:0] = [str(HERE), str(REPO / "src")]
    os.environ.setdefault("INSTANCE", "1")
    from locomani import paths

    scene = ROOT / "scenes" / f"obj_{oid:03d}"
    paths.adopt_sim_dir(str(scene))
    import grasp_planned as gp

    from locomani.skills.profiles import load_profiles
    from locomani.tasks.bottle.ready_grasp_policy import OFFSET

    profiles = load_profiles()
    g = gp.GraspPlanned(profiles, 0, 0, ROOT / "cache" / "replay" / "tmp")
    cfg = profiles["bottle_simple"].config.reach.whole_body
    ik = g._ik_setup(cfg)
    run = REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{args.tag}" / "grasp_planned"
    rows = []
    for rd in sorted(run.glob("round_*")):
        rec_path = rd / "record.json"
        if not rec_path.exists():
            continue
        rec = json.loads(rec_path.read_text())
        if "start_joints" not in rec or "cache" not in rec:
            continue
        q0 = np.asarray(rec["start_joints"], dtype=float)
        pel_t = np.asarray(rec["pelvis"], dtype=float)
        yaw = float(rec["align"]["yaw"])
        pel_R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        heading = pel_R[:2, 0]
        cands, obj_pts, info = gp.cached_candidates(
            scene, rec["object_pos"], rec["object_quat"], rec["table_top"], heading
        )
        row = dict(round=rec["round"], recorded=dict(ok=rec["plan"].get("ok"), index=rec["plan"].get("index"),
                   seconds=rec["plan"].get("plan_seconds"), row=(rec.get("grasp") or {}).get("row"),
                   rejected=len(rec["plan"].get("rejected") or []), lifted=rec.get("ok")), candidates=len(cands))  # fmt: skip
        for mode in ("off", "on", "staged", "feasfirst"):
            gp.PRESORT["on"] = mode != "off"
            gp.PRESORT["staged"] = mode in ("staged", "feasfirst")
            gp.PRESORT["tier_first"] = mode != "feasfirst"
            t0 = time.time()
            plan = g._plan_ik(ik, q0, pel_R, pel_t, cands, rec["table_top"], obj_pts, OFFSET, lambda *_a: None)
            idx = plan.get("index")
            row[mode] = dict(
                seconds=time.time() - t0, ok=bool(plan["ok"]), index=idx, row=cands[idx]["row"] if idx is not None else None,
                tier=cands[idx]["tier"] if idx is not None else None, rejected=len(plan.get("rejected") or []),
                presort=plan.get("presort"),
            )  # fmt: skip
        rows.append(row)
        print(
            f"[replay] obj {oid:03d} round {row['round']}: recorded {row['recorded']['seconds']:.1f} s idx {row['recorded']['index']}; "
            f"off {row['off']['seconds']:.1f} s idx {row['off']['index']} ok {row['off']['ok']}; "
            f"on {row['on']['seconds']:.1f} s idx {row['on']['index']} ok {row['on']['ok']}; staged {row['staged']['seconds']:.1f} s idx {row['staged']['index']} "
            f"(presort {row['on']['presort']['seconds']:.1f} s, feasible {row['on']['presort']['feasible']}/{row['candidates']})",
            flush=True,
        )
    out = ROOT / "cache" / "replay"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"obj_{oid:03d}.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
