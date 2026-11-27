"""Probe (2026-11-25, design only): build each hardgrasp task in the benchmark host, print the robot spawn, the
task's objects (obj_body_id) with distance / bearing from the pelvis, and whether a right_hand key in the VLA action
closes the Dex3 fingers. Not part of the episode pipeline."""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "locomanilab_20261117"))
import lml_tasks as LT  # noqa
import lml_selfmodel as L  # noqa

SCENES = {
    "IThorKitchenApproachTableAndLiftApple": 13, "IThorKitchenApproachTableAndLiftBowl": 13,
    "IThorKitchenApproachTableAndLiftSaltShaker": 13, "IThorLivingRoomApproachTableAndLiftDumbbell": 207,
    "IThorLivingRoomApproachTableAndLiftStatue": 209, "IThorLivingRoomApproachTableAndLiftWateringCan": 207,
    "IThorKitchenApproachTableAndStandCannedDrinkUpright": 13, "IThorKitchenTakeStatueFromCabinet": 13,
    "IThorKitchenOpenLowMicrowaveAndLiftApple": 13, "IThorKitchenOpenDrawerTakeApple": 13,
    "IThorBathroomLiftSoapBottleFromCabinet": None, "IThorBathroomLiftCandleFromFloor": 401,
}

if __name__ == "__main__":
    task = sys.argv[1]
    sc = SCENES[task]
    if sc is None:
        sys.path[:0] = [str(LT.H.HB), str(L.G1B)]
        import robocasa  # noqa
        from robosuite.environments.base import REGISTERED_ENVS
        sc = int(REGISTERED_ENVS[task].SUPPORTED_SCENE_IDS[0])
    LT.TASKS[task] = dict(scene=sc, max_steps=0, genuine=0)
    t0 = time.time()
    dr = LT.TaskDriver(task, seed=0)
    dr.reset_episode(0)
    env = dr.env
    x = dr.sample()
    p = x["root"][:3]; yaw = L.yaw_of(L._qmat(x["root"][3:]))
    out = dict(task=task, scene=sc, build_s=round(time.time() - t0, 1), pelvis=p.round(3).tolist(), yaw_deg=round(math.degrees(yaw), 1),
               lang=env.get_ep_meta().get("lang"), action_dim=int(env.action_dim), objs={})
    for n, b in env.obj_body_id.items():
        q = dr.d.xpos[b]
        dv = q[:2] - p[:2]
        out["objs"][n] = dict(pos=q.round(3).tolist(), dist=round(float(np.linalg.norm(dv)), 3),
                              bearing_deg=round(math.degrees((math.atan2(dv[1], dv[0]) - yaw + math.pi) % (2 * math.pi) - math.pi), 1))
    for a in ("TARGET_NAME", "SOURCE_NAME", "MIN_LIFT_DELTA", "HOLD_STEPS", "SPEC_TASK"):
        if hasattr(env, a):
            out[a] = getattr(env, a)
    print(json.dumps(out), flush=True)
