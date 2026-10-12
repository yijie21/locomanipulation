"""TODO 21.3.4: where curobo_server.py's ~17 s per plan goes, on a request a run saved (curobo_request.json).

Replays the request through the server's Planner with timers around each stage, and prints CuRobo's own breakdown
of the batched MotionGen call (IK, graph, trajopt, attempts). Then the same MotionGen call twice more under one change
each, to see what the time depends on: CUDA graphs on, and one candidate instead of the batch.

    CUDA_VISIBLE_DEVICES=1 third_party/SIMPLE/.venv/bin/python scripts/experiments/graspnet_bottle_20261008/profile_curobo.py \
        outputs/runs/gnb_obj_017/grasp_planned/round_01_curobo/curobo_request.json
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curobo_server import Planner  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROBOT = os.path.join(REPO, "outputs", "graspnet_bottle_20261008", "curobo", "g1_right_curobo.json")


def timed(label, fn):
    torch.cuda.synchronize()
    t = time.time()
    out = fn()
    torch.cuda.synchronize()
    print(f"{label:48s} {time.time() - t:7.2f} s", flush=True)
    return out


def batch_call(planner, req, n=None):
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    pre = np.asarray(req["pregrasp"], dtype=np.float32)[:n]
    b = len(pre)
    start = torch.tensor([planner._to_curobo(req["start"])] * b, device="cuda", dtype=torch.float32)
    goals = Pose(position=torch.tensor(pre[:, :3], device="cuda"), quaternion=torch.tensor(pre[:, 3:], device="cuda"))
    return planner.motion_gen.plan_batch(
        JointState.from_position(start, joint_names=planner.names),
        goals,
        MotionGenPlanConfig(enable_finetune_trajopt=False, num_trajopt_seeds=12, max_attempts=10),
    )


def report(result):
    keys = ("solve_time", "ik_time", "graph_time", "trajopt_time", "finetune_time", "attempts", "trajopt_attempts")
    parts = {k: getattr(result, k, None) for k in keys}
    parts = {k: (round(float(v), 2) if isinstance(v, (int, float)) else v) for k, v in parts.items()}
    print("   curobo breakdown:", parts, "successes", int(result.success.sum()), "of", len(result.success), flush=True)


def main():
    req = json.load(open(sys.argv[1]))
    print(f"request: {len(req['pregrasp'])} candidates", flush=True)
    planner = Planner(ROBOT, os.path.dirname(ROBOT))
    timed(
        "configure (MotionGen + IK solver build)",
        lambda: planner.configure(req["names"], req["lower"], req["upper"], req["start"]),
    )
    rep = timed("plan(): the whole server call", lambda: planner.plan(req))
    print("   ->", {k: rep[k] for k in ("ok", "index", "successes", "status")}, flush=True)
    r = timed("plan_batch again (warm)", lambda: batch_call(planner, req))
    report(r)
    r = timed("plan_batch, first candidate only", lambda: batch_call(planner, req, 1))
    report(r)
    i = rep.get("index") or 0
    pre, gra = req["pregrasp"][i], req["grasp"][i]
    q_end = planner._to_curobo(rep["pregrasp_path"][-1]) if rep.get("ok") else planner._to_curobo(req["start"])
    timed("entry: 10 CuRobo IK steps of 1 cm", lambda: planner._ik_line(q_end, pre, gra))
    print("--- same call with CUDA graphs on (warm-up excluded)", flush=True)
    import curobo_server

    original = curobo_server.MotionGenConfig if hasattr(curobo_server, "MotionGenConfig") else None
    del original
    from curobo.wrap.reacher import motion_gen as mg

    load = mg.MotionGenConfig.load_from_robot_config

    def with_graph(*a, **kw):
        kw["use_cuda_graph"] = True
        return load(*a, **kw)

    mg.MotionGenConfig.load_from_robot_config = with_graph
    planner2 = Planner(ROBOT, os.path.dirname(ROBOT))
    planner2.configure(req["names"], req["lower"], req["upper"], req["start"])
    from curobo.geom.types import Cuboid, Mesh, WorldConfig

    t = req["table"]
    world = WorldConfig(
        cuboid=[Cuboid(name="table", pose=[*t["center"], *t["quat"]], dims=[2 * h for h in t["half"]])],
        mesh=[
            Mesh(
                name="target",
                pose=[0, 0, 0, 1, 0, 0, 0],
                vertices=req["object"]["vertices"],
                faces=req["object"]["faces"],
            )
        ],
    )
    planner2.motion_gen.update_world(world)
    timed("plan_batch with CUDA graphs, first call", lambda: batch_call(planner2, req))
    r = timed("plan_batch with CUDA graphs, second call", lambda: batch_call(planner2, req))
    report(r)


if __name__ == "__main__":
    main()
