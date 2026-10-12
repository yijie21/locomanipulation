"""The route service's self-healing on the request that broke its solvers (curobo_service module doc, 2026-10-09).

In SIMPLE's venv, one RoutePlanner at the service's batch (2): a request a fresh build solves (gnb_obj_005's first goal), the GraspNet
debug request that left every later plan failing IK (gnb_obj_000_smoke10), then the first request again. Prints each
reply's ok / rebuilt / poisoning and whether the last answer equals the first.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src TMPDIR=/workspace/tmp third_party/SIMPLE/.venv/bin/python \
        scripts/experiments/curobo_route_20261009/poison_check.py outputs/graspnet_bottle_20261008/curobo/g1_right_curobo.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src"), str(Path(__file__).resolve().parent)]

import replay_saved  # noqa: E402

from locomani.primitives import curobo_service  # noqa: E402
from locomani.primitives.arm_route import world_message  # noqa: E402


def load(name):
    return json.loads((REPO / f"outputs/runs/{name}/grasp_planned/round_01_curobo/curobo_request.json").read_text())


def main() -> int:
    good, poison = load("gnb_obj_005"), load("gnb_obj_000_smoke10")
    planner = curobo_service.RoutePlanner(sys.argv[1], 2, tempfile.mkdtemp(prefix="poison_check_"))
    planner.configure(good["names"], good["lower"], good["upper"], good["start"])
    replies = []
    for tag, req in (("good", good), ("poison", poison), ("good again", good)):
        rep = planner.plan(
            dict(start=req["start"], goals=req["pregrasp"][:1], world=world_message(replay_saved.obstacles(req)))
        )
        replies.append(rep)
        print(
            f"{tag}: ok {rep['ok']} status {rep['status']} canary {rep.get('canary')} rebuilt {rep.get('rebuilt')} "
            f"poisoning {rep.get('poisoning')}",
            flush=True,
        )
    a, c = np.asarray(replies[0]["path"]), np.asarray(replies[2]["path"])
    same = a.shape == c.shape and float(np.abs(a - c).max()) if a.shape == c.shape else f"shapes {a.shape} {c.shape}"
    print(f"rebuilds {planner.rebuilds}; last vs first: {same}")
    return 0 if replies[0]["ok"] and replies[2]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
