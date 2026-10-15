"""`CoordinatedReach.route_planner` for visual.reach_route planned: the CuRobo route service plans the waist and both
arms from where they are to the reach's own pregrasp solve, around the scene's collidable geoms (user, 2026-10-14).

The service (`primitives/curobo_service.py`, SIMPLE's venv) is configured once with the 17 planned joints, the
solver's joint limits and a retract posture inside them; every plan sends the joint-space start and goal (protocol 2:
a pose goal let CuRobo pick its own waist and wrist solution, which the closed loop then carried, single-arm route plan
section 6) and the obstacles in the pelvis frame, built from the scene model with the free bodies where the state
says they are now, every box grown by `clearance` on each side: the planned hands track their path closed loop with a
few centimetres of error, and with CuRobo's own 1.3 cm margin the Dex1 fingers brushed the box's near rim in 11 of 49
diagnostic rounds (2026-10-14). A hanging hand already inside that margin makes CuRobo refuse the start
(INVALID_START_STATE_WORLD_COLLISION, 1 of 50); the planner then plans once more without the start check and says so
in `last`. Legs are not planned: the path keeps the start's leg joints. A robot model for the service is
`outputs/curobo_warehouse_20261008/mars_curobo.json` (scripts/experiments/curobo_warehouse_20261008: export_geometry.py,
then curobo_robot.py).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src third_party/SIMPLE/.venv/bin/python -m locomani.primitives.curobo_service \\
        outputs/curobo_warehouse_20261008/mars_curobo.json --port 5795 --batch 2
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_route import ArmRouteClient, Box, scene_obstacles
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
PLANNED = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [
    f"{side}_{joint}_joint" for side in ("left", "right") for joint in ARM
]
PLANNED_INDEX = np.array([MUJOCO_JOINT_NAMES.index(name) for name in PLANNED])


class ReachRoutePlanner:
    """Callable(start_joints, goal_joints, state) -> (T, 29) joint path or None.

    `lower` / `upper` are the 29 solver limits in MuJoCo joint order; `free_bodies` maps each free body of `model`
    to the key of its pose in `state["objects"]`; `clearance` (m) grows every box obstacle on each side (meshes are
    sent as they are). `last` keeps the latest plan's facts for the round record."""

    def __init__(self, client: ArmRouteClient, model, lower, upper, free_bodies: dict, clearance: float = 0.0):
        self.client, self.model, self.free_bodies = client, model, dict(free_bodies)
        if not 0.0 <= clearance <= 0.1:
            raise ValueError("route clearance must be within [0, 0.1] m")
        self.clearance = float(clearance)
        self.lower = np.asarray(lower, dtype=float)[PLANNED_INDEX]
        self.upper = np.asarray(upper, dtype=float)[PLANNED_INDEX]
        self.configured = False
        self.last = {}

    def configure(self) -> None:
        if not self.configured:
            self.client.configure(PLANNED, self.lower, self.upper, np.clip(0.0, self.lower, self.upper))
            self.configured = True

    def obstacles(self, state) -> list:
        pelvis = state["pelvis"]
        q = np.asarray(pelvis["quat"], dtype=float)
        frame = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        free = {
            body: (np.asarray(state["objects"][key]["pos"], float), np.asarray(state["objects"][key]["quat"], float))
            for body, key in self.free_bodies.items()
        }
        found = scene_obstacles(self.model, free, np.asarray(pelvis["pos"], dtype=float), frame)
        grow = 2 * self.clearance
        return [Box(o.name, o.pose, tuple(d + grow for d in o.dims)) if isinstance(o, Box) else o for o in found]

    def __call__(self, start, goal, state):
        self.configure()
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        # measured or reference joints can sit a hair outside the limits the service was configured with
        request = (
            np.clip(start[PLANNED_INDEX], self.lower, self.upper),
            np.clip(goal[PLANNED_INDEX], self.lower, self.upper),
            self.obstacles(state),
        )
        route = self.client.plan_to_joints(*request)
        seconds, start_in_collision = route.seconds, "INVALID_START_STATE_WORLD_COLLISION" in route.status
        if start_in_collision:
            route = self.client.plan_to_joints(*request, check_start_validity=False)
            seconds += route.seconds
        self.last = dict(
            ok=route.ok,
            seconds=round(seconds, 3),
            waypoints=len(route.path),
            status=route.status,
            start_in_collision=start_in_collision,
            clearance=self.clearance,
            cached=route.cached,
        )
        if not route.ok:
            return None
        path = np.tile(start, (len(route.path), 1))
        path[:, PLANNED_INDEX] = route.path
        return path
