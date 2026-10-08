"""Collision-filtered hand targets for the coordinated reach (visual.collision_guard mink).

A kinematic shadow of the waist and both arms tracks the reach's intended grip targets with mink (differential IK
as a QP) under a collision-avoidance limit between the hands and every non-robot collision geom of the scene (the
box walls, the platform). Each tick the shadow takes the measured pelvis, legs and object poses, moves only its own
17 joints, and returns where its grip points and palms ended up: the targets, held back from whatever the hands
would otherwise pass through. The reach then solves and streams its own whole-body reference toward those, with its
feedback unchanged. Offline on the 119 warehouse_mars_50 rounds (2026-10-08, TODO 20.4) the same shadow kept the
hands off the walls in 103 against 18 untracked, at 2.1 ms a tick.

Two workarounds, both measured on MuJoCo 3.11 / mink 1.3.0 (memory: mujoco-geomdistance-and-mink-collision-bugs):
- mj_geomDistance returned exactly 0.0 for separated pairs and the wrong sign (a mesh, and box-box pairs up to
  11 cm apart reported as -11 cm); the hands collide as boxes (one per collision geom, its bounding box in the geom
  frame, so never closer than the real shape), an exact 0.0 is re-queried with the geom shifted by ~20 um, and for
  two boxes the sign comes from a separating-axis test;
- mink's CollisionAvoidanceLimit bounds the displacement the QP solves for by a velocity (gain (d - d_min) / dt)
  and below d_min only forbids approaching; the rows here bound the displacement by gain (d - d_min) on both sides
  of d_min, so a hand that crept or was pushed in backs out. When backing out is infeasible (the measured body moved
  a hand in faster than the arms can follow) a tick falls back to "never closer", then to no collision rows.
mink, qpsolvers and daqp are imported only when the guard is built.
"""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.primitives.g1_joints import ARM_MJ, MUJOCO_JOINT_NAMES

SIDES = ("left", "right")
HAND_BODIES = ("wrist_pitch_link", "wrist_yaw_link", "dex1_finger_1_link", "dex1_finger_2_link")
OPEN_JAW = 0.0245  # Dex1 finger slide while the reach holds the jaws open (measured, warehouse_mars_50)
_JITTER = np.random.default_rng(0).normal(0.0, 2e-5, (6, 3))


def _matrix(quat_wxyz):
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat_wxyz, dtype=float))
    return out.reshape(3, 3)


def _bounding_box(m, g):
    """(centre, half extents) in the geom frame of a box containing collision geom g."""
    kind, size = int(m.geom_type[g]), m.geom_size[g]
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = m.geom_dataid[g]
        verts = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        return (lo + hi) / 2, (hi - lo) / 2
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.zeros(3), size.copy()
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return np.zeros(3), np.array([size[0], size[0], size[1]])
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return np.zeros(3), np.array([size[0], size[0], size[1] + size[0]])
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return np.zeros(3), np.full(3, size[0])
    raise ValueError(f"geom {g}: no bounding box for type {kind}")


def _boxes_overlap(m, d, g1, g2):
    """Separating-axis test of two box geoms as placed in d (15 axes); None unless both are boxes."""
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    if int(m.geom_type[g1]) != box or int(m.geom_type[g2]) != box:
        return None
    A, B = d.geom_xmat[g1].reshape(3, 3), d.geom_xmat[g2].reshape(3, 3)
    a, b = m.geom_size[g1], m.geom_size[g2]
    t = d.geom_xpos[g2] - d.geom_xpos[g1]
    axes = [A[:, i] for i in range(3)] + [B[:, i] for i in range(3)]
    axes += [np.cross(A[:, i], B[:, j]) for i in range(3) for j in range(3)]
    for axis in axes:
        n = np.linalg.norm(axis)
        if n < 1e-9:
            continue
        axis = axis / n
        ra = np.sum(a * np.abs(A.T @ axis))
        rb = np.sum(b * np.abs(B.T @ axis))
        if abs(t @ axis) > ra + rb:
            return False
    return True


def robust_geom_distance(m, d, g1, g2, distmax, fromto):
    """mj_geomDistance with two MuJoCo 3.11 errors corrected.

    - exactly 0.0 for separated geoms: re-queried with geom 1 shifted by ~20 um;
    - the wrong sign (box-box -111.7 mm where the boxes are 111.7 mm apart, 2026-10-08): a negative distance between
      two boxes is checked with a separating-axis test and flipped when they are apart (magnitude and witness points
      from MuJoCo). No overlapping pair was seen read positive (2470 pairs on perturbed warehouse postures).
    """
    dist = mujoco.mj_geomDistance(m, d, g1, g2, distmax, fromto)
    if dist == 0.0:
        pos = d.geom_xpos[g1].copy()
        values = []
        for shift in _JITTER:
            d.geom_xpos[g1] = pos + shift
            value = mujoco.mj_geomDistance(m, d, g1, g2, distmax, fromto)
            if value != 0.0:
                values.append(value)
        d.geom_xpos[g1] = pos
        dist = float(np.median(values)) if values else 0.0
    if dist < 0 and _boxes_overlap(m, d, g1, g2) is False:  # the flips seen were all separated pairs read negative
        dist = -dist
    return dist


def _shadow_model(scene_xml, joint_limits):
    """The scene with one box per hand collision geom (`<body>_geom<id>_guard`), the reach's joint limits
    (`joint_limits` = (lo, hi), 29 each, MuJoCo order) on the 17 planned joints and no limit on any other joint.
    -> (model, hand boxes per side, obstacle geoms, the 29 hinge joint ids, planned dofs)"""
    base = mujoco.MjModel.from_xml_path(scene_xml)
    spec = mujoco.MjSpec.from_file(scene_xml)
    boxes = {s: [] for s in SIDES}
    for side in SIDES:
        for part in HAND_BODIES:
            name = f"{side}_{part}"
            b = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, name)
            for g in range(base.ngeom):
                if base.geom_bodyid[g] != b or not (base.geom_contype[g] or base.geom_conaffinity[g]):
                    continue
                centre, half = _bounding_box(base, g)
                box = f"{name}_geom{g}_guard"
                spec.body(name).add_geom(
                    name=box,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=half,
                    pos=base.geom_pos[g] + _matrix(base.geom_quat[g]) @ centre,
                    quat=base.geom_quat[g],
                    contype=1,
                    conaffinity=1,
                    group=4,
                )
                boxes[side].append(box)
    model = spec.compile()
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in MUJOCO_JOINT_NAMES}
    hinges = [jid[n] for n in MUJOCO_JOINT_NAMES]
    lo, hi = (np.asarray(v, dtype=float) for v in joint_limits)
    for i, j in enumerate(hinges):
        model.jnt_range[j] = [max(model.jnt_range[j, 0], lo[i]), min(model.jnt_range[j, 1], hi[i])]
    planned = [12, 13, 14, *ARM_MJ["left"], *ARM_MJ["right"]]
    dofs = [int(model.jnt_dofadr[hinges[i]]) for i in planned]
    for j in range(model.njnt):  # a frozen joint measured just outside its range must not make the QP infeasible
        if model.jnt_dofadr[j] not in set(dofs):
            model.jnt_limited[j] = 0
    robot = set()
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for b in range(model.nbody):
        a = b
        while a not in (0, pelvis):
            a = model.body_parentid[a]
        if a == pelvis:
            robot.add(b)
    obstacles = [
        g
        for g in range(model.ngeom)
        if (model.geom_contype[g] or model.geom_conaffinity[g])
        and model.geom_bodyid[g] not in robot
        and int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
    ]
    hand = {s: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in boxes[s]] for s in SIDES}
    return model, hand, obstacles, [jid[n] for n in MUJOCO_JOINT_NAMES], dofs


class HandCollisionGuard:
    """Kinematic shadow of the waist and arms that keeps the hands `clearance` metres off the scene's obstacles."""

    def __init__(
        self,
        scene_xml,
        *,
        tool_offset,
        joint_limits,
        joint_speed,
        clearance=0.003,
        detect=0.08,
        substeps=4,
    ):
        import mink
        import mink.limits.collision_avoidance_limit as mink_collision
        from mink.limits.limit import Constraint

        class DisplacementCollisionLimit(mink.CollisionAvoidanceLimit):
            """Rows `-dd <= gain (d - d_min)` in the QP's displacement units, also below d_min, where mink 1.3.0
            only forbids approaching (`dd >= 0`): the linearisation then lets a rotating hand creep in (3.0 mm ->
            0.2 mm in 320 steps on the fixture table) and nothing brings it back. Mirrors mink's own loop."""

            def compute_qp_inequalities(self, configuration, dt):
                data = configuration.data
                h = np.full((self.max_num_contacts,), np.inf)
                G = np.zeros((self.max_num_contacts, self.model.nv))
                indices = (
                    self._broadphase_survivors(data)
                    if self.broadphase and self.max_num_contacts >= self.broadphase_min_pairs
                    else range(self.max_num_contacts)
                )
                for idx in indices:
                    g1, g2 = self.geom_id_pairs[idx]
                    dist = robust_geom_distance(
                        self.model, data, g1, g2, self.collision_detection_distance, self._fromto
                    )
                    if abs(dist - self.collision_detection_distance) < 1e-12:
                        continue
                    row = mink_collision.compute_contact_normal_jacobian(
                        self.model, data, g1, g2, self._fromto, self._normal, self._jac1, self._jac2
                    )
                    h[idx] = self.gain * (dist - self.minimum_distance_from_collisions)
                    G[idx] = (-1.0 if dist >= 0 else 1.0) * row
                return Constraint(G=G, h=h)

        class NoApproachCollisionLimit(DisplacementCollisionLimit):
            """The fallback when backing out is infeasible (the measured body pushed a hand in): never closer."""

            def compute_qp_inequalities(self, configuration, dt):
                c = super().compute_qp_inequalities(configuration, dt)
                return Constraint(G=c.G, h=np.maximum(c.h, 0.0))

        self.mink = mink
        self.model, self.hand, self.obstacles, self.hinges, self.dofs = _shadow_model(scene_xml, joint_limits)
        m = self.model
        self.offset = np.asarray(tool_offset, dtype=float)
        self.substeps = int(substeps)
        self.hinge_qadr = np.array([m.jnt_qposadr[j] for j in self.hinges])
        self.planned_qadr = self.hinge_qadr[[12, 13, 14, *ARM_MJ["left"], *ARM_MJ["right"]]]
        self.jaw_qadr = [
            m.jnt_qposadr[j] for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
        ]
        self.free_objects = {}  # state["objects"] key (the body name without "_object") -> qpos address
        for j in range(m.njnt):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]) or ""
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) and body != "pelvis":
                self.free_objects[body.removesuffix("_object")] = int(m.jnt_qposadr[j])
        self.palms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in SIDES]
        self.freeze = mink.DofFreezingTask(m, dof_indices=[i for i in range(m.nv) if i not in set(self.dofs)])
        self.frames = [
            mink.FrameTask(f"{s}_wrist_yaw_link", "body", position_cost=1.0, orientation_cost=0.6) for s in SIDES
        ]
        self.posture = mink.PostureTask(m, cost=1e-2)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.hinges]
        planned_names = [names[i] for i in (12, 13, 14, *ARM_MJ["left"], *ARM_MJ["right"])]
        common = [mink.ConfigurationLimit(m), mink.VelocityLimit(m, {n: joint_speed for n in planned_names})]
        pairs = [(self.hand[s], self.obstacles) for s in SIDES]
        self.limit_chain = [
            [*common, kind(m, pairs, minimum_distance_from_collisions=clearance, collision_detection_distance=detect)]
            for kind in (DisplacementCollisionLimit, NoApproachCollisionLimit)
        ] + [common]
        self.fallbacks = 0  # QP solves that had to drop to a weaker collision limit (see filter)
        self.configuration = mink.Configuration(m)
        self._fromto = np.zeros(6)
        self.started = False

    def reset(self):
        """Start again from the next measured posture (entering align, or after a retry)."""
        self.started = False

    def _measured(self, state):
        q = self.configuration.q.copy() if self.started else self.model.qpos0.copy()
        q[:3] = state["pelvis"]["pos"]
        q[3:7] = state["pelvis"]["quat"]
        joints = np.asarray(state["joints"], dtype=float)
        if self.started:
            q[self.hinge_qadr[:12]] = joints[:12]  # legs measured, waist and arms the shadow's own
        else:
            q[self.hinge_qadr] = joints
            m = self.model
            for j in self.hinges:
                if m.jnt_limited[j]:
                    a, (lo, hi) = m.jnt_qposadr[j], m.jnt_range[j]
                    margin = min(1e-3, (hi - lo) / 2)  # a locked axis (lo == hi, waist_dofs pitch) sits on its value
                    q[a] = np.clip(q[a], lo + margin, hi - margin)
        q[self.jaw_qadr] = OPEN_JAW
        for key, adr in self.free_objects.items():
            obj = (state.get("objects") or {}).get(key)
            if obj is not None:
                q[adr : adr + 3], q[adr + 3 : adr + 7] = obj["pos"], obj["quat"]
        return q

    def filter(self, tools, attitudes, state, dt):
        """Shadow-track grip targets `tools` (2x3, world) with palm `attitudes` (2x3x3) for `dt` seconds.
        -> (grip points, palm attitudes, smallest hand-box clearance in m)"""
        mink = self.mink
        self.configuration.update(self._measured(state))
        if not self.started:
            self.posture.set_target_from_configuration(self.configuration)
            self.started = True
        for frame, tool, R in zip(self.frames, tools, attitudes):
            R = np.asarray(R, dtype=float)
            frame.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(R), np.asarray(tool, dtype=float) - R @ self.offset
                )
            )
        step = max(float(dt), 1e-3) / self.substeps
        for _ in range(self.substeps):
            for level, limits in enumerate(self.limit_chain):
                try:
                    v = mink.solve_ik(
                        self.configuration,
                        [*self.frames, self.posture],
                        step,
                        solver="daqp",
                        damping=1e-3,
                        limits=limits,
                        constraints=[self.freeze],
                    )
                    break
                except mink.NoSolutionFound:
                    if level == len(self.limit_chain) - 1:
                        raise
                    self.fallbacks += 1
            self.configuration.integrate_inplace(v, step)
        d = self.configuration.data
        rotations = np.array([d.xmat[b].reshape(3, 3).copy() for b in self.palms])
        grips = np.array([d.xpos[b] + R @ self.offset for b, R in zip(self.palms, rotations)])
        return grips, rotations, self.clearance()

    def clearance(self, distmax=0.1):
        """Smallest hand-box to obstacle distance, capped at distmax (pairs farther apart are not queried)."""
        d, m = self.configuration.data, self.model
        best = distmax
        for s in SIDES:
            for g in self.hand[s]:
                for o in self.obstacles:
                    if np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[o]) > m.geom_rbound[g] + m.geom_rbound[o] + best:
                        continue
                    best = min(best, robust_geom_distance(m, d, g, o, distmax, self._fromto))
        return best
