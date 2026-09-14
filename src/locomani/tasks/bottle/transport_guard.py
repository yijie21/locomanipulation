"""Kinematic sweep of collision geoms on the active arm, wrist and fingers.

Uses a private MjData, never the simulator or the IK scratch data. This is a
local reference-motion screen, not a dynamic or whole-body safety guarantee.
Only use during transport: intended grasp contact must not be rejected.
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class SweepResult:
    clear: bool
    distance: float | None
    link: str | None = None
    obstacle: str | None = None
    fraction: float | None = None
    samples: int = 0

    def to_dict(self):
        return asdict(self)


class TransportGuard:
    def __init__(
        self,
        ik,
        target_key,
        target_body,
        surface_bodies=(),
        margin=0.004,
        joint_step=0.04,
        include_body=False,
        finger_only_target_contact=False,
    ):
        if not (math.isfinite(margin) and margin >= 0 and math.isfinite(joint_step) and joint_step > 0):
            raise ValueError("Invalid transport sweep margin/step")
        self.ik, self.model = ik, deepcopy(ik.model)
        self.data = mujoco.MjData(self.model)
        self.target_key, self.margin, self.joint_step = target_key, margin, joint_step
        self.finger_only_target_contact = finger_only_target_contact
        m = self.model

        def body(name):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Missing collision body: {name}")
            return bid

        def descendants(root):
            result = {root}
            for bid in range(root + 1, m.nbody):
                if int(m.body_parentid[bid]) in result:
                    result.add(bid)
            return result

        def geoms(bodies):
            return [
                g
                for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
            ]

        self.hands, self.finger_qadr, self.finger_geoms = {}, {}, set()
        for side in ["left", "right"]:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, side + "_shoulder_pitch_link")
            if bid < 0:
                continue
            self.hands[side] = geoms(descendants(bid))
            self.finger_geoms.update(
                g
                for g in self.hands[side]
                if m.body(int(m.geom_bodyid[g])).name.startswith((side + "_dex1_finger_", side + "_hand_"))
            )
            names = ["gripper_1", "gripper_2"]
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, side + "_hand_gripper_1_joint") < 0:
                names = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{n}_joint") for n in names]
            self.finger_qadr[side] = [int(m.jnt_qposadr[j]) for j in ids if j >= 0]
        if include_body:
            base_joint = int(np.flatnonzero(m.jnt_qposadr == ik.base_qadr)[0])
            robot_geoms = set(geoms(descendants(int(m.jnt_bodyid[base_joint]))))
            arm_geoms = set(g for group in self.hands.values() for g in group)
            self.hands["body"] = sorted(robot_geoms - arm_geoms)
        target = body(target_body)
        self.obstacles = geoms(descendants(target))
        self.target_geoms = frozenset(self.obstacles)
        for name in surface_bodies:
            self.obstacles.extend(geoms(descendants(body(name))))
        self.obstacles = sorted(set(self.obstacles))
        jid = int(m.body_jntadr[target])
        if jid < 0 or m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Transport target must have a free joint")
        self.target_qadr = int(m.jnt_qposadr[jid])
        if not self.obstacles:
            raise ValueError("No obstacle collision geometry")

        # Private collision model: arm↔obstacle pairs only. geomDistance can
        # report spurious zero for separated mesh pairs in our runtime; use
        # the simulation's contact generation with an explicit margin.
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        for arm_geoms in self.hands.values():
            m.geom_contype[arm_geoms] = 1
            m.geom_conaffinity[arm_geoms] = 2
            m.geom_margin[arm_geoms] = margin
        m.geom_contype[self.obstacles] = 2
        m.geom_conaffinity[self.obstacles] = 1
        m.geom_margin[self.obstacles] = margin

    def check(self, state, candidate, side, *, allow_target_contact=False, candidate_base=None):
        """Sweep joints and optional base motion from the supplied measured state.

        Without candidate_base the measured base remains fixed. Finger poses
        always remain measured; closure screening is a separate query.

        Candidate is a full MuJoCo-order body joint vector. Each interval is
        at most joint_step rad in any joint; margin covers a finite sampling
        approximation, not a proven continuous collision bound.
        """
        if side not in self.hands or not self.hands[side]:
            raise ValueError(f"No arm collision geometry for {side}")
        q0 = np.asarray(state["joints"], dtype=float)
        q1 = np.asarray(candidate, dtype=float)
        if q0.shape != self.ik.qadr.shape or q1.shape != q0.shape or not np.isfinite([q0, q1]).all():
            raise ValueError("Invalid sweep joint vectors")
        d, m = self.data, self.model
        d.qpos[:] = m.qpos0
        base = self.ik.base_qadr
        d.qpos[base : base + 7] = [*state["pelvis"]["pos"], *state["pelvis"]["quat"]]
        base_start = d.qpos[base : base + 7].copy()
        rotations = None
        if candidate_base is not None:
            from scipy.spatial.transform import Rotation, Slerp

            base_end = np.asarray(candidate_base, dtype=float)
            if base_end.shape != (7,) or not np.isfinite(base_end).all():
                raise ValueError("candidate base must contain xyz and a finite quaternion")
            rotations = Slerp([0, 1], Rotation.from_quat(np.array([base_start[3:], base_end[3:]])[:, [1, 2, 3, 0]]))
        obj = state["objects"][self.target_key]
        d.qpos[self.target_qadr : self.target_qadr + 7] = [*obj["pos"], *obj["quat"]]
        for hand, adrs in self.finger_qadr.items():
            values = state["palms"][hand]["hand_q"]
            if len(values) != len(adrs):
                raise ValueError(f"Missing measured {hand} finger joints")
            d.qpos[adrs] = values
        if not np.isfinite(d.qpos).all():
            raise ValueError("Non-finite sweep state")
        steps = max(1, int(math.ceil(float(np.max(np.abs(q1 - q0))) / self.joint_step)))
        if rotations is not None:
            translation_steps = np.linalg.norm(base_end[:3] - base_start[:3]) / 0.02
            rotation_steps = np.linalg.norm((rotations(1) * rotations(0).inv()).as_rotvec()) / self.joint_step
            steps = max(steps, int(math.ceil(translation_steps)), int(math.ceil(rotation_steps)))
        # Bound unexpected invalid commands without doing unbounded work.
        if steps > 200:
            raise ValueError("Sweep reference jump exceeds 200 intervals")
        for i in range(steps + 1):
            fraction = i / steps
            d.qpos[self.ik.qadr] = q0 + fraction * (q1 - q0)
            if rotations is not None:
                d.qpos[base : base + 3] = base_start[:3] + fraction * (base_end[:3] - base_start[:3])
                d.qpos[base + 3 : base + 7] = rotations(fraction).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            hits = []
            for contact in d.contact:
                g, other = int(contact.geom1), int(contact.geom2)
                if other in self.hands[side]:
                    g, other = other, g
                if (
                    g in self.hands[side]
                    and other in self.obstacles
                    and contact.dist <= self.margin
                    and not (
                        allow_target_contact
                        and other in self.target_geoms
                        and (not self.finger_only_target_contact or g in self.finger_geoms)
                    )
                ):
                    hits.append((float(contact.dist), g, other))
            if hits:
                dist, g, other = min(hits)
                return SweepResult(
                    False,
                    dist,
                    m.body(int(m.geom_bodyid[g])).name,
                    m.body(int(m.geom_bodyid[other])).name,
                    fraction,
                    steps + 1,
                )
        return SweepResult(True, None, samples=steps + 1)
