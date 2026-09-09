"""Motion primitives for the scripted collector (P1.2).

LEGS's insight is that a whole-body controller already solves the hard part:
given a heading, a walk command and a wrist target, SONIC produces plausible
whole-body motion. What the data engine has to supply is only *when* to issue
which command — and privileged simulator state is enough to decide that. So
there is no grasp generator here, just a fixed pre-grasp offset expressed
relative to the object's known pose, exactly as LEGS does.

Five primitives, each a closed loop on the privileged state:

    walk_to(xy)     drive the pelvis to a standing spot, in a straight line
    navigate()      follow the occupancy map's distance field around obstacles
    turn_to(yaw)    rotate in place
    reach(pos)      put a palm at a world point, via a root-frame wrist target
    grasp(side)     close the hand and confirm contact
    release(side)   open the hand

Frames. Wrist targets go out in the *root* (pelvis) frame — see the wire-format
note in agent_server.py — so every world-frame target is rotated by the pelvis
yaw before it is sent. `movement` and `facing` go out in the world frame, which
is what pico_manager_thread_server sends (its YawAccumulator integrates a
world-frame heading vector, planner loop line 1826-1830).

Split across five files (refactor step 4b family 4, pure extraction;
decision 9's file-size cap; zero method-body changes throughout): this file
keeps `__init__`/`close` and composes `Primitives` from four mixins --
`PrimitivesObservationMixin` (primitives_observation.py: `obs`/`pelvis_xy`/
`pelvis_yaw`/`object_pos`/`world_to_root`/`_rot`/`world_to_root_full`/`log`/
`_facing`), `PrimitivesWalkMixin` (primitives_walk.py: `walk_to`/`turn_to`/
`navigate`/`goto_object`/`face_for_grasp`/`approach`), `PrimitivesArmsMixin`
(primitives_arms.py: `_legacy_seed`/`reach`/`servo_grasp`/`grasp`/`release`),
`PrimitivesCrouchMixin` (primitives_crouch.py: `crouch`/`stand_up`/`track`/
`crouch_needed`/`_hold_pelvis`/`_maintain_crouch`/`tuck_arms`/`carry_pose`).
All the tuning constants and kinematic helper functions (`walking_arm_pose`,
`upper_body_17`, `upper_body_from_mj`, `_opposed`, `yaw_of`, `_axis_angle`,
`_from_axis_angle`, `_rot_scale`, `_rot_lerp`, `_quat_of`, `wrap`, and every
`ALL_CAPS` tuning knob) moved to `primitives_tuning.py`, the leaf every one
of these five files (this one included) imports from -- see that module's
own docstring for why this needed no circular-import handling, unlike
families 2/3/5. Every name that used to live here and moved is re-exported
below, so `from locomani.backends.sonic.primitives import X` still works;
`Primitives.__init__` is monkeypatched directly by three diagnostic
tools (`grasp_gym.Primitives.__init__ = ...` and similar) and stays here
unaffected by the mixin split, since a class-attribute patch mutates the
one shared class object regardless of which base class actually defines the
method being replaced.
"""

from __future__ import annotations

import numpy as np

from locomani.backends.sonic.agent_server import (
    HAND_OPEN,
    AgentCommander,
    LocomotionMode,
    StreamMode,
)
from locomani.backends.sonic.primitives_arms import PrimitivesArmsMixin
from locomani.backends.sonic.primitives_crouch import PrimitivesCrouchMixin
from locomani.backends.sonic.primitives_observation import PrimitivesObservationMixin
from locomani.backends.sonic.primitives_tuning import (
    CONTROL_HZ,
    DT,
    GRASP_ABOVE,
    GRASP_CENTRE_OFFSET,
    HAND_CLOSED_LEFT,
    HAND_CLOSED_RIGHT,
    LIFT_HEIGHT,
    PREGRASP_ABOVE,
    PREGRASP_BACK,
    READY_ARM,
    READY_ARM_NEAR,
    READY_UPPER,
    READY_WRISTS,
    STAND_OFF,
    STANDING_PELVIS_Z,
    TUCK_WRISTS,
    UPPER_BODY_NAMES,
    upper_body_17,
    upper_body_from_mj,
    walking_arm_pose,
    wrap,
    yaw_of,
)
from locomani.backends.sonic.primitives_walk import PrimitivesWalkMixin
from locomani.primitives import navigator as nav
from locomani.sim.privileged_pub import PrivilegedSubscriber


class Primitives(PrimitivesObservationMixin, PrimitivesWalkMixin, PrimitivesArmsMixin, PrimitivesCrouchMixin):
    """One episode's worth of scripted control over the live stack."""

    def __init__(
        self,
        commander: AgentCommander | None = None,
        state: PrivilegedSubscriber | None = None,
        verbose: bool = True,
        meta: dict | None = None,
    ):
        self.cmd = commander or AgentCommander()
        self.state = state or PrivilegedSubscriber()
        self.verbose = verbose
        self.aborted: str | None = None
        self.last_cmd: np.ndarray | None = None  # for reach(resume=True)
        self.crouch_cmd: np.ndarray | None = None  # the arm holding a squat
        # Yaw anchor. Every loop that sends facing=(current yaw) is a random
        # walk — "hold whatever heading I have now", re-read each tick, drifts
        # freely. Measured: 50 deg of rotation between planning a grasp and
        # reaching for it, which put the pregrasp behind the shoulder. When
        # set, every stationary loop faces THIS heading instead.
        self.hold_yaw: float | None = None
        # Active crouch: (side, target pelvis z, cmd at convergence). A static
        # pelvis-frame wrist command cannot hold a squat — when the pelvis
        # rises, the command rises with it and the controller happily stands
        # back up (measured: 0.65 -> 0.78 during the staging reach). The hold
        # is closed-loop: each tick of any loop, the holding arm's z is biased
        # by how far the pelvis has risen above its target.
        self.crouch_hold: tuple[str, float, np.ndarray] | None = None
        # The map is static — the surfaces never move — so it is built once per
        # process and the per-episode cost is one fast-march per leg.
        self.occ = self.planner = None
        if meta is not None:
            self.occ, self.planner = nav.build(meta)

    def close(self) -> None:
        self.cmd.close()
        self.state.close()


__all__ = [
    "Primitives",
    "CONTROL_HZ",
    "DT",
    "STAND_OFF",
    "PREGRASP_ABOVE",
    "GRASP_ABOVE",
    "LIFT_HEIGHT",
    "HAND_CLOSED_LEFT",
    "HAND_CLOSED_RIGHT",
    "LocomotionMode",
    "StreamMode",
    "yaw_of",
    "wrap",
]
