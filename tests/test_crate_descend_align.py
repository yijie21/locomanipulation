"""Staged reach: align across the wall before descending, and hold the across-wall feedback below the rim.

Offline warehouse pilots (2026-09-24): the across-wall grip error was 2-6 cm when the descend started; at
3 cm grasp depth a pad met the wall face and the lateral correction wound up to tracking_cap against it,
tipping the box in 6/10 rounds. The fixture's crate is rotated 90 deg, so "across the wall" is world y.
"""

from __future__ import annotations

import numpy as np
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani import profiles
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.coordinated_reach import CoordinatedReach
from locomani.tasks.crate.perception import CrateEstimate

ACROSS = np.array([0.0, 1.0, 0.0])


def setup(*overrides):
    config = profiles.load("task", "crate_visual_low_reach").with_overrides(
        ["visual.approach_speed=0.1", "visual.reach_path=staged", *overrides]
    )
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.1, 0.2, -0.1
    q[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
    q[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    state = dict(joints=q, pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    solver = BimanualIK(scene_ik(), waist_pitch_max=0.52, waist_dofs="all")
    return CoordinatedReach(
        solver, config, state, reference_joints=q, reference_quaternion=state["pelvis"]["quat"]
    ), state


def estimate(now, state):
    base = Rotation.from_quat(np.array(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
    return CrateEstimate(
        base.T @ (np.array([0.36, 0.02, 0.385]) - state["pelvis"]["pos"]),
        base.T @ Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        now,
        None,
        {},
    )


def run(policy, state, seconds, offset, phases, dt=0.05):
    """Feed the reference back as the measurement, displaced `offset` in world (a tracking error the reference
    cannot see until the feedback covers it); record (phase, across error, feedback) per tick."""
    log = []
    now = 0.0
    while now < seconds and policy.phase not in ("close", "lift"):  # the fixture has no object to lift
        command = policy.step(now, estimate(now, state), state, [0.08] * 2)
        assert not command.finished, command.reason
        log.append(
            (
                policy.phase,
                policy.diagnostics.get("align_across"),
                policy.feedback.copy(),
                policy.diagnostics.get("lateral_hold"),
            )
        )
        state = dict(
            joints=command.joints, pelvis=dict(pos=policy.x[:3] + offset(policy.phase), quat=command.root_quaternion)
        )
        phases.add(policy.phase)
        now += dt
    return log, state


def test_alignment_holds_above_the_rim_until_the_across_error_is_covered():
    policy, state = setup("visual.descend_align_tolerance=0.01")
    phases = set()
    # the error appears once the hands hold at the pregrasp point (the reach's own feedback would cover an earlier one)
    log, _ = run(
        policy,
        state,
        12.0,
        lambda phase: np.array([0.0, 0.03, 0.0]) if phase in ("align", "descend") else np.zeros(3),
        phases,
    )
    assert "align" in phases and "descend" in phases, phases
    aligning = [entry for entry in log if entry[0] == "align" and entry[1] is not None]
    assert np.max(np.abs(aligning[0][1])) > 0.02  # entered with the 3 cm across error
    assert np.max(np.abs(aligning[-1][1])) < 0.02  # the error was being covered, tick by tick
    assert policy.align_residual is not None and np.max(np.abs(policy.align_residual)) <= 0.01  # left it centred
    assert len(aligning) * 0.05 < policy.config.visual.descend_align_timeout  # not on the timeout


def test_alignment_is_off_by_default():
    policy, state = setup()
    phases = set()
    run(policy, state, 8.0, lambda phase: np.zeros(3), phases)
    assert "descend" in phases and "align" not in phases


def test_across_wall_feedback_stops_changing_below_the_rim():
    policy, state = setup("visual.lateral_hold_below_rim=true", "visual.descend_z_guard=false")
    phases = set()
    # Apply the across offset only once the hands are descending; below the rim the feedback must not chase it.
    log, _ = run(
        policy, state, 12.0, lambda phase: np.array([0.0, 0.03, 0.0]) if phase == "descend" else np.zeros(3), phases
    )
    descend = [entry for entry in log if entry[0] == "descend"]
    assert len(descend) > 10
    held = [entry[2] for entry in descend if entry[3] is not None and np.all(entry[3])]
    assert len(held) > 3  # both hands reported below the rim
    assert np.max(np.abs(np.diff(np.array(held)[:, :, 1], axis=0))) < 1e-4  # across component frozen there
    free = [entry[2] for entry in descend if entry[3] is not None and not np.any(entry[3])]
    assert free and np.max(np.abs(np.diff(np.array(free)[:, :, 1], axis=0))) > 1e-4  # but it moved above the rim
