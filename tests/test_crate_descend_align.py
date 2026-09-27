"""Staged reach: align across the wall before descending, and hold the across-wall feedback below the rim.

Offline warehouse pilots (2026-09-24): the across-wall grip error was 2-6 cm when the descend started; at
3 cm grasp depth a pad met the wall face and the lateral correction wound up to tracking_cap against it,
tipping the box in 6/10 rounds. The fixture's crate is rotated 90 deg, so "across the wall" is world y.
"""

from __future__ import annotations

import numpy as np
import pytest
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


def test_reach_feedforward_shifts_the_commanded_hands_outside_the_feedback():
    plain, state = setup()
    shifted, _ = setup("visual.reach_feedforward_forward=0.03", "visual.reach_feedforward_up=-0.04")
    for policy in (plain, shifted):
        s = dict(
            joints=state["joints"].copy(),
            pelvis=dict(pos=list(state["pelvis"]["pos"]), quat=list(state["pelvis"]["quat"])),
        )
        for now in np.arange(0, 2.0, 0.05):  # well into the reach, no tracking error injected
            command = policy.step(now, estimate(now, s), s, [0.08] * 2)
            s = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3].copy(), quat=command.root_quaternion))
        assert policy.phase == "reach"
    assert np.allclose(plain.feedback, 0, atol=1e-6) and np.allclose(shifted.feedback, 0, atol=1e-6)
    delta = shifted.body.tools(shifted.x) - plain.body.tools(plain.x)
    assert np.allclose(delta[:, 2], -0.04, atol=0.01)  # both hands commanded 4 cm lower
    assert np.all(delta @ shifted.outward > 0.015)  # and forward along the body-to-object axis


def test_common_mode_hold_lets_the_pair_chase_a_shared_drift_below_the_rim():
    """2026-09-27: the stalled sweep rounds had the hand PAIR drifting 3-5 cm to one side during the descend while
    the hold froze the correction. With lateral_hold_common_mode the shared shift stays corrected below the rim."""
    policy, state = setup(
        "visual.lateral_hold_below_rim=true", "visual.lateral_hold_common_mode=true", "visual.descend_z_guard=false"
    )
    phases = set()
    log, _ = run(
        policy, state, 12.0, lambda phase: np.array([0.0, 0.03, 0.0]) if phase == "descend" else np.zeros(3), phases
    )
    held = [entry[2] for entry in log if entry[0] == "descend" and entry[3] is not None and np.all(entry[3])]
    assert len(held) > 3
    assert np.max(np.abs(np.diff(np.array(held)[:, :, 1], axis=0))) > 1e-4  # the pair's across feedback still moves


def test_common_mode_hold_still_removes_the_spread_change():
    policy, state = setup(
        "visual.lateral_hold_below_rim=true", "visual.lateral_hold_common_mode=true", "visual.descend_z_guard=false"
    )
    policy.step(0, estimate(0, state), state, [0.08] * 2)
    tools = policy.body.tools(policy.x)
    both = np.array([True, True])
    # Opposite per-hand offsets across the wall: a pure spread error. A held hand must not chase it.
    spread = tools + np.array([[0.0, 0.03, 0.0], [0.0, -0.03, 0.0]])
    policy.advance(tools, policy.attitudes, spread, 0.05, hold_across=(both, ACROSS))
    assert np.max(np.abs(policy.feedback @ ACROSS)) < 1e-9
    # The same offset on both hands: a shared drift. The pair's across feedback moves by it.
    shared = tools + np.array([[0.0, 0.03, 0.0], [0.0, 0.03, 0.0]])
    policy.advance(tools, policy.attitudes, shared, 0.05, hold_across=(both, ACROSS))
    across = policy.feedback @ ACROSS
    assert across[0] == pytest.approx(across[1]) and abs(across[0]) > 1e-4


def _stall_the_descend(*overrides, seconds=30.0):
    """Drive the staged reach with a measurement 5 cm across the wall from the reference during the descend (more
    than the profile's tracking cap covers, so the grips never enter the capture window and the error stops
    improving); return (policy, phases seen, last command)."""
    policy, state = setup("visual.descend_z_guard=false", *overrides)
    phases, now, command = [], 0.0, None
    while now < seconds:
        command = policy.step(now, estimate(now, state), state, [0.08] * 2)
        phases.append(policy.phase)
        if command.finished or policy.phase in ("close", "lift"):
            break
        offset = np.array([0.0, 0.05, 0.0]) if policy.phase == "descend" else np.zeros(3)
        state = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3] + offset, quat=command.root_quaternion))
        now += 0.05
    return policy, phases, command


def test_descend_stall_is_terminal_without_retries():
    policy, phases, command = _stall_the_descend()
    assert command.finished and command.reason == "reach_stalled"
    assert policy.retries == dict(descend=0, reseat=0)


def test_descend_stall_backs_up_to_the_pregrasp_point_and_descends_again():
    policy, phases, command = _stall_the_descend("visual.descend_retries=1")
    transitions = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    # the profile has no align pass of its own (descend_align_tolerance 0); the retry always backs up through one
    assert transitions[:4] == ["reach", "descend", "align", "descend"], transitions
    assert policy.retries["descend"] == 1 and policy.diagnostics["retry"]["after"] == "reach_stalled"
    assert command.finished and command.reason == "reach_stalled"  # the second attempt stalls the same way, once


def test_empty_close_reopens_then_re_aligns_when_a_reseat_is_allowed():
    policy, state = setup("visual.reseat_retries=1")
    policy.step(0, estimate(0, state), state, [0.08] * 2)
    policy.closed_targets = policy.body.tools(policy.x)
    policy.phase, policy.elapsed, policy.closure = "close", policy.config.visual.close_seconds + 0.2, 1.0
    command = policy.step(0.05, estimate(0.05, state), state, [0.001] * 2)  # jaws shut on nothing
    assert not command.finished and policy.phase == "reopen" and policy.retries["reseat"] == 1
    now = 0.05
    while policy.phase == "reopen":
        now += 0.05
        command = policy.step(now, estimate(now, state), state, [0.001] * 2)
        state = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3], quat=command.root_quaternion))
    assert policy.phase == "align" and command.closure == 0.0
    # the second empty close ends the round
    policy.closed_targets = policy.body.tools(policy.x)
    policy.phase, policy.elapsed, policy.closure = "close", policy.config.visual.close_seconds + 0.2, 1.0
    command = policy.step(now + 0.05, estimate(now + 0.05, state), state, [0.001] * 2)
    assert command.finished and command.reason == "empty_gripper"


def _closed(policy, state, openings=(0.015, 0.015)):
    """Put the policy at the end of its close with the fixture's hands where they are (far above the grasp points)."""
    policy.closed_targets = policy.body.tools(policy.x)
    policy.phase, policy.elapsed, policy.closure = "close", policy.config.visual.close_seconds + 0.2, 1.0
    return policy.step(0.05, estimate(0.05, state), state, list(openings))


def test_seat_check_spends_a_reseat_on_a_shallow_grip_and_lifts_when_none_is_left():
    policy, state = setup("visual.seat_depth=0.01", "visual.reseat_retries=1")
    policy.step(0, estimate(0, state), state, [0.08] * 2)
    command = _closed(policy, state)
    assert policy.phase == "reopen" and policy.retries["reseat"] == 1 and not command.finished
    assert policy.diagnostics["seat"]["ok"] is False and min(policy.diagnostics["seat"]["depth"]) < 0.01
    assert policy.diagnostics["retry"]["after"] == "seat_rejected"
    # no reseat left: the same shallow grip goes on to the lift, as it always did
    policy.phase = "close"
    command = _closed(policy, state)
    assert policy.phase == "lift" and not command.finished


def test_seat_check_is_off_by_default():
    policy, state = setup("visual.reseat_retries=1")
    policy.step(0, estimate(0, state), state, [0.08] * 2)
    _closed(policy, state)
    assert policy.phase == "lift" and "seat" not in policy.diagnostics
