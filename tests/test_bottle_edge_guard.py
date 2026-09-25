"""CPU checks for the simple expert's guard walk: the table edge, the steering, the table-band front and the record."""

import mujoco
import numpy as np
import pytest

from locomani.primitives.foot_watch import FootWatch
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
from locomani.tasks.bottle.edge_guard import EdgeGuard, facing_edge, straight_walk
from locomani.tasks.bottle.simple_config import SimpleGraspConfig

# A quarter arc in the x-z plane below the pelvis: x = 0.5 sin(a), z = -0.5 cos(a). Strictly convex, so every
# vertex survives MuJoCo's hull; its bounding box reaches x 0.5 at any height, the arc itself only at the bottom.
_ARC = " ".join(
    f"{0.5 * np.sin(a):.5f} {y} {-0.5 * np.cos(a):.5f}" for a in np.radians(np.arange(0, 91)) for y in (-0.01, 0.01)
)


def _robot(extra=""):
    """A 'pelvis' free body carrying every G1 joint (all at the origin) plus the given geoms."""
    chain = "".join(
        f'<body name="{n.replace("_joint", "_link")}"><joint name="{n}" axis="0 1 0"/>'
        '<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'
        for n in MUJOCO_JOINT_NAMES
    )
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><asset><mesh name="arc" vertex="{_ARC}"/></asset><worldbody>
        <geom type="plane" size="2 2 0.1"/>
        <body name="pelvis" pos="0 0 0.8"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
        {extra}{chain}{"</body>" * len(MUJOCO_JOINT_NAMES)}</body></worldbody></mujoco>"""
    )


def _state(x=0.0, joints=None):
    return dict(
        pelvis=dict(pos=[x, 0.0, 0.8], quat=[1.0, 0.0, 0.0, 0.0]), joints=np.zeros(29) if joints is None else joints
    )


def test_facing_edge_is_the_edge_the_heading_walks_into():
    # Walking -y toward a 1 m table from its +y side (the bottle scene's approach, heading -86 deg).
    edge = facing_edge([0.0, 0.0], [0.503, 0.503], [0.07, -0.998, 0.0])
    np.testing.assert_allclose(edge.axis, [0.0, -1.0])
    assert edge.line == pytest.approx(-0.503)
    assert edge.clearance([0.3, 0.70]) == pytest.approx(0.197)  # robot side: positive
    assert edge.clearance([0.3, 0.45]) == pytest.approx(-0.053)  # over the top: negative
    shifted = facing_edge([1.0, 2.0], [0.4, 0.3], [1.0, 0.1])  # walking +x into its -x edge
    np.testing.assert_allclose(shifted.axis, [1.0, 0.0])
    assert shifted.clearance([0.5, 2.0]) == pytest.approx(0.1)


def test_straight_walk_heads_at_the_stand_and_falls_back_to_the_heading():
    heading = [0.0, -1.0, 0.0]
    np.testing.assert_allclose(straight_walk([0.2, 1.5], [0.2, 0.7], heading), [0.0, -1.0], atol=1e-12)
    diagonal = straight_walk([0.34, 1.5], [0.2, 0.7], heading)  # 0.14 m off: a 10 deg diagonal back onto the line
    np.testing.assert_allclose(diagonal, np.array([-0.14, -0.8]) / np.hypot(0.14, 0.8))
    # A stand beside the start, or behind it, is not walked at sideways: the heading instead.
    np.testing.assert_allclose(straight_walk([0.2, 0.75], [0.6, 0.7], heading), [0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(straight_walk([0.2, 0.5], [0.2, 0.7], heading), [0.0, -1.0], atol=1e-12)


def test_frontmost_band_measures_mesh_vertices_at_the_band_height():
    model = _robot('<geom type="mesh" mesh="arc" pos="0 0 0"/><geom type="box" size="0.02 0.02 0.02" pos="0.1 0 0.1"/>')
    watch = FootWatch(model=model)
    # Band 0.35-0.45 m below the pelvis: the arc's front there is at 0.5 sin(acos(0.7)) = 0.357, where its box says 0.5.
    front, body = watch.frontmost_band(_state(), np.zeros(2), np.array([1.0, 0.0]), 0.35, 0.45)
    assert front == pytest.approx(0.357, abs=0.012)
    assert body == "pelvis"
    # The box (0.88-0.92 m) is the only geometry above the arc's top (0.8 m); below the arc there is nothing.
    front, _ = watch.frontmost_band(_state(x=1.0), np.zeros(2), np.array([1.0, 0.0]), 0.89, 0.91)
    assert front == pytest.approx(1.12)
    assert watch.frontmost_band(_state(), np.zeros(2), np.array([1.0, 0.0]), 0.0, 0.25) == (-np.inf, None)


class _Watch:
    """FootWatch's landing/stride interface on scripted values."""

    def __init__(self, ticks, band_front):
        self.ticks, self.band_front = iter(ticks), band_front
        self.landings, self.last_strides = [], []

    def update(self, _s, _x0, _h):
        landed, front, stride = next(self.ticks)
        if landed:
            self.landings.append(front)
        return landed, front, stride

    def frontmost_band(self, *_args):
        return self.band_front, "right_wrist_yaw_link"


def test_edge_guard_brakes_at_the_deciding_landing_and_records_the_realised_extra():
    edge = facing_edge([0.0, 0.0], [0.5, 0.5], [1.0, 0.0])  # line at x -0.5, walking +x
    guard = EdgeGuard(_robot(), edge, 0.62, SimpleGraspConfig(approach_stop="guard"))
    assert guard.band == pytest.approx((0.50, 0.65)) and guard.walk_speed == 0.25
    # -1.00 + 0.14 floor + 0.20 < -0.53: walk on. -0.80 + 0.20 stride + 0.20 >= -0.53: brake on that landing.
    guard.watch = _Watch([(True, -1.00, None), (False, -1.00, None), (True, -0.80, 0.20)], band_front=-0.90)
    state = dict(_state(x=-0.9), t=1.0, seq=7)
    assert guard.step(state) is None
    assert guard.step(state) is None
    assert guard.step(dict(state, t=1.5, seq=9)) == "guard brake"
    assert guard.brake["seq"] == 9 and guard.brake["stride_m"] == pytest.approx(0.20)
    assert guard.brake["front_ankle_gap_m"] == pytest.approx(0.30)
    assert guard.brake["pelvis_gap_m"] == pytest.approx(0.40)
    guard.watch.band_front = -0.62  # the stand after the post-IDLE step
    assert guard.finish(_state(x=-0.72)) == pytest.approx(0.12)
    record = guard.record()
    assert record["realised_extra_m"] == pytest.approx(0.18)  # band front 18 cm past the braking ankle
    assert record["closest_band_gap_m"] == pytest.approx(0.12)
    assert record["closest_band_body"] == "right_wrist_yaw_link"
    assert record["final"]["pelvis_gap_m"] == pytest.approx(0.22)


def test_edge_guard_stops_the_walk_when_the_band_is_already_past_the_edge():
    edge = facing_edge([0.0, 0.0], [0.5, 0.5], [1.0, 0.0])
    guard = EdgeGuard(_robot(), edge, 0.62, SimpleGraspConfig(approach_stop="guard"))
    guard.watch = _Watch([(False, -0.7, None)], band_front=-0.45)
    assert guard.step(dict(_state(), t=0.0)) == "guard crossed"
    assert guard.brake is None and guard.closest[0] == pytest.approx(-0.05)


def test_edge_guard_reads_its_parameters_from_the_simple_config():
    cfg = SimpleGraspConfig(
        approach_stop="guard",
        walk_speed=0.3,
        guard_extra_m=0.15,
        guard_margin_m=0.02,
        guard_stride0_m=0.10,
        guard_band_below_m=0.20,
        guard_band_above_m=0.05,
    )
    guard = EdgeGuard(_robot(), facing_edge([0, 0], [0.5, 0.5], [0, 1]), 0.70, cfg)
    assert (guard.rule.extra, guard.rule.margin, guard.rule.stride0) == (0.15, 0.02, 0.10)
    assert guard.band == pytest.approx((0.50, 0.75)) and guard.walk_speed == 0.3
    # The direction is fixed on the first walking tick and then held wherever the pelvis goes.
    first = guard.direction([0.1, 0.0], [0.0, 1.0], [0, 1])
    np.testing.assert_allclose(guard.direction([0.5, 0.4], [0.0, 1.0], [0, 1]), first)
    assert guard.record()["walk_direction"] == pytest.approx(first.tolist())
    guard = EdgeGuard(_robot(), facing_edge([0, 0], [0.5, 0.5], [0, 1]), 0.70, cfg)
    record = guard.record()
    assert record["brake"] is None and record["final"] is None and record["realised_extra_m"] is None
