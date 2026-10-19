"""Scene checks before any loop runs (scene_discovery.lint, plan docs/handoff/scene-rsi-plan-20261016.md §12)."""

from __future__ import annotations

import numpy as np
import pytest

from locomani.scene_discovery import lint

ROT_X_90 = [float(np.cos(np.pi / 4)), float(np.sin(np.pi / 4)), 0.0, 0.0]  # body y up


def scene():
    surfaces = {
        "table": dict(top=0.70, xy=[0.0, 0.0], half_extent=[0.5, 0.4], yaw=0.0),
        "shelf": dict(top=0.40, xy=[3.0, 0.0], half_extent=[0.4, 0.2], yaw=0.0),
    }
    objects = {
        "wallet": dict(free=True, surface="table", init_pos=[0.1, 0.0, 0.7138], init_quat=[1.0, 0, 0, 0],
                       box=dict(half_extent=[0.054, 0.043, 0.0138], centre=[0, 0, 0]),
                       semantic=dict(upright_axis=[0.0, 0.0, 1.0])),
        "book": dict(free=True, surface="shelf", init_pos=[3.0, 0.0, 0.50], init_quat=ROT_X_90,
                     box=dict(half_extent=[0.07, 0.10, 0.012], centre=[0, 0, 0]),
                     semantic=dict(upright_axis=[0.0, 0.0, 1.0])),
        "mug": dict(free=True, surface="table", init_pos=[-0.2, 0.1, 0.75], init_quat=[1.0, 0, 0, 0],
                    box=dict(half_extent=[0.04, 0.04, 0.05], centre=[0, 0, 0]), semantic=dict(upright_axis=[0, 0, 1])),
    }  # fmt: skip
    return dict(surfaces=surfaces, objects=objects)


def test_an_object_whose_resting_half_height_is_under_a_pad_is_flat():
    flat = lint.flat_objects(scene())
    assert set(flat) == {"wallet"}  # the book stands on its 20 cm side, the mug is 10 cm tall
    assert flat["wallet"]["margin"] == pytest.approx(0.0138 - lint.DEX1_PAD_HALF_M)
    assert flat["wallet"]["support_top"] == 0.70


def test_the_resting_height_is_along_the_body_axis_nearest_up():
    book = scene()["objects"]["book"]
    assert lint.resting_half_height(book) == pytest.approx(0.10)  # body y is up
    half = book["box"]["half_extent"]
    assert lint.resting_footprint(book) == pytest.approx(sorted([2 * half[0], 2 * half[2]]))  # the other two axes


def test_an_upright_axis_off_the_resting_pose_is_reported_without_changing_the_scene():
    meta = scene()
    found = lint.upright_mismatch(meta)
    assert set(found) == {"book"}
    assert found["book"][1] == [0.0, 1.0, 0.0]
    assert meta["objects"]["book"]["semantic"]["upright_axis"] == [0.0, 0.0, 1.0]


def test_a_goal_that_holds_at_the_scene_pose_or_a_start():
    meta = scene()
    assert lint.goal_holds(["on(mug, table)"], meta)
    assert not lint.goal_holds(["on(mug, shelf)"], meta)
    start = dict(robot_xy=[1.0, 1.0], placements={"mug": [[3.1, 0.0, 0.45], [1.0, 0, 0, 0]]})
    assert lint.goal_holds(["on(mug, shelf)"], meta, start)
    assert not lint.goal_holds(["held(mug)"], meta, start)  # nothing touches it at a start


def test_the_robot_has_no_scene_pose():
    meta = scene()
    meta["room"] = dict(rooms={"bedroom": dict(polygon=[[-5, -5], [5, -5], [5, 5], [-5, 5]])})
    assert not lint.goal_holds(["in_room(robot, bedroom)"], meta)
    assert lint.goal_holds(["in_room(robot, bedroom)"], meta, dict(robot_xy=[0.0, 0.0], placements={}))
    assert lint.goal_holds(["in_room(robot, bedroom)", "on(mug, table)"], meta)  # the robot's part left out


def test_an_object_resting_tilted_off_its_upright_axis():
    meta = scene()
    a = np.radians(36)
    meta["objects"]["mug"]["init_quat"] = [float(np.cos(a / 2)), float(np.sin(a / 2)), 0.0, 0.0]
    assert lint.tilted_at_rest(meta) == {"book": 90.0, "mug": 36.0}
    assert "mug" not in lint.upright_mismatch(meta)  # z is still its body axis nearest up
