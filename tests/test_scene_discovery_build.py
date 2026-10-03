"""MJCF numbers for a scene file's build, and the drop test on a two-object world (one rests, one slides)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from locomani.scene_discovery import drop_test, mjcf


def test_a_slab_top_is_the_surface_top_and_follows_its_yaw():
    sl = mjcf.slab({"xy": [1.0, 2.0], "top": 0.75, "half_extent": [0.6, 0.3], "yaw": np.pi / 2})
    assert sl["pos"][2] + sl["size"][2] == pytest.approx(0.75)
    assert sl["size"][:2] == [0.6, 0.3]
    assert sl["quat"] == pytest.approx([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], abs=1e-6)


def test_container_boxes_line_the_inside_of_the_box():
    box = {"half_extent": [0.15, 0.10, 0.05], "centre": [0.0, 0.0, 0.05], "wall": 0.01}
    parts = {b["name"]: b for b in mjcf.container_boxes(box)}
    assert parts["floor"]["pos"][2] - parts["floor"]["size"][2] == pytest.approx(0.0)  # floor flush with the bottom
    assert parts["wall_px"]["pos"][0] + parts["wall_px"]["size"][0] == pytest.approx(0.15)
    assert parts["wall_ny"]["pos"][1] - parts["wall_ny"]["size"][1] == pytest.approx(-0.10)
    inertia = mjcf.box_inertia(1.2, box)
    assert inertia["diaginertia"][2] == pytest.approx(1.2 / 12 * (0.3**2 + 0.2**2))


XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="flat" type="box" pos="0 0 0.49" size="0.4 0.4 0.01"/>
    <geom name="ramp" type="box" pos="2 0 0.49" size="0.4 0.4 0.01" euler="0 0.6 0" friction="0.05 0.005 0.0001"/>
    <body name="rests_object" pos="0 0 0.55"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="0.3"/></body>
    <body name="slides_object" pos="2 0 0.58"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="0.3" euler="0 0.6 0"
      friction="0.05 0.005 0.0001"/></body>
  </worldbody>
</mujoco>
"""


def test_the_drop_test_tells_a_resting_object_from_a_sliding_one():
    result = drop_test.settle(mujoco.MjModel.from_xml_string(XML), seconds=1.0)
    assert result["rests_object"]["ok"] and result["rests_object"]["moved_m"] < 0.01
    assert not result["slides_object"]["ok"] and result["slides_object"]["moved_m"] > drop_test.MOVE_MAX


def test_interior_lights_sit_inside_each_room_under_the_ceiling():
    from locomani.scene_discovery import usd_scene

    rooms = {
        "kitchen": {"polygon": [[0, 0], [5, 0], [5, 4], [0, 4]]},
        "hall": {"polygon": [[5, 0], [6, 0], [6, 4], [5, 4]]},  # narrower than one grid step: still gets none
    }
    points = usd_scene.light_points(rooms, spacing=2.5)
    assert [p[0] for p in points] == ["kitchen"] * 4
    assert all(0 < x < 5 and 0 < y < 4 for _r, x, y in points)
    text = usd_scene.scene_usda("/abs/apartment.usdc", points, intensity=1000)
    assert text.count("def SphereLight") == 4 and "references = @/abs/apartment.usdc@</root>" in text
    assert f"({points[0][1]}, {points[0][2]}, {usd_scene.HEIGHT})" in text
