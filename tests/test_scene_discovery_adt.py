"""ADT object -> sim body: z up at the initial pose, origin at the box centre (the grasp code's convention)."""

from __future__ import annotations

import numpy as np
import pytest

from locomani.scene_discovery import adt


def mug(rot=None, pos=(1.0, 2.0, 0.7)):
    # ADT frame is y-up with the origin at the base: the box runs 0..0.10 along y
    return adt.AdtObject(
        uid="1", name="Mug", prototype="Mug", category="cup", motion="dynamic", rigidity="rigid",
        rot=adt.ADT_TO_SIM @ (np.eye(3) if rot is None else np.asarray(rot)), pos=np.asarray(pos, dtype=float),
        box_lo=np.array([-0.04, 0.0, -0.03]), box_hi=np.array([0.04, 0.10, 0.03]), up_local=np.array([0.0, 1.0, 0.0]),
    )  # fmt: skip


def test_the_body_origin_is_the_box_centre_and_the_mesh_moves_with_it():
    b = mug().body()
    assert b["centre"] == [0.0, 0.0, 0.0]
    assert b["half_extent"] == pytest.approx([0.03, 0.04, 0.05])  # body z = ADT y: the 10 cm height
    assert b["init_pos"] == pytest.approx([1.0, 2.0, 0.75])  # 5 cm above the base origin
    assert b["upright_axis"] == pytest.approx([0.0, 0.0, 1.0])
    # a mesh vertex at the ADT origin (the mug's base centre) lands half a height below the body origin
    o = mug()
    base = np.zeros(3) @ o.body_axes.T + np.asarray(b["mesh_offset"])
    assert base == pytest.approx([0.0, 0.0, -0.05])


def test_a_tipped_object_keeps_its_box_centre_in_the_world():
    tipped = mug(rot=[[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # ADT +90 deg about x: the mug lies on its side
    b = tipped.body()
    world_corners = tipped.corners() @ tipped.rot.T + tipped.pos
    assert b["init_pos"] == pytest.approx(world_corners.mean(axis=0).round(4).tolist())
