from pathlib import Path

import numpy as np
import pytest

from locomani.tasks.crate import warehouse_box_scene as scene

PLATFORM = {"xy": [0.0, -0.105], "flat_half_extent": [0.6, 0.45]}  # the warehouse pallet


@pytest.mark.parametrize(
    "side, centre, half",
    [("ym", (0.0, -0.555), 0.6), ("yp", (0.0, 0.345), 0.6), ("xm", (-0.6, -0.105), 0.45), ("xp", (0.6, -0.105), 0.45)],
)
def test_each_heading_meets_the_edge_on_its_side(side, centre, half):
    edge, length = scene.edge_line(PLATFORM, scene.EDGES[side])
    assert edge == pytest.approx(centre)
    assert length == pytest.approx(half)


def test_the_minus_y_camera_is_the_rooms_reverse_room_shot():
    camera = scene.edge_camera(PLATFORM, scene.EDGES["ym"])
    assert [float(v) for v in camera["pos"].split()] == pytest.approx([-1.4, 1.33, 1.71], abs=1e-3)
    quat = [float(v) for v in camera["quat"].split()]  # the room's own camera, as MJCF writes it
    assert quat == pytest.approx([-0.232547, -0.184719, 0.593921, 0.747702], abs=2e-3) or quat == pytest.approx(
        [0.232547, 0.184719, -0.593921, -0.747702], abs=2e-3
    )


def test_clearance_stops_at_the_nearest_geom_and_skips_the_pallets_own_overhang():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <geom type="box" pos="0 -0.105 0.2" size="0.6 0.45 0.2"/>
        <geom type="box" pos="0.61 -0.105 0.05" size="0.02 0.3 0.05"/>
        <geom type="box" pos="1.5 -0.105 1" size="0.1 3 1"/>
        </worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert scene.clearance(model, data, PLATFORM, scene.EDGES["xp"]) == pytest.approx(0.8)
    assert scene.clearance(model, data, PLATFORM, scene.EDGES["xm"]) == scene.CLEAR_LIMIT


def test_build_resizes_the_carton_and_measures_every_edge(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    if not (scene.SOURCE / "scene_meta.json").exists():
        pytest.skip("warehouse_textured scene not built on this machine")
    meta = scene.build(tmp_path / "scene", width=0.45, length=0.20, height=0.18)
    model = mujoco.MjModel.from_xml_path(str(Path(meta["scene_xml"])))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body("nor_cal_box_object").id
    origin = data.xpos[body]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for g in np.flatnonzero(model.geom_bodyid == body):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[start : start + count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            low, high = np.minimum(low, points.min(0)), np.maximum(high, points.max(0))
    assert low - origin == pytest.approx([-0.225, -0.10, 0.0], abs=2e-3)  # the photo meshes
    assert high - origin == pytest.approx([0.225, 0.10, 0.18], abs=2e-3)
    front = model.geom("nor_cal_ym").id  # the collision walls
    assert data.geom_xpos[front][1] - model.geom_size[front][1] == pytest.approx(-0.555, abs=1e-6)
    assert meta["objects"]["nor_cal_box"]["box"]["half_extent"] == pytest.approx([0.225, 0.10, 0.09])
    edges = meta["surfaces"]["A"]["edges"]
    assert set(edges) == set(scene.EDGES)
    for side, edge in edges.items():
        assert model.camera(edge["video_camera"]).id >= 0
        assert 1.0 < edge["clear_m"] < 1.7, side  # the room's racks, cart and walls (1.05-1.63 m on 2026-09-28)
