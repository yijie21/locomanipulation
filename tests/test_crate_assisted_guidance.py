import json
from pathlib import Path

from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.prepare import prepare_scene
from locomani.tasks.crate.visual_scene import build


def test_assisted_pilot_restores_the_original_basket_in_its_own_scene_copy(tmp_path):
    painted = tmp_path / "painted"
    source = tmp_path / "source"
    source.mkdir()
    (source / "scene_locomani.xml").write_text(
        '<mujoco><worldbody><body name="table"><geom type="box" size=".5 .5 .02"/></body>'
        '<body name="blue_crate_object"><geom type="box" size=".2 .13 .08" mass=".604"/></body></worldbody></mujoco>'
    )
    (source / "scene_meta.json").write_text(
        json.dumps({"surfaces": {"A": {"body": "table"}}, "objects": {"blue_crate": {"surface": "A"}}})
    )
    build(source, painted)
    original = (painted / "scene_locomani.xml").read_bytes()
    config = CrateTaskConfig().with_overrides(["visual.pose_source=assisted", "carry.reach_control=whole_body"])
    scene = prepare_scene(painted, tmp_path / "run", config, "visual")
    assert config.visual.pose_source == "assisted"
    assert "crate_visual_label" not in (scene / "scene_locomani.xml").read_text()
    assert "crate_visual_floor_label" not in (scene / "scene_locomani.xml").read_text()
    assert "crate_visual" not in json.loads((scene / "scene_meta.json").read_text())
    assert (painted / "scene_locomani.xml").read_bytes() == original
    import xml.etree.ElementTree as ET

    tree = ET.parse(scene / "scene_locomani.xml")
    assert tree.find('.//body[@name="table"]/geom').get("type") == "mesh"
    assert tree.find('.//body[@name="blue_crate_object"]/geom').get("type") == "box"


def test_assisted_source_validation_and_round_trip(tmp_path):
    import pytest

    from locomani.tasks.crate.config import CrateTaskConfig

    config = CrateTaskConfig.load("configs/tasks/crate_visual_low_reach.json")
    path = tmp_path / "config.json"
    config.dump(path)
    assert CrateTaskConfig.load(path).visual.pose_source == "assisted"
    assert config.with_overrides(["visual.pose_source=markers"]).visual.pose_source == "markers"
    with pytest.raises(ValueError, match="pose_source"):
        config.with_overrides(["visual.pose_source=truth"])
    with pytest.raises(ValueError, match="requires"):
        config.with_overrides(["carry.reach_control=planner"])


def test_simulation_pose_is_admitted_only_inside_a_visible_camera_view(tmp_path):
    import xml.etree.ElementTree as ET

    import numpy as np
    from geometry_fixture import DIRECTORY

    from locomani.primitives.arm_ik import ArmIK
    from locomani.tasks.crate.assisted_guidance import AssistedGuidance

    tree = ET.parse(DIRECTORY / "scene_locomani.xml")
    world = tree.find("worldbody")
    body = ET.SubElement(world, "body", name="blue_crate_object", pos="0.6 0 0.6")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", type="box", size=".2 .13 .085", rgba=".1 .2 .8 1", mass=".604")
    # A movable opaque blocker in the scene, initially parked behind the robot.
    blocker = ET.SubElement(world, "body", name="blocker", pos="-3 0 0")
    ET.SubElement(blocker, "geom", type="box", size=".02 2 2")
    scene = tmp_path / "scene.xml"
    tree.write(scene)
    guidance = AssistedGuidance(ArmIK(str(scene)))
    state = dict(
        joints=np.zeros(29),
        pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]),
        objects={"blue_crate": dict(pos=[0.6, 0, 0.6], quat=[1, 0, 0, 0])},
    )
    images = {"ego_view": np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)}
    result = guidance.update(images, state, 1.0)
    assert result is not None
    np.testing.assert_allclose(result.position, [0.6, 0, -0.18])
    assert result.reprojection_px is None
    guidance.ik.model.body_pos[guidance.ik.model.body("blocker").id] = [0.3, 0, 0]
    assert guidance.update(images, state, 1.1) is None
    guidance.ik.model.body_pos[guidance.ik.model.body("blocker").id] = [-3, 0, 0]
    state["objects"]["blue_crate"]["pos"] = [-1, 0, 0.6]
    assert guidance.update(images, state, 1.2) is None
    state["objects"]["blue_crate"]["pos"] = [0.6, 0, 0.6]
    assert guidance.update({"ego_view": images["ego_view"] * 0}, state, 1.3) is None
