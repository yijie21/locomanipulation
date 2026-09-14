import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from locomani.tasks.crate.bimanual_scene import set_crate_setback


def test_setback_changes_physics_and_reset_metadata_once(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    meta_path = scene / "scene_meta.json"
    meta = {"objects": {"blue_crate": {"body": "blue_crate_object", "init_pos": [-0.36, 0, 0.387]}}}
    meta_path.write_text(json.dumps(meta))
    xml = scene / "scene_locomani.xml"
    xml.write_text(
        '<mujoco><worldbody><geom type="plane" size="2 2 .1"/><body name="blue_crate_object" pos="-.36 0 .387"><freejoint/><geom type="box" size=".2 .12 .08" mass=".6"/></body></worldbody></mujoco>'
    )
    before = mujoco.MjModel.from_xml_path(str(xml))
    original = np.array(meta["objects"]["blue_crate"]["init_pos"])
    set_crate_setback(scene, 0.04)
    set_crate_setback(scene, 0.04)
    after = mujoco.MjModel.from_xml_path(str(xml))
    final = json.loads(meta_path.read_text())
    np.testing.assert_allclose(final["objects"]["blue_crate"]["init_pos"], original + [0.04, 0, 0])
    bid = before.body(meta["objects"]["blue_crate"]["body"]).id
    expected = before.body_pos.copy()
    expected[bid, 0] += 0.04
    np.testing.assert_allclose(after.body_pos, expected)
    for field in (
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "jnt_range",
        "actuator_gear",
    ):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))


@pytest.mark.parametrize("value", [-0.01, 0.3, float("nan")])
def test_bad_setback_rejected_before_scene_access(value):
    with pytest.raises(ValueError, match="setback"):
        set_crate_setback(Path("/nonexistent"), value)
