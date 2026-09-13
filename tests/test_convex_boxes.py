import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from locomani.sim.convex_boxes import convert_body_boxes


@pytest.mark.parametrize("body_name", ["crate", "table"])
def test_resting_basket_has_no_phantom_deep_contact_or_ejection(body_name):
    # Exact pre-ejection pose captured from seed 43; no robot or controller needed.
    pose = [
        -0.32191603098207366,
        -0.018248986980301623,
        0.3849943987162302,
        0.7360953394382325,
        -6.078765756655225e-12,
        6.6105569503748254e-12,
        0.6768778702670922,
    ]
    tree = ET.ElementTree(
        ET.fromstring("""<mujoco><option timestep="0.005"/>
      <worldbody><body name="table"><geom type="box" size=".488008 .488489 .02" pos="0 0 .28"/></body>
      <body name="crate"><freejoint/><geom type="box" pos="0 0 -.08"
      size=".1933 .1267 .005" mass=".222" solref=".002 1" solimp=".998 .998 .001"/>
      </body></worldbody></mujoco>""")
    )
    original = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    convert_body_boxes(tree, body_name)
    once = ET.tostring(tree.getroot())
    convert_body_boxes(tree, body_name)
    assert ET.tostring(tree.getroot()) == once
    model = mujoco.MjModel.from_xml_string(once.decode())
    np.testing.assert_allclose(model.body_mass, original.body_mass, rtol=1e-6)
    np.testing.assert_allclose(np.sort(model.body_inertia), np.sort(original.body_inertia), rtol=1e-6)
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    for _ in range(200):
        mujoco.mj_step(model, data)
        assert all(c.dist > -0.001 for c in data.contact[: data.ncon])
        assert np.linalg.norm(data.qvel[:3]) < 0.01
    assert abs(data.qpos[2] - pose[2]) < 0.001
