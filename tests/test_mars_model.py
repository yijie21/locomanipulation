"""The mars robot model (make_dex1_model.py) against the lab robot's URDF (assets/unitree_G1_with_wrist_camera).

mars is the stock dex1 tree with the URDF's numbers written in, so SONIC's sim bridge drives it unchanged:
same joints, actuators and sensors in the same order; kinematics, mass and torque limits the URDF's; the ego
camera at the D455 on the head's pan/tilt bracket (held at the stock aim's pitch), the wrist cameras at the D405s.
"""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from locomani.sim import make_dex1_model as M
from locomani.sim import robot_look, scene_lib

pytestmark = pytest.mark.skipif(
    not os.path.exists(f"{M.G1}/g1_29dof_with_hand.xml"), reason="SONIC's model data is not installed"
)


@pytest.fixture(scope="module")
def mars_xml():
    """A fresh build next to the installed meshes, so the test never reads a stale generated file."""
    if not os.path.isdir(M.MARS_MESHES):
        pytest.skip("mars meshes not copied: run make_dex1_model.py")
    path = f"{M.G1}/test_mars_{os.getpid()}.xml"
    M.build_mars(M.build_stock()).write(path)
    yield path
    os.remove(path)


def test_mars_matches_its_urdf(mars_xml):
    report = M.check_mars(mars_xml)  # raises on FK, mass or torque-limit mismatch
    assert report["fk_worst_mm"] < 1e-3
    assert report["mass_kg"] == pytest.approx(35.02, abs=0.01)


def test_the_stock_model_is_still_what_make_dex1_model_writes():
    """The stock robot is untouched by mars: a fresh build equals the installed file the stock scenes use."""
    path = f"{M.G1}/test_stock_{os.getpid()}.xml"
    M.build_stock().write(path)
    try:
        assert open(path, "rb").read() == open(f"{M.G1}/g1_29dof_with_dex1.xml", "rb").read()
    finally:
        os.remove(path)


def test_mars_is_driven_like_the_stock_robot(mars_xml):
    stock = mujoco.MjModel.from_xml_path(f"{M.G1}/g1_29dof_with_dex1.xml")
    mars = mujoco.MjModel.from_xml_path(mars_xml)
    for obj, count in (
        (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
        (mujoco.mjtObj.mjOBJ_SENSOR, "nsensor"),
        (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
    ):
        names = [[mujoco.mj_id2name(m, obj, i) for i in range(getattr(m, count))] for m in (stock, mars)]
        assert names[0] == names[1], count
    assert np.array_equal(stock.actuator_trnid, mars.actuator_trnid)
    assert np.allclose(stock.dof_armature, mars.dof_armature)
    assert np.allclose(stock.dof_damping, mars.dof_damping)
    limits = {mars.actuator(a).name: mars.actuator_ctrlrange[a, 1] for a in range(mars.nu)}
    assert limits["left_ankle_pitch"] == 35.0  # the firmware limit
    assert limits["left_hip_roll"] == 139.0 and limits["right_wrist_yaw"] == 13.4
    assert limits["waist_pitch"] == limits["waist_roll"] == 50.0  # MARS_EFFORT_OVERRIDES; the URDF says 35


def _pitch_deg(model, data, cam_id):
    look = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    torso = data.xmat[model.body("torso_link").id].reshape(3, 3)
    v = torso.T @ look
    return np.degrees(np.arctan2(-v[2], v[0]))


def test_mars_cameras_are_the_urdf_cameras(mars_xml):
    mars = mujoco.MjModel.from_xml_path(mars_xml)
    urdf = M.mars_urdf_model(fuse=False)
    ds, du = mujoco.MjData(mars), mujoco.MjData(urdf)
    mujoco.mj_forward(mars, ds)
    mujoco.mj_forward(urdf, du)
    ps, pu = mars.body("pelvis").id, urdf.body("pelvis").id
    cams = {"head_camera": M.MARS_EGO_LINK, **{f"{s}_wrist": link for s, link in M.MARS_WRIST_LINKS.items()}}
    assert M.MARS_EGO_LINK == "d455_head_link"  # on the tilt bracket, not d435_head_link in the face
    for cam_name, link in cams.items():
        cam, body = mars.camera(cam_name).id, urdf.body(link).id
        assert np.allclose(ds.cam_xpos[cam] - ds.xpos[ps], du.xpos[body] - du.xpos[pu], atol=1e-6), cam_name
        look = -ds.cam_xmat[cam].reshape(3, 3)[:, 2]
        assert np.allclose(look, du.xmat[body].reshape(3, 3)[:, 0], atol=1e-6), cam_name  # along the link's x
    head = mars.camera("head_camera").id
    assert _pitch_deg(mars, ds, head) == pytest.approx(52.0, abs=0.05)  # the stock aim's pitch
    assert mars.cam_fovy[head] == M.MARS_EGO_FOVY == 65.0
    assert mars.cam_fovy[mars.camera("left_wrist").id] == M.MARS_WRIST_FOVY == 58.0


def test_mars_colours_sort_into_the_robot_look(mars_xml):
    """Every visual mesh is drawn shell or dark, and the parts the real robot has black are dark."""
    root = ET.parse(mars_xml).getroot()
    rgba = {}
    for g in root.iter("geom"):
        if g.get("group") == "1" and g.get("mesh"):
            rgba.setdefault(g.get("mesh"), set()).add(g.get("rgba"))
    fingers = {"Dex1_finger_link_1", "Dex1_finger_link_2"}  # the stock finger geoms, kept with their colours
    for mesh, colours in rgba.items():
        assert len(colours) == 1, mesh
        if mesh not in fingers:
            assert colours <= {M.SHELL_RGBA, M.DARK_RGBA}, mesh
    for stem in robot_look.DARK_MESHES:
        (filename,) = [f for f in os.listdir(f"{os.path.dirname(M.MARS_URDF)}/meshes") if f.startswith(f"{stem}.")]
        assert rgba[M.mars_mesh_name(f"meshes/{filename}")] == {M.DARK_RGBA}, stem
    assert rgba["torso_link_rev_1_0"] == {M.SHELL_RGBA}
    pelvis = next(b for b in root.iter("body") if b.get("name") == "pelvis")
    assert {g.get("rgba") for g in pelvis.findall("geom")} == {M.DARK_RGBA}  # the kept stock contour too
    # collision geoms (group 0, drawn by MuJoCo's cameras) wear their link's visual colour
    for body in root.iter("body"):
        if body.get("name", "").endswith("wrist_yaw_link"):
            assert {g.get("rgba") for g in body.findall("geom") if g.get("group") != "1"} == {M.DARK_RGBA}


def test_meshes_identical_to_the_stock_files_keep_the_stock_names():
    """Code that picks geoms by mesh name (the bottle guard's Dex1_base_link envelope) reads mars as stock."""
    assert M.mars_mesh_name("meshes/Dex1_base_link.STL") == "Dex1_base_link"
    assert M.mars_mesh_name("meshes/left_knee_link.STL") == "left_knee_link"
    assert M.mars_mesh_name("meshes/left_wrist_yaw_link_5010.STL") == "mars_left_wrist_yaw_link_5010"
    assert M.mars_mesh_name("meshes/pelvis_ver0529.STL") == "mars_pelvis_ver0529"


def test_the_d405_mounts_collide(mars_xml):
    mars = mujoco.MjModel.from_xml_path(mars_xml)
    for side in ("left", "right"):
        body = mars.body(f"{side}_wrist_yaw_link").id
        shapes = sorted(
            mujoco.mjtGeom(mars.geom_type[g]).name
            for g in range(mars.ngeom)
            if mars.geom_bodyid[g] == body and mars.geom_contype[g]
        )
        assert shapes == ["mjGEOM_BOX", "mjGEOM_CYLINDER", "mjGEOM_MESH", "mjGEOM_MESH"], side


def test_torque_limits_are_the_scene_motors_in_actuator_order(tmp_path):
    motors = "".join(
        f'<general name="m{i}" joint="j{i}" ctrlrange="-{e} {e}"/>' for i, e in enumerate([88, 35, 13.4] + [1] * 30)
    )
    xml = tmp_path / "scene.xml"
    xml.write_text(f"<mujoco><actuator>{motors}</actuator></mujoco>")
    limits = scene_lib.motor_effort_limits(str(xml))
    assert limits[:3] == [88.0, 35.0, 13.4] and len(limits) == 33
    short = motors[: motors.index('<general name="m32"')]
    xml.write_text(f"<mujoco><actuator>{short}</actuator></mujoco>")
    with pytest.raises(SystemExit):
        scene_lib.motor_effort_limits(str(xml))


def test_a_scene_names_its_robot_and_mars_needs_dex1(monkeypatch):
    assert set(scene_lib.ROBOT_SCENES) == {"stock", "mars"}
    with pytest.raises(SystemExit, match="unknown robot"):
        scene_lib.build("/nonexistent", robot="nao")
    monkeypatch.setattr(scene_lib, "HAND", "dex3")
    with pytest.raises(SystemExit, match="Dex1"):
        scene_lib.build("/nonexistent", robot="mars")


def test_the_urdf_edits_each_apply_once():
    text = open(M.MARS_URDF).read()
    for joint in M.MARS_HELD_JOINTS:
        assert f'name="{joint}"' in text
        assert 'type="fixed"' in M._hold(text, joint, 0.1).split(f'<joint name="{joint}"')[1].split(">")[0]
    with pytest.raises(AssertionError):
        M._sub_once(r"no such thing", "", text)
