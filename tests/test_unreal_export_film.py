"""export_mjcf: the Unreal import carries the third-person video's camera rig and ghost grippers.

URLab moves only what the import's qpos holds, so the rig and the ghosts are free bodies, parked under the floor at
qpos0; the camera takes the scene's shot fovy and the video's size, and a ghost placed at a wrist's own pose must
cover that gripper exactly, as the Isaac ghost built from the same `ghost.hand_parts` does."""

import json

import mujoco
import numpy as np

from locomani.pipeline import sim_video
from locomani.sim.unreal import export_mjcf as X

TETRA = "v 0 0 0\nv 0.05 0 0\nv 0 0.05 0\nv 0 0 0.05\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
ARM = """<body name="{s}_wrist_yaw_link" pos="0.3 {y} 0.2" euler="0.3 0.2 {yaw}">
  <geom type="mesh" mesh="{s}_palm" group="1"/>
  <body name="{s}_dex1_finger" pos="0.08 0 0" euler="0 0.4 0"><geom type="mesh" mesh="{s}_finger" group="1"/></body>
  {cam}</body>"""
SCENE = """<mujoco><asset>{meshes}</asset><worldbody>
  <body name="pelvis" pos="0 0 0.8"><freejoint name="floating_base_joint"/><geom size="0.1"/>
    <camera name="head_camera" pos="0.06 0 0.45" fovy="70"/>
    {left}{right}</body>
  <body name="box" pos="1 0 0.5"><freejoint name="box_free"/><geom type="box" size="0.1 0.1 0.1"/></body>
  <camera name="edge_ym" pos="-1.4 1.3 1.7" xyaxes="1 -1 0 0.3 0.3 1" fovy="73.74"/>
  <camera name="edge_xp" pos="1.4 -1.5 1.7" xyaxes="-1 1 0 -0.3 -0.3 1" fovy="73.74"/>
</worldbody></mujoco>"""
META = {
    "target_object": "box",
    "objects": {"box": {"body": "box", "surface": "A"}},
    "surfaces": {"A": {"edges": {"ym": {"heading_deg": 90.0, "video_camera": "edge_ym"},
                                 "xp": {"heading_deg": 180.0, "video_camera": "edge_xp"}}}},
}  # fmt: skip


def scene(tmp_path):
    tmp_path.mkdir()
    meshes = ""
    for s in ("left", "right"):
        for part in ("palm", "finger"):
            (tmp_path / f"{s}_{part}.obj").write_text(TETRA)
            meshes += f'<mesh name="{s}_{part}" file="{s}_{part}.obj"/>'
    arms = {
        s: ARM.format(s=s, y=y, yaw=yaw, cam=f'<camera name="{s}_wrist" pos="0.05 0 0" fovy="80"/>')
        for s, y, yaw in (("left", 0.2, 0.5), ("right", -0.2, -0.4))
    }
    (tmp_path / "scene_locomani.xml").write_text(SCENE.format(meshes=meshes, **arms))
    (tmp_path / "scene_meta.json").write_text(json.dumps(META))
    return tmp_path


def test_the_import_carries_a_parked_camera_rig_at_the_shot_and_two_ghosts_that_cover_the_grippers(tmp_path):
    exported = X.export(scene(tmp_path / "scene"), tmp_path / "out", "lvl")
    assert exported["film"] == {"camera": X.FILM_CAMERA, "fovy": 73.74, "size": [sim_video.WIDTH, sim_video.HEIGHT]}
    m = mujoco.MjModel.from_xml_path(exported["xml"])
    assert m.nq == 14 + 3 * 7
    cam = m.camera(X.FILM_CAMERA)
    assert m.cam_bodyid[cam.id] == m.body(X.RIG).id and np.isclose(m.cam_fovy[cam.id], 73.74)
    assert list(m.cam_resolution[cam.id]) == [sim_video.WIDTH, sim_video.HEIGHT]
    assert [c for c in exported["cameras"] if c.startswith("edge_")] == []  # the world cameras stay out
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for name in (X.RIG, *X.GHOSTS.values()):
        assert d.xpos[m.body(name).id][2] == X.PARK[2]
    # a ghost at its wrist's own pose lies on the real gripper, part by part
    for side, name in X.GHOSTS.items():
        adr = m.jnt_qposadr[m.joint(name).id]
        wrist = m.body(f"{side}_wrist_yaw_link").id
        d.qpos[adr : adr + 3], d.qpos[adr + 3 : adr + 7] = d.xpos[wrist], d.xquat[wrist]
    mujoco.mj_forward(m, d)
    for side, name in X.GHOSTS.items():
        ghost = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(name).id]
        assert len(ghost) == 2 and all(m.mat_rgba[m.geom_matid[g]][3] < 1 for g in ghost)
        for g in ghost:
            mesh = m.mesh(m.geom_dataid[g]).name.removeprefix(f"{name}_")
            real = next(h for h in range(m.ngeom) if m.geom_dataid[h] == m.mesh(mesh).id)
            assert np.allclose(d.geom_xpos[g], d.geom_xpos[real], atol=1e-9)
            assert np.allclose(d.geom_xmat[g], d.geom_xmat[real], atol=1e-9)


def test_collision_geoms_stay_out_of_the_import_where_the_body_has_visuals(tmp_path):
    """URLab draws a primitive collision geom as an engine shape in its default grey checker material: the mars
    wrist's D405 collision box (link-coloured, alpha 1) boxed the camera in. A body drawn only by its colliders (the
    target box here) keeps them."""
    scene_dir = scene(tmp_path / "scene")
    xml = (scene_dir / "scene_locomani.xml").read_text()
    xml = xml.replace(
        '<geom type="mesh" mesh="left_palm" group="1"/>',
        '<geom type="mesh" mesh="left_palm" group="1" contype="0" conaffinity="0"/>'
        '<geom type="box" size="0.021 0.021 0.013" pos="0.09 0 0.06" rgba="0.2 0.2 0.2 1"/>'
        '<geom type="mesh" mesh="left_palm" rgba="0.2 0.2 0.2 1"/>',
    )
    (scene_dir / "scene_locomani.xml").write_text(xml)
    m = mujoco.MjModel.from_xml_path(X.export(scene_dir, tmp_path / "out", "lvl")["xml"])
    wrist = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("left_wrist_yaw_link").id]
    assert [int(m.geom_type[g]) for g in wrist] == [mujoco.mjtGeom.mjGEOM_MESH] and not m.geom_contype[wrist[0]]
    box = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("box").id]
    assert len(box) == 1 and m.geom_contype[box[0]]  # its only geom collides: kept, or the target would vanish


def test_every_free_body_is_exported_not_only_the_target(tmp_path):
    """A scene_discovery apartment has 64 free objects and no fixed target: any of them can be picked up or knocked,
    and URLab moves only what the import holds. The room's fixed geometry stays out (the level bakes it)."""
    scene_dir = scene(tmp_path / "scene")
    xml = (scene_dir / "scene_locomani.xml").read_text()
    xml = xml.replace(
        '<camera name="edge_ym"',
        '<body name="cup" pos="0.5 0.5 0.4"><freejoint name="cup_free"/><geom type="box" size="0.03 0.03 0.05"/></body>'
        '<body name="shelf" pos="2 0 0"><geom type="box" size="0.3 0.1 0.5"/></body><camera name="edge_ym"',
    )
    (scene_dir / "scene_locomani.xml").write_text(xml)
    exported = X.export(scene_dir, tmp_path / "out", "lvl")
    assert exported["moving"] == ["box", "cup"]
    m = mujoco.MjModel.from_xml_path(exported["xml"])
    assert m.body("cup").id > 0 and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shelf") == -1
