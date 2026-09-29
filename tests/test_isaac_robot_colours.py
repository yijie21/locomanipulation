from locomani.sim.isaac.stage import _mjcf_mesh_colours

MJCF = """<mujoco>
  <worldbody>
    <body name="pelvis">
      <geom type="mesh" mesh="pelvis" group="1" rgba="0.2 0.2 0.2 1"/>
      <geom type="mesh" mesh="pelvis_contour_link" group="1" rgba="0.7 0.7 0.7 1"/>
      <body name="left_dex1_finger_1_link">
        <geom type="mesh" mesh="Dex1_finger_link_1" group="1" rgba="0.3 0.3 0.3 1"/>
        <geom type="mesh" mesh="dex1_col_1" rgba="0.3 0.3 0.3 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_visual_prims_named_after_their_body_find_the_mesh_colour(tmp_path):
    # The importer names /visuals/left_dex1_finger_1_link after the body, not the mesh
    # asset it references; the fingers used to miss and render in the near-white fallback.
    xml = tmp_path / "scene.xml"
    xml.write_text(MJCF)
    colours = _mjcf_mesh_colours(str(xml))
    assert colours["Dex1_finger_link_1"] == (0.3, 0.3, 0.3)
    assert colours["left_dex1_finger_1_link"] == (0.3, 0.3, 0.3)
    # A body with two visual colours gets no body entry; its own mesh keeps its colour.
    assert colours["pelvis"] == (0.2, 0.2, 0.2)
    assert colours["pelvis_contour_link"] == (0.7, 0.7, 0.7)
    assert "dex1_col_1" not in colours
