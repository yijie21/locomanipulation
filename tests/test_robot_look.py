"""The G1's materials are one definition (`locomani.sim.robot_look`) that Isaac and Unreal both draw.

Before 2026-10-01 each renderer carried its own copy of the numbers (Isaac's stage.G1_LOOKS, Unreal's
robot_materials.py preset); the user asked for the same materials in both, for the dataset cameras and the video.
"""

from __future__ import annotations

from locomani.sim import robot_look


def test_isaac_binds_the_robot_look_as_omnipbr():
    from locomani.sim.isaac.stage import G1_LOOKS

    for key, material in (("shell", robot_look.SHELL), ("dark", robot_look.DARK)):
        assert G1_LOOKS[key] == {
            "diffuse_color_constant": tuple(material["base"]),
            "metallic_constant": material["metallic"],
            "reflection_roughness_constant": material["roughness"],
        }
        assert isinstance(G1_LOOKS[key]["diffuse_color_constant"], tuple)  # assemble_stage binds a tuple as Color3f


def test_unreal_levels_are_built_with_the_robot_look_and_record_it():
    import inspect

    from locomani.sim.unreal import build_level

    source = inspect.getsource(build_level.build)
    # robot_materials.py has no numbers of its own
    assert '"SILVER": robot_look.SHELL, "DARK": robot_look.DARK' in source
    assert '"robot_look": robot_look.NAME' in source  # source.json, which run_context.unreal_scene checks
    assert '"robot": robot_look.scene_robot(scene_dir)' in source  # and the robot the import carries


def test_a_scene_names_its_robot_and_old_scenes_are_stock(tmp_path):
    assert robot_look.scene_robot(tmp_path) == "stock"  # no scene_meta.json
    (tmp_path / "scene_meta.json").write_text('{"objects": {}}')
    assert robot_look.scene_robot(tmp_path) == "stock"  # built before 2026-10-02
    (tmp_path / "scene_meta.json").write_text('{"robot": "mars"}')
    assert robot_look.scene_robot(tmp_path) == "mars"


def test_isaac_draws_the_robot_usd_of_the_robot_the_scene_carries(tmp_path):
    from locomani.sim.isaac import stage

    scene = tmp_path / "scene"
    scene.mkdir()
    cfg = stage.StageCfg(assets=str(tmp_path / "isaac"), robot_mjcf=str(scene / "scene_locomani.xml"))
    assert stage.robot_usd(cfg) == str(tmp_path / "isaac" / "robot" / "g1_dex1.usda")  # stock: the original place
    (scene / "scene_meta.json").write_text('{"robot": "mars"}')
    assert stage.robot_usd(cfg) == str(tmp_path / "isaac" / "robot" / "mars" / "g1_dex1.usda")
    cfg.robot_usd = str(tmp_path / "given.usda")
    assert stage.robot_usd(cfg) == str(tmp_path / "given.usda")
