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
