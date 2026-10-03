"""The G1's materials in every renderer that draws the policy's pictures and the videos: Isaac and Unreal, one source.

Two PBR materials (linear base colour, metallic, roughness), picked per mesh by its MJCF colour: the light-grey
shells (rgba >= 0.6), and the dark set -- head, pelvis, hip pitch, ankle roll, Dex1 base and fingers (rgba < 0.4).
Isaac binds them as OmniPBR whenever it assembles a stage (`isaac.stage.G1_LOOKS`). Unreal writes them into the
level's material instances when it builds the level (`unreal.build_level` -> `in_editor/robot_materials.py`). The
level's `source.json` records NAME, and `run_context.unreal_scene` refuses a level built with another look. NAME is
also part of the dataset-camera marker's look in both renderers, so `view_render.plan` never fills one session with
two looks.

The values (user, 2026-10-01; A/B against the URLab hero still in outputs/g1_look_ab_20261001/ab_sheet.png, variant
v4a) start from Unitree's Isaac Lab asset for the G1 with Dex1, the look of 2026-09-29 (`g1_29dof_with_dex1_rev_1_0.usd`,
NAME `unitree_20260929`: shells roughness 0.75; dark set metallic 0.42, base 0.011, roughness 0.5). Two changes:
- The shells' roughness drops 0.75 -> 0.45. At 0.75 the metal blurs the room into flat grey paint; at 0.45 it
  reflects. Two reasons for not going lower. Unreal's Lumen traces dedicated reflection rays only below 0.4. Below
  about 0.5 the shells speckle at the video's 720x540 (0.35 speckles more than 0.45). Neither recomputed smooth
  normals nor 32 captures removed the speckle.
- The dark set becomes black plastic (metallic 0, base 0.03, roughness 0.35). At metallic 0.42 it mirrored its own
  near-black base colour, so the head had no highlight.
A hard side key light was tried too and rejected: it washed out the white room rather than shaping the robot.

The mars robot (make_dex1_model.py, the lab's real G1) is drawn the same way, its meshes sorted by the colour the
generator writes: DARK_MESHES dark whatever the URDF colours them (user, 2026-10-02: on the real robot the 5010 wrist,
the Dex1 base and the D405 mount are black; the URDF has the wrist white and the Dex1 base pale lavender), the rest
by the URDF colour's brightness. Its fingers keep the stock model's geoms and colours. SHELL_MESHES go the other way,
silver whatever the URDF says: the pelvis, the triangular part joining the legs to the upper body (user, 2026-10-03;
dark in the URDF and in Unitree's own). The D455 on top of the head (`head_tilt_link`) stays silver-white, as on the
real robot (user, 2026-10-03).

Changing a value changes the policy's pictures. A checkpoint trained before the change saw the older look, so
change NAME along with the values.

Which robot a scene carries is `scene_robot`: both renderers take their robot from it (Isaac the robot USD built for
that robot, Unreal the level imported from that scene), record it in their `source.json` and dataset-camera
markers, and `run_context` refuses a stage or level built for another robot.
"""

import json
from pathlib import Path

NAME = "g1_gloss_20261001"
SHELL = {"base": (0.675, 0.675, 0.6916), "metallic": 1.0, "roughness": 0.45}
DARK = {"base": (0.03, 0.03, 0.03), "metallic": 0.0, "roughness": 0.35}
# URDF mesh file stems (assets/unitree_G1_with_wrist_camera/urdf/meshes) drawn dark.
DARK_MESHES = frozenset(
    {
        *(f"{side}_wrist_{part}_link_5010" for side in ("left", "right") for part in ("roll", "pitch", "yaw")),
        "Dex1_base_link",
        "unitree_dex1-1_d405_wrist_mount",
    }
)
# URDF mesh file stems drawn silver (the shell look).
SHELL_MESHES = frozenset({"pelvis_ver0529"})


def scene_robot(scene_dir) -> str:
    """The robot a MuJoCo scene directory is built on: its scene_meta.json `robot` (scene_lib.ROBOT_SCENES).
    Scenes built before 2026-10-02 record none and carry the stock robot."""
    meta = Path(scene_dir) / "scene_meta.json"
    return json.loads(meta.read_text()).get("robot", "stock") if meta.is_file() else "stock"
