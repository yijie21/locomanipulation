"""The G1's materials in every renderer that draws the policy's pictures and the videos: Isaac and Unreal, one source.

Two PBR materials (linear base colour, metallic, roughness), picked per mesh by its MJCF colour: the light-grey
shells (rgba >= 0.6), and the dark set -- head, pelvis, hip pitch, ankle roll, Dex1 base and fingers (rgba < 0.4;
mars adds its waist motors).
Isaac binds them as OmniPBR whenever it assembles a stage (`isaac.stage.G1_LOOKS`). Unreal writes them into the
level's material instances when it builds the level (`unreal.build_level` -> `in_editor/robot_materials.py`). The
level's `source.json` records NAME, and `run_context.unreal_scene` refuses a level built with another look. NAME is
also part of the dataset-camera marker's look in both renderers, so `view_render.plan` never fills one session with
two looks.

The shell values (user, 2026-10-04) match a photo of the lab's real robot, assets/mars_real_model.jpg: metallic 0.75,
roughness 0.6, a neutral base 0.45. Matched in Unreal over 19 variants (outputs/mars_shell_match_20261004/) by two
exposure-free measures on a front portrait like the photo: each silver part's brightness over a white wall's in the
same picture (photo 0.62-0.83), and how uniform the surface is. At metallic 1 the arms mirror the room's dark floor
and shelves as mottled streaks, 1.5-2x the photo's surface variation, which the ego view shows on the forearms;
raising roughness alone keeps the streaks, and metallic 0.5 reads as white plastic. Roughness 0.5 looks the same as
0.6. Before (2026-10-01 to 2026-10-04, NAME `g1_gloss_20261001`, A/B against the URLab hero in
outputs/g1_look_ab_20261001/ab_sheet.png, variant v4a): shells metallic 1, roughness 0.45, base 0.675. That started
from Unitree's Isaac Lab asset for the G1 with Dex1, the look of 2026-09-29 (`g1_29dof_with_dex1_rev_1_0.usd`, NAME
`unitree_20260929`: shells roughness 0.75; dark set metallic 0.42, base 0.011, roughness 0.5). Unreal's Lumen traces
dedicated reflection rays only below roughness 0.4, and below about 0.5 the shells speckled at 720x540.
The dark set is black plastic (metallic 0, base 0.03, roughness 0.35, since 2026-10-01). At metallic 0.42 it mirrored
its own near-black base colour, so the head had no highlight.
A hard side key light was tried too and rejected: it washed out the white room rather than shaping the robot.

The mars robot (make_dex1_model.py, the lab's real G1) is drawn the same way, its meshes sorted by the colour the
generator writes: DARK_MESHES dark whatever the URDF colours them, the rest by the URDF colour's brightness. On the
real robot the Dex1 base and the D405 mount are black (user, 2026-10-02; the URDF has the Dex1 base pale lavender),
and so are the waist yaw and roll motors (user, 2026-10-04, from the photo; white in the URDF). The pelvis keeps its
URDF colour, dark, as in the photo (silver from 2026-10-03 to 2026-10-04). The 5010 wrist (the forearm) is silver
like the stock robot's arm (user, 2026-10-04; it was black from 2026-10-02). Its fingers keep the stock model's geoms
and colours. The D455 on top of the head (`head_tilt_link`) stays silver-white, as on the real robot (user,
2026-10-03).

Changing a value changes the policy's pictures. A checkpoint trained before the change saw the older look, so
change NAME along with the values.

Which robot a scene carries is `scene_robot`: both renderers take their robot from it (Isaac the robot USD built for
that robot, Unreal the level imported from that scene), record it in their `source.json` and dataset-camera
markers, and `run_context` refuses a stage or level built for another robot.
"""

import json
from pathlib import Path

NAME = "g1_satin_20261004"
SHELL = {"base": (0.45, 0.45, 0.45), "metallic": 0.75, "roughness": 0.6}
DARK = {"base": (0.03, 0.03, 0.03), "metallic": 0.0, "roughness": 0.35}
# URDF mesh file stems (assets/unitree_G1_with_wrist_camera/urdf/meshes) drawn dark.
DARK_MESHES = frozenset(
    {"Dex1_base_link", "unitree_dex1-1_d405_wrist_mount", "waist_yaw_link_rev_1_0", "waist_roll_link_rev_1_0"}
)


def scene_robot(scene_dir) -> str:
    """The robot a MuJoCo scene directory is built on: its scene_meta.json `robot` (scene_lib.ROBOT_SCENES).
    Scenes built before 2026-10-02 record none and carry the stock robot."""
    meta = Path(scene_dir) / "scene_meta.json"
    return json.loads(meta.read_text()).get("robot", "stock") if meta.is_file() else "stock"
