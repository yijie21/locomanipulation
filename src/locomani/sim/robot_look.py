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

Changing a value changes the policy's pictures. A checkpoint trained before the change saw the older look, so
change NAME along with the values.
"""

NAME = "g1_gloss_20261001"
SHELL = {"base": (0.675, 0.675, 0.6916), "metallic": 1.0, "roughness": 0.45}
DARK = {"base": (0.03, 0.03, 0.03), "metallic": 0.0, "roughness": 0.35}
