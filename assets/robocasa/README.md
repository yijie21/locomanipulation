# Third-party mesh assets

## `bottled_water_1`

A drink bottle from **RoboCasa**'s Objaverse-derived object set, unmodified
(`robocasa/robocasa-assets` on Hugging Face, `objaverse.zip`,
`objaverse/bottled_water/bottled_water_1/`).

Why this one, out of the 69 drink containers in that set: measured on its own
collision meshes, the barrel is **4.3 cm across and the bottle is 15.9 cm
tall**. dex1's jaw opens 6.1 cm and the alignment servo converges to ~6 mm, so
4.3 cm clears with 9 mm either side -- it needs no scaling to be grippable, and
`scene_lib.MESH_TARGETS` scales it only to bring the height down to the 12 cm
the rest of the scene is calibrated around (`scene_lib.py`'s note on a 14 cm
cylinder being tippy enough that a passing arm knocks it over).

It ships as MJCF with 33 convex collision submeshes already decomposed, which
is the part that is expensive to reproduce: Google Scanned Objects, the other
MuJoCo-ready library, is a retail scan set with no single beverage bottle in
it at all.

- RoboCasa code and packaging: MIT (`robocasa/robocasa` LICENSE)
- Underlying 3-D asset: Objaverse 1.0, ODC-BY 1.0

See <https://robocasa.ai/> and <https://objaverse.allenai.org/>.
