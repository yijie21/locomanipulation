# Portable geometry fixture

Tests select this directory through `tests/geometry_fixture.py`; it is not a
simulation default. The XML is data, not implementation or startup reading.

`scene_locomani.xml` retains the captured robot joints, body transforms, collision
geometries and inertial values. Meshes are embedded convex hulls; textures/materials
are removed. This removes external mesh dependencies and generated output paths.
FK at zero and ready poses matches the source model within 1e-8. Geometry tests
still run real MuJoCo FK/IK, swept collision queries and the table support check.
It is not a rendering or dynamic-behavior regression reference.

`source.json` records the original scene hash. `scene_meta.json` supplies explicit
table/object inputs; `hand_points_dex1.npy` is the existing 400-point hand envelope
used by geometric planning tests. Live behavior gates use their own pinned assets.

To intentionally refresh the XML/metadata fixture, run
`python tests/fixtures/geometry/build_fixture.py <source-scene.xml> <output-directory>`
in the simulation environment and inspect the resulting geometry test changes.
