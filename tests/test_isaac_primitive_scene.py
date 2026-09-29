"""scripts/build_isaac_primitive_scene.py: the MJCF box colours reach the Isaac stage as linear albedo.

An MJCF rgba is a display (sRGB) value; RTX reads UsdPreviewSurface diffuseColor as linear. Taken as-is the
box_platform carton (0.86, 0.78, 0.62) rendered near white (2026-09-29, TODO 10.15).
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_isaac_primitive_scene.py"
spec = importlib.util.spec_from_file_location("build_isaac_primitive_scene", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

MJCF = """<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 1"><freejoint/><geom type="box" size="0.1 0.1 0.1" rgba="0.7 0.7 0.7 1"/></body>
    <geom name="platform_col" type="box" size="0.3 0.3 0.05" pos="1 0 0.05" rgba="0.45 0.47 0.5 1"/>
    <body name="nor_cal_box" pos="1 0 0.4"><freejoint/>
      <geom name="nor_cal_floor" type="box" size="0.2 0.15 0.005" rgba="0.86 0.78 0.62 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _colours(usd: Path) -> list[tuple[float, ...]]:
    return [tuple(float(v) for v in m.split(",")) for m in re.findall(r"diffuseColor = \(([^)]*)\)", usd.read_text())]


def test_srgb_to_linear_matches_the_standard_curve():
    assert builder.srgb_to_linear((0.0, 1.0)) == (0.0, 1.0)
    assert builder.srgb_to_linear((0.04,))[0] == pytest.approx(0.04 / 12.92)
    assert builder.srgb_to_linear((0.5,))[0] == pytest.approx(0.21404, abs=1e-5)


def test_box_colours_are_written_linear_and_the_ground_as_given(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scene_locomani.xml").write_text(MJCF)
    (scene / "scene_meta.json").write_text(
        json.dumps({"target_object": "nor_cal_box", "objects": {"nor_cal_box": {"body": "nor_cal_box"}}})
    )
    out = tmp_path / "stage"
    builder.build(scene, out)
    static = _colours(out / "scene.usda")
    carton = _colours(out / "usd" / "nor_cal_box" / "nor_cal_box.usd")
    assert builder.GROUND_RGB in static
    assert (0.17064, 0.18732, 0.21404) in static  # the platform's 0.45 / 0.47 / 0.50
    assert carton == [(0.71057, 0.57048, 0.34239)]  # the carton's 0.86 / 0.78 / 0.62, cardboard rather than white
