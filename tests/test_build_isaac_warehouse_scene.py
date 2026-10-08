"""The warehouse scene builder's height-map normal block, the parts that need no pxr.

Blender's exporter wires a Bump node's grey height map into UsdPreviewSurface `normal`; on the carton photos that
reads as an about-zero normal and renders as white speckle (2026-09-28, docs/ledger/blender_scenes.md). The builder
blocks a `normal` connection when a single-channel texture feeds it. These tests pin that rule's two pure halves: the
channel count read from a PNG header, and the nested `over`s that block each shader's connection.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_isaac_warehouse_scene as B  # noqa: E402


def _png(path: Path, colour_type: int) -> Path:
    ihdr = struct.pack(">IIBBBBB", 4, 4, 8, colour_type, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk)
    return path


def test_channels_come_from_the_png_colour_type(tmp_path):
    assert B.texture_channels(str(_png(tmp_path / "height.png", 0))) == 1
    assert B.texture_channels(str(_png(tmp_path / "grey_alpha.png", 4))) == 2
    assert B.texture_channels(str(_png(tmp_path / "normal.png", 2))) == 3
    assert B.texture_channels(str(_png(tmp_path / "rgba.png", 6))) == 4


def test_anything_but_a_readable_png_is_left_connected(tmp_path):
    jpeg = tmp_path / "photo.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 40)
    assert B.texture_channels(str(jpeg)) is None
    assert B.texture_channels(str(tmp_path / "missing.png")) is None
    (tmp_path / "short.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert B.texture_channels(str(tmp_path / "short.png")) is None


def test_overs_share_parents_and_block_each_shader():
    lines = B.blocked_normal_overs(["_materials/lid/Principled_BSDF", "_materials/side/Principled_BSDF"], "    ")
    text = "\n".join(lines)
    assert text.count('over "_materials"') == 1
    assert text.count("normal3f inputs:normal.connect = None") == 2
    assert text.count("{") == text.count("}") == 5
    assert lines[0] == '    over "_materials"'
    assert '            over "Principled_BSDF"' in lines


def test_no_height_maps_writes_nothing():
    assert B.blocked_normal_overs([], "    ") == []


def _scene(tmp_path: Path, scales: list[str | None]) -> Path:
    meshes = "".join(
        f'<mesh name="nor_cal_{i:02d}" file="m.obj"' + (f' scale="{s}"' if s else "") + "/>"
        for i, s in enumerate(scales)
    )
    xml = tmp_path / "scene_locomani.xml"
    xml.write_text(f'<mujoco><asset><mesh name="room_00" file="r.obj" scale="2 2 2"/>{meshes}</asset></mujoco>')
    return xml


def test_the_carton_scale_is_the_one_its_photo_meshes_share(tmp_path):
    assert B.carton_scale(_scene(tmp_path, ["1 0.625 0.545455", "1 0.625 0.545455"])) == [1.0, 0.625, 0.545455]
    assert B.carton_scale(_scene(tmp_path, [None, None])) == [1.0, 1.0, 1.0]  # the room's own carton


def test_a_carton_scaled_two_ways_or_absent_is_refused(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="one scale"):
        B.carton_scale(_scene(tmp_path, ["1 0.625 1", None]))
    with pytest.raises(SystemExit, match="none"):
        B.carton_scale(_scene(tmp_path, []))
