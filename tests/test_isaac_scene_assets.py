import gzip
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from locomani.sim.isaac.scene_assets import background_source, build_key, mesh_source, prepare_obj

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def installed(tmp_path):
    """Only the committed installation archive; no local EmbodiedGen outputs."""
    with tarfile.open(REPO / "assets/scene_locomani_v1.tar.gz") as archive:
        for item in archive:
            if item.isfile() and (
                item.name.startswith(("scene/mjcf/dining_table/", "scene/mjcf/counter/"))
                or item.name == "scene/layout/layout.json"
            ):
                destination = tmp_path / item.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(item).read())
    return tmp_path / "scene"


def vertices(path):
    return np.array(
        [list(map(float, line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith("v ")]
    )


def test_shipped_table_and_counter_retain_geometry_and_texture_for_isaac(installed, tmp_path):
    for name in ("dining_table", "counter"):
        source = mesh_source(installed, name)
        assert source is not None and source.z_up
        obj = prepare_obj(source, tmp_path / "converted" / name)
        y_up = vertices(obj)
        # The existing place_asset rotates +90 degrees about X before seating.
        reconstructed = np.column_stack((y_up[:, 0], -y_up[:, 2], y_up[:, 1]))
        np.testing.assert_allclose(reconstructed, vertices(source.path), atol=1e-8)
        assert "map_Kd material_0.png" in (obj.parent / "material.mtl").read_text()
        assert (obj.parent / "material_0.png").read_bytes() == source.texture.read_bytes()


def test_fresh_install_uses_complete_tracked_world_space_room(installed):
    path, baked = background_source(installed, REPO)
    manifest = json.loads(path.with_name("empty_room_world.json").read_text())
    assert baked and path.stat().st_size == manifest["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    with gzip.open(path, "rb") as stream:
        header = []
        while (line := stream.readline()) != b"end_header\n":
            assert line
            header.append(line.decode().strip())
    assert f"element vertex {manifest['vertices']}" in header
    assert f"element face {manifest['triangles']}" in header


def test_cache_checks_missing_geometry_and_texture_content(installed, tmp_path):
    meta = tmp_path / "scene_meta.json"
    xml = tmp_path / "scene.xml"
    meta.write_text('{"surfaces": {}}')
    xml.write_text("<mujoco/>")
    key = build_key(installed, REPO, meta, xml, tmp_path / "cache")
    assert build_key(installed, REPO, meta, xml, tmp_path / "moved_cache") != key
    texture = mesh_source(installed, "dining_table").texture
    texture.write_bytes(texture.read_bytes() + b"changed")
    assert build_key(installed, REPO, meta, xml, tmp_path / "cache") != key
    mesh_source(installed, "dining_table").path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing surface A"):
        build_key(installed, REPO, meta, xml, tmp_path / "cache")


def test_missing_shipped_room_is_an_error(installed, tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing tracked room"):
        background_source(installed, tmp_path)


def test_launcher_rejects_missing_sources_before_starting_kit(tmp_path):
    (tmp_path / "src").symlink_to(REPO / "src", target_is_directory=True)
    (tmp_path / "assets").symlink_to(REPO / "assets", target_is_directory=True)
    snapshot = tmp_path / "outputs/scene/sim"
    snapshot.mkdir(parents=True)
    (snapshot / "scene_meta.json").write_text('{"surfaces": {}}')
    (snapshot / "scene_locomani.xml").write_text("<mujoco/>")
    result = subprocess.run(
        ["bash", str(REPO / "scripts/build_isaac_assets.sh")],
        env={**os.environ, "LOCOMANI_ROOT": str(tmp_path), "RENDER_PY": "/bin/false"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Missing surface A" in result.stderr
    assert "scene source validation failed" in result.stderr
    assert "robot  + building" not in result.stdout
