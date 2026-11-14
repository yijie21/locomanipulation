"""One logistics_room_1030 instance end to end (scripts/gso_preview.py, user 2026-11-03: five instances, one picture
each -- here one, into a temporary directory): build, settle, drop test, lint with 0 errors, the two families, and
the side-by-side render. Skipped without the scene pack, the catalog, the object meshes or an offscreen GL."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gso_preview as P  # noqa: E402

ASSETS = ROOT / "assets/mujoco_scanned_objects"


def offscreen_gl() -> bool:
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
        with mujoco.Renderer(m, 16, 16) as r:
            r.update_scene(mujoco.MjData(m))
            r.render()
        return True
    except Exception:  # no EGL / OSMesa in this process (set MUJOCO_GL=egl before pytest imports mujoco)
        return False


needs = [
    (P.prepare_logistics_scene.DEFAULT_SOURCE / "scene_locomani.xml").is_file() or "the logistics scene pack (git lfs pull)",
    (ASSETS / "catalog.json").is_file() or "the catalog (scripts/gso_catalog.py)",
    (ASSETS / "models").is_dir() or "the object meshes (scripts/install_gso.sh)",
    offscreen_gl() or f"an offscreen GL (MUJOCO_GL={os.environ.get('MUJOCO_GL')})",
]  # fmt: skip


@pytest.mark.skipif(any(n is not True for n in needs), reason="; ".join(n for n in needs if n is not True))
def test_one_instance_builds_lints_clean_and_renders_its_preview(tmp_path):
    index = P.preview([0], tmp_path / "scenes", tmp_path / "preview", "train", 10, 4, tmp_path / "families")
    ((name, entry),) = index.items()
    assert name == "logistics_room_1030_s0" and entry["checks"] == {"lint_errors": 0, "drop_failures": 0}
    assert Path(entry["png"]).is_file() and (tmp_path / "preview/index.json").is_file()
    kinds = [o["graspable"] for o in entry["objects"].values()]
    assert len(kinds) == 11 and kinds.count("single") == 6 and kinds.count("bimanual") == 5  # 4 drawn + the carton
    assert entry["families"] == {"gso_grasp_s0": 6, "gso_bimanual_s0": 5, "gso_pack_s0": 6, "gso_pack_bimanual_s0": 4,
                                 "gso_all_s0": 21, "gso_pack_all_s0": 10}  # fmt: skip
    assert sorted(p.name for p in (tmp_path / "families").iterdir()) == [
        "gso_all_s0.json", "gso_bimanual_s0.json", "gso_grasp_s0.json", "gso_pack_all_s0.json",
        "gso_pack_bimanual_s0.json", "gso_pack_s0.json", "gso_s0_tasks.jsonl"]  # fmt: skip
    meta = json.loads((tmp_path / "scenes/logistics_room_1030_s0/scene_meta.json").read_text())
    assert meta["gso"]["seed"] == 0 and len(meta["gso"]["keys"]) == 10
    assert all(o["semantic"]["drop_test"]["ok"] for o in meta["objects"].values() if o.get("free"))
    from PIL import Image

    w, h = Image.open(entry["png"]).size
    assert w == 2 * P.VIEW_W and h > P.VIEW_H
