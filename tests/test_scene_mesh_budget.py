"""TODO 4.7.8: the scene build refuses a drawn mesh over MuJoCo's fast render path (349,525 faces)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scene_mesh_budget as S  # noqa: E402


def _obj(path: Path, quads: int) -> Path:
    path.write_text("".join(f"v {i} 0 0\n" for i in range(4)) + "f 1 2 3 4\n" * quads)
    return path


def test_faces_count_an_ngon_as_its_triangles(tmp_path):
    assert S.faces(_obj(tmp_path / "a.obj", 5)) == 10


def test_a_drawn_mesh_over_the_limit_is_named_and_a_hidden_one_is_not(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "LIMIT_FACES", 8)
    _obj(tmp_path / "big.obj", 5)
    _obj(tmp_path / "small.obj", 2)
    xml = tmp_path / "scene_locomani.xml"
    xml.write_text(
        '<mujoco><asset><mesh name="big" file="big.obj"/><mesh name="small" file="small.obj"/>'
        '<mesh name="hidden" file="big.obj"/></asset><worldbody><geom type="mesh" mesh="big" group="1"/>'
        '<geom type="mesh" mesh="small"/><geom type="mesh" mesh="hidden" group="3"/></worldbody></mujoco>'
    )
    assert [(n, k) for n, k, _ in S.over_budget(xml)] == [("big", 10)]
    assert S.main([str(xml)]) == 1
    monkeypatch.setattr(S, "LIMIT_FACES", 10)
    assert S.main([str(xml)]) == 0
