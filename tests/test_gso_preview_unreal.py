"""scripts/gso_preview_unreal.py's run over the instances (2026-11-13: the editor grew to 22 GB by the eighth and the OOM
kill took it down): an up-to-date picture is skipped, the index is merged and written as each instance is done, and
the editor is restarted every --restart-every instances. The editor, the level builds and the captures are stubbed."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

G = pytest.importorskip("gso_preview_unreal")


@pytest.fixture
def run(tmp_path, monkeypatch):
    scenes, assets, out = tmp_path / "scenes", tmp_path / "assets", tmp_path / "out"
    for seed in range(4):
        name = G.SCENE.format(seed=seed)
        (scenes / name).mkdir(parents=True)
        (scenes / name / "scene_locomani.xml").write_text("<mujoco/>")
        (assets / name).mkdir(parents=True)
        (assets / name / "source.json").write_text("{}")
        os.utime(assets / name / "source.json", (1000, 1000))
    monkeypatch.setattr(G.paths, "UNREAL_ASSETS", str(assets))
    monkeypatch.setattr(G, "level_state", lambda scene_dir: "ok")
    calls = []

    def render(scene_dir, png, log=print):
        calls.append(scene_dir.name)
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"png")
        return {"png": str(png)}

    monkeypatch.setattr(G, "render_instance", render)
    monkeypatch.setattr(G, "restart_editor", lambda gpu: calls.append(f"restart {gpu}"))

    def main(*args):
        assert G.main(["--seeds", "4", "--scenes", str(scenes), "--out", str(out), *args]) == 0
        return calls, json.loads((out / "index_unreal.json").read_text())

    return main, out


def test_restarts_every_n_and_skips_up_to_date_pictures(run):
    main, out = run
    calls, index = main("--restart-every", "2", "--gpu", "1")
    s = [G.SCENE.format(seed=i) for i in range(4)]
    assert calls == [s[0], s[1], "restart 1", s[2], s[3]]
    assert sorted(index) == s
    calls.clear()
    (out / f"{s[3]}_unreal.png").unlink()  # the run that died before the last one
    calls, index = main()
    assert calls == [s[3]] and sorted(index) == s  # the others kept, in the index too
    calls.clear()
    calls, _ = main("--force", "--restart-every", "0")
    assert calls == s


def test_a_picture_older_than_its_level_is_drawn_again(run):
    main, out = run
    main("--restart-every", "0")
    s0 = G.SCENE.format(seed=0)
    os.utime(out / f"{s0}_unreal.png", (500, 500))  # the level was rebuilt after the picture
    calls, _ = main("--restart-every", "0")
    assert calls[4:] == [s0]
