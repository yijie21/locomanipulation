"""`images.source`: one run-level choice of the renderer behind the policy's pictures, migrated and checked at load.

It decides the dataset cameras, convert's renderer check, the eval sim's rendering and the closed loop's images.
Before 2026-09-27 it was `eval.image_source`, and the fields it has to agree with were not checked anywhere: an
Isaac run collected live trained on MuJoCo frames without a word, and a MuJoCo run with an offline eval sim failed
every round on `camera/state streams never became ready`.
"""

from __future__ import annotations

import pytest
import yaml

from locomani.pipeline import run_context


def _run(**blocks) -> dict:
    cfg = {"backend": {"render": "offline"}, "eval": {"client": "blocking"}}
    cfg.update(blocks)
    return cfg


def test_a_historical_file_reads_its_eval_image_source_as_the_run_source():
    cfg = run_context.migrate_images(_run(eval={"client": "blocking", "image_source": "isaac"}))
    assert cfg["images"] == {"source": "isaac"} and "image_source" not in cfg["eval"]
    assert run_context.migrate_images(_run())["images"] == {"source": "mujoco"}
    assert run_context.migrate_images(cfg) is cfg  # idempotent: every stage may call it
    stale = run_context.migrate_images(_run(eval={"client": "blocking", "backend": {"render": "offline"}}))
    assert stale["eval"]["backend"]["render"] == "live"  # migrated into a valid current-format file
    assert run_context.migrate_images(stale) is stale


def test_the_eval_sim_renders_what_the_source_needs():
    # A historical MuJoCo run with an offline eval sim (97 of the files on 2026-09-27) failed every round:
    # the policy reads MuJoCo frames only from a live sim. The migrated run renders them.
    mujoco = run_context.evaluation_of(_run(eval={"client": "blocking", "backend": {"render": "offline"}}))
    assert (mujoco.image_source, mujoco.backend.render) == ("mujoco", "live")
    isaac = run_context.evaluation_of(_run(images={"source": "isaac"}))
    assert (isaac.image_source, isaac.backend.render) == ("isaac", "offline")


@pytest.mark.parametrize(
    "cfg, message",
    [
        (_run(images={"source": "isaac"}, eval={"image_source": "isaac"}), "old spelling"),
        (_run(images={"source": "unreal"}), "images must be"),
        (_run(images={"source": "isaac", "gpu": 1}), "images must be"),
        (_run(images={"source": "isaac"}, backend={"render": "live"}), "backend.render: offline"),
        (_run(images={"source": "isaac"}, backend={}), "backend.render: offline"),  # live is the default
        (_run(eval={"image_source": "isaac"}, backend={"render": "live"}), "backend.render: offline"),
        (_run(images={"source": "mujoco"}, eval={"backend": {"render": "offline"}}), "contradicts"),
        (_run(images={"source": "isaac"}, eval={"backend": {"render": "live"}}), "contradicts"),
    ],
)
def test_a_pairing_no_stage_can_serve_is_refused_at_load(cfg, message):
    with pytest.raises(SystemExit, match=message):
        run_context.migrate_images(cfg)


def test_every_stage_reads_the_source_through_the_migration(tmp_path):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(_run(eval={"client": "blocking", "image_source": "isaac"})))
    assert run_context.load_run_config(tmp_path)["images"]["source"] == "isaac"
    experiment = tmp_path / "exp.yaml"
    experiment.write_text(yaml.safe_dump({"name": "exp", **_run(images={"source": "isaac"})}))
    assert run_context.open_run(experiment, write=False)[0]["images"]["source"] == "isaac"


def test_an_isaac_run_needs_its_scene_stage_before_a_stack_starts(tmp_path, monkeypatch):
    registry = tmp_path / "scenes.yaml"
    registry.write_text(yaml.safe_dump({"plain": {"dir": "outputs/scene/plain"}}))
    monkeypatch.setattr(run_context, "SCENES", registry)
    run_context.require_images({"images": {"source": "mujoco"}, "scene": {"name": "plain"}})  # nothing to check
    with pytest.raises(SystemExit, match="has no Isaac stage"):
        run_context.require_images({"images": {"source": "isaac"}, "scene": {"name": "plain"}})


def test_a_stage_built_from_another_scene_is_refused(tmp_path, monkeypatch):
    import hashlib
    import json

    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scene_locomani.xml").write_text("<mujoco/>")
    stage = tmp_path / "assets" / "scene" / "box"
    stage.mkdir(parents=True)
    (stage / "scene.usda").write_text("#usda 1.0\n")
    registry = tmp_path / "scenes.yaml"
    registry.write_text(yaml.safe_dump({"box": {"dir": str(scene), "isaac": {"dir": "box", "build": "b.sh"}}}))
    monkeypatch.setattr(run_context.paths, "ISAAC_ASSETS", str(tmp_path / "assets"))
    cfg = {"scene": {"name": "box", "dir": str(scene)}}
    sha = hashlib.sha256(b"<mujoco/>").hexdigest()
    (stage / "source.json").write_text(json.dumps({"scene_xml_sha256": sha}))
    assert run_context.isaac_scene(cfg, registry)["dir"] == str(stage)
    (scene / "scene_locomani.xml").write_text("<mujoco><!-- another box --></mujoco>")
    with pytest.raises(SystemExit, match="built from another"):
        run_context.isaac_scene(cfg, registry)
