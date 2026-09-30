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
    unreal = run_context.evaluation_of(_run(images={"source": "unreal"}))
    assert (unreal.image_source, unreal.backend.render) == ("unreal", "offline")


@pytest.mark.parametrize(
    "cfg, message",
    [
        (_run(images={"source": "isaac"}, eval={"image_source": "isaac"}), "old spelling"),
        (_run(images={"source": "blender"}), "images must be"),
        (_run(images={"source": "unreal"}, backend={"render": "live"}), "backend.render: offline"),
        (_run(images={"source": "unreal"}, eval={"backend": {"render": "live"}}), "contradicts"),
        (_run(images={"source": "isaac", "gpu": 1}), "images must be"),
        (_run(images={"source": "isaac", "third_person": "no"}), "images must be"),
        (_run(images={"source": "isaac", "third_person": None}), "images must be"),
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


REGISTRY = {
    "room": {"dir": "outputs/scene/room", "unreal": {"level": "room", "repeats": 8, "build": "build_level room"}},
    "floor": {"dir": "outputs/scene/floor", "unreal": {"unsupported": "no Blender room"}},
    "plain": {"dir": "outputs/scene/plain"},
}


def test_an_unreal_experiment_on_a_scene_without_a_level_is_refused_at_load(tmp_path, monkeypatch):
    registry = tmp_path / "scenes.yaml"
    registry.write_text(yaml.safe_dump(REGISTRY))
    monkeypatch.setattr(run_context, "SCENES", registry)
    experiment = tmp_path / "exp.yaml"
    for scene, message in (
        ("floor", "cannot use images.source unreal: no Blender room"),
        ("plain", "has no Unreal level"),
    ):
        experiment.write_text(
            yaml.safe_dump({"name": "exp", "scene": {"name": scene}, **_run(images={"source": "unreal"})})
        )
        with pytest.raises(SystemExit, match=message):
            run_context.open_run(experiment, write=False)
    # a level in the registry is enough at load; whether it is built on this machine is asked before rendering
    experiment.write_text(
        yaml.safe_dump({"name": "exp", "scene": {"name": "room"}, **_run(images={"source": "unreal"})})
    )
    assert run_context.open_run(experiment, write=False)[0]["images"]["source"] == "unreal"
    experiment.write_text(
        yaml.safe_dump({"name": "exp", "scene": {"name": "floor"}, **_run(images={"source": "isaac"})})
    )
    run_context.open_run(experiment, write=False)  # other sources do not read the unreal entry


def test_an_unreal_run_needs_its_level_built_from_this_scene(tmp_path, monkeypatch):
    import hashlib
    import json

    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scene_locomani.xml").write_text("<mujoco/>")
    project = tmp_path / "host" / "Host.uproject"
    registry = tmp_path / "scenes.yaml"
    registry.write_text(yaml.safe_dump({**REGISTRY, "room": {**REGISTRY["room"], "dir": str(scene)}}))
    monkeypatch.setattr(run_context.paths, "UE_PROJECT", str(project))
    monkeypatch.setattr(run_context.paths, "UNREAL_ASSETS", str(tmp_path / "assets"))
    cfg = {"images": {"source": "unreal"}, "scene": {"name": "room", "dir": str(scene)}}
    with pytest.raises(
        SystemExit, match="room.umap is missing; build the Unreal level of scene 'room': build_level room"
    ):
        run_context.unreal_scene(cfg, registry)
    umap = project.parent / "Content" / "Levels" / "room.umap"
    umap.parent.mkdir(parents=True)
    umap.write_bytes(b"")
    with pytest.raises(SystemExit, match="source.json is missing"):
        run_context.unreal_scene(cfg, registry)
    source = tmp_path / "assets" / "room" / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"scene_xml_sha256": hashlib.sha256(b"<mujoco/>").hexdigest()}))
    assert run_context.unreal_scene(cfg, registry) == {"level": "room", "level_path": "/Game/Levels/room", "repeats": 8}
    (scene / "scene_locomani.xml").write_text("<mujoco><!-- another carton --></mujoco>")
    with pytest.raises(SystemExit, match="built from another"):
        run_context.unreal_scene(cfg, registry)


@pytest.mark.parametrize("repeats", [0, 65, 2.5, True])
def test_the_unreal_capture_count_is_a_whole_number_the_plugin_accepts(tmp_path, repeats):
    registry = tmp_path / "scenes.yaml"
    registry.write_text(yaml.safe_dump({"room": {"dir": "x", "unreal": {"level": "room", "repeats": repeats}}}))
    with pytest.raises(SystemExit, match="unreal.repeats"):
        run_context.unreal_scene({"images": {"source": "unreal"}, "scene": {"name": "room"}}, registry)


def test_the_registry_gives_every_warehouse_scene_a_level_and_says_why_the_others_have_none():
    scenes = yaml.safe_load(run_context.SCENES.read_text())
    levels = {name for name, entry in scenes.items() if "level" in (entry.get("unreal") or {})}
    assert levels == {name for name in scenes if name.startswith("warehouse_")}
    for name, entry in scenes.items():
        unreal = entry.get("unreal") or {}
        assert unreal.get("unsupported") or unreal.get("level") == name, name
        if "level" in unreal:
            assert unreal["build"] == f"$BRIDGE_PY -m locomani.sim.unreal.build_level {name}"


@pytest.mark.parametrize("name", ["warehouse_smoke", "warehouse_200"])
def test_the_unreal_warehouse_runs_differ_from_the_isaac_default_in_the_image_source_only(name):
    # docs/warehouse.md offers Isaac and Unreal side by side: choosing a renderer is choosing one of these two files,
    # so nothing but who draws the pictures may differ
    experiments = run_context.ROOT / "configs" / "experiments"
    isaac, unreal = (yaml.safe_load((experiments / f"{n}.yaml").read_text()) for n in (name, f"{name}_unreal"))
    assert (isaac["images"]["source"], unreal["images"]["source"]) == ("isaac", "unreal")
    assert {k: v for k, v in isaac["images"].items() if k != "source"} == {
        k: v for k, v in unreal["images"].items() if k != "source"
    }  # the third-person switch alike
    assert unreal["name"] == f"{name}_unreal"
    rest = [{k: v for k, v in cfg.items() if k not in ("name", "images")} for cfg in (isaac, unreal)]
    assert rest[0] == rest[1]


def test_third_person_is_on_unless_the_run_turns_it_off():
    assert run_context.third_person(_run(images={"source": "isaac"}))  # the default, and every copy before 09-30
    assert run_context.third_person(_run())  # the historical format too
    assert not run_context.third_person(_run(images={"source": "unreal", "third_person": False}))
    cfg = _run(images={"source": "isaac", "third_person": False})
    run_context.migrate_images(cfg)
    assert cfg["images"] == {"source": "isaac", "third_person": False}  # checked, not rewritten


def test_the_current_warehouse_files_state_the_switch():
    for name in ("warehouse_smoke", "warehouse_smoke_unreal", "warehouse_smoke_fixed_start", "warehouse_200",
                 "warehouse_200_unreal", "warehouse_200_fixed_start"):  # fmt: skip
        cfg = yaml.safe_load((run_context.ROOT / "configs" / "experiments" / f"{name}.yaml").read_text())
        assert cfg["images"]["third_person"] is True, name
