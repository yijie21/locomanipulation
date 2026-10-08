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
from locomani.sim import robot_look


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
    # the stage records its scene's robot (none: stock); the renderer adds that robot's USD to it
    (scene / "scene_meta.json").write_text(json.dumps({"robot": "mars"}))
    with pytest.raises(SystemExit, match="built for the stock robot, the scene carries mars; rebuild it: b.sh"):
        run_context.isaac_scene(cfg, registry)
    (stage / "source.json").write_text(json.dumps({"scene_xml_sha256": sha, "robot": "mars"}))
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
    built = {"scene_xml_sha256": hashlib.sha256(b"<mujoco/>").hexdigest(), "robot_look": robot_look.NAME}
    source.write_text(json.dumps(built))
    assert run_context.unreal_scene(cfg, registry) == {"level": "room", "level_path": "/Game/Levels/room", "repeats": 8}
    # a level built before robot_look existed, or with another one, draws another robot than Isaac does
    for look in (None, "unitree_20260929"):
        source.write_text(json.dumps({**built, "robot_look": look}))
        with pytest.raises(SystemExit, match=f"carries the robot look {look}, not {robot_look.NAME}.*build_level room"):
            run_context.unreal_scene(cfg, registry)
    # the robot: a level records the one its scene carries (none recorded: stock, built before 2026-10-02)
    source.write_text(json.dumps({**built, "robot": "mars"}))
    with pytest.raises(SystemExit, match="built for the mars robot, the scene carries stock.*build_level room"):
        run_context.unreal_scene(cfg, registry)
    (scene / "scene_meta.json").write_text(json.dumps({"robot": "mars"}))
    assert run_context.unreal_scene(cfg, registry)["level"] == "room"
    source.write_text(json.dumps(built))
    with pytest.raises(SystemExit, match="built for the stock robot, the scene carries mars"):
        run_context.unreal_scene(cfg, registry)
    (scene / "scene_meta.json").unlink()
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
    # warehouse_textured_mars(_d435) is only the base the mars boxes are built from; it says so
    bases = {"warehouse_textured_mars", "warehouse_textured_mars_d435"}
    # and the scene_discovery apartment, on its own room level (unreal.room_level, built by unreal.room_build)
    apartments = {"adt_apartment_mars"}
    assert levels == {name for name in scenes if name.startswith("warehouse_") and name not in bases} | apartments
    for name in apartments:
        assert "--room-level " + scenes[name]["unreal"]["room_level"] in scenes[name]["unreal"]["room_build"]
    for name, entry in scenes.items():
        unreal = entry.get("unreal") or {}
        assert unreal.get("unsupported") or unreal.get("level") == name, name
        if "level" in unreal:
            assert unreal["build"] == f"$BRIDGE_PY -m locomani.sim.unreal.build_level {name}"


@pytest.mark.parametrize(
    "isaac_name, unreal_name",
    [
        ("warehouse_smoke", "warehouse_smoke_unreal"),
        ("warehouse_200", "warehouse_200_unreal"),
        # the mars runs, the ones docs/warehouse.md runs, have Unreal as the default (user, 2026-10-03)
        ("warehouse_mars_smoke_isaac", "warehouse_mars_smoke"),
        ("warehouse_mars_200_isaac", "warehouse_mars_200"),
        ("warehouse_mars_smoke_d435_isaac", "warehouse_mars_smoke_d435_unreal"),
    ],
)
def test_the_unreal_and_isaac_warehouse_runs_differ_in_the_image_source_only(isaac_name, unreal_name):
    # docs/warehouse.md offers Isaac and Unreal side by side: choosing a renderer is choosing one of these two files,
    # so nothing but who draws the pictures may differ
    experiments = run_context.ROOT / "configs" / "experiments"
    isaac, unreal = (yaml.safe_load((experiments / f"{n}.yaml").read_text()) for n in (isaac_name, unreal_name))
    assert (isaac["images"].pop("source"), unreal["images"].pop("source")) == ("isaac", "unreal")
    assert isaac["images"] == unreal["images"]  # the rest of the block (camera_delay) is the same
    assert (isaac["name"], unreal["name"]) == (isaac_name, unreal_name)
    rest = [{k: v for k, v in cfg.items() if k not in ("name", "images")} for cfg in (isaac, unreal)]
    assert rest[0] == rest[1]


@pytest.mark.parametrize(
    "d455_name, d435_name",
    [
        ("warehouse_mars_smoke", "warehouse_mars_smoke_d435_unreal"),
        ("warehouse_mars_smoke_isaac", "warehouse_mars_smoke_d435_isaac"),
        ("warehouse_mars_smoke_mujoco", "warehouse_mars_smoke_d435_mujoco"),
    ],
)
def test_every_renderer_has_a_d435_ego_run_that_differs_in_the_scene_only(d455_name, d435_name):
    # the D435 ego view is chosen by the scene (user, 2026-10-06: on Unreal, Isaac and MuJoCo alike), so each
    # renderer's D435 run is its D455 run with the _d435 scene and nothing else; the D435 runs name their renderer
    experiments = run_context.ROOT / "configs" / "experiments"
    d455, d435 = (yaml.safe_load((experiments / f"{n}.yaml").read_text()) for n in (d455_name, d435_name))
    assert (d455["name"], d435["name"]) == (d455_name, d435_name)
    assert (d455["scene"].pop("name"), d435["scene"].pop("name")) == (
        "warehouse_box_45x32x33_mars",
        "warehouse_box_45x32x33_mars_d435",
    )
    rest = [{k: v for k, v in cfg.items() if k != "name"} for cfg in (d455, d435)]
    assert rest[0] == rest[1]


def test_the_withdrawn_third_person_switch_is_read_as_nothing():
    # images.third_person (4144452, pushed) switched the fill's filming; the fill no longer films, so run copies that
    # carry it load, either value meaning what no key means (scripts/render_third_person.sh films on request)
    for value in (True, False):
        cfg = run_context.migrate_images(_run(images={"source": "isaac", "third_person": value}))
        assert cfg["images"] == {"source": "isaac"}


def test_the_dataset_rate_is_read_once_at_load_and_defaults_to_the_historical_25():
    # `dataset.fps` (2026-10-06): convert's row pick, the re-render's rows and the audit's trace lookup all read it
    cfg = run_context.migrate_dataset(_run())
    assert cfg["dataset"] == {"fps": 25} and run_context.dataset_fps(cfg) == 25
    assert run_context.migrate_dataset(cfg) is cfg  # idempotent: every stage may call it
    assert run_context.dataset_fps(run_context.migrate_dataset(_run(dataset={"fps": 50}))) == 50
    for bad in ({"fps": 30}, {"fps": 25, "codec": "h264"}, 25):  # 30 does not divide 50; no other fields
        with pytest.raises(SystemExit, match="dataset must be"):
            run_context.migrate_dataset(_run(dataset=bad))
