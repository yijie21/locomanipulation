"""An offline run's dataset cameras: the exporter's stand-in frames, and convert's refusal to train on them."""

import json

import pytest

from locomani.pipeline.render_dataset_cameras import MARKER, pending, renderer_of, require_rendered, unrendered
from locomani.pipeline.run_context import migrate_images
from locomani.sim.dataset_cameras import CAMERA_CONFIGS, PlaceholderCamera, install_exporter_placeholder


def _session(tmp_path, lengths):
    session = tmp_path / "session"
    (session / "meta").mkdir(parents=True)
    (session / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": ep, "length": n}) + "\n" for ep, n in lengths.items())
    )
    return session


def test_placeholder_frames_match_the_published_cameras():
    frames = PlaceholderCamera().read()["images"]
    assert set(frames) == set(CAMERA_CONFIGS)
    for name, image in frames.items():
        assert image.shape == (CAMERA_CONFIGS[name]["height"], CAMERA_CONFIGS[name]["width"], 3) and not image.any()


def test_the_exporter_gets_the_placeholder_after_its_own_constructor():
    class Collector:
        def __init__(self, camera_port):
            self._image_subscriber = f"live client on {camera_port}"

    install_exporter_placeholder(Collector)
    assert isinstance(Collector(5555)._image_subscriber, PlaceholderCamera)


def test_convert_refuses_an_offline_run_until_every_saved_episode_is_rendered(tmp_path):
    # no failed round here, so every saved episode is one convert reads
    session = _session(tmp_path, {0: 12, 1: 30, 2: 9})
    offline = {"backend": {"render": "offline"}}
    assert pending(session) == [0, 1, 2]
    with pytest.raises(SystemExit, match="3 episode"):
        require_rendered(offline, [str(session)])
    require_rendered({"backend": {"render": "live"}}, [str(session)])  # live frames are real
    (session / MARKER).write_text(json.dumps({"episodes": [0, 1, 2]}))
    assert pending(session) == []
    require_rendered(offline, [str(session)])


def test_failed_rounds_are_never_rendered_and_convert_asks_only_for_the_rest(tmp_path):
    # a failed round is saved but discarded, and run_convert never passes --include-discarded
    session = _session(tmp_path, {0: 12, 1: 30, 2: 9})
    (session / "meta" / "info.json").write_text(json.dumps({"discarded_episode_indices": [1]}))
    offline = {"backend": {"render": "offline"}}
    assert pending(session) == [0, 2]
    (session / MARKER).write_text(json.dumps({"episodes": [0]}))
    with pytest.raises(SystemExit, match="1 episode"):
        require_rendered(offline, [str(session)])
    (session / MARKER).write_text(json.dumps({"episodes": [0, 2]}))
    require_rendered(offline, [str(session)])


def test_training_images_come_from_the_renderer_the_policy_is_evaluated_on(tmp_path):
    session = _session(tmp_path, {0: 12})
    mujoco_eval = {"backend": {"render": "offline"}, "images": {"source": "mujoco"}}
    isaac_eval = {"backend": {"render": "offline"}, "images": {"source": "isaac"}}
    (session / MARKER).write_text(json.dumps({"episodes": [0]}))  # a marker from before `renderer` existed
    assert renderer_of(session) == "mujoco"
    require_rendered(mujoco_eval, [str(session)])
    # a file from before `images` existed and without eval.image_source: MuJoCo
    require_rendered(migrate_images({"backend": {"render": "offline"}, "eval": {}}), [str(session)])
    with pytest.raises(SystemExit, match="rendered by another renderer"):
        require_rendered(isaac_eval, [str(session)])
    (session / MARKER).write_text(json.dumps({"episodes": [0], "renderer": "isaac"}))
    require_rendered(isaac_eval, [str(session)])
    with pytest.raises(SystemExit, match="rendered by another renderer"):
        require_rendered(mujoco_eval, [str(session)])
    with pytest.raises(SystemExit, match="render_dataset_cameras_isaac.sh"):
        require_rendered(isaac_eval, [str(_session(tmp_path / "fresh", {0: 5}))])


def test_the_converter_refuses_a_failed_round_it_was_told_to_include(tmp_path):
    # --include-discarded on an offline run: the failed round kept its placeholders
    session = _session(tmp_path, {0: 12, 1: 30, 2: 9})
    (session / "meta" / "info.json").write_text(json.dumps({"discarded_episode_indices": [1]}))
    assert unrendered(session, [0, 1, 2]) == []  # no marker: live frames, or refused earlier by require_rendered
    (session / MARKER).write_text(json.dumps({"episodes": [0, 2]}))
    assert unrendered(session, [0, 2]) == []
    assert unrendered(session, [0, 1, 2]) == [1]
