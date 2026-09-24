"""An offline run's dataset cameras: the exporter's stand-in frames, and convert's refusal to train on them."""

import json

import pytest

from locomani.pipeline.render_dataset_cameras import MARKER, pending, require_rendered
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
    # every saved episode, failed rounds included: their priv_seq says discarded, their parquet exists
    session = _session(tmp_path, {0: 12, 1: 30, 2: 9})
    offline = {"backend": {"render": "offline"}}
    assert pending(session) == [0, 1, 2]
    with pytest.raises(SystemExit, match="3 episode"):
        require_rendered(offline, [str(session)])
    require_rendered({"backend": {"render": "live"}}, [str(session)])  # live frames are real
    (session / MARKER).write_text(json.dumps({"episodes": [0, 1, 2]}))
    assert pending(session) == []
    require_rendered(offline, [str(session)])
