"""`ObsBuilder(image_source="isaac")`: the images come from the mirror, requested after the state is read.

The live stream sockets are replaced; the robot model and the observation assembly are the real ones,
so the observation checked here is the one the policy server would receive.
"""

import numpy as np
import pytest

pytest.importorskip("gear_sonic")

from locomani.backends.sonic.vla_obs import ObsBuilder  # noqa: E402
from locomani.schema import SONIC_CRATE  # noqa: E402
from locomani.sim import mirror as M  # noqa: E402
from locomani.sim.dataset_cameras import CAMERA_CONFIGS  # noqa: E402

PROPRIO = {"body_q": [0.0] * 29, "left_hand_q": [0.0] * 7, "right_hand_q": [0.0] * 7, "base_quat": [1.0, 0, 0, 0]}


class FakeState:
    def __init__(self, host, port):
        self.reads = 0

    def get_msg(self):
        self.reads += 1
        return PROPRIO

    def close(self):
        pass


class FakeMirror:
    def __init__(self, port):
        self.port, self.fail, self.calls, self.last_error = port, False, [], ""

    def render(self):
        self.calls.append(self.state_reads())
        if self.fail:
            self.last_error = "no reply from the Isaac mirror"
            return None
        images = {k: np.full((c["height"], c["width"], 3), 9, np.uint8) for k, c in CAMERA_CONFIGS.items()}
        return images, {"seq": 11 + len(self.calls), "state_age_s": 0.004, "render_ms": 88.0}

    def ping(self):
        return True

    def close(self):
        pass


@pytest.fixture
def builder(monkeypatch):
    import gear_sonic.utils.data_collection.zmq_state_subscriber as zs

    monkeypatch.setattr(zs, "ZMQStateSubscriber", FakeState)
    monkeypatch.setattr(M, "MirrorClient", FakeMirror)
    monkeypatch.setenv("INSTANCE", "1")
    monkeypatch.delenv("TF_ISAAC_MIRROR_PORT", raising=False)
    ob = ObsBuilder("pick up the box", schema=SONIC_CRATE, image_source="isaac")
    ob.mirror.state_reads = lambda: ob.state.reads
    return ob


def test_each_observation_renders_after_reading_the_state_and_keeps_the_mirror_header(builder):
    assert builder.cam is None and builder.mirror.port == 5568  # instance 1's mirror, no camera stream opened
    obs = builder.build()
    assert builder.mirror.calls == [1]  # the state was read before the render was requested
    for key in SONIC_CRATE.video_keys:
        assert obs[f"video.{key}"].shape == (1, 480, 640, 3) and int(obs[f"video.{key}"].max()) == 9
    assert builder.image_meta["seq"] == 12
    builder.build()
    assert builder.image_meta["seq"] == 13  # a fresh render per observation, never a cached image


def test_a_mirror_that_cannot_answer_drops_the_observation_instead_of_reusing_the_last_images(builder, capsys):
    assert builder.build() is not None
    builder.mirror.fail = True
    assert builder.build() is None
    assert "mirror: no reply" in capsys.readouterr().out
