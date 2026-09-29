"""The mirror protocol's request loop and client (locomani.sim.mirror) and the Isaac server's run check, without Kit.

The server half is driven with a stand-in view: what is under test is the protocol -- which
state a render uses, what an error looks like, that a client survives a mirror that never
answered -- not the pixels, which bench_render and the view check cover.
"""

import json
import socket
import threading

import numpy as np
import pytest
import zmq

from locomani.sim import mirror as M
from locomani.sim.dataset_cameras import CAMERA_CONFIGS
from locomani.sim.isaac import mirror as isaac_mirror

NQ = 5


class FakeView:
    """Each stream is filled with qpos[0], so a reply shows which state it was rendered from."""

    def __init__(self, fail=False):
        self.fail = fail
        self.rendered = []

    def render(self, qpos):
        if self.fail:
            raise RuntimeError("Isaac returned missing or stale camera streams for this state")
        self.rendered.append(np.asarray(qpos).copy())
        return {k: np.full((c["height"], c["width"], 3), int(qpos[0]), np.uint8) for k, c in CAMERA_CONFIGS.items()}


def state(seq, first=7.0, n=NQ, qpos=True):
    msg = {"seq": seq, "t": 0.02 * seq, "wall": 1.0e9, "pelvis": {}}
    if qpos:
        msg["qpos"] = [first] + [0.0] * (n - 1)
    return msg


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mirror():
    """A REP server on a free port, run by `serve` in a thread; yields (port, box to set state/view)."""
    port = free_port()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{port}")
    box = {"state": state(3), "view": FakeView(), "records": []}
    stop = threading.Event()
    th = threading.Thread(
        target=M.serve,
        args=(sock, box["view"], lambda: box["state"], NQ, box["records"].append, stop.is_set),
        daemon=True,
    )
    th.start()
    yield port, box
    stop.set()
    th.join(timeout=2)
    sock.close(0)
    ctx.term()


def test_a_render_returns_every_camera_of_the_newest_state_and_records_it(mirror):
    port, box = mirror
    client = M.MirrorClient(port, timeout_ms=2000)
    try:
        images, header = client.render()
        assert set(images) == set(CAMERA_CONFIGS)
        for k, c in CAMERA_CONFIGS.items():
            assert images[k].shape == (c["height"], c["width"], 3) and images[k].dtype == np.uint8
            assert int(images[k][0, 0, 0]) == 7
        assert header["seq"] == 3 and header["ok"]
        box["state"] = state(9, first=42.0)  # the next request renders the state current THEN
        images, header = client.render()
        assert header["seq"] == 9 and int(images["ego_view"][0, 0, 0]) == 42
        assert [r["seq"] for r in box["records"]] == [3, 9]
        assert client.ping()
    finally:
        client.close()


@pytest.mark.parametrize(
    "bad, why",
    [
        (None, "no privileged state"),
        (state(1, qpos=False), "carries no qpos"),
        (state(1, n=NQ + 2), "different scenes"),
    ],
)
def test_a_state_the_mirror_cannot_render_is_an_error_not_an_image(mirror, bad, why):
    port, box = mirror
    box["state"] = bad
    client = M.MirrorClient(port)
    try:
        assert client.render() is None
        assert why in client.last_error
        assert box["records"] == [] and box["view"].rendered == []
    finally:
        client.close()


def test_a_failed_render_is_reported_rather_than_half_answered():
    frames, header = M.render_reply(FakeView(fail=True), state(2), NQ)
    assert header == {} and "missing or stale" in json.loads(frames[0])["error"] and len(frames) == 1


def test_an_unknown_or_malformed_request_is_refused():
    view = FakeView()
    for request in (b"not json", json.dumps({"op": "draw"}).encode()):
        frames, header = M.handle(request, view, lambda: state(1), NQ)
        assert not json.loads(frames[0])["ok"] and header == {}
    assert view.rendered == []


def test_a_client_outlives_a_mirror_that_never_answered():
    port = free_port()
    client = M.MirrorClient(port, timeout_ms=300)
    try:
        assert client.render() is None and "no reply" in client.last_error
        # the REQ socket that timed out was replaced: once a mirror is up, the same client works
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{port}")
        stop = threading.Event()
        th = threading.Thread(
            target=M.serve, args=(sock, FakeView(), lambda: state(5), NQ, lambda h: None, stop.is_set)
        )
        th.start()
        try:
            client.timeout_ms = 2000
            _, header = client.render()
            assert header["seq"] == 5
        finally:
            stop.set()
            th.join(timeout=2)
            sock.close(0)
            ctx.term()
    finally:
        client.close()


def test_the_mirror_refuses_a_run_whose_policy_would_never_ask_it(tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({"eval": {"client": "async"}, "collect": {"instance": 0}}))
    with pytest.raises(SystemExit, match="images.source is 'mujoco'"):
        isaac_mirror.main([str(tmp_path), "--gpu", "1", "--dry-run"])


def test_a_lockstep_request_names_the_boundary_state_and_the_mirror_renders_exactly_it(mirror):
    """The lockstep client asks for the frozen boundary's seq: a stale newest state is waited for, a newer one is
    an error -- the world must not have moved while the policy looked."""
    port, box = mirror
    box["state"] = state(4)
    client = M.MirrorClient(port, timeout_ms=3000)
    try:
        images, header = client.render(seq=4)
        assert header["seq"] == 4
        assert client.render(seq=3) is None and "the world moved" in client.last_error
    finally:
        client.close()


def test_the_mirror_waits_for_a_boundary_state_still_in_flight():
    states = iter([state(5), state(5), state(6)])
    got, why = M.state_at(lambda: next(states, state(6)), 6, wait_s=1.0)
    assert why is None and got["seq"] == 6
    got, why = M.state_at(lambda: state(5), 6, wait_s=0.05)
    assert got is None and "never arrived" in why
