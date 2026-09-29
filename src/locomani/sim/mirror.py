"""The mirror protocol: the newest MuJoCo state in, the dataset cameras out, when the policy asks.

A mirror renders the eval sim's state in another renderer for a closed loop whose `images.source` is not
mujoco: `locomani.sim.isaac.mirror` (Isaac, TODO 4.7.9) and `locomani.sim.unreal.mirror` (Unreal / URLab,
TODO 12.9) serve this same protocol from the same port, so the policy client (`MirrorClient`, used by
`backends/sonic/vla_obs.ObsBuilder`) does not know which renderer answers. MuJoCo physics keeps the wall clock
and never waits for the renderer: the sim publishes its complete configuration on the privileged stream every
control tick (`PrivilegedPublisher.state()["qpos"]`, the row StateTrace stores under the same seq), and the
mirror renders the newest one each time the client asks. On demand, not continuously: the policy looks every
0.4 s, and a request is rendered from the state current at request time, so the image and the proprioception
the client read just before are the same instant, and the render time falls inside the client's obs->chunk
latency compensation (`vla_client`: t0 is taken before the observation is built).

Protocol, REQ/REP on TF_ISAAC_MIRROR_PORT (`ports.isaac_mirror()`; the name predates the second renderer). The
request is one JSON frame, {"op": "render"} or {"op": "ping"}; the reply is a JSON header frame and, for a
render, one raw uint8 HxWx3 frame per stream in the header's order. A render may name the state it wants,
{"op": "render", "seq": N}: a lockstep client (tasks/crate/vla_lockstep.py) asks for the frozen boundary's own
state, so its images and its proprioception are one instant by construction. The mirror waits up to SEQ_WAIT_S
for that state to arrive and refuses a newer one -- the world must not move while the policy looks:

    {"ok": true, "seq": int, "t": sim s, "state_wall": unix s, "state_age_s": s, "render_ms": ms,
     "streams": [...], "shapes": {stream: [h, w, 3]}}
    {"ok": false, "error": "..."}

`seq` is the privileged seq, so with a state trace running the frame the policy saw can be rendered again
offline from the trace row with that seq (the consistency gate, TODO 4.7.9.8).

A server renders every dataset camera (`CAMERA_CONFIGS`), as the MuJoCo sim publishes every one on :5555; the
client keeps its schema's keys. This module is plain zmq + numpy: the policy client's venv, the Isaac render venv
and the Unreal bridge venv all import it. A view is any object with `render(qpos) -> {stream: HxWx3 uint8}`
that raises RuntimeError rather than return a partial set.
"""

from __future__ import annotations

import json
import time

import numpy as np
import zmq

from locomani.sim.dataset_cameras import CAMERA_CONFIGS

STREAMS = tuple(CAMERA_CONFIGS)
# The run's `images.source` values a mirror serves; "mujoco" is the sim's own camera stream.
MIRRORED_SOURCES = ("isaac", "unreal")
SEQ_WAIT_S = 1.0  # a lockstep boundary's state is published before its step returns; this covers the hop


def _header(**fields) -> bytes:
    return json.dumps(fields).encode()


def error_reply(message: str) -> list[bytes]:
    return [_header(ok=False, error=message)]


def state_problem(state: dict | None, nq: int) -> str | None:
    """Why `state`, a privileged message, cannot be rendered by a mirror whose scene has `nq`; None if it can."""
    if state is None:
        return "no privileged state yet: is the stack's sim running?"
    qpos = state.get("qpos")
    if qpos is None:
        return "the privileged state carries no qpos: the sim predates the mirror; restart the stack from this checkout"
    if len(qpos) != nq:
        return f"the sim's qpos has {len(qpos)} values and the mirror's scene {nq}: they run different scenes"
    return None


def render_reply(view, state: dict | None, nq: int) -> tuple[list[bytes], dict]:
    """Answer one render request from `state`, a privileged message: (frames to send, header)."""
    problem = state_problem(state, nq)
    if problem:
        return error_reply(problem), {}
    age = time.time() - float(state["wall"])
    t0 = time.perf_counter()
    try:
        images = view.render(np.asarray(state["qpos"], dtype=np.float64))
    except RuntimeError as exc:  # a view refuses a partial set rather than return one
        return error_reply(str(exc)), {}
    streams = list(images)
    header = dict(
        ok=True,
        seq=int(state["seq"]),
        t=float(state["t"]),
        state_wall=float(state["wall"]),
        state_age_s=round(age, 4),
        render_ms=round((time.perf_counter() - t0) * 1e3, 2),
        streams=streams,
        shapes={k: list(images[k].shape) for k in streams},
    )
    frames = [np.ascontiguousarray(images[k], dtype=np.uint8).tobytes() for k in streams]
    return [json.dumps(header).encode(), *frames], header


def state_at(latest_state, seq: int, wait_s: float = SEQ_WAIT_S) -> tuple[dict | None, str | None]:
    """The privileged state numbered `seq`: (state, None), or (None, why not)."""
    end = time.monotonic() + wait_s
    state = latest_state()
    while state is None or int(state.get("seq", -1)) < seq:
        if time.monotonic() > end:
            newest = None if state is None else state.get("seq")
            return None, f"state {seq} never arrived within {wait_s} s (newest {newest})"
        time.sleep(0.001)
        state = latest_state() or state
    if int(state["seq"]) != seq:
        return None, f"asked for state {seq} but the newest is {state['seq']}: the world moved while it waited"
    return state, None


def handle(request: bytes, view, latest_state, nq: int) -> tuple[list[bytes], dict]:
    """One request -> (reply frames, the header when a render happened, else {})."""
    try:
        req = json.loads(request)
        op = req.get("op")
    except (ValueError, AttributeError):
        return error_reply("a request is one JSON object with an 'op'"), {}
    if op == "ping":
        return [_header(ok=True, streams=list(STREAMS))], {}
    if op == "render":
        if req.get("seq") is None:
            return render_reply(view, latest_state(), nq)
        state, why = state_at(latest_state, int(req["seq"]))
        return (error_reply(why), {}) if why else render_reply(view, state, nq)
    return error_reply(f"unknown op {op!r}; the mirror answers 'render' and 'ping'"), {}


def serve(sock, view, latest_state, nq: int, record, stopped) -> None:
    """Answer requests until `stopped()`; `record(header)` sees every render."""
    while not stopped():
        if not sock.poll(200):
            continue
        frames, header = handle(sock.recv(), view, latest_state, nq)
        sock.send_multipart(frames)
        if header:
            record(header)


class MirrorClient:
    """The policy client's side. `render()` returns ({stream: HxWx3 uint8}, header) or None.

    None on an error reply or a timeout; a timeout also replaces the socket, because a REQ that
    never got its reply cannot send again (the "lazy pirate" pattern).
    """

    def __init__(self, port: int, host: str = "localhost", timeout_ms: int = 2000):
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context()
        self.last_error = ""
        self._open()

    def _open(self) -> None:
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def _ask(self, op: str, **fields) -> list[bytes] | None:
        self.sock.send(_header(op=op, **fields))
        if not self.sock.poll(self.timeout_ms):
            self.sock.close(0)
            self._open()
            self.last_error = f"no reply from the mirror at {self.addr} within {self.timeout_ms} ms"
            return None
        frames = self.sock.recv_multipart()
        header = json.loads(frames[0])
        if not header.get("ok"):
            self.last_error = str(header.get("error"))
            return None
        return frames

    def ping(self) -> bool:
        return self._ask("ping") is not None

    def render(self, seq: int | None = None) -> tuple[dict, dict] | None:
        """Images of the newest state, or of state `seq` exactly (a lockstep boundary)."""
        frames = self._ask("render") if seq is None else self._ask("render", seq=int(seq))
        if frames is None:
            return None
        header = json.loads(frames[0])
        images = {
            k: np.frombuffer(buf, dtype=np.uint8).reshape(header["shapes"][k])
            for k, buf in zip(header["streams"], frames[1:])
        }
        return images, header

    def close(self) -> None:
        self.sock.close(0)
        self.ctx.term()
