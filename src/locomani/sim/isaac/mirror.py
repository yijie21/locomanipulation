"""The Isaac mirror: the newest MuJoCo state in, the dataset cameras out, when the policy asks.

Closed-loop evaluation with Isaac pixels (TODO 4.7.9). MuJoCo physics keeps the wall clock and never
waits for Isaac: the sim publishes its complete configuration on the privileged stream every control
tick (`PrivilegedPublisher.state()["qpos"]`, the row StateTrace stores under the same seq), and this
process -- Kit, the render venv, its own GPU -- renders the newest one each time the policy client
asks. On demand, not continuously: the policy looks every 0.4 s, and a request is rendered from the
state current at request time, so the image and the proprioception the client read just before are
the same instant, and the render time falls inside the client's obs->chunk latency compensation
(`vla_client`: t0 is taken before the observation is built). Measured on the warehouse, three
cameras, one request per 0.4 s on GPU1: p50 89 / p90 108 ms (docs/ledger/crate.md, 2026-09-24).

    bash scripts/start_stack.sh eval <run> --gpu 0          # terminal 1
    bash scripts/run_isaac_mirror.sh <run> --gpu 1          # this; after the stack, before the rounds

Protocol, REQ/REP on TF_ISAAC_MIRROR_PORT. The request is one JSON frame, {"op": "render"} or
{"op": "ping"}; the reply is a JSON header frame and, for a render, one raw uint8 HxWx3 frame per
stream in the header's order:

    {"ok": true, "seq": int, "t": sim s, "state_wall": unix s, "state_age_s": s, "render_ms": ms,
     "streams": [...], "shapes": {stream: [h, w, 3]}}
    {"ok": false, "error": "..."}

`seq` is the privileged seq, so with a state trace running the frame the policy saw can be rendered
again offline from the trace row with that seq (the consistency gate, TODO 4.7.9.8). Every render is
also appended to `<run>/eval/isaac_mirror.jsonl`.

The server renders every dataset camera (`CAMERA_CONFIGS`), as the MuJoCo sim publishes every one on
:5555; the client keeps its schema's keys. `MirrorClient` is plain zmq + numpy, so the policy
client's venv imports this module without Kit: every Isaac import is inside a server function.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import zmq

from locomani.sim.dataset_cameras import CAMERA_CONFIGS

STREAMS = tuple(CAMERA_CONFIGS)


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
    except RuntimeError as exc:  # IsaacView refuses a partial set rather than return one
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


def handle(request: bytes, view, latest_state, nq: int) -> tuple[list[bytes], dict]:
    """One request -> (reply frames, the header when a render happened, else {})."""
    try:
        op = json.loads(request).get("op")
    except (ValueError, AttributeError):
        return error_reply("a request is one JSON object with an 'op'"), {}
    if op == "ping":
        return [_header(ok=True, streams=list(STREAMS))], {}
    if op == "render":
        return render_reply(view, latest_state(), nq)
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

    def _ask(self, op: str) -> list[bytes] | None:
        self.sock.send(_header(op=op))
        if not self.sock.poll(self.timeout_ms):
            self.sock.close(0)
            self._open()
            self.last_error = f"no reply from the Isaac mirror at {self.addr} within {self.timeout_ms} ms"
            return None
        frames = self.sock.recv_multipart()
        header = json.loads(frames[0])
        if not header.get("ok"):
            self.last_error = str(header.get("error"))
            return None
        return frames

    def ping(self) -> bool:
        return self._ask("ping") is not None

    def render(self) -> tuple[dict, dict] | None:
        frames = self._ask("render")
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


def _stop(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    from locomani.pipeline import run_context
    from locomani.pipeline.eval_config import EvalConfig
    from locomani.stack import Stack

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory whose eval stack is up")
    ap.add_argument("--gpu", required=True, help="the card Isaac renders on; not the stack's")
    ap.add_argument("--dry-run", action="store_true", help="say what would be served, start nothing")
    args = ap.parse_args(argv)

    run_dir = (args.run if args.run.is_absolute() else Path.cwd() / args.run).resolve()
    cfg = run_context.load_run_config(run_dir)
    evaluation = EvalConfig.from_dict(cfg["eval"])
    if evaluation.image_source != "isaac":
        raise SystemExit(
            f"{run_dir.name}: eval.image_source is {evaluation.image_source!r}; the policy would never ask "
            "this mirror. Set eval.image_source: isaac in the experiment file"
        )
    isaac = run_context.isaac_scene(cfg)
    instance = int(cfg["collect"]["instance"])
    try:
        stack = Stack.attach(instance)
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {instance}; the mirror renders the running sim's state, so start it first:\n"
            f"  bash scripts/start_stack.sh eval {args.run} --gpu <card>"
        )
    run_context.require_stack_of(stack, run_dir, "eval")
    scene_xml = stack.spec.scene_path() / "scene_locomani.xml"
    ports = stack.spec.ports()
    plan = dict(
        scene_xml=str(scene_xml),
        isaac_scene=isaac["dir"],
        rtx_settings=isaac["rtx_settings"],
        streams=list(STREAMS),
        privileged_port=ports["TF_PRIVILEGED_PORT"],
        mirror_port=ports["TF_ISAAC_MIRROR_PORT"],
        gpu=str(args.gpu),
        stack_gpu=str(stack.spec.gpu),
    )
    if args.dry_run:
        run_context.show_config("isaac_mirror", plan)
        return 0
    if str(args.gpu) == str(stack.spec.gpu):
        print(f"WARNING: the mirror and the stack share GPU {args.gpu}; the render then competes with the sim")
    return _run(run_dir, plan)


def _run(run_dir: Path, plan: dict) -> int:
    """Kit's lifetime. On failure os._exit(1) BEFORE app.close(): close() ends the process with status 0."""
    os.environ.update(CUDA_VISIBLE_DEVICES=plan["gpu"], OMNI_KIT_ACCEPT_EULA="YES")
    from locomani.sim.isaac import stage as S
    from locomani.sim.isaac.view import IsaacView
    from locomani.sim.privileged_pub import PrivilegedSubscriber

    log_path = run_dir / "eval" / "isaac_mirror.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = S.StageCfg(scene_dir=plan["isaac_scene"], robot_mjcf=plan["scene_xml"], absolute_camera=True)
    app = S.start_app(cfg)
    try:
        view = IsaacView(cfg, plan["scene_xml"], plan["streams"], plan["rtx_settings"])
        nq = view.replay.model.nq
        sub = PrivilegedSubscriber(port=plan["privileged_port"])
        first = sub.wait(30.0)
        problem = state_problem(first, nq)  # the check every request gets, once before any request
        if problem:
            raise SystemExit(problem)
        view.warm_up(np.asarray(first["qpos"]))
        sock = zmq.Context.instance().socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://*:{plan['mirror_port']}")
        signal.signal(signal.SIGTERM, _stop)
        print(f"isaac mirror listening on :{plan['mirror_port']} (state from :{plan['privileged_port']})", flush=True)
        with log_path.open("a") as sink:

            def record(header):
                sink.write(json.dumps(header) + "\n")
                sink.flush()

            try:
                serve(sock, view, lambda: sub.latest(0), nq, record, lambda: False)
            except KeyboardInterrupt:
                pass
    except SystemExit as exc:
        print(f"isaac mirror: {exc}", flush=True)
        os._exit(1)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
