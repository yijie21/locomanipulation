"""Opt-in diagnostic capture; never publishes control or changes observations.

All disk work runs outside the policy/control threads. Times are wall-clock
and sim time/seq are retained where available. A dropped capture is reported.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def serial(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Capture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "chunks").mkdir(exist_ok=True)
        self.queue = queue.Queue(maxsize=4096)
        self.stop = threading.Event()
        self.dropped = 0
        self.errors = []
        self.chunk_id = 0
        self.chunk_lock = threading.Lock()
        self.inference_context = threading.local()
        self.latest_debug = None
        self.latest_priv = None
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.writer.start()
        self.reader.start()

    def put(self, kind, **data):
        row = {"wall": time.time(), **data}
        try:
            self.queue.put_nowait((kind, row))
        except queue.Full:
            self.dropped += 1

    def event(self, name, **data):
        self.put("events", event=name, **data)
        print(f"[transition-audit] {name}", flush=True)

    def _write(self):
        files = {}
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    break
                kind, row = item
                if kind == "chunk":
                    np.savez_compressed(self.root / "chunks" / f"{row.pop('id'):06d}.npz", **row)
                else:
                    if kind not in files:
                        files[kind] = (self.root / f"{kind}.jsonl").open("w")
                    files[kind].write(json.dumps(row, default=serial) + "\n")
            except Exception as exc:
                self.errors.append(repr(exc))
            finally:
                self.queue.task_done()
        for fh in files.values():
            fh.close()

    def _read(self):
        import zmq
        from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.connect(f"tcp://localhost:{os.environ.get('TF_PRIVILEGED_PORT', '5581')}")
        debug = ZMQStateSubscriber(port=int(os.environ.get("TF_OUT_PORT", "5557")))
        try:
            while not self.stop.is_set():
                try:
                    row = sock.recv_json(zmq.NOBLOCK)
                    self.latest_priv = row
                    self.put("privileged", state=row)
                except zmq.Again:
                    pass
                row = debug.get_msg()
                if row is not None:
                    self.latest_debug = row
                    self.put("debug", state=row)
                self.stop.wait(0.002)
        except Exception as exc:
            self.errors.append(repr(exc))
        finally:
            debug.close()
            sock.close(0)
            ctx.term()

    def instrument(self):
        import agent_server
        import vla_round

        original_send = agent_server.AgentCommander.token_frame
        original_get = vla_round.PolicyLink.get_chunk
        capture = self

        def send(commander, token, *args, **kwargs):
            started = time.time()
            result = original_send(commander, token, *args, **kwargs)
            capture.put(
                "sent",
                started=started,
                token=np.asarray(token).copy(),
                right_hand=kwargs.get("right_hand"),
                left_hand=kwargs.get("left_hand"),
                frame_index=kwargs.get("frame_index"),
                source=kwargs.get("trace_source"),
                receipt_status="unconfirmed",
            )
            return result

        def get(link, obs):
            started = time.time()
            token, grip = original_get(link, obs)
            ended = time.time()
            with capture.chunk_lock:
                cid = capture.chunk_id
                capture.chunk_id += 1
            capture.inference_context.chunk_id = cid
            link.capture_context = capture.inference_context
            capture.put(
                "chunk",
                id=cid,
                started=started,
                ended=ended,
                token=np.asarray(token).copy(),
                grip=np.asarray(grip).copy(),
                **{k: np.asarray(v).copy() for k, v in obs.items()},
            )
            capture.put("inference", id=cid, started=started, ended=ended, rtc_applied=getattr(link, "last_rtc", None))
            return token, grip

        agent_server.AgentCommander.token_frame = send
        vla_round.PolicyLink.get_chunk = get

    def close(self):
        self.stop.set()
        self.reader.join(timeout=3)
        self.queue.put(None)
        self.writer.join(timeout=60)
        result = dict(
            dropped=self.dropped, errors=self.errors, writer_alive=self.writer.is_alive(), chunks=self.chunk_id
        )
        (self.root / "capture_status.json").write_text(json.dumps(result, indent=2))
        if self.dropped or self.errors or self.writer.is_alive():
            raise RuntimeError(f"Incomplete diagnostic capture: {result}")
