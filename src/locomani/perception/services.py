"""The perception services: SAM 3's masks of an image for text prompts (served from the sam3d conda env:
locomani.perception.sam3_server). Moved from the esc worktree's locomani.tasks.scene.services (TODO 49.2); the address comes
from the caller (SimpleConfig.sam3), there is no module default. Each server listens on 127.0.0.1 and answers
pickled dicts over multiprocessing.connection, a thread per client and one request at a time (the editor and the GPU
model are single resources); a round worker keeps one connection to each. Standard library and numpy only: both
servers' environments import this module.

    serve(address, handle)            # handle(request dict) -> reply dict
    Client(address).call(request)     # -> reply dict; raises the server's error text as a RuntimeError
"""

from __future__ import annotations

import re
import threading
import traceback
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client as _Connect
from multiprocessing.connection import Listener

import numpy as np

AUTHKEY = b"locomani-vision-rsi"


def serve(address, handle, log=print) -> None:
    """Answer requests until killed: each connection in its own thread, `handle` under one lock. A connection that fails
    the handshake (a bare port probe, a wrong key) is logged and dropped; the server keeps listening."""
    lock = threading.Lock()

    def session(conn) -> None:
        with conn:
            while True:
                try:
                    request = conn.recv()
                except (EOFError, OSError):
                    return
                try:
                    with lock:
                        reply = handle(request)
                except Exception as e:  # the client sees the error; the server keeps serving
                    reply = dict(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
                conn.send(reply)

    with Listener(tuple(address), authkey=AUTHKEY) as listener:
        log(f"serving on {address[0]}:{address[1]}")
        while True:
            try:
                conn = listener.accept()
            except (AuthenticationError, EOFError, OSError) as e:
                log(f"dropped a connection that failed the handshake: {type(e).__name__}: {e}")
                continue
            threading.Thread(target=session, args=(conn,), daemon=True).start()


class Client:
    """One connection to a service, opened on first use and once more after a broken pipe."""

    def __init__(self, address):
        self.address, self.conn = tuple(address), None

    def call(self, request: dict) -> dict:
        for attempt in (0, 1):
            try:
                if self.conn is None:
                    self.conn = _Connect(self.address, authkey=AUTHKEY)
                self.conn.send(request)
                reply = self.conn.recv()
                break
            except (EOFError, OSError, ConnectionError):
                self.conn = None
                if attempt:
                    raise
        if "error" in reply:
            raise RuntimeError(f"service {self.address}: {reply['error']}")
        return reply


def ready(address, timeout_s: float = 0.0) -> bool:
    """Whether a service answers the handshake at `address` (polling up to `timeout_s`)."""
    import time

    end = time.time() + timeout_s
    while True:
        try:
            _Connect(tuple(address), authkey=AUTHKEY).close()
            return True
        except (OSError, EOFError, AuthenticationError):
            if time.time() >= end:
                return False
            time.sleep(1.0)


def pack_mask(mask: np.ndarray) -> dict:
    return dict(shape=tuple(mask.shape), bits=np.packbits(mask.astype(bool), axis=None))


def unpack_mask(packed: dict) -> np.ndarray:
    n = int(np.prod(packed["shape"]))
    return np.unpackbits(packed["bits"], count=n).astype(bool).reshape(packed["shape"])
