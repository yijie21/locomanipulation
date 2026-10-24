"""The perception services a RenderedView calls at level L3 (docs/handoff/vision-rsi-design-20261023.md): Unreal's RGB
of a MuJoCo state (served from the URLab bridge venv: locomani.sim.unreal.rgb_server) and SAM 3's masks of an image for
text prompts (served from the sam3d conda env: scripts/rsi/sam3_server.py). Each server listens on 127.0.0.1 and answers
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
from multiprocessing.connection import Client as _Connect
from multiprocessing.connection import Listener

import numpy as np

UNREAL_ADDRESS = ("127.0.0.1", 18821)  # clear of the stacks (5590-5700+), URLab (5800-5812) and the editor (18801)
SAM3_ADDRESS = ("127.0.0.1", 18822)
AUTHKEY = b"locomani-vision-rsi"


def serve(address, handle, log=print) -> None:
    """Answer requests until killed: each connection in its own thread, `handle` under one lock."""
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
            threading.Thread(target=session, args=(listener.accept(),), daemon=True).start()


class Client:
    """One connection to a service, opened on first use and once more after a broken pipe."""

    def __init__(self, address=None):
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


def pack_mask(mask: np.ndarray) -> dict:
    return dict(shape=tuple(mask.shape), bits=np.packbits(mask.astype(bool), axis=None))


def unpack_mask(packed: dict) -> np.ndarray:
    n = int(np.prod(packed["shape"]))
    return np.unpackbits(packed["bits"], count=n).astype(bool).reshape(packed["shape"])


ORDINAL = {"first", "second", "third", "fourth", "fifth", "tallest", "smallest", "largest", "biggest", "left", "right"}


def prompt_of(obj: dict, key: str) -> str:
    """The category noun phrase a skill may name an object by (design D6): the scene description's phrase without its
    article, the words that tell one instance from another (ordinals, superlatives) and tokens with digits; the key with
    its suffixes split into words when there is no phrase."""
    phrase = ((obj.get("semantic") or {}).get("phrase") or "").strip()
    if not phrase:
        phrase = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", re.sub(r"(_[A-Za-z]?\d+|_Anon|_object)+$", "", key)).replace(
            "_", " "
        )
    words = [w for w in phrase.split() if w.lower() not in ("the", "a", "an") and w.lower() not in ORDINAL]
    words = [w for w in words if not re.search(r"\d", w)]
    return " ".join(words) or key
