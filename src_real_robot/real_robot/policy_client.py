"""A msgpack + ZMQ client for this repository's policy server (`locomani.pipeline.serve_policy`).

The server is Isaac-GR00T's `PolicyServer` (`src/gr00t`, the vendored copy the checkpoint was trained with) plus two
endpoints: `get_action_raw`, which takes the FLAT observation dict and parses it server-side so this process needs
no gr00t, and `identity`, which reports the checkpoint's file hashes and the field order of each modality.
`locomani.backends.sonic.vla_obs.PolicyLink` speaks the same protocol from the simulation; this one is kept free of
simulation imports.
"""

from __future__ import annotations

from locomani.schema import TaskSchema


class PolicyClient:
    def __init__(self, host: str = "localhost", port: int = 5590, timeout_ms: int = 30000):
        import msgpack
        import msgpack_numpy as mnp
        import zmq

        self._msgpack, self._mnp = msgpack, mnp
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def _call(self, endpoint: str, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self.sock.send(self._msgpack.packb(request, default=self._mnp.encode))
        reply = self._msgpack.unpackb(self.sock.recv(), object_hook=self._mnp.decode)
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        return reply

    def ping(self) -> bool:
        try:
            self._call("ping")
            return True
        except Exception:
            return False

    def identity(self) -> dict:
        return self._call("identity")

    def get_action_raw(self, obs: dict) -> dict:
        return self._call("get_action_raw", {"obs": obs})

    def close(self) -> None:
        self.sock.close(linger=0)
        self.ctx.term()


def identity_mismatch(identity: dict, schema: TaskSchema) -> dict:
    """The modalities whose field order the server reports differently from `schema`: empty when the checkpoint
    being served is one this client's observations and action parsing fit."""
    served = identity.get("modalities") or {}
    wanted = {"video": list(schema.video_keys), "state": list(schema.state_keys), "action": list(schema.action_keys)}
    return {kind: (served.get(kind), keys) for kind, keys in wanted.items() if list(served.get(kind) or []) != keys}
