"""Isaac-GR00T's `PolicyClient` (`gr00t/policy/server_client.py`) for a process without gr00t.

NVIDIA's client imports `gr00t.policy.server_client.PolicyClient`; SONIC's `.venv_sim`, where this client runs (it
needs gear_sonic's camera client, robot model and protocol-v4 packing), has no gr00t. This is the same REQ socket
and msgpack-numpy wire format with the same methods (`ping`, `call_endpoint`, `get_action`, `close`), plus
`identity`, the one endpoint `real_robot.run_gr00t_server` adds: the checkpoint's file hashes and the field order
of each modality, which `identity_mismatch` compares with the schema this client builds observations for.
"""

from __future__ import annotations

from typing import Any

from locomani.schema import TaskSchema
from real_robot.gripper import MODEL_UNITS, units_of_range


class PolicyClient:
    def __init__(self, host: str = "localhost", port: int = 5590, timeout_ms: int = 15000):
        import msgpack
        import msgpack_numpy as mnp
        import zmq

        self._msgpack, self._mnp, self._zmq = msgpack, mnp, zmq
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        """A fresh REQ socket: after a timeout the old one waits for a reply that never comes."""
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self.socket.send(self._msgpack.packb(request, default=self._mnp.encode))
            message = self.socket.recv()
        except self._zmq.error.Again:
            self.socket.close(linger=0)
            self._init_socket()
            raise
        response = self._msgpack.unpackb(message, object_hook=self._mnp.decode, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except self._zmq.error.ZMQError:
            return False

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        """(action, info) for a nested, batched observation; the action is {key: (B, T, D)}."""
        action, info = self.call_endpoint("get_action", {"observation": observation, "options": options})
        return action, info

    def identity(self) -> dict:
        return self.call_endpoint("identity", requires_input=False)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def identity_mismatch(identity: dict, schema: TaskSchema) -> dict:
    """The modalities whose field order the server reports differently from `schema`: empty when the checkpoint
    being served is one this client's observations and action parsing fit."""
    served = identity.get("modalities") or {}
    wanted = {"video": list(schema.video_keys), "state": list(schema.state_keys), "action": list(schema.action_keys)}
    return {kind: (served.get(kind), keys) for kind, keys in wanted.items() if list(served.get(kind) or []) != keys}


def resolve_gripper_units(identity: dict, requested: str) -> str:
    """The gripper units this client must use for the served checkpoint: "dex1" or "sim_m".

    `identity["gripper_ranges"]["action"]` (the server reads it from the checkpoint's statistics) names the units
    the checkpoint was trained in. `requested` is `--gripper-units`: "auto" takes the checkpoint's; an explicit
    value is checked against it. ValueError when they disagree, when the keys disagree among themselves, or when
    "auto" has nothing to read -- each of these used to be a wrong unit on the robot's hands."""
    if requested not in ("auto", *MODEL_UNITS):
        raise ValueError(f"gripper units {requested!r} is not one of auto, {', '.join(MODEL_UNITS)}")
    ranges = (identity.get("gripper_ranges") or {}).get("action") or {}
    found = {}
    for key, quantiles in ranges.items():
        if "q01" in quantiles and "q99" in quantiles:
            found[key] = units_of_range(quantiles["q01"], quantiles["q99"])
    if not found:
        if requested == "auto":
            raise ValueError(
                "the server reports no gripper statistics, so the checkpoint's gripper units cannot be inferred; "
                "pass --gripper-units dex1 or sim_m explicitly"
            )
        return requested
    if len(set(found.values())) != 1:
        raise ValueError(f"the checkpoint's gripper keys disagree on their units: {found}")
    inferred = next(iter(found.values()))
    if requested not in ("auto", inferred):
        raise ValueError(
            f"--gripper-units {requested} contradicts the checkpoint, whose gripper statistics are in {inferred} "
            f"({ {k: ranges[k] for k in found} }); use --gripper-units {inferred} or auto"
        )
    return inferred
