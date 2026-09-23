"""with_planner_field appends a field the deploy's header-driven decoder can find, leaving the rest intact."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("gear_sonic")
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message  # noqa: E402

from locomani.backends.sonic.agent_server import PLANNER_HEADER, PLANNER_TOPIC, with_planner_field  # noqa: E402


def decode(message):
    header = json.loads(message[len(PLANNER_TOPIC) : len(PLANNER_TOPIC) + PLANNER_HEADER].rstrip(b"\x00"))
    payload, out, at = message[len(PLANNER_TOPIC) + PLANNER_HEADER :], {}, 0
    size = {"i32": 4, "f32": 4}
    for f in header["fields"]:
        n = f["shape"][0]
        fmt = "<" + ("i" if f["dtype"] == "i32" else "f") * n
        out[f["name"]] = struct.unpack(fmt, payload[at : at + size[f["dtype"]] * n])
        at += size[f["dtype"]] * n
    assert at == len(payload)
    return out


def test_target_field_is_appended_and_the_original_fields_survive():
    base = build_planner_message(mode=1, movement=(1, 0, 0), facing=(1, 0, 0), speed=0.3, height=-1.0)
    msg = decode(with_planner_field(base, "target", (0.5, -0.1, 0.2, 7)))
    assert msg["mode"] == (1,) and msg["speed"][0] == pytest.approx(0.3)
    assert msg["target"] == pytest.approx((0.5, -0.1, 0.2, 7.0))
    assert decode(base).keys() == msg.keys() - {"target"}
