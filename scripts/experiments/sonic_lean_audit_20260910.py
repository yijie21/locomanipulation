"""Offline SONIC waist signal audit; no sockets or simulation.

Checks serialization and reference mapping plus model sensitivity.
The synthetic observation cannot measure physical tracking or explain a fall.
Run with CUDA_VISIBLE_DEVICES empty, CPU thread limits, and PYTHONPATH=src.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE

from locomani.backends.sonic.agent_server import HAND_CLOSED, HAND_OPEN, AgentCommander, StreamMode
from locomani.backends.sonic.primitives import upper_body_from_mj
from locomani.backends.sonic.sonic_decoder import ACTION_SCALE, ISAACLAB_TO_MUJOCO
from locomani.backends.sonic.sonic_encoder import DEFAULT_ANGLES_MJ, SonicEncoder, _stand_window, mj_to_isaaclab

ROOT = Path(__file__).resolve().parents[2]
deploy = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
source = (deploy / "src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp").read_text()
ilidx = np.array(
    [
        int(x)
        for x in re.search(r"upper_body_joint_isaaclab_order_in_isaaclab_index\s*=\s*\{([^}]+)", source)
        .group(1)
        .split(",")
    ]
)
messages = []
cmd = AgentCommander.__new__(AgentCommander)
cmd._hand_open, cmd._hand_closed = HAND_OPEN, HAND_CLOSED
cmd.reset_pose()
cmd.heading_offset = 0
cmd.sonic = SimpleNamespace(send=messages.append)
cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
enc = SonicEncoder.__new__(SonicEncoder)
enc.sess = ort.InferenceSession(
    str(deploy / "policy/release/model_encoder.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
)
enc.in_name = enc.sess.get_inputs()[0].name
encoder_baseline = enc.encode_window(*_stand_window())
dec = ort.InferenceSession(
    str(deploy / "policy/release/model_decoder.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
)
obs = np.zeros((1, 994), dtype=np.float32)
obs[0, -30:] = np.tile([0, 0, -1], 10)
obs[0, :64] = encoder_baseline
a0 = dec.run(None, {dec.get_inputs()[0].name: obs})[0].ravel()
rows = []
for pitch in [0, 0.1, 0.2, 0.3, 0.4, 0.52]:
    q = DEFAULT_ANGLES_MJ.copy()
    q[14] = pitch
    cmd.set_upper_body(upper_body_from_mj(q))
    cmd.frame()
    wire = messages[-1]
    assert wire.startswith(b"planner")
    header = json.loads(wire[7 : 7 + HEADER_SIZE].rstrip(b"\0"))
    offset = 7 + HEADER_SIZE
    fields = {}
    for field in header["fields"]:
        size = int(np.prod(field["shape"]))
        dtype = {"i32": "<i4", "f32": "<f4"}[field["dtype"]]
        fields[field["name"]] = np.frombuffer(wire, dtype=dtype, count=size, offset=offset)
        offset += size * 4
    u = fields["upper_body_position"]
    assert abs(u[2] - pitch) < 1e-7
    qref, dq, anchor = _stand_window()
    qref[:, ilidx] = u
    assert np.allclose(qref, mj_to_isaaclab(np.tile(q, (10, 1))))
    token = enc.encode_window(qref, dq, anchor)
    obs[0, :64] = token
    action = dec.run(None, {dec.get_inputs()[0].name: obs})[0].ravel()
    delta = (action - a0)[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
    rows.append(
        dict(
            command_rad=pitch,
            wire_waist_pitch_rad=float(u[2]),
            encoder_waist_pitch_rad=float(qref[0, 8]),
            token_changed_dims=int(np.count_nonzero(token != encoder_baseline)),
            one_step_fixed_observation_waist_target_delta_rad=float(delta[14]),
            largest_joint_target_delta_rad=float(abs(delta).max()),
        )
    )
assert rows[-1]["token_changed_dims"] > 0, "waist signal lost in encoder"
out = dict(
    scope="No physics, no sockets: real commander serialization, C++ joint-map constants, real CPU encoder/decoder sensitivity. Decoder is one-step at synthetic standing observation, NOT measured tracking.",
    rows=rows,
)
out["assets_sha256"] = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        deploy / "policy/release/model_encoder.onnx",
        deploy / "policy/release/model_decoder.onnx",
        deploy / "policy/release/observation_config.yaml",
        deploy / "src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp",
    ]
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
