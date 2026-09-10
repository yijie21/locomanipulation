"""Read PRIVATE encoder capture and align steady holds to requested references."""

import json
import struct
from pathlib import Path

import numpy as np

from locomani.primitives.g1_joints import mujoco_to_isaac

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/diagnostics/sonic_lean_phase2_20260910"
raw = (OUT / "encoder.bin").read_bytes()
header = struct.Struct("<IdIIi")
offset = 0
stamps = []
inputs = []
tokens = []
modes = []
while offset + header.size <= len(raw):
    magic, wall, ni, nt, mode = header.unpack_from(raw, offset)
    assert magic == 0x53454E43 and 0 < ni < 10000 and nt == 64, "invalid record format"
    end = offset + header.size + 4 * (ni + nt)
    if end > len(raw):
        break
    arr = np.frombuffer(raw, dtype="<f4", count=ni + nt, offset=offset + header.size)
    stamps.append(wall)
    inputs.append(arr[:ni])
    tokens.append(arr[ni:])
    modes.append(mode)
    offset = end
assert inputs, "no captured inference inputs"
x = np.stack(inputs)
t = np.asarray(stamps)
rows = []
for case in sorted(OUT.glob("*_[123].json")):
    r = json.loads(case.read_text())
    hold = [v for v in r["trace"] if v["phase"] == "hold"]
    if not hold:
        continue
    end = hold[-1]["wall"]
    steady = [v for v in hold if v["wall"] >= end - 1]
    selected = (t >= end - 1) & (t <= end)
    if not selected.any():
        continue
    actual = x[selected, 4:294].reshape(-1, 10, 29)
    expected = mujoco_to_isaac(r["requested"])
    d = [v["debug"] for v in steady if v["debug"] is not None]
    rows.append(
        dict(
            case=case.name,
            complete=r["complete"],
            encoder_samples=int(selected.sum()),
            mode_ids=np.unique(np.array(modes)[selected]).tolist(),
            reference_max_joint_error_rad=float(np.max(np.abs(actual - expected))),
            reference_waist_pitch_rad=float(actual[:, :, 8].mean()),
            requested_waist_pitch_rad=r["requested"][14],
            encoder_dq_max=float(np.max(np.abs(x[selected, 294:584]))),
            desired_torso_deg=r["desired_torso_deg"],
            actual_torso_deg=float(np.degrees(np.mean([v["torso_pitch"] for v in steady]))),
            actual_waist_deg=float(np.degrees(np.mean([v["joints"][14] for v in steady]))),
            motor_target_waist_deg=float(np.degrees(np.mean([v["last_action"][14] for v in d]))) if d else None,
            min_pelvis_z=float(min(v["pelvis"]["pos"][2] for v in steady)),
            max_observation_age_s=max(v["observation_age"] for v in steady),
        )
    )
report = dict(
    records=len(t),
    input_dim=x.shape[1],
    trailing_bytes=len(raw) - offset,
    rows=rows,
    scope="actual post-gather float buffer used by private deploy encoder; steady hold alignment by wall time",
)
(OUT / "encoder_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
