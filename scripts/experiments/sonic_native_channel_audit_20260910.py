"""Per-channel audit: which reference joint is each motor command actually tracking?

A wrong joint MAPPING permutes channels in pairs, so it shows up as several
channels matching someone else's reference. A wrong SIGN shows up as exactly one
channel anti-correlated with its own reference. This separates the two without
assuming either. Also reports the observation chain (deploy view vs privileged
truth) so an action anomaly cannot be blamed on a bad observation.
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from locomani.backends.sonic.primitives import Primitives
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.g1_joints import isaac_to_mujoco

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("SONIC_NATIVE_OUT", ROOT / "outputs/diagnostics/sonic_native_phase3_20260910/retry_reset_02"))
REF = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/squat_001__A359"
HEADER = struct.Struct("<IdIIi")
NAMES = [
    "L_hip_p",
    "L_hip_r",
    "L_hip_y",
    "L_knee",
    "L_ank_p",
    "L_ank_r",
    "R_hip_p",
    "R_hip_r",
    "R_hip_y",
    "R_knee",
    "R_ank_p",
    "R_ank_r",
    "waist_y",
    "waist_r",
    "waist_p",
    "LS_p",
    "LS_r",
    "LS_y",
    "L_elb",
    "LW_r",
    "LW_p",
    "LW_y",
    "RS_p",
    "RS_r",
    "RS_y",
    "R_elb",
    "RW_r",
    "RW_p",
    "RW_y",
]


def pitch_of(q):
    w, x, y, z = q
    m20 = 2 * (x * z - w * y)
    return float(np.arcsin(np.clip(-m20, -1, 1)))


def read_capture(path):
    raw = path.read_bytes()
    off, stamps, inputs = 0, [], []
    while off + HEADER.size <= len(raw):
        magic, wall, ni, nt, _mode = HEADER.unpack_from(raw, off)
        if magic != 0x53454E43:
            raise ValueError(f"bad record at {off}")
        end = off + HEADER.size + 4 * (ni + nt)
        if end > len(raw):
            break
        inputs.append(np.frombuffer(raw, "<f4", ni + nt, off + HEADER.size)[:ni])
        stamps.append(wall)
        off = end
    return np.asarray(stamps), np.stack(inputs)


def corr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a.T @ b) / len(a) / np.outer(a.std(0) + 1e-9, b.std(0) + 1e-9)


def main():
    ref_i = np.loadtxt(REF / "joint_pos.csv", delimiter=",", skiprows=1)
    ref_m = np.stack([isaac_to_mujoco(ref_i[k]) for k in range(len(ref_i))])
    ik = ArmIK()
    torso_id = ik.model.body("torso_link").id

    cases = []
    for case in sorted(OUT.glob("native_release_*.json")):
        r = json.loads(case.read_text())
        cap = OUT / f"encoder_{r['repeat']}.bin"
        trace = [v for v in (r.get("trace") or []) if v.get("debug")]
        if not trace or not cap.exists():
            cases.append(dict(case=case.name, skipped="no trace or capture", error=r.get("error")))
            continue
        t_enc, x = read_capture(cap)
        frame = np.linalg.norm(x[:, 4:294].reshape(-1, 10, 29)[:, 0, None, :] - ref_i[None], axis=-1).argmin(1)
        t_act = np.array([v["wall"] for v in trace])
        sel = (t_enc >= r.get("play_wall", t_act[0])) & (t_enc <= t_act[-1])
        near = np.abs(t_act[None, :] - t_enc[sel][:, None]).argmin(1)
        R = ref_m[frame[sel]]
        C = np.array([trace[i]["state"]["cmd_q"] for i in near])
        A = np.array([trace[i]["state"]["joints"] for i in near])
        M = corr(R, C)

        channels = []
        for j in range(29):
            best = int(np.argmax(np.abs(M[:, j])))
            channels.append(
                dict(
                    channel=NAMES[j],
                    own_reference_corr=round(float(M[j, j]), 3),
                    best_match=NAMES[best],
                    best_corr=round(float(M[best, j]), 3),
                    ref_range_deg=round(float(np.degrees(R[:, j].ptp())), 1),
                    cmd_range_deg=round(float(np.degrees(C[:, j].ptp())), 1),
                    act_range_deg=round(float(np.degrees(A[:, j].ptp())), 1),
                )
            )
        inverted = [c["channel"] for c in channels if c["own_reference_corr"] < -0.5 and c["ref_range_deg"] > 5]
        misrouted = [
            c["channel"]
            for c in channels
            if c["best_match"] != c["channel"]
            and abs(c["best_corr"]) - abs(c["own_reference_corr"]) > 0.3
            and c["ref_range_deg"] > 5
        ]

        # observation chain: what the deploy saw vs privileged truth
        obs = []
        for v in trace:
            st, dbg = v["state"], v["debug"]
            ik.fk(np.array(st["joints"]), "right")
            mat = Primitives._rot(st["pelvis"]["quat"]) @ ik.data.xmat[torso_id].reshape(3, 3)
            obs.append(
                (
                    pitch_of(st["pelvis"]["quat"]) - pitch_of(dbg["base_quat"]),
                    float(np.arcsin(np.clip(-mat[2, 0], -1, 1))) - pitch_of(dbg["body_torso_quat"]),
                    st["joints"][14] - dbg["body_q"][14],
                )
            )
        o = np.abs(np.array(obs)).max(0)

        cases.append(
            dict(
                case=case.name,
                policy=r.get("policy", "policy/release"),
                complete=r.get("complete"),
                samples=int(sel.sum()),
                inverted_channels=inverted,
                misrouted_channels=misrouted,
                waist_pitch=dict(
                    own_reference_corr=round(float(M[14, 14]), 3),
                    ref_peak_deg=round(float(np.degrees(R[:, 14].max())), 2),
                    cmd_at_ref_peak_deg=round(float(np.degrees(C[R[:, 14].argmax(), 14])), 2),
                    act_at_ref_peak_deg=round(float(np.degrees(A[R[:, 14].argmax(), 14])), 2),
                ),
                observation_max_error=dict(
                    pelvis_pitch_deg=round(float(np.degrees(o[0])), 3),
                    torso_pitch_deg=round(float(np.degrees(o[1])), 3),
                    waist_joint_deg=round(float(np.degrees(o[2])), 3),
                ),
                channels=channels,
            )
        )

    report = dict(out=str(OUT), reference=str(REF), cases=cases)
    (OUT / "channel_audit.json").write_text(json.dumps(report, indent=2))
    for c in cases:
        if "skipped" in c:
            print(f"{c['case']}: skipped ({c['skipped']})")
            continue
        print(
            f"{c['case']} [{c['policy']}] inverted={c['inverted_channels']} misrouted={c['misrouted_channels']} "
            f"waist corr={c['waist_pitch']['own_reference_corr']} "
            f"ref {c['waist_pitch']['ref_peak_deg']}deg -> cmd {c['waist_pitch']['cmd_at_ref_peak_deg']}deg "
            f"-> act {c['waist_pitch']['act_at_ref_peak_deg']}deg  obs_err={c['observation_max_error']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
