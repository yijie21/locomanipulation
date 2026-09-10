"""Align a native CSV replay against the official reference and report tracking error.

Reads the private encoder capture (what the policy was actually asked for, with a
wall clock) plus the privileged state trace (what the robot actually did), and
recovers the reference frame index by nearest-neighbour matching each captured
10-frame reference block against the official joint_pos.csv. Nothing here assumes
the playback rate; the recovered frame index is the measurement.
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from locomani.primitives.g1_joints import isaac_to_mujoco

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("SONIC_NATIVE_OUT", ROOT / "outputs/diagnostics/sonic_native_phase3_20260910/retry_reset_02"))
REF = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/squat_001__A359"
HEADER = struct.Struct("<IdIIi")


def rot(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def pitch_of(q):
    """Forward pitch of a body: world-z component of its local x axis. Yaw invariant."""
    return float(np.arcsin(np.clip(-rot(q)[2, 0], -1, 1)))


def read_capture(path):
    raw = path.read_bytes()
    off = 0
    stamps, inputs, modes = [], [], []
    while off + HEADER.size <= len(raw):
        magic, wall, ni, nt, mode = HEADER.unpack_from(raw, off)
        if magic != 0x53454E43 or not 0 < ni < 10000:
            raise ValueError(f"bad record at {off}")
        end = off + HEADER.size + 4 * (ni + nt)
        if end > len(raw):
            break  # trailing partial record
        arr = np.frombuffer(raw, dtype="<f4", count=ni + nt, offset=off + HEADER.size)
        stamps.append(wall)
        inputs.append(arr[:ni])
        modes.append(mode)
        off = end
    return np.asarray(stamps), np.stack(inputs), np.asarray(modes), len(raw) - off


def main():
    ref_q = np.loadtxt(REF / "joint_pos.csv", delimiter=",", skiprows=1)
    ref_bq = np.loadtxt(REF / "body_quat.csv", delimiter=",", skiprows=1)
    ref_bp = np.loadtxt(REF / "body_pos.csv", delimiter=",", skiprows=1)
    ref_torso = np.array([pitch_of(ref_bq[i, 28:32]) for i in range(len(ref_bq))])
    ref_pelvis = np.array([pitch_of(ref_bq[i, 0:4]) for i in range(len(ref_bq))])

    cases = []
    for case in sorted(OUT.glob("native_release_*.json")):
        r = json.loads(case.read_text())
        trace = r.get("trace") or []
        row = dict(
            case=case.name,
            complete=r.get("complete"),
            error=r.get("error"),
            startup_protocol=r.get("startup_protocol"),
            trace_samples=len(trace),
            preplay_pelvis_z=None if not r.get("preplay") else r["preplay"]["pelvis"]["pos"][2],
        )
        cap = OUT / f"encoder_{r['repeat']}.bin"
        if not trace or not cap.exists():
            row["skipped"] = "no trace or no capture"
            cases.append(row)
            continue
        t_enc, x, modes, tail = read_capture(cap)
        # 4:294 is the 10x29 joint reference block in ISAACLAB order (phase2 verified).
        blocks = x[:, 4:294].reshape(-1, 10, 29)
        # Recover the played frame index without assuming a playback rate.
        d = np.linalg.norm(blocks[:, :, None, :] - ref_q[None, None, :, :], axis=-1)
        frame_idx = d.argmin(axis=2)  # (records, 10)
        frame_err = d.min(axis=2)
        slot0 = frame_idx[:, 0]

        t_act = np.array([v["wall"] for v in trace])
        act_torso = np.array([v["torso_pitch"] for v in trace])
        act_waist = np.array([v["state"]["joints"][14] for v in trace])
        act_z = np.array([v["state"]["pelvis"]["pos"][2] for v in trace])
        act_pelvis_pitch = np.array([pitch_of(v["state"]["pelvis"]["quat"]) for v in trace])
        motor_waist = np.array(
            [v["debug"]["last_action"][14] if v.get("debug") and "last_action" in v["debug"] else np.nan for v in trace]
        )

        play_wall = r.get("play_wall", t_act[0])
        sel = (t_enc >= play_wall) & (t_enc <= t_act[-1])
        if sel.sum() < 10:
            row["skipped"] = f"only {int(sel.sum())} encoder records inside the play window"
            cases.append(row)
            continue
        te, fr, fe = t_enc[sel], slot0[sel], frame_err[sel]
        near = np.abs(t_act[None, :] - te[:, None]).argmin(axis=1)
        lag = np.abs(t_act[near] - te)

        played = int(fr.max())
        # Playback rate over the ascending span only: the cursor returns to frame 0
        # once the motion ends, so first-to-last would read as zero progression.
        top = int(np.argmax(fr == fr.max()))
        span = te[top] - te[0]
        rate = float((fr[top] - fr[0]) / span) if span > 0 else float("nan")
        peak = int(np.argmax(ref_torso))  # deepest torso lean in the reference
        at_peak = np.abs(fr - peak) <= 2
        row.update(
            encoder_records=int(len(t_enc)),
            trailing_bytes=int(tail),
            modes=np.unique(modes[sel]).tolist(),
            reference_match_max_err_rad=float(fe.max()),
            first_frame=int(fr[0]),
            last_frame=int(fr[-1]),
            frame_at_end_of_ascent=int(fr[top]),
            seconds_to_play_out=float(span),
            frames_played=played,
            reference_total_frames=int(len(ref_q)),
            play_rate_frames_per_s=rate,
            state_alignment_max_lag_s=float(lag.max()),
            slot_order=np.median(frame_idx[sel] - fr[:, None], axis=0).tolist(),
            ref_torso_peak_deg=float(np.degrees(ref_torso[peak])),
            ref_torso_peak_frame=peak,
            reached_peak_frame=bool(at_peak.any()),
            actual_torso_at_peak_deg=float(np.degrees(act_torso[near][at_peak].mean())) if at_peak.any() else None,
            actual_waist_at_peak_deg=float(np.degrees(act_waist[near][at_peak].mean())) if at_peak.any() else None,
            ref_waist_at_peak_deg=float(np.degrees(ref_q[peak, 8])),
            motor_target_waist_at_peak_deg=float(np.degrees(np.nanmean(motor_waist[near][at_peak])))
            if at_peak.any()
            else None,
            max_actual_torso_deg=float(np.degrees(act_torso.max())),
            max_actual_waist_deg=float(np.degrees(act_waist.max())),
            min_actual_pelvis_z=float(act_z.min()),
            ref_pelvis_z_over_played=float(ref_bp[fr, 2].min()),
            torso_rmse_deg=float(np.degrees(np.sqrt(np.mean((act_torso[near] - ref_torso[fr]) ** 2)))),
            torso_mean_deficit_deg=float(np.degrees(np.mean(ref_torso[fr] - act_torso[near]))),
            waist_rmse_deg=float(np.degrees(np.sqrt(np.mean((act_waist[near] - ref_q[fr, 8]) ** 2)))),
            pelvis_pitch_rmse_deg=float(np.degrees(np.sqrt(np.mean((act_pelvis_pitch[near] - ref_pelvis[fr]) ** 2)))),
            pelvis_z_rmse_m=float(np.sqrt(np.mean((act_z[near] - ref_bp[fr, 2]) ** 2))),
        )
        # coarse per-decile curve so the shape of the failure is visible, not just a scalar
        curve = []
        for lo in range(0, 424, 40):
            m = (fr >= lo) & (fr < lo + 40)
            if m.any():
                curve.append(
                    dict(
                        frames=[lo, lo + 40],
                        ref_torso_deg=float(np.degrees(ref_torso[fr[m]].mean())),
                        act_torso_deg=float(np.degrees(act_torso[near][m].mean())),
                        ref_waist_deg=float(np.degrees(ref_q[fr[m], 8].mean())),
                        act_waist_deg=float(np.degrees(act_waist[near][m].mean())),
                        ref_pelvis_z=float(ref_bp[fr[m], 2].mean()),
                        act_pelvis_z=float(act_z[near][m].mean()),
                    )
                )
        row["curve"] = curve
        cases.append(row)

    report = dict(
        out=str(OUT),
        reference=str(REF),
        scope="frame index recovered by nearest-neighbour match of the captured reference block "
        "against joint_pos.csv; actual state aligned to each encoder record by wall clock",
        cases=cases,
    )
    (OUT / "native_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
