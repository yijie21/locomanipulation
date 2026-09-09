"""Run SONIC's motion encoder offline: a 29-DoF reference motion -> 64-dim token.

    python -m teleop_free.tools.sonic_encoder --selfcheck [--dataset DS]

WHY THIS EXISTS

The scripted agent never emits a latent. It emits INTENT (5 planner scalars +
`upper_body_position[17]` + the gripper); the k-planner renders a 29-DoF
reference motion from that, the C++ deploy splices our 17 upper-body joints
into it verbatim, and SONIC's encoder turns the result into the 64-dim token
that IS the dataset's action label. So "author nicer motion, then convert it to
a SONIC latent" does not need a new mechanism -- the encoder is that converter.
What has been missing is the ability to run it WITHOUT the stack, which is what
this module adds: any 29-DoF trajectory, from any source (a trajectory
optimiser, a min-jerk spline, retargeted mocap), becomes a token sequence here,
offline, in-process.

THE INPUT LAYOUT -- read this before changing anything

`policy/release/model_encoder.onnx` takes ONE flat tensor `obs_dict` of 1762
floats and returns `encoded_tokens` (64). The 1762 is the concatenation of
every entry under `encoder.encoder_observations` in
`policy/release/observation_config.yaml`, IN THAT ORDER, at increasing offsets
(g1_deploy_onnx_ref.cpp:1852). Each block's width comes from the observation
registry (cpp:1705). The widths sum to exactly 1762, which is how the offsets
below were checked.

Only the blocks the ACTIVE MODE requires are filled; everything else stays zero
(cpp:2012 zero-fills, cpp:2083 fills only `required_observations`). Mode 0
("g1", the one this pipeline runs) requires four of them -- 644 live numbers:

    encoder_mode_4                          4    @    0   [mode_id, 0, 0, 0]
    motion_joint_positions_10frame_step5  290    @    4
    motion_joint_velocities_10frame_step5 290    @  294
    motion_anchor_orientation_10frame_step5 60   @  601

THE THREE THINGS THAT ARE EASY TO GET WRONG

  1. FRAME-MAJOR, AND THE FRAMES ARE IN THE FUTURE. cpp:757 writes
     `frame_offset = offset + frame_idx * 29`, so the 290 numbers are
     [f0j0..f0j28, f1j0..f1j28, ...] -- frame-major, not joint-major. And
     cpp:763 computes `target_frame = current_frame_ + frame_idx * step_size`,
     clamped to the last frame. At 50 Hz with step 5 that is a 0.9 s LOOKAHEAD
     (t, t+0.1 s, ..., t+0.9 s), not a history.

  2. ISAACLAB ORDER, RAW RADIANS. The gather copies `JointPositions(frame)`
     straight through -- no subtraction of DEFAULT_ANGLES, no scaling. The
     motion's joint array is in IsaacLab order (the upper-body splice indexes
     it with `upper_body_joint_isaaclab_order_in_isaaclab_index`). Use
     `mj_to_isaaclab()` on anything that came out of the MuJoCo-ordered side.

  3. THE ANCHOR BLOCK IS NOT PURE INTENT. cpp:632 reads the MEASURED base
     quaternion from the state logger and emits, per frame, the first two
     columns of the rotation matrix of `conj(base_quat) * (delta_heading *
     motion_root_quat[f])`, flattened row-wise. So 60 of the 644 live numbers
     depend on where the robot actually is. docs/collection.md's gloss that the
     token is "pure intent, carrying no proprioceptive state" holds for the two
     joint blocks and NOT for this one.

  Also: when the upper body is being streamed, the splice writes the SAME 17
  buffered values into all 10 lookahead frames (cpp:781) -- the arm is frozen
  across the window. Arm motion reaches the token only through the sequence of
  per-tick tokens, never through the shape of one window.

WHAT IS VERIFIED HERE, AND WHAT IS NOT

`--selfcheck` verifies the structure (offsets vs the ONNX input dim), the
codebook (the graph's own FSQ, so outputs land on the 1/16 grid and match the
grid of recorded `action.motion_token`), and the wiring end to end (perturbing
one joint in the reference moves that joint in SONIC's decoded command).

It does NOT verify token-for-token agreement with the running deploy, because
the reference motion is not recoverable from the dataset: the exporter records
the policy's OUTPUT (`action.wbc`) and the measured state, never
`current_motion_`. The live `g1_debug` stream does publish `body_q_target`
(the reference frame, MuJoCo order, output_interface.hpp:192) and
`base_quat_target`, which with a 45-tick lookahead window would give the
positions and the anchor exactly -- but NOT `JointVelocities`, which is
published nowhere. Closing that gap is a one-line addition of a
`body_dq_target` key to the ZMQ output handler plus a rebuild; until then,
treat this module as spec-faithful but not bit-verified.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# The buffer layout, in observation_config.yaml order. Widths from the
# registry at g1_deploy_onnx_ref.cpp:1705. The sum is asserted against the
# ONNX input dimension at construction time.
# ---------------------------------------------------------------------------
BLOCKS = [
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
]
OFFSETS, _o = {}, 0
for _n, _d in BLOCKS:
    OFFSETS[_n] = (_o, _d)
    _o += _d
OBS_DIM = _o  # 1762

MODE_REQUIRED = {
    0: (
        "encoder_mode_4",
        "motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5",
        "motion_anchor_orientation_10frame_step5",
    ),
}

N_JOINTS, N_FRAMES, STEP = 29, 10, 5
TOKEN_DIM = 64
FSQ_STEP = 1.0 / 16.0
CONTROL_HZ = 50.0

# g1_deploy_onnx_ref policy_parameters.hpp:80
UPPER_BODY_ISAACLAB = np.array([2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])

# Both arrays are named BACKWARDS relative to their contents -- sonic_decoder.py
# documents this and the deploy's own usage is the definition:
#     ISAACLAB_TO_MUJOCO[mujoco_index] -> isaaclab index
#     MUJOCO_TO_ISAACLAB[isaaclab_index] -> mujoco index
# so converting a MuJoCo-ordered vector INTO IsaacLab order means indexing with
# MUJOCO_TO_ISAACLAB, not with the array whose name reads that way. Getting it
# backwards is silent -- the graph still runs and still returns 64 numbers; it
# showed up here only because the decoder then answered a 20 deg elbow change
# by moving the knee.
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
assert np.array_equal(ISAACLAB_TO_MUJOCO[MUJOCO_TO_ISAACLAB], np.arange(29))

DEFAULT_ANGLES_MJ = np.array(
    [
        -0.312,
        0,
        0,
        0.669,
        -0.363,
        0,
        -0.312,
        0,
        0,
        0.669,
        -0.363,
        0,
        0,
        0,
        0,
        0.2,
        0.2,
        0,
        0.6,
        0,
        0,
        0,
        0.2,
        -0.2,
        0,
        0.6,
        0,
        0,
        0,
    ]
)

_DEFAULT_ENCODER = os.path.join(
    os.environ.get("SONIC_DEPLOY_ROOT", "third_party/GR00T-WholeBodyControl/gear_sonic_deploy"),
    "policy/release/model_encoder.onnx",
)


def mj_to_isaaclab(q_mj: np.ndarray) -> np.ndarray:
    """Reorder a MuJoCo-ordered 29-vector (or ...x29 stack) to IsaacLab order."""
    return np.asarray(q_mj)[..., MUJOCO_TO_ISAACLAB]


def isaaclab_to_mj(q_il: np.ndarray) -> np.ndarray:
    return np.asarray(q_il)[..., ISAACLAB_TO_MUJOCO]


# --- quaternion helpers, wxyz throughout (audit item C9) --------------------
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def anchor_6d(base_quat_meas, ref_root_quat, delta_heading=None) -> np.ndarray:
    """The 6 numbers of one `motion_anchor_orientation` frame.

    Mirrors GatherMotionAnchorOrientationMutiFrame, orientation_mode 0
    (g1_deploy_onnx_ref.cpp:613): the first two COLUMNS of the rotation matrix
    of `conj(base_quat) * (delta_heading * motion_root_quat)`, flattened
    row-wise -- [R00, R01, R10, R11, R20, R21].
    """
    ref = np.asarray(ref_root_quat, dtype=float)
    if delta_heading is not None:
        ref = quat_mul(np.asarray(delta_heading, dtype=float), ref)
    rel = quat_mul(quat_conj(np.asarray(base_quat_meas, dtype=float)), ref)
    R = quat_to_mat(rel)
    return np.array([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]])


class SonicEncoder:
    """The deploy's motion encoder, in-process."""

    def __init__(self, onnx_path: str | None = None):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is required; install it into the venv that runs the stack (.venv_sim)"
            ) from e
        path = onnx_path or _DEFAULT_ENCODER
        if not os.path.exists(path):
            raise FileNotFoundError(f"encoder ONNX not found at {path}; pass onnx_path or set SONIC_DEPLOY_ROOT")
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        dim = self.sess.get_inputs()[0].shape[-1]
        if dim != OBS_DIM:
            raise RuntimeError(
                f"this encoder wants {dim} inputs but the block table sums to "
                f"{OBS_DIM}. The deploy is running a different "
                f"observation_config.yaml -- rebuild the table from the one it "
                f"actually loads before trusting anything here."
            )

    # -- low level ----------------------------------------------------------
    def pack(self, q_win_il, dq_win_il, anchor_win, mode: int = 0) -> np.ndarray:
        """Build the 1762-vector from the three live blocks.

        q_win_il / dq_win_il : (10, 29) IsaacLab order, raw radians and rad/s,
                               frame 0 = now, frame k = now + k*STEP ticks.
        anchor_win           : (10, 6) from anchor_6d().
        """
        q = np.asarray(q_win_il, dtype=np.float64)
        dq = np.asarray(dq_win_il, dtype=np.float64)
        a = np.asarray(anchor_win, dtype=np.float64)
        for name, arr, shape in (
            ("positions", q, (N_FRAMES, N_JOINTS)),
            ("velocities", dq, (N_FRAMES, N_JOINTS)),
            ("anchor", a, (N_FRAMES, 6)),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {arr.shape}")
        buf = np.zeros(OBS_DIM, dtype=np.float64)
        o, _ = OFFSETS["encoder_mode_4"]
        buf[o] = float(mode)  # [mode, 0, 0, 0]
        o, d = OFFSETS["motion_joint_positions_10frame_step5"]
        buf[o : o + d] = q.reshape(-1)  # frame-major
        o, d = OFFSETS["motion_joint_velocities_10frame_step5"]
        buf[o : o + d] = dq.reshape(-1)
        o, d = OFFSETS["motion_anchor_orientation_10frame_step5"]
        buf[o : o + d] = a.reshape(-1)
        return buf

    def encode_window(self, q_win_il, dq_win_il, anchor_win, mode: int = 0) -> np.ndarray:
        buf = self.pack(q_win_il, dq_win_il, anchor_win, mode)
        out = self.sess.run(None, {self.in_name: buf[None, :].astype(np.float32)})[0]
        return out.reshape(-1)

    # -- trajectory ---------------------------------------------------------
    def encode_trajectory(
        self, q_mj, base_quat, ref_root_quat, dq_mj=None, hz: float = CONTROL_HZ, mode: int = 0, upper_body_mj=None
    ) -> np.ndarray:
        """Token per tick for a whole reference trajectory.

        q_mj          : (T, 29) reference joints, MuJoCo order, raw radians.
        base_quat     : (T, 4) MEASURED base quaternion, wxyz. Pass the
                        reference root quaternion here to get the "robot is
                        exactly on the reference" case.
        ref_root_quat : (T, 4) reference root quaternion, wxyz.
        dq_mj         : (T, 29) reference velocities. Defaults to a finite
                        difference of q_mj at `hz` -- an APPROXIMATION: the
                        deploy reads velocities stored in the motion, which
                        are not necessarily the difference of its positions.
        upper_body_mj : optional (T, 17) upper-body stream, MuJoCo indices
                        12..28, applied the way the deploy applies it -- the
                        value at tick t written into ALL 10 lookahead frames.

        Returns (T, 64).
        """
        q_mj = np.asarray(q_mj, dtype=float)
        T = len(q_mj)
        if dq_mj is None:
            dq_mj = np.gradient(q_mj, 1.0 / hz, axis=0)
        q_il = mj_to_isaaclab(q_mj)
        dq_il = mj_to_isaaclab(np.asarray(dq_mj, dtype=float))
        base_quat = np.asarray(base_quat, dtype=float)
        ref_root_quat = np.asarray(ref_root_quat, dtype=float)

        toks = np.zeros((T, TOKEN_DIM))
        for t in range(T):
            idx = np.minimum(t + np.arange(N_FRAMES) * STEP, T - 1)
            qw, dqw = q_il[idx].copy(), dq_il[idx].copy()
            if upper_body_mj is not None:
                ub = np.asarray(upper_body_mj, dtype=float)[t]
                # the deploy freezes the streamed upper body across the window
                qw[:, UPPER_BODY_ISAACLAB] = ub[None, :]
                dqw[:, UPPER_BODY_ISAACLAB] = 0.0
            aw = np.stack([anchor_6d(base_quat[t], ref_root_quat[i]) for i in idx])
            toks[t] = self.encode_window(qw, dqw, aw, mode)
        return toks


# ---------------------------------------------------------------------------
def _stand_window(offset_mj=None):
    q = np.tile(DEFAULT_ANGLES_MJ, (N_FRAMES, 1))
    if offset_mj is not None:
        q = q + offset_mj
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    return (mj_to_isaaclab(q), np.zeros((N_FRAMES, N_JOINTS)), np.stack([anchor_6d(ident, ident)] * N_FRAMES))


def selfcheck(enc: SonicEncoder, dataset: str | None) -> int:
    bad = 0

    def ok(label, cond, detail=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not cond:
            bad += 1

    print("1. structure")
    ok("block widths sum to the ONNX input dim", OBS_DIM == 1762, f"{OBS_DIM}")
    live = sum(OFFSETS[n][1] for n in MODE_REQUIRED[0])
    ok("mode 0 fills 644 of them", live == 644, f"{live}")
    q, dq, a = _stand_window()
    buf = enc.pack(q, dq, a, 0)
    ok(
        "everything outside the mode-0 blocks stays zero",
        np.count_nonzero(buf) <= live,
        f"{np.count_nonzero(buf)} non-zero of {live} live slots",
    )

    print("2. codebook")
    tok = enc.encode_window(q, dq, a, 0)
    idx = tok / FSQ_STEP
    ok(
        "output lands on the 1/16 grid (the FSQ is inside the graph)",
        np.allclose(idx, np.round(idx), atol=1e-4),
        f"max off-grid {np.abs(idx - np.round(idx)).max():.2e}",
    )
    ok(
        "indices inside the 32-level range [-16, 15]",
        idx.min() >= -16.5 and idx.max() <= 15.5,
        f"[{np.round(idx.min())}, {np.round(idx.max())}]",
    )
    ok("deterministic", np.array_equal(tok, enc.encode_window(q, dq, a, 0)))

    print("3. sensitivity")
    off = np.zeros(N_JOINTS)
    off[25] = np.deg2rad(20.0)  # right elbow, MuJoCo order
    q2, dq2, a2 = _stand_window(off)
    tok2 = enc.encode_window(q2, dq2, a2, 0)
    nd = int((np.round(tok2 / FSQ_STEP) != np.round(tok / FSQ_STEP)).sum())
    ok("a 20 deg elbow change moves the token", nd > 0, f"{nd}/64 dims differ")
    off_s = np.zeros(N_JOINTS)
    off_s[25] = np.deg2rad(0.5)
    tok3 = enc.encode_window(*_stand_window(off_s), mode=0)
    ns = int((np.round(tok3 / FSQ_STEP) != np.round(tok / FSQ_STEP)).sum())
    print(f"       (for scale: 0.5 deg moves {ns}/64 dims -- below this the label cannot represent the change)")

    print("4. against recorded labels")
    if not dataset:
        print("       skipped (pass --dataset to run it)")
    else:
        import pyarrow.parquet as pq

        p = os.path.join(dataset, "data", "chunk-000", "episode_000000.parquet")
        rec = np.stack(pq.read_table(p)["action.motion_token"].to_numpy(zero_copy_only=False))
        ridx = rec / FSQ_STEP
        ok("recorded tokens sit on the same grid", np.allclose(ridx, np.round(ridx), atol=1e-4))
        ok(
            "recorded tokens sit in the same range",
            rec.min() >= -1.0 - 1e-6 and rec.max() <= 1.0 - FSQ_STEP + 1e-6,
            f"[{rec.min():.4f}, {rec.max():.4f}]",
        )
        ok(
            "our synthetic token is in the recorded per-dim range",
            bool(((tok >= rec.min(0) - 4 * FSQ_STEP) & (tok <= rec.max(0) + 4 * FSQ_STEP)).mean() > 0.9),
            f"{((tok >= rec.min(0) - 4 * FSQ_STEP) & (tok <= rec.max(0) + 4 * FSQ_STEP)).mean():.0%} of dims",
        )

    print("5. end-to-end through the decoder")
    if not dataset:
        print("       skipped (needs --dataset for a realistic observation)")
    else:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from sonic_decoder import SonicDecoder

            dec = SonicDecoder()
        except Exception as e:
            print(f"       skipped: {e}")
        else:
            # The decoder reads 930 numbers of proprioception besides the
            # token. Feeding it zeros there is far off-distribution and its
            # response to a token swap means nothing, so the observation comes
            # from a real episode and only the token is swapped.
            import pyarrow.parquet as pq

            t = pq.read_table(os.path.join(dataset, "data", "chunk-000", "episode_000000.parquet"))
            g = lambda k: np.stack(t[k].to_numpy(zero_copy_only=False))
            pre = SonicDecoder.prepare(
                g("observation.state"),
                g("action.wbc"),
                g("observation.root_orientation"),
                g("observation.projected_gravity"),
            )
            frames = list(range(60, len(pre["q"]) - 60, 20))
            n = len(frames)
            d = dec.joint_delta_deg(pre, frames, np.tile(tok, (n, 1)), np.tile(tok2, (n, 1)))
            d = np.abs(np.asarray(d)).reshape(n, -1).mean(0)
            # NOT "joint 25 must rank first": SONIC is a whole-body tracker,
            # so lifting the elbow 20 deg moves the palm and the shoulder
            # answers too -- the honest claim is that the response stays in
            # the arm that was perturbed.
            groups = {"r_arm": range(22, 29), "l_arm": range(15, 22), "waist": range(12, 15), "legs": range(0, 12)}
            gm = {k: float(d[list(v)].mean()) for k, v in groups.items()}
            top2 = set(np.argsort(-d)[:2].tolist())
            ok(
                "the decoder's response stays in the perturbed arm",
                top2 <= set(range(22, 29)) and gm["r_arm"] == max(gm.values()),
                "per group deg: " + ", ".join(f"{k} {v:.1f}" for k, v in gm.items()) + f"; top-2 joints {sorted(top2)}",
            )

    print(f"\n{'ALL CHECKS PASSED' if bad == 0 else f'{bad} CHECK(S) FAILED'}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument(
        "--dataset", default=None, help="a LeRobot dataset root, to compare against its recorded action.motion_token"
    )
    args = ap.parse_args()
    enc = SonicEncoder(args.onnx)
    if args.selfcheck:
        sys.exit(1 if selfcheck(enc, args.dataset) else 0)
    ap.error("nothing to do -- pass --selfcheck, or import SonicEncoder")


if __name__ == "__main__":
    main()
