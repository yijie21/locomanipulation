"""Translate token-space error into joint-space degrees, using SONIC's decoder.

WHY THIS EXISTS

`action.motion_token` is a learned FSQ latent (see sonic_token.py). Euclidean
distance in that space has no guaranteed relation to how differently the robot
moves: an MSE of 0.01 could be a millimetre or a metre. Every open-loop number
reported in that space is uninterpretable on its own -- and we already know the
stronger version of this, that replaying the GROUND TRUTH token open loop still
drifts 0.77 m (diag_token_replay.py), so token accuracy and task success are
only loosely coupled.

The decoder is what gives token error a unit. The ACTIVE checkpoint's
model_decoder.onnx (locomani.paths.SONIC_POLICY, default policy/sonic_v1_1 since
2026-09-11) maps obs_dict[994] -> action[29], and the token is 64 of those 994
inputs. Which checkpoint MATTERS: a token is a checkpoint-private latent, so
tokens recorded before the switch (every dataset collected under policy/release)
must be decoded with `SonicDecoder(<release path>)`, not with the default. Hold
the observation fixed, swap the ground-truth token for the predicted one, and
the difference between the two joint commands is "how many degrees does this
prediction move the robot". That is comparable with numbers this project
already has: 0.03 cm IK residual, 0.2-0.9 deg token-replay reproduction.

TWO JOINT ORDERS AND A SCALE -- THE PART THAT IS EASY TO GET WRONG

The decoder does NOT emit joint angles, and it does NOT read them. Three
transforms sit between the parquet and the graph, all read out of the deploy:

  observation  (g1_deploy_onnx_ref.cpp:2840)
      body_q[i] = q_hw[MUJOCO_TO_ISAACLAB[i]] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB[i]]
      body_dq[i] = dq_hw[MUJOCO_TO_ISAACLAB[i]]
    i.e. IsaacLab order, and positions are OFFSETS FROM THE STANDING POSE.
    GatherHisBodyJointPositions copies this straight into the buffer, so there
    is no further scaling anywhere.

  action       (g1_deploy_onnx_ref.cpp:3133)
      q_target[i] = DEFAULT_ANGLES[i] + out[ISAACLAB_TO_MUJOCO[i]] * ACTION_SCALE[i]
    i.e. the output is a per-joint-SCALED RESIDUAL in IsaacLab order.

  his_last_actions is the policy's own previous raw output, so it has to be
  inverted back out of the recorded joint target:
      raw[i] = ((wbc[i] - DEFAULT_ANGLES[i]) / ACTION_SCALE[i])  then permuted

Getting these wrong is silent: the graph still runs and still returns 29
numbers. It cost a first version of this file, which fed raw MuJoCo-order
angles and correlated 0.20 with the recorded command.

Both arrays are named backwards relative to their contents; the deploy's own
usage is the definition, and it is unambiguous:
    MUJOCO_TO_ISAACLAB[isaaclab_index] -> mujoco index
    ISAACLAB_TO_MUJOCO[mujoco_index]   -> isaaclab index
DEFAULT_ANGLES and ACTION_SCALE are indexed in MuJoCo order, which is the same
order as observation.state's body block (left_leg 6, right_leg 6, waist 3,
left_arm 7, right_arm 7) -- verified against the data: the leg entries track
DEFAULT_ANGLES joint for joint (ankle pitch -0.354 measured vs -0.363 default).

THE 994-DIM LAYOUT

From the checkpoint's observation_config.yaml -- the order of `observations:` IS
the buffer order, the deploy fills at increasing offsets -- confirmed against
the ONNX input shape. This half is the one thing release and v1.1 share
BYTE FOR BYTE: same six blocks, same order, same 994 (only the ENCODER side
differs, sonic_encoder.py), so this table needs no per-checkpoint variant:

    token_state                              64
    his_base_angular_velocity_10frame_step1  30   = 10 frames x 3
    his_body_joint_positions_10frame_step1  290   = 10 frames x 29
    his_body_joint_velocities_10frame_step1 290
    his_last_actions_10frame_step1          290
    his_gravity_dir_10frame_step1            30
                                            ---
                                            994

Each history block is FRAME-MAJOR and OLDEST-FIRST: GatherHisBodyJointPositions
(cpp:1451) writes `offset + f * joints` over GetLatest(..., newest_first=false).
The yaml's header comment says 436 and "4frame"; it is stale -- the registered
function for the enabled name is the 10-frame one (cpp:1791), and 994 is what
the graph declares.

TWO CHANNELS ARE RECONSTRUCTED, NOT RECORDED

The exporter logs joint POSITIONS and orientations, not velocities, so
his_body_joint_velocities and his_base_angular_velocity are finite-differenced
here at the source 50 Hz.

That approximation would matter if we wanted the decoder's absolute output. We
do not: every use compares decoder(obs, token_a) against decoder(obs, token_b)
on the SAME reconstructed obs, so the approximation enters both terms
identically and largely cancels. What is measured is the decoder's sensitivity
to the token. Absolute joint targets from this module are not a validated
quantity; joint DIFFERENCES are what it is for.
"""

from __future__ import annotations

import os

import numpy as np

from locomani import paths

ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)

MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)

ACTION_SCALE = np.array(
    [
        0.35066146637882428,
        0.35066146637882428,
        0.54754646521423045,
        0.35066146637882428,
        0.43857731392336724,
        0.43857731392336724,
        0.35066146637882428,
        0.35066146637882428,
        0.54754646521423045,
        0.35066146637882428,
        0.43857731392336724,
        0.43857731392336724,
        0.54754646521423045,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.074500870329507141,
        0.074500870329507141,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.074500870329507141,
        0.074500870329507141,
    ]
)

DEFAULT_ANGLES = np.array(
    [
        -0.312,
        0,
        0,
        0.66900000000000004,
        -0.36299999999999999,
        0,
        -0.312,
        0,
        0,
        0.66900000000000004,
        -0.36299999999999999,
        0,
        0,
        0,
        0,
        0.20000000000000001,
        0.20000000000000001,
        0,
        0.59999999999999998,
        0,
        0,
        0,
        0.20000000000000001,
        -0.20000000000000001,
        0,
        0.59999999999999998,
        0,
        0,
        0,
    ]
)

# Body-joint indices inside the exporter's 43-wide configuration vector: the
# 29 actuated body joints, dropping the two 7-wide hand blocks.
BODY_IDX = np.array(list(range(0, 22)) + list(range(29, 36)))
N_BODY = 29
HIST = 10  # frames, step 1
SRC_HZ = 50.0  # the exporter's rate, and the deploy's control rate
OBS_DIM = 994

_DEFAULT_DECODER = paths.sonic_policy_file("model_decoder.onnx")


def _quat_ang_vel(q, dt):
    """Body-frame angular velocity from a wxyz quaternion track.

    omega ~= 2 * vec(conj(q[t-1]) * q[t]) / dt -- the rotation between
    consecutive samples expressed in the moving frame, which is what an IMU
    gyro reports. The first sample repeats the second so the array keeps its
    length, matching how a history buffer looks at start-up.
    """
    q = np.asarray(q, dtype=np.float64)
    w0, v0 = q[:-1, 0], q[:-1, 1:]
    w1, v1 = q[1:, 0], q[1:, 1:]
    vec = w0[:, None] * v1 - w1[:, None] * v0 - np.cross(v0, v1)
    omega = 2.0 * vec / dt
    return np.vstack([omega[:1], omega])


class SonicDecoder:
    """The SONIC decoder as a callable, plus the obs assembly it needs."""

    def __init__(self, onnx_path: str | None = None, *, num_threads: int | None = None):
        import onnxruntime as ort

        path = onnx_path or _DEFAULT_DECODER
        if not os.path.exists(path):
            raise FileNotFoundError(f"decoder ONNX not found at {path}; pass onnx_path or set SONIC_POLICY")
        options = ort.SessionOptions()
        if num_threads is not None:
            if num_threads < 1:
                raise ValueError("num_threads must be positive")
            options.intra_op_num_threads = options.inter_op_num_threads = num_threads
        self.sess = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        dim = self.sess.get_inputs()[0].shape[-1]
        if dim != OBS_DIM:
            raise ValueError(f"decoder expects obs {dim}, this module builds {OBS_DIM}")

    # -- observation assembly -------------------------------------------------

    @staticmethod
    def prepare(state43, action_wbc43, root_quat_wxyz, gravity3, hz: float = SRC_HZ):
        """Per-episode precompute: everything in the obs except the token.

        Args:
            state43:        [T, 43] observation.state   (raw angles, MuJoCo order)
            action_wbc43:   [T, 43] action.wbc          (joint targets, MuJoCo order)
            root_quat_wxyz: [T, 4]  observation.root_orientation
            gravity3:       [T, 3]  observation.projected_gravity

        Returns [T, ...] arrays already in the decoder's own conventions.
        """
        dt = 1.0 / hz
        q_mj = np.asarray(state43, dtype=np.float64)[:, BODY_IDX]
        wbc_mj = np.asarray(action_wbc43, dtype=np.float64)[:, BODY_IDX]
        # Backward difference, first sample zero -- a live buffer has no
        # velocity estimate on its first tick either.
        dq_mj = np.vstack([np.zeros((1, N_BODY)), np.diff(q_mj, axis=0)]) / dt
        return {
            # offsets from the standing pose, IsaacLab order
            "q": (q_mj - DEFAULT_ANGLES)[:, MUJOCO_TO_ISAACLAB],
            "dq": dq_mj[:, MUJOCO_TO_ISAACLAB],
            # the policy's own raw output, inverted out of the joint target
            "a": ((wbc_mj - DEFAULT_ANGLES) / ACTION_SCALE)[:, MUJOCO_TO_ISAACLAB],
            "omega": _quat_ang_vel(root_quat_wxyz, dt),
            "g": np.asarray(gravity3, dtype=np.float64),
        }

    @staticmethod
    def obs_at(pre, t: int, token):
        """Assemble one 994-vector at frame `t` with the given 64-d token.

        History runs t-9 .. t, oldest first, clamped at the episode start --
        which is what a real buffer holds during the first nine control ticks.
        """
        idx = np.clip(np.arange(t - HIST + 1, t + 1), 0, len(pre["q"]) - 1)
        return np.concatenate(
            [
                np.asarray(token, dtype=np.float64).ravel(),  # 64
                pre["omega"][idx].ravel(),  # 30
                pre["q"][idx].ravel(),  # 290
                pre["dq"][idx].ravel(),  # 290
                pre["a"][idx].ravel(),  # 290
                pre["g"][idx].ravel(),  # 30
            ]
        ).astype(np.float32)

    # -- decoding -------------------------------------------------------------

    def __call__(self, obs):
        """Raw graph output: [N, 29] scaled residuals in IsaacLab order.

        The exported graph fixes its batch axis at 1 (obs_dict is [1, 994]), so
        a stack is fed row by row -- passing [N, 994] raises INVALID_ARGUMENT
        rather than broadcasting. The loop is cheap: the decoder is 7 MatMuls.
        """
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        return np.concatenate([self.sess.run(None, {self.iname: row[None]})[0] for row in obs], axis=0)

    def joint_targets(self, obs):
        """[N, 29] commanded joint angles in radians, MuJoCo order.

        Applies the deploy's own output transform (cpp:3133). Absolute values
        depend on the reconstructed velocity channels and are not validated --
        use joint_delta_deg() for anything quantitative.
        """
        return DEFAULT_ANGLES + self(obs)[:, ISAACLAB_TO_MUJOCO] * ACTION_SCALE

    def joint_delta_deg(self, pre, frames, token_a, token_b):
        """Degrees between the joint commands two token tracks produce.

        DEFAULT_ANGLES cancels in the difference but ACTION_SCALE does not --
        it is a per-joint multiplier, so the residual difference has to be
        scaled before it means anything in degrees.

        Args:
            pre:      output of prepare()
            frames:   iterable of frame indices, length N
            token_a:  [N, 64] e.g. ground truth
            token_b:  [N, 64] e.g. the model's prediction, snapped

        Returns:
            [N, 29] signed difference in degrees (b - a), MuJoCo joint order.
        """
        frames = np.asarray(frames)
        oa = np.stack([self.obs_at(pre, int(t), token_a[i]) for i, t in enumerate(frames)])
        ob = np.stack([self.obs_at(pre, int(t), token_b[i]) for i, t in enumerate(frames)])
        d = (self(ob) - self(oa))[:, ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        return np.degrees(d)
