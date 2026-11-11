# Reach world-model comparison: implementation spec

## Goal

Learn what frozen ScaleBFM actually delivers when a single task-space reach command is held: start body state plus
command in, final jaw-centre pose out. Then compare candidate model designs on accuracy, generalization and
uncertainty.

The models will later be inverted (find the command that lands on a desired pose), so every learned model must be
differentiable with respect to the command. Do not implement the inversion now.

## Data

The data is produced by `gen_reach.py` (read it; do not modify it). Chunks live in
`outputs/reachwm_20261030/data/w<worker>_<chunk>.npz`. Generation is still running; load whatever chunks exist by
globbing. `merge` writes `all.npz` at the end. The loader must accept either a list of chunk files or `all.npz`.

**Per-step arrays**, concatenated over segments, one row per 0.1 s:

| Array | Width | Contents |
|---|---|---|
| `S` | 65 | self state: joint pos 29 (ScaleBFM order), joint vel 29, gravity in pelvis frame 3, pelvis angular vel 3, pelvis height 1 |
| `W` | 13 | world: pelvis xyz, pelvis yaw, right jaw xyz, left jaw xyz, head camera xyz |
| `J` | 6 | acting jaw rotation, world frame, first two columns |
| `C` | 4 | commanded jaw centre in the current pelvis frame (3), time since command in s (-1 before the command) |

**Per-segment arrays:**

| Array | Meaning |
|---|---|
| `n_steps` | rows in the segment |
| `k0` | row index, within the segment, at which the command is issued. Rows 0..k0 are before or at the command. |
| `side` | +1 right, -1 left |
| `target` | (3,) x ahead and y left in the heading frame at command time, z = world height |
| `dyaw`, `pitch` | goal wrist yaw offset and pitch, rad |
| `rot6` | (6,) goal wrist rotation in that heading frame, first two columns |
| `pelvis_z` | -1 means stand; otherwise the commanded pelvis height |
| `start` | (4,) pelvis xyz and yaw at command time (world) |
| `walked`, `chain`, `fell`, `ok` | flags. `chain` is 0 for the first reach after a reset, 1 for the second, which starts where the first left the body. |
| `jaw_err_cm`, `att_err_deg`, `ik_err_cm`, `reach_s` | from the controller's own report |
| `reset` | reset id |

Each segment runs from 0.3 s before the command to 1 s after the reach call returns. The robot falls in a few
segments; keep them for fall prediction only.

**Heading frame of a segment:** origin at (start.x, start.y, 0), rotated by `start.yaw` about z. A world point
maps to heading coordinates as `p_h = Rz(yaw)^T (p_w - [x, y, 0])`, so z stays the world height.

### Derived per-segment quantities

Build these in `data.py` and check them on a few segments: final error vs `jaw_err_cm` should agree within about
0.3 cm when settled. On the pilot it did.

- **Delivered pose:** jaw position from `W` (right is cols 4:7, left is 7:10, chosen by side) and rotation from
  `J`, both at the LAST row, expressed in the heading frame.
- **`settled`:** max jaw speed over the last 0.5 s below 2 cm/s.
- **Command vector u (12):**
  - `side`
  - `target` (3)
  - `rot6` (6)
  - stand flag: 1 if `pelvis_z == -1`, else 0
  - `pelvis_z`, or 0 when standing
- **Start features x0:**
  - `S[k0]`, `S[k0-1]`, `S[k0-2]` (195)
  - acting jaw position at k0 in the heading frame (3)
  - acting jaw rotation at k0 in the heading frame, rot6 (6)
  - `walked`, `chain` (2)
- **Error frame:** the goal wrist frame `Rg` (from `rot6`). Express any position error e as `Rg^T e`. Components
  are depth (wrist x), closing (wrist y) and third (wrist z). Report all three.

### Splits

Deterministic, by command value:

- **Bands.** Never train or validate on these. Each is its own test set; a segment may fall in several.
  - `band_x`: target x in [0.48, 0.54)
  - `band_z`: target z in [0.40, 0.48)
  - `band_yaw`: dyaw in [30, 42) deg
  - `band_pel`: pelvis_z in [0.60, 0.64)
- **Edge set.** Never train on this; it is a test set. Target x > 0.85, or z < 0.08, or the target is more than
  0.50 m to the hand's outer side (y < -0.50 for the right hand, y > 0.50 for the left).
- **The rest, by reset id:** `hash(reset) % 10` gives 8 train, 1 val, 1 `test_iid`.
- **Nested train fractions** 0.1, 0.3 and 1.0, chosen by reset id, for data-efficiency runs.

## Candidates

Put everything in `models.py` with one interface. Normalize all inputs and outputs. For `Norm`, copy the class from
`selfwm_20261029/wm.py`, or import it.

`predict(batch) -> dict` returns:
- `pos` (B, 3) and `rot6` (B, 6): heading frame, final jaw pose
- `pos_std` (B, 3): optional
- `p_settled`, `p_fell`: optional

**Baselines**

| Name | Design |
|---|---|
| `B0_command` | pos = target, rot = goal rot. Nothing learned. |
| `B1_knn` | k=8 nearest training segments in normalized [u, pelvis height at k0, walked, chain]. Prediction = target plus the inverse-distance-weighted mean offset (delivered - target). `pos_std` = weighted std of the neighbour offsets. This is a data-driven version of the main line's bias table. |

**End-state models.** One network call, input [x0, u].

| Name | Design |
|---|---|
| `E1_direct` | MLP (`wm.mlp`, width 512, depth 4, SiLU) predicts delivered pos and rot6 directly. Also outputs settled and fell logits. |
| `E2_residual` | Same MLP, but predicts the offset: pos = target + d, and rot = Rgoal @ R(d6) with R from a 6D residual that starts at identity. Initialize the last layer to zero so the untrained model equals `B0_command`. |
| `E3_ensemble` | 5 independent `E2_residual` members (different seeds, each on a bootstrap resample of train). pos = mean; pos_std = std across members. |
| `E4_joints` | Kinematics in the loop. The MLP predicts the final joint vector q_f (as a delta from q at k0) and the final pelvis pose in the heading frame (position 3, plus yaw change, roll and pitch via the gravity vector, or a rot6 directly). The jaw pose comes from EXACT forward kinematics (`selfwm_20261029/fk.py`, pelvis frame) composed with the predicted pelvis pose. Loss = joint and pelvis terms plus FK jaw position in cm. Training targets: final q from `S`, pelvis from `W` (xyz, yaw) and `S` gravity (roll, pitch). Build the FK exactly as `wm2.build` does: `R.Sim` on the walk_mars scene, then `FKM.FK(sim.m, sim.qadr)`. |

**Step models.** Autoregressive at 10 Hz from k0 to the last row, with the episode command u and the time since
command as the action at every step.

| Name | Design |
|---|---|
| `S1_mlp` | The `selfwm_20261029/wm2.py` design. Input [s_{t-1}, s_t, u, tc] gives Δs (65) and odometry (4: pelvis displacement in the previous step's heading frame, yaw change; see `env.odometry`). Training rolls out 8 steps with the `wm2.py` loss: state + odometry + 0.1 × FK jaw in cm². Windows start anywhere with a full 8-step future, including pre-command rows (tc = -1). For the final pose, roll out from (k0-1, k0) to the last row, integrate odometry into the pelvis pose in the heading frame, and get the jaw from exact FK of the predicted q plus the pelvis pose. For the pelvis roll and pitch use the predicted gravity, as in `E4`. |
| `S2_ensemble` | 5 `S1_mlp` members, mean and std of the final pos. |
| `S3_gru` | GRU, hidden 512, input [s_t, u, tc], same outputs and loss as `S1`. Warm up on rows 0..k0, then roll out. Train on 16-step windows: the first 4 are warm-up, and the loss is on the last 12. |

**Training.** Use the same budget for every model:
- AdamW, lr 1e-3, weight decay 1e-5, cosine schedule, grad clip 1.0.
- End-state models: 30k iterations, batch 1024.
- Step models: 40k iterations, batch 512 windows.
- Evaluate on val every 2k iterations and keep the best val final-position error.
- Fixed seeds.
- `train.py <name> [--frac 0.1|0.3|1.0] [--seed] [--iters]` saves to
  `outputs/reachwm_20261030/models/<name>_f<frac>/` (`model.pt` and `train.json`).
- `train.py all` trains every model at frac 1.0, plus `E2_residual`, `E3_ensemble` and `S1_mlp` at 0.1 and 0.3.

## Bench

`bench.py` evaluates every trained model on `test_iid`, the four bands and the edge set. Non-fallen segments only,
except fall AUROC. It writes `outputs/reachwm_20261030/bench/results.json` and `results.md`.

Metrics per model and test set:
- final position error in cm: median, p90, p99, mean
- per-direction median absolute error in cm: depth, closing, third
- fraction under 0.75, 1 and 2 cm
- rotation error in degrees: median, p90
- also on settled-only segments
- **For models with `pos_std`:**
  - Spearman correlation of |std| with the error
  - AUROC of |std| for flagging error > 1 cm
  - mean |std| per test set, which should rise on bands and edge
- **For models with `p_fell` / `p_settled`:** AUROC
- **Data efficiency:** `test_iid` median and p90 against frac
- **Speed:** ms per 1024 predictions on GPU

## Constraints

- Code goes only in `scripts/experiments/reachwm_20261030/` (new files `data.py`, `models.py`, `train.py`,
  `bench.py`). Import from `selfwm_20261029` (`env.py`, `fk.py`, `wm.py`) via sys.path. Never edit those files or
  `gen_reach.py`.
- Run with `PY=/workspace/code/locomanipulation-selfwm/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python`, cwd
  `/workspace/code/locomanipulation-selfwm`, and `OMP_NUM_THREADS=1`.
- Data generation is using both GPUs. Tests must be tiny (a few hundred iterations on `CUDA_VISIBLE_DEVICES=1`) until
  generation finishes. Do not launch full training.
- No git commits.
- Match the style of `selfwm_20261029/wm2.py`: module docstring with what and how to run, terse comments.
