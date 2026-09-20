# G1 paired-ghost reachability preview — 2026-09-20

Read this for the proposed target distribution before running the reachability
experiment. This task renders the environment and targets only; it does not test
IK, reach execution, standing stability, contact, or grasp success.

**Current revision: V2.** The V1 ghost assembly had a mesh-frame conversion defect
reported by the user; its wide sampling limits are superseded. Earlier sections
below preserve the V1 record. See [V2](#v2-corrected-assembly-and-task-conditioned-bounds)
for the correction and current proposal.

## Scope and coordinates

Use a floor-only MuJoCo environment with the current Unitree G1/dex1 model. Freeze
the original bimanual ready posture throughout rendering: waist zero,
left arm `[-0.35, 0.2, 0, 0.8, 0, 0, 0]`, right arm
`[-0.35, -0.2, 0, 0.8, 0, 0, 0]`, and OPEN grippers. Leg posture follows the
original SONIC default. This is a static pose, not a dynamically settled observation.

Robot facing is +x, left is +y, ground-up is +z. The xy origin is the ground
projection of the pelvis. Forward and lateral offsets below describe the **pair
midpoint**, not each hand separately. Heights are above the ground.

| Sampling variable | Proposed interval |
|---|---|
| Pair midpoint forward x | 0.25–0.65 m |
| Pair midpoint lateral y | −0.15–+0.15 m |
| Common grasp-center height z | 0.35–1.05 m |
| Distance between grasp centers | 0.25–0.65 m |
| Pair-line yaw relative to robot left-right direction | −35–+35 degrees |
| Wrist inclination toward the body from vertical | 0–45 degrees |

These are editable candidate sampling limits, **not measured reachable limits**.
The union of individual target positions can extend beyond midpoint bounds: its
axis-aligned envelope is x approximately [0.0636, 0.8364] m, y [−0.475, 0.475] m,
z [0.35, 1.05] m. Not every point in that envelope is an allowed paired sample.

## Fixed-center orientation convention

For yaw ψ, let `line=(-sinψ, cosψ, 0)` and `forward=(cosψ, sinψ, 0)`.
Targets are `midpoint ± separation/2 * line`. Both retain the same height and
positions during an example; changes between examples are hard cuts.

For inclination θ, wrist-to-grasp direction is
`sinθ * forward − cosθ * up`. Given the existing tool offset 0.1515 m,
`wrist = grasp_center − 0.1515 * direction`. Thus the wrist moves toward the
body while the grasp center stays fixed. At zero inclination the fingers point
vertically downward. Real gripper meshes visualize the orientation, and colored
center markers identify the fixed target positions.

## Artifacts and verification

Artifact root: `outputs/g1_reachability_preview_20260920/`.
CPU OSMesa rendering is available and avoids any live controller or GPU instance.
Completed with 14 examples: six named boundary configurations and eight random
samples with seed 42. Each example lasts 1.5 s and sweeps wrist inclination 0–45
degrees with fixed capture centers. The final full-duration preview is
`g1_reachability_preview.mp4`: **445,606 bytes**, **21.0 s**, **252 frames at 12 fps**,
960×540. The original 1280×720 source is retained. Six decoded frames spanning
initial/final inclination, near/far/low/high/yawed examples, and the final frame
were visually reviewed. Full decode and exact source/preview duration/frame-count
checks passed (`video_verification.json`).

`render_preview.py` rebuilds the environment and applies the explicit ready qpos;
`range_config.json` is the editable range/seed/camera configuration; `targets.json`
contains all paired capture centers and orientation endpoint poses.
`scene_floor_g1_ghosts.xml` is the compiled geometry, and `manifest.json` records the
initial qpos and model provenance. Read the artifact `README.md` for CPU render
commands. Runtime artifacts and assets remain untracked.

Validation: independent static checks confirmed the configured bounds, midpoint,
spacing, paired equal height and bodyward wrist inclination. Maximum reconstructed
capture-center error was 1.11e-16 m (`geometry_verification.json`). Top-level bodies
are only the robot and two collision-free ghost overlays; the sole world geom is
the floor. Every rendered frame passed the bit-identical robot-qpos assertion;
`mj_step` is never called. The original default leg angles are unchanged; seating
moves only the pelvis down 0.035509 m, leaving the foot mesh bottom 0.25 micrometres
above the floor. No claim of dynamic standing stability or reachability follows.

No production code or controller was modified and no GPU/live stack was launched.
Next action: user review of candidate target ranges and gripper orientation, then
implement the separately authorized reach experiment using this explicit geometry.


## V2: corrected assembly and task-conditioned bounds

Artifact root: `outputs/g1_reachability_preview_20260920_v2/`. V1 source/video remain
untouched. No production controller changes or live/GPU simulation were made.

### Mesh defect and verification

MuJoCo compiled geom poses include the mesh compiler's centering/principal-axis
transform. V1 subtracted `mesh_pos` using the effective rotation before undoing
`mesh_quat`; components could be displaced by 0.129049 m. Correct inversion is:

```text
R_raw = R_effective @ R_mesh.T
p_raw = p_effective - R_raw @ p_mesh
```

`verify_ghost_geometry.py` compares the assembled compiled ghost with the original
source hand at translated identity, vertical, 45-degree tilt and 35-degree yaw
poses. Both hands retain all four source visual meshes. Across 302,300 corresponding
world vertices, maximum corrected error is 3.16e-16 m; maximum effective geom
position error is 1.11e-16 m. The previous formula reproduces 0.129049 m error.
The capture marker radius is reduced to 7 mm so it does not obscure the fingers.
`geometry_verification.json` records the checks. Source-vs-ghost closeups in
`fixed_ghost_geometry_closeup.png` were visually reviewed: wrist, base and both
fingers form the same assembly. V1 capture-center checks alone did not detect this
mesh assembly error; the new comparison exercises the actual visual geometry.

### Proposed front-of-body box-grasp task region

Use the current robot yaw-aligned frame, xy origin at pelvis ground projection,
height above the floor. Sample a **paired virtual box arrangement**, then derive
both hands from its midpoint, separation and yaw. Hands are not independently
randomized. A useful nominal proposal is midpoint `(0.40, 0, 0.70)` m, separation
0.44 m and yaw zero. The prior box geometry had a rim near 0.695 m and opposing
grasp-center separation 0.44 m, so this focuses the experiment on that task family.

| Parameter | V2 proposed range |
|---|---|
| Pair midpoint forward x | 0.32–0.48 m |
| Pair midpoint lateral y | −0.05–+0.05 m |
| Common height z above floor | 0.60–0.80 m |
| Grasp-center separation | 0.35–0.50 m |
| Pair-line yaw relative to body left-right | −15–+15 degrees |
| Allowed bodyward wrist inclination from downward | 0–45 degrees |

These limits imply individual target x approximately 0.255–0.545 m, with each hand
remaining on its own body side. This is an analytic consequence of the parameter
bounds, not evidence of reachability or collision clearance. `sampling_proposal.json`
records the rationale and derived bounds; `range_config.json` drives rendering.
`target_range_verification.json` checks actual sampled midpoints, spacing, height,
bounds and non-crossed left/right assignment. Capture centers remain fixed while
the allowed wrist inclination is illustrated.

For the later reach experiment, validate the nominal case first, then sample the
core and extend distance/height/separation/yaw individually around observed
success/failure transitions. Do not interpret this narrower parameter box as a
proven reachable region or use visual appearance alone to label reachability.

### V2 export

Completed `g1_reachability_preview.mp4`: **443,306 bytes**, full **21.0 seconds**,
252 frames at 12 fps, 960×540. The original source recording is preserved. Decode,
source/preview duration and frame-count checks passed. Six decoded frames covering
the start/end, 0/45-degree inclination, and different target configurations were
reviewed; the repaired hands remain assembled. Every rendered frame passed the
frozen-robot-qpos invariant. The source-vs-ghost closeup is 267,950 bytes. See
`video_verification.json` and `artifact_manifest.json` for export/evidence records.
No new dynamic reach trial has been run.
