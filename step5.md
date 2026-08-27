# Step 5 — Deep-Dive Verified Records (2026-08-26)

## LEGS (arXiv:2606.01458, 2026-05-31, Stanford; arXiv preprint)
- Problem framing (verified): finetune pretrained VLAs (ψ0, π0.5, GR00T N1.6) for Unitree G1 loco-manipulation with ZERO teleoperation; real-robot eval, 3 pick-and-place tasks, 10 trials/condition, 1110 total trials.
- Core mechanism (verified): hybrid simulator = MuJoCo physics + mesh foreground composited over 3DGS background + two-stage color calibration; procedural motion-primitive generator using privileged sim state (Walk/Pick/Place decomposed into primitives, scene-level args); per-motion verification check, only verified episodes saved; re-render one motion dataset under new appearances (aug at GPU cost). Action space: SONIC 18-D KINEMATIC command (6-D wrist SE3 + grip scalar per arm + 4-D base vx,vy,ωz,h). Same SONIC binary sim & real.
- Key insight (verified): "photorealistic rendering is a key enabler for synthetic data transfer" — 3DGS vs mesh-only (SAM3D) ablation is the scientific core. Scripted generation is instrumental, not analyzed.
- Application domain (verified): Unitree G1 + SONIC + Dex3 hands, real-robot transfer, pick-and-place (orange→plate), incl. long-horizon walk-turn-squat task.
- Venue: arXiv preprint (CoRL-style formatting).
- Assumptions & scope: randomization envelope TINY (robot ±10 cm, heading ±10°, object ±5 cm, object yaw ±180°); fixed AE/AWB cameras; static background; "procedural generator produces feasible motion but does not optimize for dynamic efficiency" (their only nod to motion quality); single platform/controller.
- Closest-passage evidence: "LEGS uses a procedural motion-primitive generator to synthesize labeled demonstrations at scale without human teleoperation"; "Each motion ends with a verification check, and only verified successful episodes are saved"; Table 2: Task 3 Teleop 0/10 all backbones vs LEGS(200) 2–6/10.
- ZERO occurrences of: pause/cadence/hesitation/DAgger/recovery/covariate/compounding. §3.3 is ONE paragraph + Appendix D (tiny table) + G (primitive list).
- Refined overlap: framing MATCH, domain MATCH, mechanism PARTIAL, insight DIFFER → 3 axes (worst case) → Level 2.

## SIMPLE (arXiv:2606.08278, 2026-06-06, USC PSI Lab; arXiv preprint)
- Framing: BENCHMARK for humanoid loco-manip policy learning+evaluation (60 tasks, 50 scenes, 1000+ assets).
- Mechanism: MuJoCo physics + Isaac Sim offline replay/re-render for photorealistic obs + domain rand; data gen = CuRobo motion-planning pipeline (BoDex grasp synthesis, scripted decomposition) + VR teleop; 36-D explicit action space (32-D upper targets + 4-D base cmd); 3 WBC impls incl. SonicWBC.
- Insight: sim↔real performance correlation makes sim eval a valid proxy.
- Domain: humanoid (G1-class) loco-manip, sim benchmark + zero-shot real transfer.
- Overlap: domain MATCH, mechanism PARTIAL (planner-driven gen ≠ behavior-shaped scripted expert; offline re-render = our ROADMAP mirror idea), framing DIFFER, insight DIFFER → 2 axes → Level 3.

## DemoHLM (arXiv:2510.11258, 2025-10, PKU/BeingBeyond)
- Framing: ONE human seed demo in sim → MimicGen-style synthesis → per-task small manipulation policies (NOT a VLA) over a universal WBC; real G1 transfer, 10 tasks.
- Mechanism: hierarchical low-dim WBC command + data generation from single demo.
- Overlap: domain MATCH, others DIFFER (needs human seed; not VLA; no cadence) → 1 axis → Level 4.

## WholeBodyVLA (arXiv:2512.11047, 2025-12, OpenDriveLab/AgiBot)
- Framing: unified latent VLA for large-space loco-manip on AgiBot X2; latent actions learned from action-free egocentric HUMAN VIDEO + grounded via teleop trajectories.
- Overlap: domain MATCH (humanoid loco-manip), mechanism DIFFER (video LAM, human data), framing/insight DIFFER → 1 axis → Level 4.

## LeVERB (arXiv:2506.13751, 2025-06)
- Framing: vision-language WBC benchmark (sim-rendered kinematic demos); latent "verb" vocabulary co-trained: high-level VL policy emits latent, RL WBC consumes it. NO object grasping/loco-manip (state-space + visual WBC commands: walk/turn/sit/reach).
- Overlap vs claim 3: mechanism PARTIAL-MATCH (latent action interface to WBC — but co-designed, not a fixed pretrained controller's wire), insight PARTIAL-MATCH (latent beats hand-crafted EE vocab for WB feasibility), framing DIFFER, domain PARTIAL (humanoid WBC, no manipulation) → 2 axes → Level 3.

## HAF (arXiv:2608.16837, 2026-08)
- Framing: adapt FROZEN flow-matching VLA to humanoid WB loco-manip; hierarchical 3-stage action denoising + spectral latent offline-to-online RL; 7 real-world tasks; data = VR teleop (TWIST2/AMO lineage).
- Overlap: domain MATCH, rest DIFFER → 1 axis → Level 4.

## ISR / Trajectory Standardization (arXiv:2606.22907, 2026-06)
- Framing: offline preprocessing of HUMAN TELEOP demos for arm manipulation; pauses/variable speed/inconsistent density degrade IL.
- Mechanism: information-standardized resampling — velocity+acceleration information field on Riemannian manifold, geodesic-equidistant resample; +25% success vs 3× uniform downsampling; explicitly claims accel term preserves fine-manipulation phases (the exact hazard that made us park --drop-idle).
- Insight: temporal irregularity in demos is a data-quality problem worth fixing (post-hoc).
- Overlap vs claim 2: insight PARTIAL-MATCH, mechanism DIFFER (post-hoc resample vs source-side behavior engineering; no chunk-horizon analysis; no scripted-vs-human comparison; no closed-loop freeze causal chain), framing/domain DIFFER → 1 axis → Level 4.
