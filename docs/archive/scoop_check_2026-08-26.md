# 新颖性核查工作记录(scoop check,2026-08-26)

> `scoop_check` skill 的逐步工作笔记(step 1–7),2026-08-27 随 b41692a 被误提交到仓库根目录,
> 2026-09-10 移入归档。内容未改动。结论:LEGS(arXiv:2606.01458)与本仓库在框架/领域/机制上高度重叠。

# Step 1 — Decompose the Novelty (2026-08-26 ~08:00)

- **Problem framing**: Finetune a humanoid loco-manipulation VLA (walk to table → grasp → lift, language-conditioned, RGB+proprio in) **entirely from scripted privileged-agent demonstrations in simulation, zero human teleoperation**. Evaluation regime: closed-loop success in sim (100-round protocol) as primary; open-loop nMSE as auxiliary (and shown to be misleading).
- **Core mechanism**:
  (a) self-verifying unattended data engine: privileged scripted expert (reach maps, GT poses, visibility gates, grasp-causality audit, per-episode manifest), chunked collection with stall watchdog;
  (b) **cadence governance**: measure demonstration pause distribution against the action-chunk horizon (40 frames @30Hz = 1.33s), engineer the expert to eliminate super-chunk pauses (4.6s → 3.0s median) and gait hesitation;
  (c) VLA action space = **WBC latent motion tokens** (SONIC 64-dim FSQ token + gripper), feasibility delegated to the pretrained whole-body controller.
- **Key insight**: The gap between scripted and human demonstrations is **temporal texture (cadence), not spatial correctness** — BC clones cadence artifacts; action chunking only absorbs pauses shorter than the chunk horizon; hence demonstration cadence must be a first-class dataset-quality metric. Secondary: the latent-WBC-token action space reduces humanoid failure modes from "correct AND balanced" to "correct only".
- **Application domain**: Humanoid (Unitree G1 + dex1 gripper) loco-manipulation, MuJoCo sim-first, GR00T N1.7 VLA + SONIC WBC, aimed at real transfer.

---

# Step 2 — Search and Deduplicate (2026-08-26)

Queries (paper-search, all 6 sources):
- Q1 original-problem: "humanoid loco-manipulation vision-language-action scripted demonstration generation simulation" (2023-2026) → 30 unique
- Q2 broad-domain: "automated demonstration data generation robot manipulation without teleoperation" (2022-2026) → 30 unique
- Q3 method-signature: "whole-body controller latent action space humanoid vision language action model" (2024-2026) → 30 unique
- Q4 claim-2 probe: "demonstration quality pauses idle imitation learning action chunking temporal" (2022-2026) → 25 unique

Cross-query duplicates removed: Manipulate-Anything, CHILD, Physically Consistent Humanoid Loco-Manipulation, Humanoid WB Manipulation via Active Spatial Brain, Inference-Time Attention Steering (≈5).

Model-recall additions (Source: model-recall):
- MimicGen (arXiv:2310.17596, 2023-10) — automated demo generation from few human demos, arm-based, segment-and-transform
- DexMimicGen (arXiv:2410.24185, 2024-10) — bimanual dexterous extension of MimicGen
- RoboCasa (2024-06) — large-scale sim kitchen tasks; data via MimicGen-style generation, arm/mobile-arm
- GR00T N1 (arXiv:2503.14734, 2025-03) — NVIDIA VLA foundation model; data pyramid incl. human teleop for humanoid post-training
- SRPO (arXiv:2511.15605, 2025-11) — self-referential RL for VLAs (from user's discussion doc)

---

# Step 3+4 — Triage & Candidate Selection (2026-08-26)

High-potential candidates (deep dive, reasons):
1. LEGS arXiv:2606.01458 — title claims "Teleop-Free VLAs for Humanoid Loco-manipulation": potential match on ALL FOUR axes. User flags its scripted-data treatment as superficial — verify from body.
2. SIMPLE arXiv:2606.08278 — sim-based policy learning AND evaluation for humanoid loco-manip; possible match on framing+domain.
3. DemoHLM arXiv:2510.11258 — "from one demonstration to generalizable humanoid loco-manipulation": automated sim data gen for humanoid; framing+domain threat.
4. WholeBodyVLA arXiv:2512.11047 — unified latent VLA for whole-body loco-manip; claim-3 (latent WBC action space) threat.
5. LeVERB arXiv:2506.13751 — latent vision-language instruction for humanoid WBC; claim-3 threat, well-known.
6. HAF arXiv:2608.16837 — adapting generalist VLAs to humanoid WB loco-manip via hierarchical action flow + spectral latent RL; newest (2026-08), claim-3 + RL.
7. Trajectory Standardization arXiv:2606.22907 — only surfaced work near claim 2 (temporal demo quality); verify whether it covers pauses/cadence.

Triage notes on non-candidates: MimicGen/DexMimicGen/SkillMimicGen/HumanoidGen/DemoGen = automated data gen but ARM-based or bimanual-static (no locomotion, no WBC tokens, no cadence analysis — mechanism differs, domain differs). OmniRetarget = human MoCap retargeting (needs human motion source, not scripted agent). VIRAL = visual sim-to-real scaling for humanoid loco-manip (rendering axis, not data-gen-mechanism axis; couldn't resolve arXiv id from search output — openalex only). GR00T N1 = human-teleop post-training (our baseline recipe). SENTINEL/MotionWAM/ω-0/PhysiFlow/WOLF-VLA = humanoid VLA architectures, no scripted-data claim. Learning-to-Discern / Quantifying-Demo-Quality = human demo quality scoring (spatial/strategy heterogeneity, not cadence-vs-chunk-horizon). How-to-Train-Your-Robots = demonstration MODALITY comparison (ICRA'25) — adjacent to claim 2, listed but not deep-dived (paywalled IEEE).

---

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

---

# Step 6+7 — Comparison, Levels, Delta (2026-08-26)

Per-paper levels (axes matching → level):
- LEGS: 3 (framing, domain, mechanism-partial-counted-worst-case) → Level 2 High Overlap
- SIMPLE: 2 → Level 3
- LeVERB: 2 → Level 3
- DemoHLM: 1 → Level 4
- WholeBodyVLA: 1 → Level 4
- HAF: 1 → Level 4
- ISR: 1 → Level 4
- (triage-only) MimicGen/DexMimicGen/SkillMimicGen/HumanoidGen/DemoGen: 1 → Level 4 (mechanism family, arm domain)
- GR00T N1 recipe: baseline, not competing claim.

OVERALL VERDICT: Level 2 — High Overlap (minimum across prior works, driven by LEGS).
Consequence: standalone claim 1 ("scripted replaces teleop for humanoid loco-manip VLA") is ALREADY CLAIMED by LEGS (2026-05-31) with real-robot evidence. Delta must be carried by claim 2 (cadence) + the engineering science LEGS glossed + the latent-token-wire recipe difference + wide-envelope closed-loop analysis.

Delta (one sentence):
Unlike LEGS, which shows procedurally-generated demonstrations can replace teleoperation for humanoid loco-manip VLA finetuning *provided photorealistic 3DGS rendering* — while treating the scripted expert as an unproblematic given (one paragraph, ±10cm/±10° spawn randomization, no analysis of demonstration timing, recovery data, or closed-loop failure) — the proposed work makes the scripted expert itself the object of study: it measures a categorical cadence gap between scripted and human demonstrations (median arrival pause 4.6s vs 0.2s against a 1.33s chunk horizon), shows BC clones this artifact into a closed-loop freeze attractor that DAgger alone cannot remove, and demonstrates source-side cadence governance (vs ISR-style post-hoc resampling) at a 4×-wider spawn envelope, through the WBC's latent-motion-token deployment wire (vs LEGS's 18-D kinematic command).
