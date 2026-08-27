# Step 1 — Decompose the Novelty (2026-08-26 ~08:00)

- **Problem framing**: Finetune a humanoid loco-manipulation VLA (walk to table → grasp → lift, language-conditioned, RGB+proprio in) **entirely from scripted privileged-agent demonstrations in simulation, zero human teleoperation**. Evaluation regime: closed-loop success in sim (100-round protocol) as primary; open-loop nMSE as auxiliary (and shown to be misleading).
- **Core mechanism**:
  (a) self-verifying unattended data engine: privileged scripted expert (reach maps, GT poses, visibility gates, grasp-causality audit, per-episode manifest), chunked collection with stall watchdog;
  (b) **cadence governance**: measure demonstration pause distribution against the action-chunk horizon (40 frames @30Hz = 1.33s), engineer the expert to eliminate super-chunk pauses (4.6s → 3.0s median) and gait hesitation;
  (c) VLA action space = **WBC latent motion tokens** (SONIC 64-dim FSQ token + gripper), feasibility delegated to the pretrained whole-body controller.
- **Key insight**: The gap between scripted and human demonstrations is **temporal texture (cadence), not spatial correctness** — BC clones cadence artifacts; action chunking only absorbs pauses shorter than the chunk horizon; hence demonstration cadence must be a first-class dataset-quality metric. Secondary: the latent-WBC-token action space reduces humanoid failure modes from "correct AND balanced" to "correct only".
- **Application domain**: Humanoid (Unitree G1 + dex1 gripper) loco-manipulation, MuJoCo sim-first, GR00T N1.7 VLA + SONIC WBC, aimed at real transfer.
