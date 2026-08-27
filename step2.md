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
