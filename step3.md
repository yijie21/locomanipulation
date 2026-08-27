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
