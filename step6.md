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
