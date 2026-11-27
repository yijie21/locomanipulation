---
name: selfaim-framing
description: "User's framing decisions for the RSS paper (title SelfAim, core claim, H4 binding is an engineering detail and is not foregrounded)"
metadata:
  node_type: memory
  type: project
  originSessionId: 08e310a7-62a6-4bc9-a5f6-4f5f3133059a
  modified: 2026-11-26T12:35:51.195Z
---

Framing decisions the user made for the RSS 2027 paper on 2026-11-26:

- **Title (final, 2026-11-26):** "SelfAim: Learned Self-Models for Aiming Frozen Humanoid Whole-Body Controllers".
  - The user briefly switched to "SelfAim: Aiming … by Fixing the Command", then reverted. That title dropped "learned" and made it sound as if we only fix the command, when the fix comes from a trained model.
  - Titles must keep "learned" (or an equivalent).
  - Also rejected: the user's own "WBC-WM" and the handle-led title "Fix the Command, Not the Controller: …".
  - Do not put "any controller" in the title.
- **Core claim (the user's own):**
  - We neither train a more accurate WBC nor change its weights. The missing ability goes into a separate model.
  - That model can be added to essentially any WBC and improves its one-shot reach cheaply.
- **Binding to controller + interface + simulator is an engineering detail; the principle is general.** Do not foreground it, and do not hedge "any WBC" because of it.
  - H4 becomes a deployment point: a new setting costs one re-collection with the same recipe.
  - Table 2 and the velocity / variant F / unexplained-factor details shrink or go.
- **Wording to keep accurate:**
  - Present it as one recipe that applies to any WBC, with one model trained per controller.
  - The paper never says that one trained model is reused across controllers or simulators without new data.
  - The gain is "aiming" (where the hand lands), not loco-manipulation in general.
- **Used in the 11-26 rewrite.** H2, H4, the H3 falsifier and removing SIMPLE fixed geometry were confirmed by the user on 11-26. The handle sentence stays in the intro; the user has not objected:
  - the handle "fix the command, not the controller" (intro, paragraph 4);
  - the positive H2 ("aiming raises grasp success … depends on task");
  - H4 as the deployment cost.

- **11-26 late:** with LML out, there are three hypotheses: H1 aim, H2 success, H3 deployment. Former H3 (training support) was dropped because in SIMPLE R = S (all targets reachable); the reachable restriction remains a design choice.
- **Figure 1 taste (user, 11-26):**
  - v1 and v2 (benchmark scenes: LML candle and salt) were rejected: the head was not visible, the viewpoint was poor, and a salt shaker "looks low".
  - The user wants a fancy custom-built scene: the robot reaches up to an object on a high shelf, the camera looks down obliquely from the front, and the head, body, hand and object are all visible.

**Why:** the user wants a claim- and contribution-driven paper rather than an experiment report (they raised this on 2026-11-26).

**How to apply:** use this framing in the rewrite. Rewrite only after the user says go ([[bestpaper-skill-first]]); numbers still follow [[rss-paper-workflow]].
