---
name: rss-paper-workflow
description: "RSS 2027 paper: numbers only from rss_status.md, experiments via the \"5090 机器 SSH 连接\" session; draft at ~/Research/rss/paper"
metadata:
  node_type: memory
  type: project
  originSessionId: 08e310a7-62a6-4bc9-a5f6-4f5f3133059a
  modified: 2026-11-26T12:04:48.904Z
---

RSS 2027 Stage 1: a 6-page extended abstract due 2026-12-04 AoE. The paper is about learned self-models of frozen humanoid WBCs (ScaleBFM / SONIC / HEFT), inverted to aim the hand.

- **Draft:** ~/Research/rss/paper/main.tex. Every number carries a `% src:` comment. `\dnote{}` marks red draft notes; set `\draftfalse` before submitting.
- **Number sources:** only ~/research-notebook/paper/rss_status.md and the experiment entries it links. Read-only; never edit the entries.
  - Where the status file and an entry disagree, the entry's "复核后的更正" section wins. Report the mismatch to the experiment session.
- **Experiments:** new numbers or experiments are requested by SendMessage to the session named 「5090 机器 SSH 连接」. When it sends an update, re-read the status file and fix the affected parts of the draft.
- **Paused (user, 2026-11-26 ~20:20):** stop updating the draft for each incoming 5090 message, because the data there is changing too fast. Once that session's work is finished, ask it once for the consolidated state, then update the draft in a single pass. Incoming update messages until then: acknowledge at most, do not edit main.tex.
  - Open questions already sent to 5090 at ~20:15 go into that batch: the H3 falsifier change, whether r = 0.85–0.89 is still citable, the numbers not in the status file, the HEFT gait note and per-controller bowl counts.
  - H3 falsifier change: confirmed by the user on 11-26, together with H2 positive, H4 as deployment and removing SIMPLE fixed geometry.
  - 11-26 exception (user "都做"): the reviewed parts of 5090's interim snapshot (SIMPLE table, LML per-task figure, plan) were written into main.tex; everything else still waits for the consolidated message.
  - Answers and decisions received during the pause are queued in ~/Research/rss/paper/pending_updates.md. That includes the user's scope change "不要再发散": only the benchmark × controller A-vs-R table remains.
- **LocoManiLab out of the paper (user, 2026-11-26 ~23:50):** "先不报 locomanilab 上的结果"; confirmed here: "包括论文，先移出". All LML results moved out of main.tex (the full LML version is main_with_lml.tex), LML experiments stopped. The paper is now SIMPLE-first, with HumanoidArena in the plan. Do not reintroduce LML numbers until the user says so. The user then extended this (11-26): delete ALL LML-related content, including results that only used an LML-trained model (HEFT's SIMPLE cell, LML -> SIMPLE / HA transfer ratios). Do not paraphrase LML as "another simulator".
- **HumanoidArena stopped (user, 2026-11-27 09:45 Beijing, relayed by 5090):** "现在先停下所有 humanoidarena 上相关的任务". No HEFT/SONIC HA results will come unless the user resumes. Whether HA stays in the paper is open: on 11-27 the user said "先不动，等我想好", so main.tex keeps its HA text until they decide. The ScaleBFM HA arm was already run (no gain), so a plan that keeps HA must not present that arm as future work.
- **User decisions (11-25/26):**
  - No constant-offset baseline arm.
  - The 2×2 interface × solver experiment is not to be run, and not to be mentioned either.
  - See [[no-defensive-writing]].

**Why:** The experiment session maintains the status file and syncs the entries from the repo. Stale copies and status-file errors were found on 11-26.

**How to apply:** Cross-check status-file claims against the entries' corrections before writing them.
