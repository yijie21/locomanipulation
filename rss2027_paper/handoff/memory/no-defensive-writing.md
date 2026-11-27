---
name: no-defensive-writing
description: "In papers, never explain or justify experiments/baselines we didn't run; state claims positively"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 08e310a7-62a6-4bc9-a5f6-4f5f3133059a
  modified: 2026-11-26T06:11:58.008Z
---

In paper drafts, do not write text that explains why a baseline or experiment is absent, and do not list un-run experiments as limitations or future work. State hypotheses and results positively, as what the method does.

**Why:** The user (2026-11-26, RSS 2027 draft) called such text "完全是防御性写作，我们不用为我们没做的事情解释". It came up when I justified the missing constant-offset baseline and listed a 2×2 (command interface × solver) as a limitation.

**How to apply:**
- Scope claims to what the evidence shows, e.g. "the model does not transfer when interface and solver both differ", rather than apologizing for what wasn't separated.
- If a reviewer-risk worth raising remains, tell the user in chat. Do not write it into the paper.
- Related: [[rss-paper-workflow]].
