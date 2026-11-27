---
name: bestpaper-skill-first
description: Use the bestpaper-writing skill as the FIRST guideline whenever writing/revising paper prose
metadata:
  node_type: memory
  type: feedback
  originSessionId: 08e310a7-62a6-4bc9-a5f6-4f5f3133059a
  modified: 2026-11-26T06:26:54.384Z
---

Use the `bestpaper-writing` skill (~/.claude/skills/bestpaper-writing, from github.com/gulucaptain/Best-Paper-Writing-Skill @ b9eff99, installed 2026-11-26) as the first guideline for writing paper text: title, abstract, intro, method, experiments and figures.

**Why:** The user asked for this explicitly on 2026-11-26 for the RSS 2027 paper. They also said not to start revising yet, because the draft content still needs many adjustments.

**Status:** the user said go on 2026-11-26. The first Mode B rewrite was done that day: main.tex, report.md in ~/Research/rss/paper, pre-rewrite copy main_pre_rewrite.tex. Later edits keep its structure and check_draft passes.

**How to apply:**
- (Done 11-26) Waited until the user said to start revising. Then invoke the skill in Mode B (existing draft):
  - reverse-engineer the draft into the intake fields;
  - keep facts, numbers and terms verbatim;
  - produce report.md with a change table;
  - run scripts/check_draft.py and check_figures.py.
- Keep the project rules on top:
  - numbers come only from rss_status.md (see [[rss-paper-workflow]]);
  - [[no-defensive-writing]] overrides the skill's "Limitations section" step: do not list un-run experiments as limitations.
- The skill assumes ML/CV venues. RSS Stage 1 is hypothesis-driven (5+1 pages), so adapt the routing to it.
