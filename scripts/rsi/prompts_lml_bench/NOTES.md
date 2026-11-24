# Notes on the LocoManiLab texts (2026-11-20)

Written from `rules_simple.md` version 3 and `prompts_simple/`; same placeholder set per file as the SIMPLE prompts (checked
with a diff of the `{...}` names). Checked against the code on 2026-11-20 (see below).

## What changed against the SIMPLE texts

- `rules_lml_bench.md`: rules 1-38 kept, words adapted (twenty tasks, `genuine`, `navigate` class, no `comply` class, no hand-over file,
  no table edge). Rule 30 allows branching on the instruction's words and on camera measures, never on the task's id. Rule 39
  adds "whether to walk at all" to level 2 and names the stance keys. Rule 40 has the per-task table and the three compliance
  rules as `comply_lml.py` implements them. Rule 43 follows `loss_lml.py`. Rule 44 follows `guard_lml.py`.
- Prompts: five-task lists became pointers to rule 40; the Engineer text names the four editable files and no new file.

- 2026-11-20: the texts follow `rules_lml_bench.md` version 2: `sweep` and `tune`, `untested` for a hypothesis that needs code, `tunable` in the Engineer's answer, `{REVIEW_NOTES}` in manager.md, engineer.md and composer.md (the only new placeholder), drafts run on their declared scope.

## Checked against the code

- The Engineer's guard and override commands run on the seed package (`backend_lml.run_guard`, `load_skills`, `apply_set`).
- Stance: `set` keys `stance_d`, `stance_t`, `stance_yaw`, `stance_here` (pick), `stance_d2`, `stance_t2`, `stance_stay`
  (placing); the backend splits the `stance_` keys off and hands them to the runner as `--stance`; any other `stance_` key is
  refused by `apply_set`. The walking tasks read no stance key.
- The twenty time limits and instructions agree with the bed's records and the family's tasks file.
- The guard refuses a new import inside a function body as it does at the top of a file.

## Still open

- "About half a minute" per round is measured wall time of 12 to 18 s per start plus the first look and grasp pass of a start.
- Pool size: the context gives the number of starts; the texts say "small".
