# Notes on the BiGym 2.0 texts (2026-11-24)

Version 1, 2026-11-24, TODO 49.7a. Adapted from `prompts_lml_bench/` (written from `rules_lml_bench.md` version 9) for the bench adapter
`locomani.bench.bigym2`, family `bigym2_reach_drawer`, rules `rules_bigym2.md` version 1. Same placeholder set per file as the source
(`{REVIEW_NOTES}` is the one the Engineer, Manager and Composer read the Reviewer's files through; checked with a diff of the `{...}` names).

## What changed against the LocoManiLab texts

- Benchmark and robot: BiGym 2.0, G1 with Dex1 jaws, BiGym's own GR00T-WBC lower body, the 20-D native action; two tasks.
- A win is the row's `success` (rule 40): BiGym's check, no fall, and the hands did it (`body_push`, `no_hand`); no `genuine`, no `comply` dict.
- The package is one editable file `skill.py` with a frozen `__init__.py`; the facade `env` is named in the Engineer text (rule 44); the target is the truth view.
- Levels 1 and 3 only; a `stance_` key is refused. Failure classes: error, timeout, fall, comply, short (rule 43).
- A round takes seconds, one simulator at a time; the pools are seed pools P, A, S.

## Checked against the code

- The Engineer's guard and override commands are those of the source (`backend_bench.load`, `run_guard`, `load_skills`).
- The facade members, the forbidden names and the 300-line growth limit follow `bigym2/guard.py` as quoted in rule 44; the compliance numbers (5 and 3 steps) follow `bigym2/judge.py` as quoted in rule 40.

## Still open

- The wall time per round (1 to 8 s) is from the adapter's smoke runs on 5 practice starts per task.
