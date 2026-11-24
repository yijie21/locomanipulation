# Notes on the HumanoidArena texts (2026-11-24, version 1)

TODO 49.7a. These are the role prompts of the RSI loop for the family `humanoidarena_pp_box` on the bench adapter
`locomani.bench.humanoidarena`. Adapted from `prompts_lml_bench/` (written for `rules_lml_bench.md` version 9); the
rules they cite are `rules_humanoidarena.md` version 1, which keeps the same numbering.

## What changed against the LocoManiLab bench texts

- Benchmark and robot: HumanoidArena, one task (`pp_box`), a G1 under the benchmark's native TWIST2 tracker, target=truth.
- A win: the benchmark's check, no fall, the hands did it (`success`; `why` is `no_hand`, `body_push`, `not_moved`, ...). No
  `genuine` flag, no `comply` dict.
- Package: one editable file `skill.py`, `__init__.py` frozen; the facade `env` of rule 44; positions come from `env.view.goal()`.
- Levels: 1 and 3 only; a `stance_` key is refused. Failure classes: error, timeout, fall, comply, short (rule 43).
- Round time: one to three minutes, one Isaac server at a time on GPU 1. Pools are seed pools P, A, S.
- The `{...}` placeholder set of every file is the same as in the source file (`{REVIEW_NOTES}` in manager, engineer and composer).

## Checked against the code

- Rule numbers cited in the texts exist in `rules_humanoidarena.md` (1, 12-17, 18, 25, 28, 30, 32, 33, 38-49).
- Guard rules R1-R8, forbidden names and the facade list follow `humanoidarena/guard.py` as quoted in rule 44.
- The guard and override commands are the generic `backend_bench` calls of the source texts.
- The seed's flow (settle ... place) and the 1450-step budget follow `seed/skill.py` and rule 16.

## Still open

- Round time is quoted from the bring-up notes (7 to 10 steps a second); the Manager's pool size comes from the context.
