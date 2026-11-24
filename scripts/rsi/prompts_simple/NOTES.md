# Notes on prompts_simple/ and rules_simple.md (2026-11-19)

Written against `scripts/rsi/loop.py` as it stands in the working tree (line numbers below are from it; they will move).
Nothing here was run. The files keep the names of `prompts/`; `rules_simple.md` keeps rules.md's numbering (dropped
rules stay as one-line stubs) because the driver's messages and `CLASS_SUGGEST` cite rules by number; rules 39-45 are new.

## 1. What the loop fills or parses

All templates are filled by `fill()` (loop.py:768), a plain `str.replace` of `{KEY}` for each key given; any other brace
text (`{object}`, JSON examples such as `{"set": ...}`) is left as written.

| file | placeholders it uses | filled at |
|---|---|---|
| reviewer.md | RULES, PLAN, CONTEXT, CARDS, DIAG_INC, POOL, SKILLS, SKILLS_NOTE, FAMILY, FAMILY_FILE, N_DIAG, SIM_PY, LIBRARY | loop.py:1604, with `common` (1592-1594) and `prompt_vars()` (404-420) |
| manager.md | RULES, PLAN, CONTEXT, FAMILY, FAMILY_FILE, N_ACCEPT, CANDIDATES_NOTE, REVIEW_NOTES | 1721 |
| engineer.md | RULES, PLAN, CONTEXT, ITER, SKILLS, DIAG_INC, FAMILY, FAMILY_FILE, N_DIAG, SIM_PY, LIBRARY, REVIEW_NOTES | 1793 and 2720 (parallel drafts: SKILLS and TMP overridden per draft) |
| composer.md | RULES, PLAN, CONTEXT, ITER, SKILLS, DIAG_INC, FAMILY, FAMILY_FILE, N_DIAG, SIM_PY, REVIEW_NOTES | 2817-2821 |
| composer_imagined.md | CONTEXT | appended to composer.md only when `imagined` exists (2818-2819) |
| repair.md | BRIEF, ITER, SKILLS, DIAG_INC, FAMILY, N_DIAG, SIM_PY | 2535-2537 |
| reflector.md | FAMILY, CONTEXT_DIR (only these two, `.replace` at reflect.py:153) | reflect.py |

`{TMP}` (the roles' scratch directory) is no longer used: the original needed it only for the infeasibility check files.
`{LOOP}` and `{INC_DIR}` are given by the loop but unused, as in the originals' most prompts.

Schema fields the roles must answer in (the texts name them; spelling kept exactly):
- Reviewer, HYPOTHESES_SCHEMA (loop.py:262-312): `hypotheses[]` {id, claim, instead, evidence, delta{set}, sweep[{key, values}], needs_code, measure{feature, op, value, expect, tasks, scope}, design, predicted_fixed, predicted_broken}, `notes`, `infeasible[]`. `status` is the driver's. The text tells the Reviewer to leave `infeasible` empty and not to use `design`.
- Manager, MANAGER_SCHEMA (314-354) through `manager_schema()` (2384): `decision` in engineer / review / stop (reopen is removed only when `reopen_blocked`; the text says it is unused), `hypothesis`, `hypotheses`, `tune`, `base`, `skip{id: check}` (enum interface_kept, safety_checks_kept, changed_code_reached, evidence_not_refuting), `repeats[{now, same_as}]`, `focus`, `reason`, `escalation`, `infeasible`, `gap_proposals`, `summary_zh`.
- Composer, COMPOSER_SCHEMA (356-366): `decision` merge / adopt / keep, `adopted`, `adopted_pool`, `set`, `mechanism`, `notes` (line format `kept|revised|dropped -> ... -> ...`).
- Engineer (and repair), ENGINEER_SCHEMA (369-387): `mechanism`, `files_changed`, `set`, `tests`, `scope{when, starts}`, `tunable[{key, values}]` (optional).
- Reflector, reflect.py SCHEMA (32-58): `edits[{iteration, component, change, observed, verdict}]`, `experience{cue, lesson, evidence, confidence}`, `no_entry`, `why`.

Context keys the texts name and the loop writes (checked against loop.py): `recurring`, `repair_notes`, `mechanisms`, `history`, `candidate_pool`, `parameters`, `focus`, `parked`, `attribution`, `designs_last`, `draft`, `base`, `rejected_hashes`, `hypothesis`, `probe`, `manager`, `candidates` with `kind`, `ran`, `fixed`, `broken`, `scope`, `footprint`, `patch`, `imagined`; repair brief: `fixed`, `broken`, `first_difference`, `changed`, `outside`, `missed`, `draft_rounds`, `incumbent_rounds`.

Exact phrases kept because code or the driver reads them:
- "propose at most 4 interventions" in reviewer.md is string-replaced by `scoped_review` (loop.py:1671).
- Rule 28's "forced" slot logic, `skip` check names and `repeats` format are as in the original manager.md.
- Commands in engineer / composer / repair: the `try.py {FAMILY} --skills {SKILLS} --set ... --labels ... --ref {DIAG_INC} --out {ITER}/smoke | compose_smoke | repair_smoke` lines are unchanged (the out directory names may be checked by the leak-path logic around loop.py:1071-1130).
- "Never open anything under a directory named scene_rsi_20261016_hidden" is kept (the driver's leak check, loop.py:1059-1079, matches paths, not the sentence).
- `LEVEL 3` is a label I introduced in the Reviewer's `notes`; nothing parses it.

## 2. Where I was unsure whether the loop parses a phrase or supports what the text says

1. Level 3 requests: the schemas have no field for them. I route them through the Reviewer's `notes` and the Manager's `escalation` (ledgered at loop.py:1731 whatever the decision). If the Manager's `escalation` is only read when `decision == stop`, the request is still recorded but nothing acts on it.
2. Stance parameters (d, t, yaw, d2, t2, yaw2, here) are a `stance` argument of the bed, not obviously keys of the skill's parameter dict. The texts say "parameters" and "a `delta` that sets the factor" and rely on `parameters_of(inc)` listing them. If the backend passes the stance separately, `set` needs a key for it, or level 2 hypotheses cannot be probed.
3. The Engineer's guard check: I do not know the guard command or the base-package path inside the sandbox, so the text says "if the context gives you its command". The `load_skills` check line is the scene one (`scene_round.load_skills`, `p.apply_set`, `p.parameters`) which the seam plan says the SIMPLE package keeps; I dropped its `plan_for` print because the SIMPLE seed may have no `plan_for`.
4. `records.py features`: unknown whether it has columns for the SIMPLE rounds; the Reviewer text makes `measure` conditional on it.
5. Cards: I describe a card from the seam plan's record fields (class, phase, palm error, the four marks, release error, stance, time against limit, comply reason). The text says "where a card names an image, read it". If `backend_simple.cards` writes differently, the Reviewer's item 4 needs a sentence changed.
6. `{PLAN}`: for a SIMPLE family it may be None or the default scene plan (loop.py:402, `configure`). The texts say "if it names sections for you" instead of citing section numbers.
7. Gate rule (rules.md 25): I describe "net threshold (3 unless the verdict says otherwise), losses at most a third of wins" from caps `min_net`; the SIMPLE gate may differ, and with `repeat` > 1 or a non-exact sentinel the wording of rule 41 is only a paraphrase.
8. "Pool of 30 episodes per task" and "a round takes a minute or more" are from the brief and the seam plan's cost line; the real `N_DIAG` is the driver's (`{N_DIAG}` is used in the prompts wherever a count is needed).
9. Time limits as steps: 400 / 800 / 800 / 800 / 1200 at 50 Hz give 8 / 16 / 16 / 16 / 24 s. The "about 9 cm / 9 cm / 2 cm" lifts are the task files' reward at 0.9 of 10 / 10 / 2 cm (the XMove and TabletopGrasp files declare 0.9; BendPick 0.9).
10. Compliance wording comes from `comply_simple.py` (the Handover rule: right hand, then left hand alone for at least 0.2 s, before the first container contact; Locomotion: in a hand and touching only the robot at some moment; pick tasks: since 2026-11-20 a run of 0.2 s with the check holding, the object on a hand and off every support, starting inside the limit; a check-true step without that does not end the episode). The seam plan says `comply` is still being added to the bed; if its rules change, rule 40 changes.
11. The reflector.md and composer_imagined.md are generic: whether the SIMPLE loop runs the Reflector or the imagined model at all is not known.

## 3. Scene-specific passages of the originals that I dropped (or replaced)

reviewer.md: the apartment, "mars" and two-finger grippers; the plan's sections 0, 2, 3, 7, 8 and D13; the failure-class list too_wide / side_reach / pinch_reach / walk_off / knocked / slip / fell / other and the card contents (support heights, cover above, hand corridor, top-down and side images, wrist path against reference); `scene_meta.json`, `scene/`, `world`, `stage.py`; the NPZ trace format; graspability.py, body.py (15,120-reach table, AUC numbers), choose.py, keypoints.py, prep.py and their `pred_*` fields; the "jaw" and the jaw geometry; design probes (`design`, factors spawn_bearing, spawn_distance, object_yaw, primitive) and `designs_last`; infeasibility claims (kind, bound with a check file, exits, attempts) and the `{TMP}` check files; `--steps` / `--npz` of try.py and the 40-starts / 240-rounds numbers (replaced by "the tool prints it").

manager.md: `reopen` and `reopen_claim`, parks, `gap_candidates`, `claims`, the D13 stop logic and the physical-claim note; the acceptance-gate line "b - c >= 3 and c <= b / 3 on {N_ACCEPT}" with the example about one object's starts (replaced by a general statement); "new skills the family file's `about` names" (rule 24).

engineer.md: `stage()` and its checks and `on_fail`; graspability, choose, two_hand, prep (`@prep.register`), floor, keypoints, `body.best_level`; `recipes/` with SIGNATURE and `match`, and the string-constant ban on object keys (kept in spirit: no names, no episode identity); the `plan_for(task)` and `held(<object>)` load check; `src/locomani/tasks/scene` and `scene_meta`; upper-case module constants as the parameter convention (the SIMPLE parameters are keys of a dict); the `reopen` paragraph.

composer.md: wording about the 6/10 vs 2/10 repeat and rules.md 8, "same model biases" kept; `adopted_pool`'s `older_rule` remark (rule 25's version history) dropped.

composer_imagined.md: all measured numbers (640 rounds, 58 % / 78 %, per-class counts, rank correlation 0.17) and the `extra_starts`, `doubtful`, `body_p_miss`, `check_on_diagnostic_starts` description; replaced by a short text that says it is unmeasured here.

repair.md: only "first step" -> "first phase", the added noise caution and the guard sentence.

reflector.md: "apartment", the jaw/shelf example kept as a generic cue example; added the compliance and level remarks.

rules_simple.md: rules 2 (diagnostic redraw, 300 s limit), 12-17 (Dex1 jaw geometry, walking at 0.25 m/s versions), 21-24 (infeasibility, parks, `held`), 26-27, 29's library numbers, 30's recipe/transfer checks, 31's `stage()`, 32's class definitions and tools (rewritten as rules 32 and 43), 34-36, 37's mode-000 detail, and the version history in every rule.
