# SelfAim rewrite report (bestpaper-writing, Mode B, 2026-11-26)

- **Draft:** `main.tex`; the pre-rewrite version is kept in `main_pre_rewrite.tex`.
- **Numbers:** unchanged from the status file at the time of the pause. Items received during the pause are still in `pending_updates.md` and are applied in the batch update.

## 1. Draft reverse-engineered into the intake fields

| Field | Old draft | Status |
|---|---|---|
| Task | Aim a frozen humanoid WBC's hand for loco-manipulation grasping | present |
| Dominant prior approach | Retrain or fine-tune the WBC; closed-loop hand feedback; simulate the controller at run time (TTCO) | vague (scattered in related work) → now intro paragraph 3 |
| Gap and its reason | "hand does not stop where it was told"; tracking trades accuracy for balance | present, no concrete example → route-deviation 6.4–14.3 cm added (status §3) |
| Closest work + checkable difference | Related work only; no contrast in the intro | missing → intro paragraph 5 (OSRAM, TTCO) |
| Key insight | Implicit | missing → "the miss repeats in a given state, so it can be predicted and cancelled" |
| Method components | Self-model, inversion S/G/R | present (G dropped earlier) |
| Results with numbers | Present, but four hypotheses listed in parallel | present → restructured around the claim |
| Efficiency / cost | None | missing → `\TODO{}` in Method (data and training time) |
| What is surprising | Not stated | "no task data, yet grasp success rises on unseen tasks" |

## 2. Claim, tension, insight, handle, route

- **Claim:** We show that inverting a self-model, learned from a frozen WBC's own free-space reaches, aims the hand of three frozen controllers with different command interfaces. It raises grasp success for all three without task data, hand sensing or retraining, because the controller's miss repeats for a given body state and command.
- **Tension type:** obvious-fix-too-costly (secondary: tradeoff, tracking vs. balance).
- **Insight:** "The miss itself points to a cheaper route, because it repeats: in a given body state, a frozen controller sends the same command to nearly the same place."
  - Evidence: r = 0.85–0.89 (status §1.1); model error 0.7–1.7 cm on reachable commands (status §1.3).
- **Handle:** "fix the command, not the controller". Used once, in intro paragraph 4. The title keeps the user's wording.
- **Closest work:**
  - OSRAM: mechanism, but one bipedal self-trained policy, low-dimensional commands, response measured at run time, judged by tracking error.
  - TTCO: simulates the controller at run time; SelfAim moves the simulation offline.
  - HERO: uses measured end-effector feedback.
- **Stance:** "Aiming a frozen controller thus costs one data collection per controller and leaves its weights untouched." (end of intro)
- **Route:** L1 method / `cost` (card: mip-nerf).
  - Abstract: Setting → Gap → Method → Result.
  - Figure 1: contrast schematic (direct execution vs. SelfAim).

## 3. Claim map

| # | Claim (as written) | Evidence | Wording level |
|---|---|---|---|
| 1 | SelfAim brings the hand closer for all three controllers | status §1.3c: paired R − A, 7 tasks, CIs exclude 0 | "brings … closer" (consistent across three) |
| 2 | Outperforms direct execution in grasp success for all three | Table 1, McNemar p ≤ 0.006, pre-registered decision set | "outperforms", scoped to LocoManiLab |
| 3 | More than doubles ScaleBFM's success | 15.4 → 36.9 % (2.40×) | headline number |
| 4 | Needs no task data | Method: data are free-space resets | **to confirm with 5090** (dnote) |
| 5 | Same recipe works across command interfaces | three controllers, three interfaces | "applies to controllers with different command interfaces" |
| 6 | New simulator = one data collection | SIMPLE re-collection: 0.36 / 0.40 / 1.07 cm on 3 tasks | scoped to "three tasks" |
| 7 | The miss repeats from reach to reach | r = 0.85–0.89 (status §1.1) | observation |
| 8 | Hands stray from collision-free routes, 6.4–14.3 cm | status §3 lmlroute, reviewed | concrete example |
| 9 | Inversion must stay inside support | ~20 cm vs 0.7–1.7 cm; 19.87 cm n 75 | results section |

## 4. Change table (main changes)

| Original | Revised | Reason |
|---|---|---|
| Abstract: "We study a simple, sensing-free remedy … We state four hypotheses …"; then H1–H4 as separate sentences, ending "We detail the experiments …" | Setting → gap (obvious fixes costly) → "We present SelfAim, which corrects the command of a frozen WBC instead of its weights" → mechanism → black box, no task data → result (closer hand, success for all three, 15.4 → 36.9 %) → deployment (SIMPLE) → hypotheses in one sentence | Award abstracts follow Setting → Gap → Method → Result with 1–2 headline numbers; the old one listed hypotheses in parallel and never stated a thesis |
| Abstract H2: "Accuracy only partly turns into success … third controller's gain comes entirely from one task" | "outperforms direct execution in grasp success for all three" | The positive main result (pre-registered) is the claim. HEFT's candle-only gain stays, stated plainly in Sec. V-B, not in the abstract (skill: scope mixed results to the wins) |
| Abstract H4: "tied to the controller, its command interface and the simulator, not the body" | "A new simulator costs one data collection" | User: binding is an engineering detail |
| Intro paragraph 2: no example | Route-deviation example (6.4–14.3 cm) | Concrete example in paragraph 1–2 (36/55 award intros) |
| Intro: no obvious-fix paragraph | Paragraph 3: retraining, hand feedback, run-time simulation, each with its cost | `cost` route: rule out the obvious fix |
| Intro paragraph 4: "We ask whether a frozen controller can instead be aimed" | Insight as an observation, handle, method, reachability as the answer to the support problem | The insight comes before the method, stated directly |
| Intro: no contrast with closest work | Paragraph 5: OSRAM, TTCO | Reviewers look for this paragraph (32/55) |
| Intro last paragraph: "organize the paper around four hypotheses … a negative result we consider useful" | Results with headline numbers, hypotheses in one sentence, stance | The last paragraph carries results or a stance, not an outline |
| Related work: itemized list | Prose; ends with SelfAim's three differences | Space; same citations |
| H1: S/R | R only in the falsifier | No S arm in the main run |
| H2: "Hand accuracy turns into task success only partly …; falsified if success gains are proportional to the accuracy gains" | "Aiming raises grasp success over A for every controller without task data, by an amount that depends on the task; falsified if R does not raise success on a controller's pre-registered task set" | Testable and matches the thesis. **User to confirm** |
| H4 (specificity) | H4 (deployment): one data collection; falsified if the re-collected model fails the gate on most tasks | User framing. **User to confirm** |
| Sec. V subsection titles: descriptive | Claim-first ("SelfAim aims all three controllers", "aiming raises grasp success for every controller", …) | Results paragraphs open with the finding |
| Routes paragraph | Framed as the obvious alternative; deviation = the command–execution gap | Ties the C/CR arms to the thesis. No "stack/combine"; no R-vs-C claim (not tested) |
| Table 2 + interface/velocity/variant F paragraph | One sentence: ratios 1.06–6.62, five of six above the gate | User: engineering detail; saves space |
| Plan E1 free space, E2 SIMPLE + HA, E3 failure diagnosis, E4 HA routes | E1 SIMPLE (A vs R), E2 HA (A vs R), E3 free-space map | Scope "不要再发散" (user, relayed 20:25) |
| Discussion "If E1 shows …" | Division of labour stance; what E1/E2 will establish | End with a stance, not conditionals |
| — | Figure 1 (TikZ contrast schematic); Figure 2 placeholder (per-task gains) | No figures before; Figure 1 type matches the `cost` route |

## 5. Removed from the text

The numbers stay in `main_pre_rewrite.tex`.

| Removed | Why |
|---|---|
| SIMPLE fixed geometry (3.97 / 1.74 / 1.28 / 1.26 cm; model ties a calibration) | Does not support the claim, and invites a constant-offset baseline the user decided not to run. **Ask user** whether to keep it as one sentence |
| "In SIMPLE, success cannot reward ScaleBFM's aim (route 28/28)" | Defensive; SIMPLE success now comes from E1 |
| Bowl palm-down paragraph | Not about aim; the bowl stays only as the exclusion rule (55 of 90) |
| SIMPLE second look / stance correction | Not about the self-model |
| Old E3 (SONIC turn, HEFT drift) | Out of scope per "不要再发散" |
| Old E4 (HA routes) | C/CR in HA not planned (5090 20:25) |
| Data recipe subsection | Kept as two clauses in Method (24–32 %, 14–26 %); learning-curve sentence dropped |
| "For SONIC no model contribution is shown; underpowered" | Self-undermining |

## 6. To confirm or fill

1. ~~H2 and H4 rewording; H3 falsifier; SIMPLE fixed-geometry removal~~: **all confirmed by the user (11-26)** as recommended.
2. **"No task data"**: confirm with 5090 that training data contain no task-episode data.
3. **Cost numbers** (`\TODO{}` in Method): simulation time per controller for data collection; training time; optional inversion latency per reach. Ask 5090 in the batch update.
4. **SAW (arXiv 2409.10308):** read it, then add the bib entry and replace `\TODO{cite SAW}`, or drop it.
5. **Figure 2 data:** per-task A-only / R-only counts for all 7 tasks × 3 controllers from the lmlroute entry. Batch update.
6. **E3 scope:** is the SIMPLE free-space part still running? (5090)
7. **Pending batch items** (`pending_updates.md`):
   - r dnote removal;
   - 7-task sign p;
   - bowl 55/90 for all three;
   - first-contact counts;
   - simplecmp numbers after review.
8. **References run > 1 page** (refs start on page 4, end on page 6). The body is now ~3.6 pages, so there is room. The RSS reference limit still needs a decision, e.g. abbreviated author lists (`et al.`) or a compact bib style.

## 7. Notes for internal limitations

These are not written in the paper.

- HEFT's gain comes from one task (candle); on its other tasks R is worse (0 vs 7, post hoc). This is stated in Sec. V-B but not in the abstract or intro.
- Absolute success stays low: R 36.9 / 10.4 / 13.3 %.
- SIMPLE fixed geometry: the model ties a tracked-reference calibration (removed from text).
- TabletopGraspMP: free-space data miss table-edge contact (28–40 cm); this one stays in the text as a scoped result.
- Transfer ratios 1.06–6.62; SONIC's remaining factor of ~3 in HA is unexplained (removed).
- The free-space pilot is not reviewed or pre-registered (dnote stays).
- The SIMPLE re-collection uses one seed and ScaleBFM only.
- Everything is in simulation.

## 8. Figure plan (revised 11-26: teaser as the user asked)

| Fig | Place | Claim | Type | Status | Material needed | Caption takeaway |
|---|---|---|---|---|---|---|
| 1 | Page 2 top, full width (figure*) | SelfAim aims a frozen WBC by aiming off the target | contrast-results + mechanism + plot | (a) placeholder; (b) schematic drawn; (c) drawn from Table I | (a) 5090 render of one paired lmlroute episode chosen by rule (A fails, R succeeds, errors near medians): A solid, R translucent blue, hand inset. (b) real p*, c*, two landings (batch) | "SelfAim aims a frozen whole-body controller by aiming off the target." |
| 2 | Sec. IV top, column width | Overview: offline self-model, run-time inversion, frozen WBC | overview (TikZ) | drawn | none | "SelfAim overview." |
| 3 | Sec. V-B | The gain concentrates on a few tasks, differently per controller | grouped paired bars | placeholder | per-task discordant counts, 7 tasks x 3 controllers | "Where aim pays off." |

- The user's idea was a G1 reaching with a miss, overlaid by a translucent G1 that hits.
- Three changes were added to it:
  - a hand zoom, since cm-level offsets are invisible at full-body scale;
  - the "aim off" mechanism panel, since a ghost overlay alone conventionally reads as the commanded pose;
  - the success plot, so the first page carries the result.
- The render request is queued in pending_updates.md; the user chose to bundle it with the batch.

## 9. Checks

- `check_draft.py`:
  - abstract: no issues (203 words, 9 sentences, mean 22.6 words);
  - introduction: 0 errors, 0 warnings, 1 note (a mechanism sentence without a number, which is fine). 592 words, 6 paragraphs.
- `check_figures.py`: 0 errors, 0 warnings, 2 notes (Fig. 1a and Fig. 3 placeholders).
- Compiles without errors or overfull boxes; 6 pages; body ends on page 4.
