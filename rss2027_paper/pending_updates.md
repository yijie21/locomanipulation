# Pending draft updates (batch; do not apply until 5090 says its work is finished)

The user paused per-message updates (2026-11-26 ~20:20). Apply all of these in one pass after asking 5090 for the consolidated state. Re-check every item against rss_status.md at that time.

## From 5090, ~20:25 (answers to my 20:15 questions)
- Q1 (H3 falsifier change): RESOLVED. The user confirmed the shortened falsifier on 11-26 in the paper session, and it is applied in main.tex.
- Q2: r = 0.85–0.89 is citable (property of the model's correction, not perception; noted in status §1.3c). Remove its dnote in H1.
- Q3: added to status §1.3c, now citable:
  - lmlroute A, decision set, first arm contact before the target: ScaleBFM 379/503; SONIC 84/422 (+149 before the first command); HEFT 368/501.
  - 7-task sign p for the paired R − A close-time hand distance: 2.61e-27, 1.66e-32, 3.43e-6.
  - The lmlroute entry is synced to ~/research-notebook/experiments/ (final, review pasted, c4718446). Cross-check Table 1 and H1 against it.
- Q4: the HEFT gait mismatch still holds (lmlroute HEFT task episodes use gait_heft, lrt_run.py:98, walk_ctrl :241; training data use hg_walk). Keep it as an internal dnote only; no paper text. The nowalk follow-ups are paused.
- Q5: bowl in the formal run = 55/90 (61 %) for all three controllers, plan failed after the look, identical across four arms, excluded by rule. Write "55 of 90" for all three; drop the "offline recount" dnote.

## Scope change from the user (relayed by 5090, ~20:25): "不要再发散"
- Remaining work is only the benchmark × controller table, self-model vs. direct command, in SIMPLE, HA and LML.
  - LML: done (Table 1).
  - SIMPLE: ScaleBFM done (simplecmp, under review); HEFT and SONIC being set up.
  - HA: A vs. R on task episodes, three controllers, after HA data collection.
- Paused, not run: E2c (HA free-space), walkmix / in-place collection, the reviewers' falsification experiments.
- E4 in the paper becomes "A / R columns of Table 1 in HumanoidArena"; C / CR in HA not planned. dnote: "HA: A vs R planned, no date yet".
- Implications to check at batch time:
  - E1 (free-space 3×3): is it still in scope? E2c is paused. Ask 5090 which parts of E1 (E2a LML, E2b SIMPLE) still run.
  - E3 (faildiag): does it stay in the plan?
  - The new SIMPLE/HA benchmark × controller table may replace Table 2 or join Table 1.

## Figure 1 render request (the user chose to bundle it with the batch, 11-26; SENT to 5090 as a queued heads-up at the user's request, msg 084c5b6f; output to ~/Research/rss/paper/fig/teaser/ + README). AMENDED (msg 31f08c2c, user): start NOW if the 5090 is idle and report the render on its own when done. MuJoCo EGL renderer (offsamples, shadows, light background); ghost via alpha + blue tint or composite layers; replay or same-seed rerun with an outcome-reproduction check.
- **Episode choice, by a fixed rule (not cherry-picked):**
  - a LocoManiLab lmlroute paired episode where A fails and R succeeds;
  - both close-time hand errors near their arm's median;
  - controller: ScaleBFM preferred;
  - record the episode id and the rule used.
- **Render:** the frame at hand closure, from one fixed camera, for A and for R:
  - same camera and lighting;
  - also an R frame with the robot only (transparent background), for the ghost overlay.
- **Numbers for panel (b):** target grasp point p*, the R command c*, the A landing and the R landing, in the camera frame (pixels) and in the world frame.
- **Also in the batch:**
  - data-collection time and training time per controller;
  - confirmation that no task episodes are in the training data;
  - discordant counts (A only / R only) for 7 tasks x 3 controllers (Fig. 2 / pertask);
  - whether the SIMPLE part of E3 still runs.

## Interim snapshot from 5090 (~14:50Z 11-26; the user asked 5090 for it; not the consolidated message)
- **SIMPLE (120 eps, 4 tasks x 30)**, cells = 45 s / official limit (TabletopGrasp 45 s / official):
  - ScaleBFM: A 83/73 (18/18), J 95/81 (30/26), kin_rr 95/82 (30/27), S 94/78 (29/23), R_D 108/78 (18/18). R_D reviewed "supported with limits".
    - R_D per task, 45 s: BendPick 30, TabletopGrasp 18, XMovePick 30, XMoveBendPick 30.
    - R_D fixes one failure mode (~6 cm sideways offset on entry), shown on 10 layouts only.
    - +25 over A at 45 s, +5 at official. BendPick official limit = 8 s, so 0 for every arm.
    - Route arms win TabletopGrasp 30 vs 18 (the direct reach hits the table edge).
  - SONIC: A 29/18 (10/10), J 34/20 (15/12), kin_rr 36/23 (17/15), S 33/17 (14/9), R_D pending (~11-27 03:00Z).
  - HEFT: A 58/55 (0/0), R_D 62/59 (0/0). Reviewed: +4, n.s. (p 0.29 / 0.125).
    - Model: LML-trained D_heft; gate passed on XMovePick and XMoveBendPick only.
    - Per task: XMovePick 29->30, XMoveBendPick 26->29, BendPick 3->3 (official 0), TabletopGrasp 0->0 (wrist hits the table edge).
    - HEFT A enters the main metric via a post-hoc relaxation of "same look" (logged deviation).
  - **My arithmetic, to confirm with 5090 (not citable yet):**
    - At 45 s on the three non-TabletopGrasp tasks: R_D 90/90 vs A, J, kin_rr and S all 65/90. The route arms lead R_D only through TabletopGrasp (table edge).
    - At the official limit on those three tasks: R_D 60 vs 55 for every other arm.
    - So aim and routes fix different failures.
- **LML Table 2b, per-task discordant pairs (A only / R only, p), all 7 tasks x 3 controllers**: this is the Figure 3 data.
  - Checked against status §1.3c: decision-set and without-candle totals match.

  | Task | ScaleBFM | SONIC | HEFT |
  |---|---|---|---|
  | candle | 0/65 (5.42e-20) | 0/4 (0.125, ceiling) | 2/26 (3.03e-6) |
  | apple | 3/4 (1) | 1/5 (0.219) | 0/0 |
  | bowl (excl.) | 0/0 | 0/3 (0.25) | 1/0 |
  | salt | 3/31 (7.66e-7) | 2/11 (0.0225) | 6/0 (0.0312) |
  | statue | 0/0 | 0/0 | 0/0 |
  | dumbbell | 8/22 (0.0161) | 0/6 (0.0312) | 1/0 |
  | watering can | 8/16 (0.152) | 1/11 (0.00635) | 0/0 |

- **HumanoidArena** (P&PBox + DoubleDesk, seeds 400-459, A vs R):
  - Primary metric: official success with a 3x time limit.
  - Pre-registered: if A <= 1/20 in the smoke test, switch to grasp-and-lift.
  - ETA (Beijing): ScaleBFM 11-27 09-12, HEFT 16-19, SONIC 11-27 22:00 to 11-28 04:00.
- **E3:** the SIMPLE part is NOT run (narrowed scope). The LML free-space part (E2a) finishes tonight. Plan text: E3 = LML only.
- **No task data:** for HEFT's SIMPLE cell, the reviewer confirmed that the training reset ids and the SIMPLE episode ids are disjoint. The full confirmation and the cost numbers come in the consolidated message.
- **Before using any of this:** verify it in rss_status.md §1.3c / §2 and the entries.

## APPLIED 11-26 (user: "都做": write the snapshot's reviewed parts now + ask 5090 for SIMPLE per-task verification, msg 95f57422)
- Q2: r dnote removed (H1). Q3: 7-task sign p moved into the src comment. Q5: bowl src comment updated (formal run 55/90, all three).
- Fig. 3 (fig:pertask) is now drawn in TikZ from status §1.3c per-task table (7 x 3, * = per-task McNemar p < 0.05).
- Table II (tab:simple), SIMPLE 45 s / official: A, Route (J), Route+ref (kin_rr), R (R_D). S (jrefwm) left out of the table. SONIC R = red "?".
- SIMPLE paragraph after the routes paragraph in V-B. dnote: the 48/60 vs 28/60, p 0.0078 and BendPick 30 vs 10 come from status §3 simplecmp raw numbers (review status of that test to confirm; the simplecmp entry is not in ~/research-notebook/experiments/ yet).
- Plan: E1 = SONIC in SIMPLE; E2 = HA with the registered protocol (P&PBox + DoubleDesk, seeds 400-459, 3x limit, switch rule); E3 = LML only.
- Still open: per-task verification (msg 95f57422) -> then decide whether to write "aim and routes fix different failures"; SONIC R_D; HA; cost numbers; "no task data" for all models; E3 results.
- Page count: 7 total (body ends p5 left column; refs p5-p7).

## Figure 1 v1 render (11-26 23:15): REJECTED by the user
- v1 = ScaleBFM candle ep 117, chosen by the median rule (fig/teaser/README.md). User: the head isn't visible and the viewpoint is poor. They want an upward reach to an object above, seen obliquely from above, showing head, hand, body and object.
- Re-render requested (msg 25363830). Episode rule: grasp point above shoulder height at closure (fallback SONIC/HEFT, then the top 10 % of grasp heights), then the same median rule. Camera: in front, within ±60° of the robot's facing; 35-55° down; head >= 80 % visible. Contact sheet first so the user can pick the view. Ghost: full body at alpha 0.30 plus an arm-only variant. Output to fig/teaser_v2/.
- v1 crops I made (fig/teaser_a.jpg, fig/teaser_b.jpg) are unused; main.tex Fig. 1(a) is still the placeholder.
- v2 episode (5090 ~23:40, chosen by the user in that session): ScaleBFM salt ep 90.
  - No LML object sits above standing shoulder height (~1.07 m); the highest is salt at 1.026 m. The 18 pairs that pass "above shoulder at closure" pass only because the torso leans forward (~29°).
  - Ep 90 by the median rule within those 18 (tie with ep 102, lower id): A 4.66 cm, fails on furniture contact; R 1.99 cm, succeeds. Reproduction bit-identical.
  - Caption must NOT say "upward/raised reach": it is a forward-leaning reach to counter height. Rule text: "among pairs whose grasp point is above the shoulder at closure, closest to both arms' median".
  - Contact sheet next (its own message) -> user picks the view -> final render in fig/teaser_v2/.

## 11-26 ~23:50: USER DECISION: LocoManiLab results out of the paper (for now)
- User in the 5090 session: "那现在先不报 locomanilab 上的结果吧。如果现在有 locomanilab 相关的实验的话就先都停掉". In this session they confirmed "包括论文，先移出". 5090 stopped all LML work (E2a, Fig. 1 v2, LML videos).
- Done in main.tex (backup with LML: main_with_lml.tex):
  - abstract and intro rewritten SIMPLE-first;
  - hypotheses H1 aim / H2 success / H3 deployment (former H3 training support removed: SIMPLE R_D = S_D bit-identical, no out-of-reach commands);
  - Method Data: empty scene, no task episodes or scenes, task-style walk displacements (5090 confirmation);
  - benchmark paragraph = SIMPLE;
  - results: V-A model accuracy plus the 6 cm offset plus HEFT entry error, V-B Table I SIMPLE plus text, V-C transfer;
  - plan E1 SONIC/SIMPLE, E2 HA; E3 removed;
  - Fig. 1(c) = SIMPLE per-task ScaleBFM A vs R at 45 s; Fig. 3 removed.
- Open dnotes: transfer models' source named "a different simulator" (confirm wording); simplecmp decision-set numbers' review status; per-task verification (msg 95f57422).
- "No task data" for the SIMPLE ScaleBFM model: confirmed (5090, simplecmp review). Abstract dnote removed.
- Fig. 1 v3: the user wants a fancier custom scene ("不使用这些场景，我们自己搭建也行"). 5090's plan:
  - hand-built MuJoCo scene (plain room, tall shelf, object on an upper shelf);
  - ScaleBFM with the SIMPLE-recollected model; in-place reach; GT target; hand closure <= 2 cm;
  - 20 placements, then the median rule; contact sheet first.
  - Estimate 3-3.5 h to the contact sheet plus 0.5 h final. NEEDS THE USER'S GO before 5090 starts.

## 11-26 ~23:45: SIMPLE per-task verification back (status §二.1, independently reviewed): APPLIED
- Reviewed mechanism wording used in V-B. The BendPick offset 5.5-6.9 -> 1.1-1.7 cm and the HEFT counter-example (4.5 cm forward offset) are in V-A.
- Table I caption: on 3 tasks the planner finds no route, so the route arms execute the direct reach (cuRobo IK_FAIL on all 90 episodes). R's official counts exclude computation time.
- Protocol: the sim is paused during SelfAim's computation (ScaleBFM median 3.3-6.7 s per episode), not counted toward official limits. Internal only: counting it, R_D official 60 -> 22 on the 3 tasks, HEFT 59 -> 2.
- The registered P1 is used (+8/-0 over 20 layouts, p 0.0078); the post-hoc three-task subset is not used. The "different failures" claim is NOT written.
- HA smoke: A official 0/20, so grasp-and-lift (>= 5 cm, 1 s) is the primary metric; E2 text updated.

## 11-26 ~23:55: user "1. 开始 2. 不用了，就直接把这个 locomanilab 的相关内容删掉"
- Fig. 1 v3: GO sent to 5090 (msg af11b4ea). Output to fig/teaser_v3/; contact sheet first.
- All LML-related content deleted from main.tex:
  - the V-C transfer section (LML -> SIMPLE / HA ratios);
  - HEFT's SIMPLE R_D (LML-trained model) everywhere: Table I row, H1 / H2 sentences.
  - Kept: H3 evidence = ScaleBFM's SIMPLE-collected model passing the gate (merged into V-A).
  - Scope sentences: "SONIC in SIMPLE and all three controllers in HumanoidArena".
- Backups: main_with_lml.tex (full LML version) and main_simple_with_heft.tex (SIMPLE-first, with HEFT + transfer).
- Now 5 pages; no "LocoManiLab" in the PDF text.

## 11-27 ~00:20: Fig. 1 v3 contact sheet; user picks (sent to 5090, msg 203cc656)
- v3 at 1.26 m: 20 placements; A-miss/R-hit 16, both hit 2, both miss 2, A-hit/R-miss 0. Chosen: placement 17 (A 2.71 cm, R 0.51 cm). A topples the vase in 19/20, R topples 0 and lifts 18/20.
- User: two panels side by side (A toppled vase | R holding vase, same view and moment), raise the grasp point to ~1.35 m and re-run the 20 placements under the same rules, view #2 (az -30, down 45). Finals next.
- When the finals arrive:
  - Fig. 1(a) = two panels (+ optional hand inset);
  - caption: "scene built for illustration, not part of the benchmarks", placement rule, counts;
  - (b) is still a schematic; consider real p*, c* from the points JSON.

## 11-27: Fig. 1 v3 at 1.34 m stopped; user chose back to 1.26 m (msg 7f222070)
- 1.34 m: A-miss/R-hit 11, both hit 8, A-hit/R-miss 1; A misses marginal (2.03-2.36 cm); lifts A 14/20, R 20/20; falls A 5/20, R 0/20; no topple on entry. Not used.
- Old "A toppled 19/20" at 1.26 m used a bad rule (counted vases held tilted). Do NOT cite it; wait for the recount (fallen = not touching hand AND tilt > 60° or below the shelf top, 2 s after close).
- Finals: 1.26 m, placement 17, about 1 s after close, view #2 or -40. The caption must say: illustrative scene, not part of the benchmarks.

## 11-27: HA ScaleBFM cell DONE + reviewed (selfmodel_hatask_20261126, 59a6aa77 / ee70aa1d). NOT applied (batch).
- Primary grasp-and-lift (>= 5 cm, >= 1 s), 3x limit, seeds 400-459 x 2 tasks: A 19/120, R 13/120, McNemar p 0.345, "no difference detected". Exact 95 % CI R-A [-13.3, +4.4] pp, so the registered prediction (+3 to +15) is essentially ruled out.
  - P&PBox A 14/60, R 8/60 (p 0.238); DoubleDesk 5/60 each. Official success 0/120 for both arms.
- Hand: R closer on the P&PBox pre-grasp (L -2.76, R -4.59 cm), not on the decisive reach (clamp +6.19 cm, descend +0.71 cm). 229/240 of those hands were in contact; "the model can't correct" is unsupported.
- Context: falls 40 % (DoubleDesk A 70 %, R 77 %); P&PBox half stop at HA's find_shelf check; seed halves differ (confounded with GPU).
- Implication for the draft: in HA, H2 fails for ScaleBFM so far. The abstract / intro / plan / discussion need reconsideration at batch time. HEFT and SONIC come next.

## 11-27 ~08:30: Fig. 1 v3 finals integrated (msg a9ec4b49 to 5090)
- (a) = fig/teaser1_A.jpg | fig/teaser1_R.jpg, cropped from teaser_v3/{A,R}_full.png (920x1140+880+190, 800 px wide). Labels "direct" / "SelfAim", "vase knocked over" / "vase grasped".
- (b) = to-scale top view from teaser_points.json (0.42 cm per cm, 2 cm hit circle).
- Caption counts: A knocks the vase over 19/20, SelfAim lifts 18/20. R's literal fallen 2/20 (shelf penetration 0.15 mm) not cited. The caption says "illustrative scene".
- check_figures: 0 warnings; 5 pages.

## 11-27 09:45 Beijing: user stopped all HumanoidArena work (relayed by 5090)
- User, verbatim: "现在先停下所有 humanoidarena 上相关的任务".
- Stopped: HEFT HA formal run (~6/120 pairs, will not be read or reported), SONIC HA data collection, HA video rendering.
- ScaleBFM HA result (reviewed, already sent) stays: grasp-and-lift A 19/120 vs R 13/120, p 0.345.
- HEFT and SONIC HA cells will not arrive unless the user resumes.
- NOT applied to main.tex. Whether HA stays in the paper is the user's call; asked the user 11-27.
- HA appears in main.tex at: abstract (l.40), intro para 6 (l.65), protocol (l.214, last sentence), E2 + dnote (l.262-263), discussion (l.271). HEFT has no experiment left in the paper without HA (its SIMPLE cell was LML-trained and already removed).
- If HA stays as a plan, the E2 text cannot describe the ScaleBFM arm as future work: that arm has already been run.
- User answer (11-27): "先不动，等我想好". main.tex keeps its HA text unchanged until the user decides.

## 11-27 ~10:10: SONIC SIMPLE R_D done and reviewed (NOT applied; falsification run pending)
Source: selfmodel_simplesonicdata_20261126 (result bd23da61, rev 2 025ee7e9, review c2922905); rss_status §二 l.323/329-330/355 cross-checked, numbers match the 5090 message.
- Paired same-machine A -> R_D (H20b, SONIC net on CPU), 45 s / official: XMovePick 7->7 / 7->7; XMoveBendPick 4->5 / 4->5; BendPick 6->16 / 0->0; TabletopGrasp 13->8 / 13->8. Total 45 s 30->36 (+5.0 pp [-6.7, +16.7], p 0.71); official 24->20 (p 0.52). No difference detected.
- Gate: only XMoveBendPick passes on re-gate (1.039x, UB 1.344; reviewer: marginal). BendPick gain is not credited to the model (gate failed). TabletopGrasp registered as "correction harmful" (palm error 6.6 -> 12.9 cm).
- First gate failed (4.4-5.9x val); `walked` column zeroed after an offline diagnostic, before any R_D episode. 52 % of reaches still out of range (leg joint velocity |z| > 6).
- Paper constraints: this cell's A (30/24) is a CPU re-run, not the main-metric 29/18 (4090, GPU); 39/120 episodes differ. Table I cannot show two "SONIC direct" numbers without saying so. Don't cite the BendPick gain; use "no difference detected" until the falsification run (two more A + R_D runs on other float paths, pre-registered) reports.
- Consequence for main.tex: E1 (SONIC in SIMPLE) has now been run, so it cannot stay as a future experiment. With HA on hold (user "先不动"), the Plan section has no un-run experiment left. Same pending decision as HA.
- Minor: status says HEFT HA formal run stopped at 9/120 pairs (5090 message said ~6). Not reported either way.

## 11-27 ~10:54: SONIC SIMPLE replication done and reviewed (NOT applied)
Source: rss_status §二 l.338-342 (revision 3 2cbc6198 committed before the runs; results 498b5d7c; review fa739c1e). Cross-checked, numbers match the 5090 message.
- Runs, A -> R_D, 45 s / official: C (CPU 1 thread) 30->36 / 24->20; G (net on GPU) 29->36 / 20->16; T (CPU 2 threads) 30->31 / 23->16.
- Pooled sign test over 40 layouts: 45 s 89->103 of 360 (+3.9 pp [-5.6, +13.9], p 0.50); official 67->52 (-4.2 pp). No difference.
- BendPick difference replicates (pooled 22->51 of 90, layouts +8/-1, p 0.039), but the gate fails in all three runs: may be reported as a difference, never as a model-driven gain.
- TabletopGrasp harm replicates (pooled 37->19 of 90, p 0.18). XMovePick -3 in T only, not replicated.
- T agrees with C on 116/120 episodes (near repeat); G is the only independent rerun (85/120 agree). The task that passes the gate changes between runs (C, T: XMoveBendPick; G: XMovePick).
- 5090's suggested reading: on SONIC in SIMPLE, the self-model correction shows no measurable gain; the model is trained on in-place data only, and post-walk start states are outside its range. Any SONIC caption carries the A-number caveat (main-metric A 29/18 vs these re-runs).
- The SONIC SIMPLE cell is now final, so it is ready for the batch update, pending the user's decision on the Plan/HA structure.
