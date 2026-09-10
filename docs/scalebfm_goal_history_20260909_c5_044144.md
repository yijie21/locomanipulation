# ScaleBFM visual VLA Goal — current checkpoint

Updated 2026-09-09 04:23 UTC. Goal ACTIVE; final VLA comparison is incomplete.
On recovery read docs/STATE.md, this file, then goal_plan/design as needed.
Do not repeat completed experiments to recover context.

## Current live work: C5 training C17; B17 video complete

- Driver1470213/session89332 LIVE; C17 trainer1890281 LIVE, step197/3000.
  C17 started04:13:01UTC. Observed total optimizer budget 6197/18000.
  A17 and B17 training/selection COMPLETE; remaining C17,A29,B29,C29, lock and
  confirmation. Old B17 trainer1474248 and evaluator1867812 ended.
- Root outputs/experiments/scalebfm_visual_goal_20260908/. Authoritativestages
  pipeline_v2_state.json; currenttrain_C_s17.driver.log. Do notrestartalivejobs.
- A17training3000/3000exit0 at2026-09-09 01:24:14UTC; runtime16832.36s (~4h40m),
  finalloss.0157,trainmeanloss.08131. Finalcheckpoint3000auditPASS (537optimizer
  states/scheduler3000,fourRNGgroups,1030tensors,processorunchanged). Onlyfinal
  checkpointretained. Allsix500-stepauditsPASS; JSONevidencekept, oldweightsremoved.
  Artifacts checkpoint_A17_3000_audit.json,model_A_s17_completion.json.
- A17selection60 COMPLETEexit0 at01:34:42UTC:13/60 (21.7%), normal11/48,
  recovery2/12 (blackout2/3,otherkinds0/3); highnear8/20,lownear3/20,deep2/20.
  Registeredpostureviolations49/60; all13successeshavesomepostureviolation.
  Failures24precloseknock,17physicswarning/fallcategory,6timeout; actualMuJoCo
  warningcases0. No-close27, stationary_without_close14. Meanvisibility.9880;
  medianofcaseinferencemedians65.52ms,medianofcasep95s67.48ms (notpooledp95).
  selection_A_s17_audit.json verifiesall60seed/edge/stratum/recoveryassignments,
  per-caseequalityandfinalcheckpoint. Onlyonetrainseed; no cross-armconclusion.
- B17actualsavedprocessorcommonkwargs EXACTLYmatchA; actionstatsmatchfinalB
  dataset within1e-12/1e-10roundofftolerance. normalization_B17_saved_processor_check.json
  PASS; priornormalization-tailprojectionnowverifiedforB. Cverifiedbelow.
- B17checkpoint500auditPASS at2026-09-09 01:59UTC:537optimizerstates/scheduler500,
  fourRNGgroups,1030tensors,processorunchanged. checkpoint_B17_500_audit.json/log;
  CPUaudit session74069completed0. No extraoptimizer/physicssteps.
  A/Bembodimentmapping parsedcontentEXACTLYequal,activeID11; differingfilehashes
  areJSONkeyorderonly, recordedinnormalization_B17_saved_processor_check.json.
- B17checkpoint1000auditPASS at2026-09-09 02:24UTC:537optimizerstates and
  scheduler1000,fourRNGgroups,1030tensors,processorconsistent. CPUaudit session16559
  exited0; checkpoint_B17_1000_audit.json/log. Onlycheckpoint1000retained,500
  auditJSONpreserved. Saved15,887,395,211bytes; free185.7GiB. No resumedupdates.
- B17checkpoint1500auditPASS at2026-09-09 02:48UTC:537optimizerstates and
  scheduler1500,fourRNGgroups,1030tensors,processorconsistent. CPUaudit session11392
  exited0; checkpoint_B17_1500_audit.json/log. Onlycheckpoint1500retained; previous
  auditJSONspreserved. Saved15,887,402,121bytes; free185.7GiB. No resumedupdates.
- B17checkpoint2000auditPASS at2026-09-09 03:14UTC:537optimizerstates and
  scheduler2000,fourRNGgroups,1030tensors,processorconsistent. CPUaudit session43743
  exited0; checkpoint_B17_2000_audit.json/log. Onlycheckpoint2000retained; previous
  auditJSONspreserved. Saved15,887,409,110bytes; free185.5GiB. No resumedupdates.
- B17checkpoint2500auditPASS at2026-09-09 03:40UTC:537optimizerstates and
  scheduler2500,fourRNGgroups,1030tensors,processorconsistent. CPUaudit session78504
  exited0; checkpoint_B17_2500_audit.json/log. Onlycheckpoint2500retained; previous
  auditJSONspreserved. Saved15,887,416,085bytes; free181.2GiB. No resumedupdates.
- B17finalcheckpoint3000auditPASS:537optimizerstates/scheduler3000,fourRNGgroups,
  1030tensors,processorconsistent;15,887,423,057bytes. CPUaudit63119 exited0.
  Onlycheckpoint3000retained; prior500-stepauditJSONspreserved. Completionrecord
  model_B_s17_completion.json. Validation60/60finite; selection COMPLETE.
- B17 selection60 complete04:13:01UTC, exit0:17/60success,0/60registeredposture
  violations,17task+posturegates. High/low/deep0/10/7 of20 each; normal14/48,
  recovery3/12 (oneeachstop_short/wrist_offset/object_shift, blackout0). Failure
  timeout24,preclosemotion15,heighttermination4(alllowdeep,noMuJoCowarnings or
  nonfiniteqpos). No-close15,stationary_without_close7. Full audit PASS:
  selection_B_s17_audit.json; failure_states.json. PairedB-only17/A-only13/both0,
  exactMcNemarp.585 (singletrainingseedonly); paired_B_vs_A_s17_selection.json.
- B17 requested video COMPLETE, CPUrender session68508 exited0.
  outputs/videos/scalebfm_a16_delivery_20260909/vla_B17_seed21001_third_person.mp4
  is560frames/25fps/22.4s,640x552; ffprobePASS andencodedframes0/280/559viewed.
  Fixedheld-out seed21001high_near, timeout24s; video skipsstartup1.6s. Do not
  present as successful grasp or same reset as expert4001. Deliverymanifest and
  inference_visual_review.json saved in delivery directory. No rerender needed.
- Next continueC17,A29,B29,C29 automatically; checkpointC17_500 audit next.
  ActualC17processor check PASS: commonkwargs/embodimentmappingequalA17, all
  state91/action29 min/max/mean/std/q01/q99 matchfinalCdataset (13groups, maxabs
  roundoff3.13e-14). normalization_C17_saved_processor_check.json/log; script
  check_saved_processor.py --arm C, no model/optimizer/physics execution.
  Lockwinneronlyaftersixfullselectionsets; confirmationthenfinalauditstillrequired.
- Olddriver1081396/session17266,trainer1081656,eval1467322/1470234 allDEAD.
  AllcheckpointauditCPUjobscompleted; no1000watcher remainslive.

## Evaluation startup repair (completed, do notrepeat)

FirstA17evalstartup failed before anyclosed-loopscene: validationdiagnostic
removedactionfroma dict sharedwithLeRobotEpisodeLoader, dropping scoringlabels.
Onevalidationpredictioncomputed,zerooptimizerupdates/physicalcases. Archive
 evaluation_startup_failure_01 containsoldconfig/source(verifiedoldhash),logs,
stageledgerandrecovery.json; pipelineinfrastructure_attempts preservescost.
eval_visual_vla.py nowgivesloaderitsowndeepcopybeforeomittingactionfrominputs.
Realdataregression redKeyError1.4s→green20episodes/60fixedframes2.7s, rawparquet
MAEparity,predictorreceivesnoactionlabels. tests/test_bfm_visual_validation.py;
validation_config_red/green.log. Realmodelvalidation60/60finite subsequently
completedandfullselectionpassed. No scientificconfig/model/runtimechange or
trainingrestart. Driverrestartpreservedallcompletedstages.

## Resources / requirements

- GPU0 explicitly released by9d18:50 and claimed by this Goal18:56. GPU1 other
  workload: leave alone. No recording stacks or ports; NEVER stop_stack.sh.
- All formal simulation/training/evaluation uses third_party/Isaac-GR00T/.venv/bin/python
  (torch2.9cu128, MuJoCo3.11). GPU env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
  MUJOCO_EGL_DEVICE_ID=0. CPU work hides CUDA; stats requires PYTHONPATH=src.
- VLA receives two RGB cameras + own proprioception/history/previous action and
  fixed task text; NO object truth, contact flags, expert phase or success inputs.
- Humanlike approach/reach overlap; upright when reachable, necessary forward
  bend, constrained side lean/twist. Decide best tested strategy by VLA closed loop.
- Preserve other sessions' dirty work/data; no commits. Shared GR00T library and
  legacy task files include other sessions' edits; our launchers use frozen hashes.

## C0–C3 completed evidence (do not rerun)

- Plans/gates/budgets and amendments: docs/scalebfm_goal_plan.md.
- a16 measured-arrival repair: keep native forward velocity when measured body
  remains >10cm short despite internal reference arrival; stop in corridor/blackout.
- dev_a16_measured_arrival12/12; qual_a16 task+audit30/30, each stratum10/10.
- recovery_a16 task+audit11/12; failure1710 high_near object_shift pre-close knock.
- qual_a16/startup_contract.json30/30: first capture1.60s, task starts1.68s;
  labels/previous actions exact. data.allow_padding=True prevents future wrap.
- qual_a16_chunk_replay10:10/10 task AND exact states/commands, same environment.
- qual_a16_perturbed_replay3:3/3 success with actual3cm wrist reset, original
  recorded actions and duration only; not an extended24s recovery evaluation.
- c3_final_acceptance.json frozen true with counts/source hashes.
- unittest_review_repair.log35tests passed; independent Spec/Standards reviews
  resolved. Original review subagents done; no more review required merely to wait.
- frozen_base_numeric_preflight: actual batch4 forward/backward finite, loss1.502294,
  all trainable gradients finite; zero optimizer updates. Base snapshots pinned in
  base_model_snapshots.json; local_files_only=True. Source snapshots71GR00T files.

## C4 data COMPLETE and exported/statistics ready

| Arm | Accepted train/val | Frames train/val | Accounted attempts |
|---|---|---|---|
| A frozen legacy task/common runtime | 200/20 | 56522/5693 | 225/880 |
| B qualified a16 | 200/20 | 50572/5102 | 367/880 incl145retired |
| C recovery replacement | 200/20 | 47616/4758 | 225/880 incl165reused |

- Cohorts are cohort_A_v2/B_v2/C_v2; datasets dataset_{A,B,C}_{train,val}.
- B new physical222 + retired145. Old cohort_B retired, never enters final data.
  Preserve cohort_B/retirement.json; old incomplete4145 counts in145started attempts.
- C reuses first150/15 B normal split, adds50/5 recovery from60new attempts.
  Train recovery: stop_short12, wrist_offset15, blackout12, object_shift11.
  Val recovery: stop_short2,wrist_offset3 only; report limited coverage.
- Actual accepted train strata A66/68/66, B67/67/66, C67/69/64 (high/low/deep).
- cohort_data_report.json contains frame/episode mixes, uniqueness, failed cost.
- dataset_B_contract.json actual loader/parser6/6: history/images, labels and
  previous actions exact, onlyvideo/state/language. dataset_A/B_readiness.json.
- All driver collect/export/stats stages complete. Do not re-export/recollect.

## C5 frozen protocol and automatic order

- A/B/C each seeds17,29; final3000 only; max6runs/18000optimizer updates.
- Batch4 × accumulation16=64, LR1e-4, AdamW8bit, decay1e-5, warmup.05;
  vision/LLMfrozen, projector+diffusion tuned, state dropout.2, fullFOV240/crop1.
- Model checkpoint every500 with optimizer/RNG, retain latest1; finalroot saved too.
- six_config_contract.json verified six configs differ only data/outputpaths/seed,
  all frozen before first actual update. Each model_*/source has actual sources.
- Driver runs A17→selection, B17→selection, C17→selection, A29,B29,C29 similarly.
- Selection60/model:21001..21060,48normal+12recovery (3/kind).
- Then selection_locked.json; winner+A both seeds100confirmation31001..31100
  (88normal+12recovery), no duplicate A if A wins. Final confirmation_report.json.
- Policy RNG900000+scene seed, 4denoising steps,40action rows, .4sreplan/.08sdelay,
  old chunk continues while pending; maximum24s including startup.
- Primary mean closed-loop task success across two train seeds. Within1/60 tie:
  fewer registered posture violations then simplerA/B/C. No checkpoint tuning.
- Per-seed paired McNemar; scene-joint bootstrap across both seeds. Validation
  diagnostic3fixedtimes/20episodes, final model only, never selects winner.

## Recorded infrastructure / performance observations

- First A17 launch failed in OmegaConf bool validation before model setup, zero
  optimizer updates. training_startup_failure_01 preserves failed log, stage ledger,
  all initial six configs/sources. pipeline infrastructure_attempts records the cost.
- train_visual_vla.py now uses bool resume; --resume path must equal latestcheckpoint.
  Dry-run exercises real OmegaConf serialization. CPU regression red2subcases→green;
  training_config_red/green.log. Scientific configs unchanged; all6 were refrozen.
- Temporary slowdown: mainCPU~onecore, GPU often idle, worker PIDs live. No growing
  major page faults in sampled interval; RAM available35GiB, CPUquota15.36cores.
  py-spy attachment denied by container (train_A17_slow_stack_01.txt), so exact
  hotspot unproven. No configuration/source/runtime change or training restart.

## Delivery and recovery links

- docs/scalebfm_visual_collection.md: operator guide and checkpoint audit command.
- docs/scalebfm_acceptance_evidence.md: requirements/evidence, final items pending.
- docs/scalebfm_vla_comparison_results.md: real data table; VLA result table pending.
- docs/scalebfm_goal_history_20260908.md: older a00..a15 development history.
- docs/scalebfm_goal_history_20260908_c4.md: archived C3/C4/C5-start checkpoints.
- training_live_checkpoint.json: latest observed step/PIDs, not a completion marker.

Goal is NOT complete: full six-model training, selection, confirmation and final
requirement-by-requirement audit remain. Expert30/30 alone cannot select a winner.
C synthetic recovery is not actual student-state DAgger; RGB localization remains
an expert proxy. Final report must retain these limitations and domain boundaries.

## Latest offline reproducibility/normalization checkpoint

Environment snapshot145packages saved environment_manifest.json/environment_versions.txt,
explicitly capturedafterA17step500. No mutationofsharedtrainingenvironment.
A actualsavedprocessor actionroundtrip all56,522frames passesstate-freeABSOLUTE
encode/decode, expectedq01/q99clippingerror<1e-6 (processor_action_roundtrip_A.json).
Normalizationtailaudit allABCtrainframes: wristpositionerror>1cm fractions
A2.39%,B1.74%,C2.00%; maxima10.25/6.00/6.21cm. Primarilyreach, butclosephase
alsoA.34%,B3.18%,C3.05%frames>1cm. Thisisrepresentationloss; actualclosed-loop
impactunproven. Noextraoptimizer/model/physics executions orconfigchanges.
B/Cderivedfromfrozencommonsettings+finaldatasetstats, notyetsavedB/Cprocessors;
B17/C17actualsavedprocessorsverified; Ccheckincludesallstate91andaction29statistics.
Artifacts normalization_tail_audit.py/json/log inexperimentroot. Firststrict
metadataequalitycheckonlyfailedonstdroundoff8.7e-15, retainedin
normalization_tail_audit_stats_precision.log; finalcomparisonuses1e-12tolerance.
CPU sessions19624/87119/97606allcompleted0. Checkpoint1000retentioncheckpassed;
A17training/selectioncomplete, B17trainingrunning. No repeated experiments needed.

## Expert visual-delivery checkpoint (rendering only)

posture_ab_v1/{high_near,low_near,low_deep}.png generatedandvisuallyreviewedall3,
pairedfirstsharedacceptedtrainseeds4001/4002/4003. Fixedcamera,4phaseframes/arm;
FK/renderfromqpos only, zero additionalphysics/optimizersteps. Manifest+manual
reviewinposture_ab_v1/manifest.json/visual_review.json. Bhighcloseupright, low
closeforwardbend, lowcarrymoreupright; lowsamplesstillnonzerosidelean. Illustrative
expertframes, notVLAresults/episodepeakmetrics. Reportlinksadded.
All three continuous B videos COMPLETE; render session10919 exited0.
posture_ab_v1/video_integrity.json: ffprobe passed all3, 640x528 at25fps;
253/242/283frames and10.12/9.68/11.32seconds. video_render_live.json complete.
Source render_expert_videos.py, log render_expert_videos.log, video_manifest.json.
Video links added to comparison report. No rerender needed. A17trainingcomplete;
newdriver1470213/session89332 runsB17training. Olddriverand1000watcherended.

## A17 student visual checkpoint

vla_selection_visuals_s17/A_first_three.png renderedandviewed: fixedfirst3
registeredscenes21001/21002/21003,t=2/4/6s+final, CPUFKonly,zerophysicsupdates.
A_manifest.json andA_visual_review.json preserveframes/sources/hash/review;
cameras.json fixedforfutureB/C17pairedvisuals. Script render_vla_selection_samples.py
--arm A completedexit0(session37842); do notrerenderA. Reportlinked.
Terminology correction: precloseknock is >5mmhorizontalbottlemotion beforeclose,
NOTnecessarilytoppling. Chinese report nowcallsit提前碰动 instead of碰倒.
B17trainingcontinues; futureB/Crenderonlyaftertheiroriginalselectionfinishes.

## User-requested a16 video delivery (2026-09-09 02:04 UTC)

- Exported one accepted a16/B training episode, seed4001 high_near, third person:
  outputs/videos/scalebfm_a16_delivery_20260909/a16_expert_seed4001_third_person.mp4.
  Verified 253 frames, 25fps, 640x528, 10.12s; copied existing recorded-qpos render.
- Corresponding trained inference video is STILL PENDING: B17 was at625/3000.
  A17 uses legacy A data and must not be presented as a16-trained inference.
- After formal selection_B_s17 seed21001_x- result exists, render its recorded
  qpos into a third-person MP4, using existing evaluation only (no new rollout).
  Label expert4001 and held-out student21001 as different scene seeds. Preserve
  final3000-only evaluation protocol. Deliver clickable inference MP4 to user.
- Delivery manifest: experiment-root a16_video_delivery.json and
  outputs/videos/scalebfm_a16_delivery_20260909/delivery.json.

Video renderer ready: experiment-root render_vla_video.py --arm B, CPU environment
CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 with third_party/Isaac-GR00T/.venv/bin/python. Requires completed
selection_B_s17/summary.json and final3000 checkpoint; fixed seed21001, frozen
A/B/C camera, 25fps recorded-state FK only. Input check --arm A --check-only PASS
(vla_video_input_check_A17.json), no A video generated. Actual B render/ffprobe/
visual inspection and delivery still pending; do not claim renderer fully verified.

## A17 termination-label clarification (completed)

selection_A_s17_failure_states.json audits all60savedqpos/warnings: noMuJoCo
warning ornonfiniteqpos; exactly17cases endwithpelvisz<0.45m, matching all17
physicswarning/falllabels. High/low/deep3/7/7; firstcross7.10–21.78s,median11.98;
finalheight0.4483–0.4499m. CurrentSim.fallenpredicateisheightonly, sourcehash
recorded withoutclaimingaretrospectivefreeze. No changedtermination/scoring.
Firstcaseperstratum21004/21005/21009 renderedat2s/end-2s/end andvisuallyreviewed:
allthree showcrouching, terminaltorsonotprone. Do notinferall17visuallyconfirmed
falls or numericalinstability. Artifacts vla_selection_visuals_s17/
A_height_termination_samples.png/json andA_height_termination_visual_review.json;
script audit_selection_failure_states.py --arm A andrender_height_termination_samples.py
completed. CPUrender sessions21704/92350/61790 exit0 (two layout corrections
archived aslayout_v1/v2); zero additionalpolicy/physics/optimizersteps. Reportupdated.

Latest delivery checkpoint: both requested MP4s ready and linked in user response.
B17 inference clip is a timeout; expert clip is accepted train seed4001.
C17 savedprocessor check and B17 fixedfirstthreecontactsheet COMPLETE;
B17 fullselection/failure-state/pairedsingle-seed audits complete. No model winner
is locked; C17 plus allseed29 runs and confirmation remain.

B17 fixedfirstthree visual checkpoint: render_vla_selection_samples.py --arm B
CPU session98024 exited0; vla_selection_visuals_s17/B_first_three.png viewed and
B_visual_review.json saved. SameA17cameras. Firsttwo timeout24s; deep21003
pronouncedcrouching atheighttermination20.22s. Allthreeillustrativecasesfail;
B17overall17/60success. No rerender needed, no addedphysics/modelsteps.
