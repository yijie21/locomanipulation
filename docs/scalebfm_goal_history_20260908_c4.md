# ScaleBFM Goal checkpoint

Updated2026-09-08 20:45UTC. Goal ACTIVE. Latest timestamped checkpoint below wins.
Read STATE then this file then
goal_plan/design on context recovery. Do not reconstruct by repeating runs.

## User requirements / resources

- VLA2RGB+proprio, NO objecttruth. Humanlike approach/reach, upright ifreachable,
  necessaryforwardbend; choose best measured strategy byfixedbudget VLAclosedloop.
- GPU0 explicitlyreleasedby9d18:50 andreservedbythisGoal. GPU1otherwork: leavealone.
  Standalonesim only; no stacks/ports, NEVER stop_stack.sh. No commits.
- GR00Tvenv for ALL formalcollection/train/eval: torch2.9cu128,MuJoCo3.11.
  Env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0.
- Experimentroot outputs/experiments/scalebfm_visual_goal_20260908/ (below).

## CURRENT: formal C4 v2 collection running

Old a15 wasqualified29/30,recovery11/12, but reviewfound a hiddenstop-short bug:
internalrootreference couldarrive andzero velocity whilebodyremained20cmshort.
Also rawrecordingstarted1.68 butVLAfirstcapture1.60. STOPPED own a15cohort_B
PID1022216 andoldpipelinePID1029457; BOTH DEAD. Preservecohort_B andretirement.json:
144completeattempts,142tasksuccesses,oneincomplete4145 =145physicalattempts.
ALL145debitB880budget. Noneofthisretireddata entersfinaltraining.
NO FORMAL OPTIMIZER STEPS. B_v2 exported; see latest checkpoint. Do not restartoldcohort/driver.

a16 repair: visual_motion keepsnativeforwardvelocity onlywhile measuredremaining
>10cm ANDinternalremaining<4cm inwalk/reach; cap.12m/s, accel.6;stopwithinarrival
corridor orvisualpause. Existinggeometry/armparametersunchanged. Newheldplant
regressionpasses (20cmshort→.072m/s, arrived/blackout→0). IndependentSpecfollowup
confirmsconvergence.
- dev_a16_measured_arrival:12/12task ANDaudit; paireddev101..112. DONE.
- qual_a16 COMPLETE30/30task and30/30audit, fullRGBrecording, overlap1.0.
  startup_contract.json verifiesALL30firstt1.60, labelsandpreviousactions exact.
- recovery_a16 COMPLETE11/12task (1710high_nearobject_shiftprecloseknock);
  auditCOMPLETE11/12; recovery_a16/audit_summary.json.
- qual_a16_chunk_replay10 COMPLETE10/10exact; perturbation3/3success.
- Allrequiredqualification/recovery/startup/exactandperturbedreplaygatescompleted.
  c3_final_acceptance.json isfrozenwithrealcountsandcriticalsourcehashes.
- NEWdriverRUNNINGsession77244 PID1045488, pipeline_v2.log. ChildcollectorB_v2
  PID1045538 completed B; A child PID1059354 now active. Do notlaunchduplicate.
  Authoritative stagefile:pipeline_v2_state.json. NextautomaticA_v2,C_v2then
  exports/stats/6trains/selection/confirmation. Inspectthisfile/logs onresume.

## Ready implementation / review repairs

- Shared25Hzfloat32native29 action,50HzBFM executor, livepelvisyawframe, wrist
  <=2cm/5deg. Taskrobotfeedback bakedintolabels; notdoubleappliedbyVLA.
- Sceneproxy actual2camera visibilitygates, .4supdate/.08latency,5mmquantization,
  pausesprecisionwhenhidden. Expertproxyonly; no objectqueryinVLApredictor.
- Observation91 ownstate(53+velocity9+previousaction29), twoRGB atpast.4s/current.
  data.allow_padding=True REQUIRED (False wrapsnegativehistoryintofuturetail).
  Bothcollectorsnowrecordfromtick80(1.60);firsttaskactiontick84(1.68).
- run_visual_bottle/visual_baseline write simulation_result.json; callerwrites
  finalresult.json onlyafterrecordcloseandscene metadata. Auditrequiresstratum
  andscene_directory. collect_visual_cohort canfinishcompletedrecordingmetadata
  withoutrerun; interruptedphysicalattemptsretained,counted,notretried.
- visual_provenance verifies6registeredassets hashes; verifiedunchanged.
- preflightnowraisesonnonfiniteloss/gradients, batchshapesMappingfixed. Earlier
  actualpreflightfiniteforward/backward:batch4peak20.63GB,optimizerestimate3.24GB,
  zerooptimizerupdates. processor_batch_contract verified16images/4samples.
- eval_visual_vla: commonobserver+ChunkScheduler+executor, firstcapture80,
  delayedrow2at84,continueoldchunkduringpendingreplans.4denoisingsteps.
  Per-sceneRNG900000+seed. Last12of60/100arefixedrecoveries, noextraevalbudget.
  --replay-perturbation wrist_offset --replay-limit3 appliescontrolled3cmwrist
  resetwhileexecutingrecordedcommands. Its horizonisoriginalrecordingduration
  (explicitlyreported; not24scompetitiveVLAhorizon).
- report_visual_comparison tieswithin1/60→posture→simplicity, lockedselection,
  pairedMcNemarpertrainseed andscenewisebootstrapjointacross2trainingseeds.
  Two selection/statistics testsPASS.
- review skill applied; agentsreview_standards/review_spec read-only. Specfollowup
  done,noblockingdefect; standardsfollowupongoing. Seeagentmessages.
- Fulltestlog unittest_review_repair.log:35testsPASS.
- StandardsfollowupfoundCbudgetguardreservedonlycurrentsplitreuse, fixedto165
  inbothsplits. Allinitialreviewfindingsresolved; Specfollowupno blockingissue.

## New formal pipeline (RUNNING; inspect stage before acting)

run_visual_comparison.py nowwrites pipeline_v2_state.json,usescohort_A_v2/B_v2/C_v2.
B automaticallyreceives --prior-attempts145 fromretirement.json. Requires
c3_final_acceptance.json beforeanyformalstage. Oldpipeline_state.json remains
paused_review_correction. Drivercurrentlyexecutingcollect_B; latestsourcefrozenbycollector.
Active driver was launched with:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 third_party/Isaac-GR00T/.venv/bin/python
src/teleop_free/scalebfm/run_visual_comparison.py --root outputs/experiments/scalebfm_visual_goal_20260908
(alreadyredirectedtopipeline_v2.log; session77244).

Driver: collectB,A,C sequentialGPU0; export200train/20val each; statsCPU; freeze6
configs; A17/B17/C17/A29/B29/C29 eachfinal3000; selection60each; lockwinnerthen
winner+baseline2seeds100confirmation. Neverselectcheckpointusingtest.
Trainingbatch4xacc16=64,LR1e-4,AdamW8bit,tuneprojector+diffusion, frozenvision/LLM,
fullFOV240,crop1,state_dropout.2. saveevery500withoptimizer/RNG,retainlatest1.
Current model outputsroot+checkpoint final, disk233GBavailablebeforetraining.
Retired145+newBphysicalattempts<=880. Pools: A/B4001..4800train,6001..6080val;
C150/15normalreusedfromB +50/5recovery8001..8400/9001..9040. Selection21001..21060,
confirmation31001..31100. Baselinegeneratorfrozeninitialsnapshotwithcommonruntime.

## Prior evidence (DONE; details in history/plan)

a00..a15developmentanddiagnostics in scalebfm_goal_history_20260908.md.
a15development12/12,qual29/30,recovery11/12. Originaltaskcommonruntimebaseline
pilot1401..1412=12/12. Sameenvdelayedchunkreplayexact3/3 bothsimvenv andGR00Tvenv;
crossPyTorchversionssmallnumericaldifferencespromptedunifiedGR00Tenv.
10qualifieda15directexecutor replaysALLexact,3GPUa15replaysALLexact.
contract_gr00t_a15/posture_contact_sheet.png visuallyreviewed:highupright,
low/deepforwardbend,moreuprightcarry. Alloldartifactsretained.

## Latest completed C3 evidence / next launch

qual_a16 30/30audit, recovery_a16 11/12audit. qual_a16_chunk_replay10:10/10task,
10/10exact(qpos/commandszero). qual_a16_perturbed_replay3:3/3successwithactual
3cmwristintervention, naturallynotexact; withinoriginalrecordedhorizon.
35testsPASS, allreviewfindingsresolved includingfull165C-reusereservation.
Base models pinnedlocal snapshots in base_model_snapshots.json; training now
local_files_only=True; train_config_frozen_base_smoke dryrunPASS.

frozen_base_numeric_preflight PASSED: samefinite loss1.502294, ALLtrainable
gradientsfinite; peakreserved21.74GB, estimatedoptimizer3.24GB. Zerooptimizersteps.
C3acceptancewrittenandv2driverlaunched; currentGPU0taskisformalcollection.
Do not reuse oldpipeline/cohort_B IDs. Newpipelineautomaticallydebits145 retired
Bphysicalattempts, startscohort_B_v2/A_v2/C_v2. Trainingbudgetstill0/18000.

Operator guide: docs/scalebfm_visual_collection.md. No finalVLAwinner yet.

## 20:04UTC live checkpoint

Driver PID1045488/session77244 alive, childBcollectorPID1045538. Stagecollect_B.
B_v2 train120accepted/121newattempts, plus145retired=266Bphysicalattempts;
20valstillpending. Budget880 preserved. Noformaloptimizersteps.
Operatorguide added docs/scalebfm_visual_collection.md.
Evalauditnowreportshigh_near first0.8saftercommandedclosemedianpitch,
stationary_without_close,last1sbasevelocity,visibility,infermedian/p95.
Tie-breakincludesregisteredhigh-near upright15deggate; planfixedbeforetraining.
Reporttestsrepassed2/2afterthis addition. No runtime/expertchanges.
Trainingfreeze nowincludesallGR00T Python sourcehashes, so inter-sessioncode
changesbetweenfreezingandtrainingwillfailratherthansilentlyaltercomparison.

Next: inspectpipeline_v2_state.json andrunningstage log; backgrounddriver
automaticallycompletesA/Ccollection,export/stats,6train+selection,confirmation.
If a stagefails, diagnose its actual log and preservecompletedstages; do not
launchduplicateprocesses or resetfrozenconfigs. Goalremainsactive.

## 20:14UTC C4 B complete; A running

B_v2 COMPLETE200train/20val. Newattempts202train+20val=222; with145retired,
Btotal367/880. Trainingstrata67high_near/67low_near/66low_deep; val7/7/6.
Failednewtrainseeds4018knock,4133timeout. Noformaloptimizersteps.
DriverPID1045488/session77244 alive; A_v2 childPID1059354 currentlycollecting.
Currentstagelogcollect_A.driver.log; authoritativepipeline_v2_state.json.
cohort_data_report.json summarizescompletedBcoverage/framecounts/cost; A/C
markednotcomplete. Rerunsummarize_visual_cohorts.py afterremainingcohortsfinish
toupdatethisofflineaggregate (doesnotrerunphysics).

Thisturnprogress: Bcompleted; training source snapshotimplementedanddryrun
validated71GR00Tfiles, includinglauncher+modality+basemodelsnapshotmanifest.
Trainingmanifestnowrecordslauncherhash. EvalverifiesGR00Tsourcehashes against
trainingbeforeloadingweights. Exportrejectsepisodes<40frames (cannotprovide
onefullconfiguredactionchunk). Source/data metadataonly; nopolicyclechanges.
Newhelper summarize_visual_cohorts.py reportsepisodesANDframesperstratum/recovery,
actualoptimizationstartpositions(n-39), reused/retired/failedcostandnosplitoverlap.
Next: monitorA/C; inspectexports/statsandfullcohort_data_report beforeassessing
training. Allmodelswillremainfinal3000steps, seeds17/29, fixed60/100evalpools.

## 20:25UTC B data readiness checkpoint

B_v2 export COMPLETE: train200episodes/50,572frames; val20/5,102frames.
Both CPU statistics commands exited0, stats.json/relative_stats.json saved.
Actual LeRobot loader and inference parser checks PASS6/6 (episodes0,99,199;
frames0,12). Past/currentRGB, labels and previousactions are exact; observation
groups onlyvideo/state/language. Evidence:dataset_B_contract.json and
dataset_B_readiness.json; logs early_export_B_*, early_stats_B_*.
Sessions18952 and20707 nowfinished exit0. No remaining B CPU job.

DriverPID1045488/session77244 and AcollectorPID1059354 verifiedalive20:24UTC;
Atrain113accepted/117attempts then. Current stagecollect_A, logcollect_A.driver.log.
Bexports were done early onCPU whileAcollects; driver safelyreuses completed
exports and cachedstatistics. No extra physics or optimizer updates.
Next: completeA/C, summarize allcohorts, allowautomatic six-configfreeze and
fixedtraining/evaluation. Do notlaunchanotherdriver. GPU0ours, GPU1untouched.

## 20:35UTC A complete; C recovery collection running

A_v2 COMPLETE200train/20val, 205train+20valphysicalattempts=225/880.
Five training failures retained; no interrupted attempts. Trainstrata
66high_near/68low_near/66low_deep; validation7/7/6.
Aexports ANDstatistics complete:56,522trainframes/5,693valframes. Both CPU
preparecommands exited0; logs early_prepare_A_train.log/early_prepare_A_val.log;
dataset_A_readiness.json. Sessions75180/97028/65308 nowfinished.
cohort_data_report.json refreshed forA/B;Cnotcomplete. Ameantrainlength11.30s,
B10.11s; fixedoptimizerbudgetaccountsfordifferentframecounts, noresamplingchange.

DriverPID1045488/session77244 alive, currentstagecollect_C; childPID1075083
verifiedalive20:34UTC. Logcollect_C.driver.log. C adds50train+5valrecovery
tofirst150/15normalBepisodes. Earlyfailure8003low_deepstop_shortknock isretained.
No formaloptimizersteps. NoactiveearlyCPUjobs. A/Bexports/stats arereadyand
willbereusedbydriverafterC. Next: finishC, refreshcohortsummary, inspectfinal
Cmix, six-configfreeze then firsttraining. No duplicate driver orcollection.

## 20:43UTC C4 complete; C5 startup repair

C_v2 COMPLETE200train/20val:150/15Bnormalreused plus50/5recovery. 60newphysical
attempts +165reused=225accounted/880. Trainrecoverymixstop_short12,wrist_offset15,
blackout12,object_shift11; validationstop_short2,wrist_offset3 only (reportthis
coverage limitation). Cframes47,616train/4,758val; allABCexports/stats COMPLETE.
cohort_data_report.json refreshed. Aframes56,522/5,693; B50,572/5,102.

Allsix initialconfigs frozenandcheckedidenticalexceptdata/outputpathsandseed.
FirstA17 failed BEFORE modelsetup/optimizerupdates dueOmegaConf boolvsNone
resume field. Driver1045488 andchild1079657 DEADconfirmed. Failureandallsix
initialsource/configsnapshots archived intraining_startup_failure_01; pipeline
infrastructure_attempts preservesfailurecost0updates. Oldsession77244terminal.
Launcherrepair: boolresume, explicitpathmustmatchlatestcheckpoint, dry-run
exercisesrealOmegaConf boundary. CPUregressionfailed2subcasesbeforefix,passes
afterfix(training_config_red/green.log). No scientificconfigchanges. Completed
collect/export/statstages retained; sixfreezestagesandfailedA17 clearedonlyafter
archivalanddeadPIDcheck. Next restartdriver torefreeze6/retryA17.

Evidenceindex added docs/scalebfm_acceptance_evidence.md; finalVLAselectionand
confirmation explicitlypending, syntheticCnotclaimedasstudent-stateDAgger.

## 20:45UTC C5 formal training live

RestarteddriverPID1081396/session17266 LIVE; train_A_s17 childPID1081656 LIVE.
Oldsession77244/driver1045488 aredead; doNOT poll/restartthose. Allsix configs
refrozen andchecked identicalexceptdatasetpath/outputpath/seed, beforefirst
formalupdate; six_config_contract.json updated, oldversionpreserved inarchive.
A17 observed36/3000 optimizerupdates; totalformalbudget36/18000atcheckpoint.
Lossatsteps10/20=1.5136/1.291, gradnorm.886/.634, finite. GPU0memory26,366MiB
of32,607MiB; initialsteadythroughputabout2.7s/update (~2.2hours/runifunchanged).
Actualfirsttrainingworking; noOOM. Fail01onlyinitialization,zerooptimizercost.
Liveevidence training_live_checkpoint.json; logtrain_A_s17.driver.log.

Next verifyfirst500checkpointincludesoptimizer/RNGandnormalcontinuedupdates.
AutomaticdrivercompletesA17final3000→60selection(+valdiagnostic), thenB17,C17,
A29,B29,C29; lockwinner→100confirmationperrequiredmodel. Noextraarmsorsteps.
Allcohort/export/stats done; noindependentCPUjobsleft. Goalactive, finalresults
notready. Currentevidenceindex docs/scalebfm_acceptance_evidence.md.

## 20:49UTC report scaffold and verified training wait

Added docs/scalebfm_vla_comparison_results.md withactualABCepisode/frame/cost
table, fixed192,000samplepositionspermodel, Crecoverycoverage limitation and
explicitlypendingVLAresulttable. Noscientificconfiguration changes.
A17latestobserved101/3000, recentloss.9835/.8693withfinitegradnorm.941/.823.
Driver1081396 andtrainer1081656 bothverifiedlive, session17266 waited50seconds
andreturnedstillrunning. Nocheckpoint500yet; donotduplicate/restarttraining.
Nextwaitfor500andsanity-checksavedoptimizer/RNGfiles; thentrainingcontinues
automatically. Throughputvariesduringdataloading; early2.7s/updateisnotafixedETA.

## 21:01UTC verified A17 training wait

A17 observed258/3000, driver1081396/session17266 andtrainer1081656 bothlive.
Latestloggedloss.1048,gradnorm.811, finite. No500checkpointyet.
TemporaryCPU-side slowdown: ~3s→~13s/update, then~6s; GPUoftenidle, mainCPU
~1core, workerslive, majorpagefaultcounterstayed2727duringobservation.
CPUquota15.36cores, availableRAM35GiB. py-spy installedviaisolateduvxcachebut
attachdeniedbycontainer (train_A17_slow_stack_01.txt); noadditionalprivileges
requested. Exacthotspotunproven. No training/runtime/sourcechanges orrestart.
Continueoriginalrun; nextvalidate500checkpointoptimizer/RNG. Thisturnverified
live-sessionwaitsanddiagnosticevidence, notablocker.
