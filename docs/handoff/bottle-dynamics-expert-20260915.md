# Bottle response, stopping and loaded motion

Date: 2026-09-15. Read this for the active continuation after the central-grasp
delivery; the previous experiments remain in
[the contact expert report](bottle-contact-expert-20260915.md).

Current status: GitHub baseline push complete. Main holds frozen `stage3_3_switch`
(worker `0e914d8`), which tolerates the ownership-switch gap in the handback
stability window. Combined fixed8 `stage3_2_aligned8` finished **0/8**: three
rounds reached loaded upright and passed the pre-switch quiet window for the first
time, then failed on the first post-switch sample (`clock_gap`); six rounds
consumed a walking stance plan before braking. After the switch fix
the user approved (2026-09-16) a diagnosis-driven order: liftoff-gated swing
supervision, loaded orientation tracking, the housing/bottle guard, then the
endpoint hold; swing-prior recalibration was rejected. Root checkpoints
`stage3_3_switch` … `stage3_8_hold` (worker `0e914d8` … `d7a768f`) each ran the
fixed8 bank; all 0/8, but the reached stage moved from 3/8 lifts to 8/8 lifts
and 3/8 real post-switch measurements. See "Stage3.3–3.8" below.
The last combined physical checkpoint `stage3_1_loaded` (`ca0abeb`) had root
focused149/149 and worker hook pass, endpoint-only CPU2/2; fixed8 finished0/8: six loaded
orientation/tracking failures and two wrist/bottle guards; candidate rejected,
handback untested. Walking planning still had0 consumptions: three feasible
results invalidated on moving candidate XY/yaw despite stable edge distance.
Worker reviews residual capture-clearance failures read-only while the root
audit covers walking adoption and loaded/handback behavior. Fresh20 reserved; no timing
reduction. Ledger: `outputs/bottle_dynamics_20260915/continuation_state.json`.

Evidence correction (2026-09-16): historical two-second post-outcome root travel
includes the explicitly commanded35cm carry backoff. It is **not evidence of
uncontrolled handback drift**. `handback_command_aligned_review.json` separates
requested IDLE and locomotion and removes pre-reset time aliases: the19 successful
runs provide only0.08–0.18s of immediate requested-IDLE observation, with net root
motion0.8–11.6mm. These short windows do not establish stable handback; Stage3
adds a dedicated post-switch observation window before carry locomotion. Older
tail numbers below are historical mixed-command measurements, superseded by this
qualification.

## Authorization and preserved baseline

The user requested a GitHub snapshot first, then implementation in this order:
tracking-response identification and terminal braking; joint stopping and
whole-body posture planning; loaded trajectories and stable handback. Shorter
reach/recovery durations follow demonstrated stability, not precede it. No VLA
training is requested.

Current code/config/tests/docs were committed as `7db3bb3`. Remote main's three
path/document fixes were merged; the new process-level `LOCOMANI_TMP` read was
registered in the environment-path check. Both installed commit-hook validations
passed. `7af58f2ba3aed132c75573ff2d6ac59146221eb4` was pushed to GitHub main and
verified against `git ls-remote`. Runtime artifacts and local Headroom tooling
were not tracked. The push receipt is in `outputs/bottle_dynamics_20260915/`.

Integration branch: `feat/bottle-dynamics-20260915`. One GPT-5.6 Sol worker uses
`fix/bottle-dynamics-20260915` in
`/workspace/code/locomanipulation-bottle-dynamics-20260915`. Main owns design,
review, integration, GPU/stack launches and final evidence. Worker runs CPU
implementation/tests, without recursively delegating. The old worker branch
contains rejected checkpoint12 and must not be blindly reintegrated.

Baseline remains checkpoint11: middle-pad capture +1.2mm ±5mm, close1.2s,
lift/recovery5s. Fresh seeds30700/30800 produced10/15 successes: three wrist/bottle
clearance guards, one pre-capture finger/table guard, one recovery slip. Ten
successful tails retained bilateral middle-pad contact. High reliability,
natural motion and stationary terminal holding remain unmet.

## Ordered implementation and review gates

1. **Response identification and terminal braking.** Separate measured dynamic
   response from static/posture bias and IK residual. Establish trace identity,
   timestamp provenance, excitation and episode-held-out fit quality before
   changing the control law. Use the identified response to test bounded terminal
   translation/rotation braking and safe central capture. Existing motion
   limits, collision guards, central capture geometry and scene physics stay
   active. A model without identification support is reported as unavailable.
2. **Stopping and whole-body posture planning.** Replace the unsupported fixed
   stopping-distance assumption with measured stopping-state evidence. Evaluate
   final grasp/comfort/support feasibility together with the candidate stance;
   account for endpoint uncertainty and the swept table clearance. The current
   stance callback checks collision but does not solve the final supported grasp.
3. **Loaded path and stable handback.** Plan the bottle/hand path through table
   clearance and comfortable carry, instead of allowing torso recovery to add an
   uncontrolled hand excursion. Add load/slip evidence where observable, and
   require measured stabilization and a compatible planner handback. Current
   bilateral contact fields identify touching links, not contact force; current
   zero-reference-speed handback is not a measured stoppedness gate.
4. **Timing.** Only after the above physical checks support stability, assess
   shorter reach/recovery timing. Record both success and actual motion metrics;
   nominal trajectory smoothness and ten selected successful videos are not
   evidence of high reliability or VLA learning benefit.

Use isolated, frozen source checkpoints for each live candidate. Preserve every
attempt and report failures, sample selection and asynchronous replay limits.
Focused CPU checks precede live tests; final integration uses the repository's
required check entry. No runtime output/weight deletion is planned.

## Current work and experiment inputs

Stage1a is instrumentation and CPU response analysis only, with no changes to
motion, acceptance, physics or task configuration. Existing sent-reference
journals retain packet endpoints and send/context times. The deploy tap exposes
the selected motion frame, but its receive timestamp is asynchronous and must
not be described as an exact controller-consumption timestamp. Frozen-upper
override semantics must also be accounted for before fitting.

Evidence root: `outputs/bottle_dynamics_20260915`. The fixed eight-scene bank
`regression8.json` includes three historical successes and all five failure
cases from the15-attempt baseline, including a long-reach success. Sampled XY and
appearance were checked against the original manifests; actual recorded object
yaw was restored. CPU reconstruction and bank provenance are retained. The
predeclared response-fit split uses four training and four held-out episodes;
fresh evaluation seeds are separate. This selected bank is for diagnosis and
regression, not a population success-rate estimate.

GPU0/instance0 is the intended live resource; it is currently released. Check
live GPU use, instance locks, CPU headroom and STATE before launch; acquire the
existing launcher lock and stop only the owned instance. Preview delivery keeps
the500000-byte cap and preserves source recordings.

Code-review hypothesis for terminal braking: `ForwardSpeedFilter` rectifies
signed speed before filtering, and the admission predictor takes the maximum
of raw-positive and filtered speed. Zero-mean oscillation can therefore acquire
a positive speed estimate. The current constant-acceleration stopping formula
may treat oscillatory speed as persistent drift. Preserve signed velocity and
separate slower movement from bounded oscillation during identification before
changing this rule. Sparse last-second journal diagnostics are saved as
`terminal_speed_journal_review.json`; their cadence cannot establish a high-
frequency spectrum or causal attribution.

## Stage1a logging calibration

The four producer paths frozen in worker commit `b6175a1` were reviewed and
integrated as `outputs/bottle_dynamics_20260915/stage1_logging`. They add measured
joints, asynchronous deploy-selected reference identity and agent post-send POSE
metadata; wire packet contents and motion/acceptance/configuration are unchanged.
The existing streamed-reference, streamed-handoff, planner-takeover and bottle
whole-body focused suite passed (75 tests).

`stage1_logging8` ran the prescribed eight cases on GPU0/instance0 and returned
**7/8**, with `30700:4` slipping during recovery. All attempts, sidecars, traces
and videos remain. The runner returned1 because the diagnostic batch did not
meet its requested eight-success quota; stack shutdown succeeded, GPU0 released.
`stage1_logging8_review.json` records results and verifies frozen producer hashes
after the run. This differs substantially from the historical3/8 and is an
asynchronous rerun observation, not evidence that instrumentation improved control.

The raw tap appears in every early inspected sidecar row. Receive age relative
to state wall time sometimes becomes negative (down to roughly55ms in those rows),
so the analyzer must select preceding snapshots instead of consuming row-local
future reference values. The declared four-train/four-held-out split is retained.

Read-only measured FK review (`review_end_motion.py`,
`stage1_logging8_end_motion_review.json`) projects the last second before capture
onto the final measured palm x-axis. Reaches lasted6.22–15.4s. Most terminal signed
mean velocities were14–30mm/s; two cases showed stronger rectification bias
(4 vs15mm/s and14 vs27mm/s). The95th percentile of200ms detrended displacement
range was1.5–5.9mm. These are descriptive projections, not the exact commanded
entry axis or an identified deceleration law. Seven successful episodes retained
the bottle but moved the root8.2–53.7cm over the two-second handback tail.

Model acceptance must compare against persistence as well as static bias, check
multi-step held-out prediction, and reject or segment irregularly filtered
sampling. Teacher-forced one-step improvement alone does not identify a useful
terminal response. No learned response parameters or control changes have been
integrated yet.

The analyzer was frozen at `e02640d` and integrated as `stage1_identification`.
Twelve analyzer/instrumentation tests pass in the integration tree; its CLI
reproduces the worker report exactly. The prescribed holdout rejects the simple
first-order model: recursive0.2s prediction improves persistence by only
1.14–4.52% (median3.10%; threshold5%). No model parameters are exported for
control. It used6655 matched samples in contiguous50Hz segments; reference
receive age median9.93ms/p9520.24ms. Joint excitation was sufficient but does
not establish predictive validity. Solver hand residuals are separately
3.60mm median/11.01mm p95; these are not measured tracking error.

`stage1_logging8_stop_review.json` records the complete physical stopping tail:
all eight lead commands saturate at0.10m, but peak travel is0.064–0.276m and
final travel0.063–0.274m. Final bottle depth spans0.186–0.371m. Similar request
speeds produced very different stopping distances, so simply raising the cap
is not an identified solution. Joint stance planning needs an uncertainty
envelope and revalidation from measured support after stopping.

Stage1b is implementing signed Cartesian terminal trend and bounded detrended
excursion, without learned response parameters. Fixed capture geometry and
close/lift durations remain the baseline. Stance and loaded handback changes
remain subsequent checkpoints.

A fresh20-case bank (seed30900) was generated in CPU sample-only mode before
stage1b integration. `fresh_evaluation_plan.json` fixes its identity and final
evaluation purpose; it is not used to tune the intermediate stages. All attempts
will be reported. Nominal timing is conditional on physical stability of the
three changes, including post-handback movement, not just the success count.

## Stage1b terminal motion candidate

Worker commits `0bbbe5a` and `6c5bc16` were integrated as `stage1_terminal`.
The observer keeps signed Cartesian velocity, a0.20s least-squares trend and
peak-to-peak detrended excursion as a conservative two-sided depth uncertainty.
It admits capture only when current geometry and the entire predicted stopped
interval fit the original centred window. Cold starts, insufficient history,
phase changes and gaps over50ms withhold admission. Goal retreat follows the
predicted interval centre; uncertainty is not added again as a one-sided bias.
The existing stopping acceleration/lookahead are explicit priors, not parameters
from the rejected response model.

The goal-retreat tracker retains continuous state and uses
`min(hand_speed, capture_brake_acceleration * capture_terminal_window_seconds)`
as its rate bound. A review caught the initial0.16m/s bound coupling to the
shared tracker's gain and slowing its90% response to3.12s; final0.04m/s bound
reaches90% of a0.024m step in0.88s. This is a CPU target-generator result, not
a measured robot stopping time. The final focused worker suite83 tests and
full CPU checks pass; five related suites pass in the integration tree.

`stage1_terminal8` is the frozen live candidate run on GPU0/instance0. The
worker is implementing joint stance planning in its separate checkout while
this run uses only the terminal-motion change. Physical results remain pending.

Stage1b physical result is **6/8**, versus the contemporary baseline7/8.
`30700:1` and `30700:8` failed measured foot-drift admission before capture;
there were no wrist/table clearance failures or recovery slips in this batch.
This is a provisional candidate, not an established improvement in success.
`stage1_terminal8_review.json` verifies all frozen source hashes after the run;
exit1 is the unmet eight-success quota, with normal stack shutdown. GPU0 released.

The two failing cases had only0.12–1.63mm solver-reported foot error and roughly
1–2mm nominal COM shift at their last reach samples, yet measured right-foot
drift reached67/80mm. The drift helper uses the full actual root quaternion.
Static supported IK feasibility therefore cannot be reported as actual support
stability; execution-time measured checks remain essential. The successful
case tails still moved6.3–55.7cm after handback. No timing reduction is admitted.

The reusable FK reviewer now also records phase hand-speed/acceleration and
height range, with phases clipped at loaded-upright/outcome physical timestamps.
Video failure cleanup must be excluded when explaining a pre-abort failure:
logging-batch `30700:4` moved its hand downward during its2.08s attempted lift;
the later visible upward swing belongs to cleanup and is not evidence for the
cause of the earlier slip. Successful baseline lift paths rose roughly28–50cm
despite the nominal6cm lift term, because torso recovery contributes additional
motion. Raw recordings and failures are preserved.

Stage2 implementation is active in the worker. CPU probes show a full-body
final-grasp solve takes127–399ms, so walking-time planning needs an asynchronous
worker with isolated IK/guard data and nonblocking consumption. Pending work
retains the current reference and any still-valid plan; measured post-stop
revalidation must feed the actual reach posture rather than a distance-only
lean heuristic. Geometry checks do not replace measured support guards.

A narrower takeover hypothesis is recorded in
`stage1_terminal8_takeover_review.json`: the two foot-drift failures start
SupportedMotion with measured root XY speeds0.174/0.149m/s after
`StreamedArm.start`; successful cases are all at or below0.044m/s. The consumer
passes measured `pelvis.vel` as the IK's virtual `root_velocity`, while POSE
packets contain root quaternion but no root-position trajectory. At the existing
0.20m/s² root acceleration limit, these initial velocities imply76/56mm braking
travel, comparable in scale to the observed80/67mm foot drift. This association
is not causal proof, but it requires checking both post-stop and post-takeover
stability and distinguishing measured rebound from an incoming reference
translation derivative. Joint/root-angular reference continuity remains required.


## Takeover-only diagnostic and handback prerequisites

The frozen `stage2_takeover_probe` changes only initial virtual root translation
velocity to zero; joint and root-angular reference velocities remain inherited.
Its fixed regression run `stage2_takeover_zero8` returned **5/8**, with two
close-phase wrist/bottle clearance failures and one lift-phase measured foot-drift
failure (50.97 mm, 6.06 s after capture with bottle rise 31.2 cm). Measured takeover XY speed was 0.154 m/s in the foot failure, but
0.168 m/s in one success. This intervention does **not** establish better success
or eliminate actual support instability. The command, source checkpoint, all
attempts and integrity checks are preserved under the run's sibling evidence
files. Runner exit1 means the eight-success quota was unmet; stack shutdown was
normal and GPU0/instance0 released. Five successful two-second handback tails
still moved 17.5–52.0 cm.

`stage3_handback_precondition_review.json` examines the last 0.40 s before
handback in all 13 successful logging/terminal cases: least-squares measured
root XY speed 0.002–0.101 m/s, hand speed 0.047–0.397 m/s, and maximum leg-joint
speed 0.108–0.287 rad/s. FK uses actual full root orientation and measured
joints. Root motion near zero can still precede 36–53 cm of post-switch travel.
A root-speed gate alone therefore does not demonstrate a stopped hand, stopped
legs or compatibility with the standing planner; stage3 must address these
separately. These are descriptive measurements, not identified dynamics or
validated new acceptance thresholds.


## Joint stopping and whole-body stance candidate

Worker commit `2d686e6` is integrated as `stage2_stance`. It adds a bottle-owned
private IK/collision planner in one background worker; walking holds its bounded
reference while a plan is pending and reuses valid results. Measured settled
posture is re-solved before and after POSE takeover, with collision-admitted
continuous braking of inherited joint/root-angular rates before the latter
check. The final solved joints and torso pose guide reach. Cooperative
cancellation between solves lets measured settled requests supersede walking
batches without waiting for all remaining candidates. Existing measured foot,
collision, capture and contact guards remain.

The frozen root CPU replay reproduces walking joint feasibility7/8 and settled
feasibility8/8. Median/max walking batch time is2.036/2.244s; settled solve
0.230/0.253s; submit/poll maximum0.774ms. `stage2_stance_root_cpu.json` and its
receipt identify input/config/scene/source hashes. The worker commit hook and
105 focused tests passed; the root integration's five related suites also
passed (103 tests). These are kinematic and software checks. The three sampled
uncertainty points do not cover a continuous nonlinear region, and replay
endpoint alternatives retain the recorded arm posture.

The fixed eight-case physical run is `stage2_stance8`, on GPU0/instance0.
Its results must be assessed before treating the candidate as a motion
improvement. Stage3 world loaded path and measured stable handback is in progress
independently; nominal durations remain unchanged.


The physical `stage2_stance8` result is **0/8** and rejects this candidate.
Seven cases fail the settled straight-joint-interpolation finger/table sweep
at fractions0.35–0.625; `30700:2` instead fails the walking execution-prediction
finger/table check while braking, before a settled solve. All are before POSE
takeover. Every final walking snapshot lacks acquired wrist vision and reports
a missing, pending or stale full-body plan; some live batches take over3s.
Thus coupling local arm preparation to completion of the full final-stance
search blocks useful preparation during the short approach. The previous
recorded-takeover CPU bank did not exercise these actual unprepared starts.

`stage2_stance8_review.json` distinguishes these gates and retains every attempt.
All frozen source hashes match; collection and owned stack shutdown exit0;
runner exit1 is the unmet quota. GPU0/instance0 is released. The worker is
repairing plan consumption and actual reach-path validation before Stage3.
No collision margins are relaxed, and the unused fresh20 bank remains reserved.


The guard diagnosis is in `stage2_live_stance_guard_review.json`: all seven
settled cases have clear start and endpoint in an **approximate** state replay,
with collision along the speculative straight joint/base interpolation. These
are nearest-state reconstructions using boot wall/monotonic mapping; they are
not exact recorded request inputs. In `30700:2`, the exact recorded execution
failure state is already inside the4mm finger/table margin (distance3.773mm),
and pending planning emitted no arm increment. The contact sheet
`stage2_pending_arm_review.png` confirms the arm stayed down throughout approach.
The repair therefore needs both independent safe wrist preparation/conservative
parking while pending and validation of the supported Cartesian path actually
used for reach. New settled request snapshots must retain exact input state,
reference, target and physical timestamp.

## 2026-09-16：联合规划修复版 Stage2.5

冻结提交 `95a83f9d2253521c5155caf2eff6e6ff96620001`，主树快照 `outputs/bottle_dynamics_20260915/stage2_5_stance`。待规划时恢复碰撞筛查的局部抬手准备；已选候选沿实际笛卡尔进入路径校验，保留已知参考/实测偏差和实时碰撞守卫。POSE期间保活显式保持POSE；准备判稳使用连续实测关节/手运动和参考速度，不以固定绝对跟踪误差排除静态偏差。

主树112项相关测试通过；CPU重放仍为步行7/8、站稳5/8。可行站稳求解中位/最大1.281/1.322秒，全部站稳候选最大2.965秒；提交/轮询最大0.729毫秒。旧快照不能验证连续判稳，三例几何失败保留，不能据此宣称物理成功率提升。主树复现记录 `stage2_5_root_cpu.json`，19个冻结源码哈希核验一致。固定8例物理回归 `stage2_5_stance8` 启动；Stage3带物/交接实现与该冻结版本分离。

Stage2.5实测固定8例 **1/8**，所有尝试保留。全部8例均抬手并获得腕视野，上一版两项均0/8。失败为2例规划等待后恢复walking触发物理时钟间隔保护、4例3秒准备预算内未判稳、1例POSE接管后规划超时；`30800:0`成功。所有8例行走规划命中率0，均有过期结果；代码检查发现有效性契约比较了随行走正常变化的ego桌边距离，需改用世界/物体相对桌边几何。修复方向为固定安全准备目标并连续制动、等待时维护物理时钟/碰撞检查、修正计划有效性；不放宽碰撞/脚漂移/抓取门槛。证据 `stage2_5_live_review.json`，19个冻结哈希一致；采集和关闭均返回0，未满足8成功配额导致runner返回1。GPU0/instance0已释放。启动期出现短时CPU限流，之后采样无持续限流且余量13–14核；不将全部失败归因于算力。

## Stage2.6：固定准备参考与桌边契约

`abd6e8aa8ccdea0d6625be0cee401c70eda5b332`连续制动走路手臂参考、等待时持续维护物理时间与碰撞检查、使用物体/桌边相对距离判定场景变化。主树四个相关测试套件通过；21个冻结文件哈希核验一致。固定8例仍 **1/8**（`30700:8`成功），时钟间隔失败消失；2例POSE后规划超时，另5例均在3秒准备预算末只有2个相隔0.26秒的判稳样本，参考速度0且测量趋势低于门槛。规划保活指令间隔通常约20毫秒，不能将该采样异常归因于长时间主循环阻塞。证据 `stage2_6_live_review.json`。下一步定位判稳调用时序；不能仅放宽判稳或时间预算。采集和关闭完成，GPU0已释放。

Stage2.6两例接管后超时的**精确请求**已离线重放：`stage2_6_post_takeover_timeout_bank.json` → `stage2_6_timeout_cpu.json`。两例最终手位置误差仅1.0–1.1毫米，但名义路径达到160步上限，耗时3.07/4.06秒，躯干残差停留0.031/0.036弧度（路径要求0.02）。终点IK与路径rollout使用不同的手/腰软目标权重，造成终点/路径目标不一致；worker正在统一，而非增加等待掩盖不收敛。

Stage2.7 `11be8ec`修复准备采样：每个物理tick观测，支撑运动时清空历史，提交仍等待0.25秒支撑静止。主树相关26项测试通过；固定8例运行中。当前进一步发现准备或支撑暂时不静止就立即清空待求解请求，三秒内可反复提交6次；计划改为保留计算、仅在原有静止和当前实测状态有效性均通过后采用结果。

Stage2.7固定8例最终 **2/8**（`30700:2,:8`），5例接管前规划超时，1例接管后超时；21个冻结哈希一致。所有行走规划仍未在停车前采用（7 pending、1 stale），因此尚不能声称实际停车由联合规划结果选定。证据 `stage2_7_live_review.json`。GPU0/instance0释放，继续修复计算结果保留与软目标一致性。

线程数反事实：CPU探针使用OpenBLAS1线程，采集入口环境未指定、同venv默认32线程。固定两个超时请求重放的站稳求解1线程4.061/3.068秒，默认32线程3.693/3.122秒，仍同样不可行；行走2.180/2.751秒对2.241/2.921秒。该诊断没有支持线程数是主要超时原因，不引入额外线程配置修改。记录 `stage2_6_timeout_cpu_threads32.json`。

## Stage2.8：实际终点、控制目标与规划生命周期一致

冻结 `f34b47208fc8958fb47b6d6aa63f5e2838a210db`。准备/支撑短暂运动期间保留求解，但采用时仍需原静止时间和实测状态/目标匹配；从首次提交开始的超时在所有等待状态中生效。终点IK、名义路径及live使用一致的手/腰权重；保留显式`control_*`目标供live执行。路径返回其实际达到并经碰撞检查的终点，对该终点检查抓点、速度、脚/重心和自然姿态限制；去掉精确复现中间软IK躯干角的内部0.02弧度条件，保留残差诊断，并要求末端参考速度≤0.05。没有放宽物理碰撞/脚漂移/抓取/成功门槛。

两例旧超时精确请求均通过：45/54路径样本，求解1.202/1.580秒，手位置误差0.9/1.2毫米。证据 `outputs/bottle_dynamics_20260916/stage2_8_terminal_endpoint_cpu.json`，SHA256 `612391b11ea0e268191fcb7d93debe35bae3df53e17c4bba8384c9f6c476019b`，主树核验实现哈希一致。worker106项相关测试和完整提交hook通过；主树四个相关测试套件通过。固定8例 `stage2_8_stance8` 启动。

Stage2.8固定8例最终 **1/8**（`30800:3`），7例接管前规划超时；21个冻结文件哈希一致。名义路径收敛修复有效，但仍因实测快照变化重算而不能推进任务。读取最后请求附近1.2秒状态窗口，超0.08弧度变化主要发生在腿关节2/6/8，部分达0.11–0.34弧度，骨盆XY大多未超0.025米；这说明全关节静态快照前提与平衡控制存在矛盾，但不是每次失效原因的完整逐条记录。下一步增加逐次失效诊断，在线使用快速全身终点IK，平滑笛卡尔运动由现有每帧碰撞/实测支撑/抓取约束监督；完整名义路径保留离线审查，不再作为长时间等待的必要条件。不得将终点可行声称为整段轨迹预验证，也不放宽物理守卫。GPU0释放。


## Stage2.9：快速在线终点检查

冻结 worker commit `fb28484d42dd2b0210793063b491705ef9df681b`。两个旧超时精确请求，在线 walking/settled 均 2/2；walking 中位/最大 0.975/1.040 秒，settled 0.169/0.178 秒。离线完整路径同样均 2/2，分别需 2.712/2.878 秒与 1.309/1.396 秒。原始 CPU 证据为 `outputs/bottle_dynamics_20260916/stage2_9_endpoint_only_cpu.json` 和 `stage2_9_full_path_cpu.json`。新增逐次失效原因、请求时间、目标/根部/关节偏差记录；主树四个相关测试文件全部通过。固定 8 例尚待实测，不据 CPU 可达性宣称物理成功。

固定 8 例物理结果 **5/8**（failure-heavy bank，不代表总体成功率）：`30700:2` 腕/瓶碰撞守卫，`30700:4` 指2/瓶碰撞守卫，`30800:3` 闭爪超时；其余成功。无 settled 规划超时，settled 求解中位 0.322 秒、最大 0.508 秒；所有 walking 计划仍 pending/stale，停车前采用率 0/8，故联合停车目标尚未验收。`30700:8` 唯一 settled 失效记录是根部平移 16.3 毫米超过 15 毫米合同，后续重算成功。所有冻结文件哈希一致；collection/shutdown 均返回 0，runner 因未满 8 个成功返回 1。证据 `outputs/bottle_dynamics_20260915/stage2_9_live_review.json`。不据此声称超过 Stage1 logging7/8，也不缩短时长。


## Stage3：世界坐标带物与实测交接（未通过物理验证）

冻结 `a9369fa795dac5c11916172a3d88f3ff0c0e83fe`，含 `778e10a`。竖直方向使用连续五次曲线，实际离桌后才内收；命令升高 0.085 米（保留实测 0.060 米要求，加实际解析配置 `stance_ik_tolerance=0.025` 米跟踪裕度），内收仍为 0.060 米。保留5秒名义时长和原有速度限制，进度由实际位置/姿态跟踪及双侧接触确认决定。站立姿态目标与固定脚、COM、碰撞终点检查同时求解；腿关节相对全零参考的残差仅作诊断，不将未经标定的0.35弧度当成物理门限。

交接前先对持有的真实参考做制动，持续持有零速POSE参考；手、根部、腿与物体相对运动必须在连续物理时间窗内稳定。仅释放一次控制权，保留证据并观察独立IDLE窗后才允许carry导航。记录切换请求前后wall、state wall/t/seq和窗末状态；不是controller消费确认。

主树五个相关测试文件及worker提交hook通过。`outputs/bottle_dynamics_20260916/stage3_loaded_endpoint_cpu.json` 的两个名义终点2/2，求解中位/最大0.1156/0.1158秒。该同步检查超过50毫秒观测连续性窗，物理测试必须核对close→lift命令连续性；CPU可达不等于实际稳定。

Stage3固定8例结果 **0/8**，候选不采用。`30700:3` 接近时髋/桌守卫；其余7例进入lift后最大进度仅0.028–0.36，其中6例超时、1例长时间等待后滑落。所有冻结文件哈希一致，collection/shutdown返回0；原始manifest与评审保留在 `outputs/bottle_dynamics_20260915/stage3_loaded8`、`stage3_loaded_live_review.json`。无人到达handback，不可据CPU测试声称物理交接稳定。

进度阻塞证据：`30700:1`/`:2` 中位置误差≤25毫米占96%/94%，但全掌姿态变化≤0.035弧度只占0.27%/2.75%；该门限原意是静止合爪窗口的姿态波动，直接用于带物全程跟踪可能语义不适合。不能盲目放宽：`:0` 后期瓶子倾角很大，确有持物风险。需区分任务所需重力方向/物体相对夹爪稳定、允许跟踪误差和静止门限。首例同步endpoint solve耗时0.200秒，跟踪基准在求解前取样，首个lift循环已偏0.054弧度；下轮还需修复参考连续性与基准时序。辅助证据 `stage3_loaded_partial_tracking_review.json`、`stage3_loaded_rotation_review.json`（后者记录FK角度基准复现残差）。fresh20仍未使用。


## Stage2.10：选中停车方案的合同

冻结 `08c426ff4a75e72b31d10ac9b19468290d62eed3`。只根据选中search identity核验合同，非选候选的排序/增删、未参与标准站姿评估的测得root-Z不再使结果失效。保留目标、目标/桌沿相对偏移、选中root XY/yaw、stop-edge和当前IK/腕视野检查。当前重新求IK与缓存候选的upper关节差仅记录，不当作真实执行状态变化。已采用计划每个规划tick做便宜几何复核；pending保守fallback允许短轮询，ready立即进入当前候选检查。新增逐次消费/失效事件和时间来源。原守卫、时长、增益与求解器不变。

主树四文件相关测试112/112与worker提交hook通过。CPU相同精确请求 walking/settled 均2/2；walking中位/最大1.046/1.165秒，settled0.170/0.179秒，证据 `outputs/bottle_dynamics_20260916/stage2_10_endpoint_only_cpu.json`。尚无物理采用率结论；等待Stage3.1完成后联合回归，避免重复运行已知带物失效版本。Stage3入口已另核对：7例同步求解0.129–0.289秒，对应附近POSE发送间隔0.175–0.333秒、样本发送时年龄0.179–0.353秒；`stage3_loaded_entry_timing_review.json`记录原始对齐，不能解释成controller消费间断。


## Stage3.1 asynchronous loaded endpoint and moving supervision — 2026-09-16

Frozen worker commit `ca0abebe11f59eb3f5803951dfa52db3bd3cd34a`, root checkpoint
`outputs/bottle_dynamics_20260915/stage3_1_loaded`. The endpoint worker owns its
ArmIK/collision model and copies the live nominal fixed-support problem. Pending
close ticks continue POSE sends; consumption validates request-to-current geometry
and initializes lift tracking from the current observation. Collision candidates
apply measured offsets to joints and root consistently. Request foot movement
is distinct from the original cumulative 5 cm support guard.

Moving speed now scales continuously with translation/angular tracking, bottle
gravity tilt, relative grip rotation, and bilateral contact. The initial grip
rotation is latched once; persistent orientation stops have bounded recovery.
The new orientation/tracking thresholds are task priors, not identified dynamics.
The original physical acceptance and collision/support guards remain.

Validation: root focused suite149/149 passed; worker installed hook passed
(845 collected, 9 skipped). Versioned worker CPU replay
`outputs/bottle_dynamics_20260916/stage3_1_loaded_endpoint_cpu.json` uses the
new runtime request and private evaluator, admits2/2, solve median/max
0.142887/0.142913 s, implementation hashes5/5 verified. This is endpoint-only
evidence. Combined fixed8 `stage3_1_loaded8` is running through the existing
launcher on GPU0/instance0; source hashes are frozen, fresh20 untouched.


Stage3.1 live result: **0/8** on the fixed difficult bank; candidate rejected.
Six reached loaded motion, two stopped at wrist/bottle guards. Five loaded
attempts ended on new orientation supervision, one on tracking-starved lift
timeout. No handback was exercised. Collection and owned stack shutdown exited0;
pilot exited1 because the quota was unmet. All25 frozen source hashes match.
GPU0/instance0 released; GPU1 untouched.

`stage3_1_loaded_live_review.json` records each attempt. The private solve took
0.340–0.854s while pending states sent5–10 POSE references; adjacent sender gaps
around consumption were0.080–0.173s. This removes the single synchronous solve
from the sender but does not prove controller-consumption timing or sufficiently
regular commands. Loaded angular feedback stayed frozen in all six attempts;
`SupportedMotion.step` currently receives measured rotations only during reach.
Case30700:2 reached progress0.829 and >6cm measured rise, then angular tracking
error~0.21rad stalled completion despite bilateral contact.

`stage3_1_grip_swing_review.json` separates cylinder-axis swing from axial twist.
Using first consumed request state as capture approximation,30700:0 had full
relative rotation0.1415rad but axis swing only0.0215rad;30800:3 had full0.2174
and swing0.2154rad, a real axis change. FK reproduction residuals0.0002–0.0035rad
are retained. Thus full-SO(3) grip rotation can falsely reject harmless axial
twist, while some failures remain genuine tilt/axis deviations. Moving thresholds
are still uncalibrated priors. The endpoint quiet admission also still compares
to the lift-entry bias, rather than measured pairwise quietness, and needs repair.


## Stage2.11 fixed navigation frame — 2026-09-16

Worker commit `485446d0fb75313ebdfe0bb4dbc93ba113ba85af`, root checkpoint
`stage2_11_stance`. Candidate geometry now uses the independent current world
target plus navigation heading and arrival lane, passed from existing navigation
and approach owners. The future root is the intersection of the selected depth
and the fixed arrival lane; its yaw follows the navigation heading. Local arm
preparation is evaluated in that candidate frame. Actual target/edge changes
still invalidate the contract, and all geometric thresholds remain unchanged.
Root walking/approach/stance tests46/46 pass. This checkpoint retains the known
failed Stage3.1 loaded behavior and is not run live separately; await the loaded
feedback/axis-swing/quiet-window fix for combined physical validation.


Additional pre-fix review: `stage3_1_motion_quiet_replay.json` replays the existing
measured stability gate on full-rate recorded FK motion, with bilateral contact
forced true only to isolate motion readiness. Cases30700:0/2/4 yielded63/309,
46/1138,0/618 ready states;30700:2 had only one ready state in the final5s.
Raw palm FK position differs from the recorded outcome by0.358mm on30700:2.
This shows both gate attainability and residual real motion; it is not a contact
acceptance or success replay. A still at video31s shows30700:2 already upright
and carrying the bottle centrally, but a still alone cannot establish smoothness.

Review found a remaining Stage2.11 integration mismatch: the selected depth is
now expressed on the fixed navigation axis, while approach stop/speed still use
`_rel_now` in current pelvis yaw. Worker will fix the coordinated path in an
independent commit before the combined run; non-coordinated behavior is preserved.

Existing post-switch validation is0.20s. It must be reported as short-window
evidence and followed by longer explicitly requested-IDLE observation once real
handback is reached. Historical carry backoff must not be counted as IDLE drift.


## Stage3.2 loaded rotation feedback and measured quietness — 2026-09-16

Worker commit `49e6b9588ab62122c5494dbe531eda4e49a89d76`, root checkpoint
`stage3_2_loaded`. Loaded ticks again provide measured rotations to the existing
bounded compensation. Closing freezes its value and clears the stale pre-close
rate; resumption preserves the held value and starts with zero correction rate.
Full grip rotation remains diagnostic; the new `loaded_grip_swing_*` controls
use bottle-axis swing, so cylinder axial twist is not treated as loss of grip.
The numerical priors are unchanged. Endpoint quietness now uses the measured
palm/root/legs/object-relative window, replacing the complete static angle/error
conjunction. Existing rise, lean, contact, collision and support criteria remain.
The async worker now stamps actual completion; delayed polling no longer hides
result age. Main loaded/whole-body/supported tests121/121 pass.

This is not yet live-validated. A separate stop/speed frame correction is required
before the combined fixed8 run. No new timing reduction or fresh20 use.


Stage3.2 aligned checkpoint `db18399823caca2c43fd3e6ba63753b17bbf07cf` also
fixes the actual stopping interface. Coordinated visible stop, complete command
speed (including the coarse/mid band), and stall progress use the fixed
navigation depth. Non-coordinated/non-visible paths keep the previous ego depth.
Diagnostics record both the chosen frame and ego depth. Main navigation-related
tests47/47 pass; the loaded tests121/121 already passed on unchanged loaded code.
Worker hooks passed. Combined fixed8 `stage3_2_aligned8` started after checking
GPUs/locks and a fresh CPU sample (15.09 free cores, zero throttling). Source27
files are frozen by receipt; no fresh20 use or timing reduction.


## Stage3.2 aligned fixed8 result and Stage3.3 switch gap — 2026-09-16

The codex session ended after launching `stage3_2_aligned8`; the run completed
at 04:31 UTC unaudited. `stage3_2_aligned8_live_review.json` (from
`review_stage3_2_aligned8.py`, read-only) records **0/8**, 27/27 frozen hashes
unchanged, collection/shutdown exit 0, pilot exit 1 (quota).

| Family | Cases | Evidence |
|---|---|---|
| handback post-switch `clock_gap` | 30700:0 / :3 / :8 | reached loaded upright (rise 0.074–0.082 m, lean 0.6–2.3°), pre-switch gate ready 11 samples / 0.20 s, first post-switch sample 0.060 s after the switch sample, gate limit 0.05 s, post-switch sample_count 1, held 0.0 |
| loaded grip axis swing | 30700:1 (recovery timeout), 30800:0 (limit) | swing 0.122 / 0.203 rad against the 0.120 rad stop prior |
| clearance guard | 30700:2 finger/table, 30700:4 finger/bottle (reach); 30800:3 wrist/bottle (close) | unchanged guards |

Walking adoption: `stance_plan_consumptions` 1 in six of eight rounds and
`prestop_stance_plan` feasible in the same six (solve 1.4–2.0 s); 30700:0 was
invalidated by `target_edge`, 30700:2 hit a reach guard first. This is the first
batch with any pre-braking consumption; it is adoption evidence, not a stopping
quality result.

Handback: the release call took 32–35 ms wall in all three cases and `obs()`
waits for the next 50 Hz state, so the first post-switch sample is one frame
beyond the 0.05 s gap limit while the sidecar shows the state stream continuous
at 0.020 s across the switch. The reset was therefore the gate's own boundary,
not physical instability; post-switch motion was never measured and handback
stability remains untested.

Stage3.3 (worker `0e914d8`, root checkpoint `stage3_3_switch`): the gate accepts
one declared boundary gap up to the quiet window (0.20 s), keeps the pre-switch
samples so motion caused by the switch is still measured against them, and the
first post-switch sample and its gap are recorded under
`handback.switch.first_post_switch`. A gap beyond the window or any later gap
still fails. No threshold, timing, guard or config change; focused tests
171/171 in the root tree, worker hook passed. Combined fixed8
`stage3_3_switch8` launched via `launch_stage3_3_switch8.sh`.

`stage3_3_switch8` result: **0/8**, 27/27 frozen hashes, collection/shutdown
exit 0, pilot exit 1 (`stage3_3_switch8_live_review.json`). No round reached the
ownership switch, so the boundary-gap change was not exercised live.

| Family | Cases | Evidence |
|---|---|---|
| loaded grip axis swing, recovery timeout | 30700:2, 30700:8, 30800:0 | swing 0.122 / 0.151 / 0.171 rad; 30700:8 and 30800:0 stopped 0.28 / 0.24 s after lift entry (6–7 lift samples), i.e. the bottle rotates in the grip as it leaves the table; stop prior 0.120 rad, recovery 0.40 s |
| wrist/bottle clearance | 30700:1 (close), 30700:4, 30800:3 (reach) | unchanged guard |
| finger/table clearance | 30700:3 (reach) | unchanged guard |
| lift timed out | 30700:0 | progress reached 1.0 at 13.5 s, object rose 0.071 m, then the held reference sagged to 0.049–0.058 m measured rise for 10 s below the 0.06 m requirement (commanded rise 0.085 with 0.025 margin); the same case held 0.073 m in the previous run |

Walking adoption: `stance_plan_consumptions` 1 and `prestop_stance_plan` feasible
in 8/8 (6/8 in the previous run). Run-to-run repeatability: only 30800:3 failed
the same way twice; the other seven cases changed family although nothing before
the switch changed between the two checkpoints. The eight-case bank therefore
does not repeat, three-of-eight handback arrivals in one run and zero in the next
are within its scatter, and it cannot by itself demonstrate the switch fix.
Open items for the user: calibrate the swing priors against the recorded
lift-entry swing, the loaded hold sag against the 0.06 m rise requirement, and
the wrist/bottle guard family; none were changed here. Fresh20 untouched; no
timing reduction.


## Stage3.3–3.8: diagnosis-driven loaded-phase fixes — 2026-09-16

All runs use the same fixed8 bank, launcher and frozen-hash receipts
(`review_fixed8_live.py <run> <checkpoint>`); each is 0/8, so the evidence is
the stage reached and the per-stage telemetry, not the success count.

| Checkpoint (worker) | Change | Reached | Families |
|---|---|---|---|
| `stage3_3_switch` (`0e914d8`) | post-switch gate tolerates one declared switch gap | 3 lifts, 0 switches | 3 lift-entry swing stops, 4 guards, 1 hold sag |
| `stage3_4_liftoff` (`3aa19d2`) | swing supervision only after measured liftoff (`loaded_liftoff_rise_m` 0.01), grip relation re-latched once at liftoff, on-table rotation recorded | 3 lifts, 3 liftoffs | 5 guards, 3 lift timeouts: `rotation_error_rad` 0.15–0.19 bound the governor after compensation resumed |
| `stage3_5_frozen` (`1b78da7`) | rotation compensation frozen through the lift | 5 lifts | palm tilted 0.30–0.40 rad, bottle followed (tilt 0.32–0.35): 3 tilt, 2 swing, 2 disturbed, 1 guard; rejected |
| `stage3_6_command` (`49895d8`, incl. `e9da5eb`) | rotation error measured against the corrected command; Dex1 housing may touch the bottle like the pads | 7 lifts, 0 housing guards | rotation error still 0.19–0.23 median, governor 0 in 5/7; tilt 0.04–0.10 |
| `stage3_7_governor` (`d8368c7`) | rotation tracking error diagnostic-only in the governor | 8 lifts, 4 full-progress, 3 switches | 3 post-switch motion, 4 lift timeouts (quiet window never formed), 1 tilt, 1 swing |
| `stage3_8_hold` (`d7a768f`) | both compensation loops frozen once at progress 1 (`stats.loaded_hold`) | 7 lifts, 6 holds, 4 upright, 4 switches | 4 post-switch motion (planner re-posture), 3 lift timeouts (2 holds never quiet, 1 stalled at 0.58 without admitted contact), 1 finger/table guard |

Evidence behind each step (all read-only reviews under the evidence root):

- Lift entry (`stage3_3_switch8` 30700:8/30800:0): swing 0.15/0.17 rad within
  0.24–0.28 s of lift entry with object rise 0 and the hand not yet raised; the
  table pins the bottle, so palm rotation reads as swing. On-table rotation at
  liftoff was later measured at 0.04–0.16 rad (`loaded_liftoff`).
- Compensation: rotation feedback already carries 0.21–0.30 rad at close in every
  case. Frozen (3.5) the palm tilts with the bottle; resumed (3.4/3.6) the
  bottle stays within 0.24 rad but the palm-vs-command offset drifts ~0.2 rad
  through the lift whichever reference it is measured against. That drift is the
  controller's orientation tracking under load and is now diagnostic only.
- Housing guard (`housing_clearance_review.py`, 8/16 rounds of 3.3+3.4): at every
  wrist/bottle failure the pads were already 1–11 mm into the bottle while the
  Dex1 housing on `right_wrist_yaw_link` was 0–4 mm from it; backing the bottle
  5–10 mm cleared the housing with the pads still engaged. After the allowance
  (3.6/3.7) the family did not recur in 16 rounds.
- Terminal entry (`review_reach_tracking.py`, `stage3_3_switch8` last 2.5 s): the
  hand sits 4–5 cm below the deploy's own reference in 6/8 cases with ±2–4 cm
  oscillation at 0.4–0.8 Hz; across 24 reaches the hand wanders 4–5 cm
  peak-to-peak sideways and ~4 cm vertically in the last 3 s against a 5 mm
  depth tolerance. This drives the remaining reach/close guards and the two
  "disturbed target" aborts (bottle pushed 3 cm by a lateral sweep). Not
  addressed; it is the next limiter.
- Switch (`review_switch_jump.py`, 3.7): after the planner takes over, legs move
  up to 0.2 rad and the hand 6–9 cm within 0.5 s, root 3–8 mm. The expert ends
  the lift 0.42 rad from the planner's standing legs, so post-switch motion is
  real re-posturing; the gate now measures it correctly. Handback needs a
  planner-compatible lower posture before release, which is unimplemented.
- Hold (`review_hold_quiet.py`, 3.7): root/legs pass the quiet limits ~100% of
  windows, the hand moves 7–17 mm per 0.2 s at 4–8 cm/s, the translation
  compensation's own rate; pre-switch windows passed only under a frozen
  reference. With the frozen hold (3.8) four of six holds were admitted upright
  1.1–4.9 s later at 8–9 cm rise and 1.4–2.1° lean, then braked, passed the
  pre-switch window and were released; every release ended in the planner
  re-posture above. The upright residual of the legs was 0.37–0.44 rad in all
  four.

Run-to-run repeatability stays poor (1/8 identical failure modes between 3.2 and
3.3 on identical pre-switch code); stage counts are reported, not success rates.
No thresholds, timings or fresh20 use; no VLA training.


## Stage4: simple single-arm expert on the crate pattern — 2026-09-16

User decision after the Stage3.3–3.8 audit: the single-arm expert had grown to
15.5k lines with 20 abort gates, most of them millimetre-level supervisors of a
controller that tracks the hand to 4–5 cm; the crate line (6.9k lines,
align → reach → close → lift) reaches 71/100 closed loop because the task is
tolerant of that error. Rebuild the single-arm expert in that shape, and lower
the bottle's centre of mass so a pad that touches first slides the bottle
instead of toppling it.

Worker `67a0122`, root checkpoint `stage4_simple`:

- `reach.strategy=simple` (`simple_policy.py`, `simple_config.py`,
  `configs/tasks/bottle_simple.json`). Walking phase: the existing hand follow
  without the stance planner. At the stop: one POSE takeover, the preview's
  own brake to rest, one `SupportedMotion` executor. Phases: pre-grasp point
  12 cm behind the grip on the pelvis-to-grip line (2.5 s minimum-jerk) →
  entry along the line at 6 cm/s, closing when the measured tool is within
  1 cm of the grip or any finger touches the bottle → 1.2 s closure, bilateral
  pad contact required within 0.8 s → 8 cm lift over 3 s with the hand riding
  the reference torso → 2 s torso recovery → planner handback with the
  reference preserved. Gates: collision sweep (housing allowed), fall, bottle
  knocked > 5 cm before the close, contact after the close, ≥ 4 cm rise,
  timeouts. Nothing is gated on quietness or tracking error.
- Scene `outputs/scene/sim_lowcom`: `build_sim_scene.py --preset bottle_only
  --harden-pads --object-com-drop 0.05`. Bottle mass 0.353 kg unchanged, centre
  of mass from mid-height (8 cm above the base) to 3 cm above the base; critical
  tilt 15.7° → 36.9°, tipping energy barrier ×2.4. The jaw's open width is at
  its joint limit (4.9 cm for a 4.5 cm bottle), so the one-pad-first contact
  cannot be removed by opening wider. Fixed bank re-stamped as
  `regression8_lowcom.json` (identical geometry, provenance file alongside).
- Tests `test_bottle_simple.py` (config, dispatch, phase geometry, walk tick,
  handback); focused suites pass in root and worker; config doc regenerated.

`stage4_simple8` first full batch (`stage4_2_simple`, after two crash fixes:
an unbound first-step flag in `SupportedMotion` when only translation feedback
is frozen, and numpy in the round record): **0/8**, all geometric, all in the
first seconds of the policy. Two hips at the table: the copied lean formula
asked for 19° at a 0.26 m stand-off. Three fingers at the table: the hand sat
4–5 cm below the grip while the fingers hang 5.8 cm under the tool point and
the grip was 7.2 cm above the table. Three misaligned captures: the jaw has
2 mm of clearance around the 4.5 cm bottle and cannot open wider (joint limit),
the hand wanders centimetres, so one pad pushed the bottle 2–9 cm (it slid, as
the low centre of mass intends) or the jaw closed near the cap and lost it in
the lift. Two rounds reached the close and one the lift, so the takeover,
entry, closure and handback path itself runs.

`stage4_3_simple` (worker commit after `d6b6781`): lean from the distance
term only; grip 3 cm above the bottle centre (`grip_below_center=-0.03`);
before closing, the bottle's offset in the palm frame must be within 1.5 cm on
the jaw axis and 3 cm in height, otherwise the hand retreats 8 cm along the
line, shifts by the measured offset and re-enters, up to three entries
(`stats.entries` records each).

`stage4_simple8_stage4_3`: **1/8**, the first complete success of the rebuilt
expert (30700:2 on its third entry: grasp, 8 cm lift, torso recovery, planner
handback, 35 cm back-off walk and terminal hold in 39 s). Three rounds used
all three entries without aligning, one lost the bottle in the lift, one
finger/table, one hip/table, one entry timeout. The records show why: during
the 2.5 s pre-grasp the measured hand lags the moving command by 5–9 cm, so
the entry starts from a hand still catching up; and the retreat-and-shift
retries chase a wandering offset that flips sign between entries (+2.8 →
−2.7 → −3.8 cm laterally in one round).

`stage4_4_simple`: pre-grasp over 4 s, then a coarse settle (measured tool
within 2 cm of the command, ≤ 3 s); the entry advances at 4 cm/s to a 6 cm
stand-off where the pads are still clear of the bottle, waits up to 4 s for
the wander to bring the bottle inside the jaw window (1.5 cm lateral, 3 cm
vertical), then dashes at 8 cm/s and closes on arrival or bilateral contact;
retreat-and-retry remains the fallback.

`stage4_simple8` (stage4_4): **1/8** again; 30800:0 completed with a 22 cm
rise. The stand-off wait worked: 3 of 4 entries that reached it aligned within
0.02–1.5 s and closed on arrival. The remaining families are calibration
trade-offs, not logic: four hips at the table at the start of the pre-grasp
(lean 0.13–0.40 rad, because the base drifts back after the stop and the
pelvis-to-grip distance is 0.4–0.5 m at takeover, so the distance term still
leans); two lifts in which both pads held but the bottle slid down (pads 3 cm
above the centre with the centre of mass 8 cm below them, a pendulum the 20 N
clamp does not hold); one alignment exhaustion. Decisions for the user:
`simple.lean_max_rad` 0.55 → ~0.25 or a farther stop that accounts for the
drift; `grip_below_center` −0.03 → −0.02 with `--object-com-drop 0.04`; the
1.5 cm alignment window against the jaw's 2 mm clearance. All batches, crashes
included, are preserved under `stage4_simple8_*`; `review_simple8.py <run>`
prints the per-round anatomy.

What the rebuild established regardless of the count: the whole path
(takeover, pre-grasp, stand-off, dash, closure, lift, torso recovery, planner
handback, back-off walk, terminal hold) runs in 40–70 s with no quiet windows
and no tracking-error gates; the two successes both went through the planner
handback that every Stage3 checkpoint failed on. The expert is 500 lines
(`simple_policy.py`) against the 3 000-line whole-body policy it replaces for
this strategy; the whole-body strategy remains selectable.

## Stage4.5–4.7: hanging-arm walk, raise after the stop — 2026-09-16

User decision: the walk-in tracks nothing. The arms hang as the deploy left
them, the body stops as close as the hands allow, and after the stop the hand
goes up beside the body, over the edge, down to the pre-grasp and in along the
approach line (the crate pattern, and how a person does it at a low table).
One fixed path, no "would the raise hit the table" branch: at any feasible
stand the hanging hand is behind the edge, and the scene has one table height,
so a branch would never vary in the data. Side effect accepted: the wrist
camera sees the bottle only once the hand is over the table.

Code (`stage4_5_hang`, worker `9b88fce`): `walk_tick` holds the ramp's
hanging reference (17-vector, zero velocity) and runs only the sweep and the
target observer; a `raise` phase precedes the pre-grasp
(`simple.raise_above_grip_m` 0.08, `raise_forward_m`, `raise_seconds` 2.5)
and re-plans grip, direction and pre-grasp from the settled base when it ends;
`lean_angle` takes the pelvis heading and uses the forward depth to the grip.
The stand puts the bottle 0.24–0.30 m to the robot's right at 0.30 m of depth,
so the straight-line distance had leaned the torso 0.13–0.40 rad for a lateral
offset no lean can shorten (the 4/8 hip/table of stage4_simple8).

`stage4_hang8` (brake 0.27 depth / 0.23 edge, 4 cm closer): **1/8** (30800:3,
rise 10.9 cm, lean 0, 46 s). Five rounds ended on the first raise tick with
the left fingers at the table, one with the right fingers at the bottle during
the raise, one with a right finger at the table in the entry. The hanging hands
sit 10 cm ahead of the pelvis at 0.53–0.72 m, the table-top band; the settled
stand was 0.22 m from the edge and the bottle 8–14 cm from the hanging right
hand.

`stage4_hang8b` (`stage4_6_hang`, worker `79e355c`: brake back to 0.31/0.27,
`raise_forward_m` 0): **0/8**, every round fingers at the table on the first
raise tick. The state traces explain it: the brake request fires at the same
0.26–0.27 m edge distance as stage4_simple8 and the base overshoots
0.10–0.13 m in both; with the raised arm the base then drifted back to
0.27–0.33 m before the takeover, with hanging arms it stays at 0.13–0.19 m.
`stage4_7_hang` (worker `17ccbb0`) moves the brake 10 cm earlier (0.41/0.37)
so the takeover stand lands at the 0.23–0.27 m of the successful rounds.

`stage4_hang8c` (`stage4_7_hang`, brake 0.41/0.37): **1/8** (30700:4, rise
16 cm, 45 s); 30800:3 grasped and lifted (rise 13 cm) and then lost the bottle
in the hold after the planner handback; six fingers at the table on the first
raise tick. Moving the brake out did not move the stand: the request fired at
0.36 m but the robot was faster there and overshot 0.19–0.26 m, so the minimum
edge distance was again 0.09–0.29 m. Across the four batches the overshoot
after the request is 0.02–0.26 m and is set by the gait speed at the brake,
not by the brake point; the two hanging-arm successes were the two rounds that
happened to be slow (≤ 0.34 m/s) at the request. The reason the deceleration
band never slows the walk is in `gait_speeds.py`: v1.1's shuffle band (0.20)
sits below its 0.28–0.30 speed step, so the band's command falls through the
step and the gait runs until the IDLE. `stage4_8_hang` (worker `be5f21f`) adds
`reach.visible_shuffle_speed` (0 = checkpoint band; the simple task uses 0.35,
0.083 m/s measured) with a 0.50 m band on the 0.31/0.27 brake, so the last
half metre is a crawl the stop can end within centimetres.

`stage4_hang8d` (`stage4_8_hang`, floor 0.35 over 0.50 m, brake 0.31/0.27):
**1/8** (30700:4, 42 s); 30800:3 again grasped and lifted (rise 7 cm) and lost
the bottle in the hold after the handback (pads 4.4 cm above the bottle centre
at capture, the pendulum case of the pending `grip_below_center` decision);
one bottle knocked in the entry; five fingers at the table on the first raise
tick. The state traces show the delivered walk is 2–3× the gait table
(0.54–0.69 m/s at command 0.36–0.40), so the 0.50 m band is crossed in under
a second and the gait barely slows before the request (0.22–0.54 m/s); the
stands landed 0.14–0.20 m from the edge, still inside the hanging hands'
0.22 m. `stage4_9_hang` (worker `17d645e`): floor 0.32 (just above the step),
band 0.60, brake 0.34/0.30 → `stage4_hang8e`.

`stage4_hang8e` (`stage4_9_hang`, floor 0.32 over 0.60 m, brake 0.34/0.30):
**2/8** (30700:0 rise 15 cm and 30700:4, 42–50 s); 30800:3 lost pad contact
in the lift; five fingers at the table. The crawl now holds the request speed
at 0.31–0.49 m/s (overshoot 0.08–0.16 m, stands 0.146–0.248 m from the edge).
Reproducing the guard offline on the takeover states
(`ArmIK` + `TransportGuard` on the state traces) settles the mechanism: the
hanging hands' geometry reaches 0.15–0.18 m ahead of the pelvis at table-top
height (0.55–0.72 m), so stands nearer than 0.22 m are in contact with the
table top (0–8 mm) before any motion, and two stands at 0.18–0.23 m still lost
1–3 cm to the first IK tick; the successes had 5–12 cm of margin. The hands
need ≥ 0.22 m of pelvis-to-edge distance, which the crawl's overshoot cannot
deliver from a 0.30 m brake. `stage4_10_hang` (worker `f367cf7`): brake
0.44/0.40 with the same crawl → `stage4_hang8f`.

`stage4_hang8f` (`stage4_10_hang`, worker `f367cf7`: brake 0.44/0.40, floor
0.32 over 0.60 m): **5/8** — 30700:0/1/2/4/8 in 42–48 s, rises 8–24 cm,
lean 0 in every round, every success through the planner handback and the
back-off walk. The stands settled 0.20–0.31 m from the edge in seven rounds
(request speeds 0.26–0.55 m/s); 30800:3 arrived at 0.79 m/s, stopped at
0.13 m and hit the table on the first raise tick as before. The two other
failures are the lift/hold family: 30700:3 grasped, lifted 20 cm and lost the
bottle in the hold after the handback; 30800:0 lost pad contact in the lift
(the pads-above-centre pendulum of the pending `grip_below_center` decision).
Eight full-duration previews (≤ 500 000 bytes, `verification.json` all_ok)
are under `outputs/bottle_dynamics_20260915/stage4_hang8f/previews/`.

State of the line: the hanging-arm expert is the configuration in
`configs/tasks/bottle_simple.json`. Its remaining limiters are (1) the stand
scatter from the gait-phase stop overshoot, which the crawl band narrows but
does not remove (1/8 fast arrivals), and (2) the grip height vs the bottle's
centre of mass in the lift and the post-handback hold (2/8). Neither is logic.

## Stage4.11–4.16: settled stance, continuous reach, inward wrap — 2026-09-16

User review of the `stage4_hang8f` videos: feet one ahead of the other, the hand
pulled back to an awkward angle before going forward, a hesitant entry, and a
twisted arm on a bottle to the right. Diagnosis and changes (worker
`c563da9`…`7d9c976`, root `stage4_11_wrap`…`stage4_16_wrap`):

- **Stance** — the takeover followed the stop request by a second, mid-step.
  `settle_for_admission` now holds the hanging arms until legs and base are at
  rest (0.5 s, timeout 4 s) and records the stance. The planner's IDLE does not
  square the feet: settled stances still show 3–19 cm of forward offset, so
  this is the planner's behaviour, recorded not fixed.
- **Path** — the raise kept the hanging xy while the fingers turned level (the
  palm swung 15 cm back) and the pre-grasp lay inboard on the pelvis-to-bottle
  diagonal. One minimum-jerk arc-length profile now runs (pull-back,) raise,
  slide; the fingers stay down through the raise and turn on the slide; the
  raise height is max(grip + 8 cm, table top + 12 cm); the pull-back (0.15 m/s,
  both hands, body-only screen) happens only when the measured hand sits in the
  table-top band within 6 cm of the edge, with the edge distance carried from
  the last walk tick by the base motion since (the stop overshoots after it).
- **Entry** — one 8 cm/s advance with the jaw-axis correction riding along;
  retreat-and-retry is a straight back-off used at most once; the height
  window is measured on the grip point; the jaw-axis window is 2.5 cm (the
  hand wanders 4–6 cm p-p and the low-CoM bottle slides to the centre).
- **Wrap** — `wrap_direction` turns the fingers inward with the bottle's
  lateral offset (1.2 rad/m, cap 20°). An offline IK sweep on the eight
  recorded stands gives 65–70° of joint margin from fingers-forward to 20°
  inward, 42–55° on the old outward bearing and 17–27° at 60° inward, so the
  human-style 60° wrap is not what this arm does comfortably.
- **Camera** — at the settled stand the bottle sits below the ego frame; the
  takeover accepts an admission within `target_grace_s` (8 s) of the walk-in.
- **Grip height** — the hand now tracks the pre-grasp to 1 cm, so the 3 cm the
  old line added for its sag put the pads 5–6 cm above the centre; the closure
  itself lifts the tool 2–3 cm at its onset (compensation active or frozen),
  so the grip aims at the centre (`grip_below_center` 0.0) and the pre-grasp
  is 18 cm back (the hand overruns the end of the slide by 6–8 cm).

Batches: `stage4_wrap8` 2/8, `8b` 1/8, `8c` 1/8 (six reached the lift),
`8d` 3/8 (seven reached the lift; four closure-rise slips), `8e` **6/8** with
the compensation active through the closure and the 2.5 cm window (one
closure-rise slip, one bottle knocked by a 4 cm entry wander), `8f` **8/8**
with the grip at the centre (40–44 s per round, rises 10–20 cm, lean 0,
pads 1–4 cm above the centre at capture). Previews of both under
`outputs/bottle_dynamics_20260915/stage4_wrap8{e,f}/previews/`. Still
recorded, not fixed: the planner's IDLE leaves the feet 3–19 cm apart
fore-aft; the entry wander (4–6 cm p-p) that the 2.5 cm window and the
sliding bottle absorb; one table height, so the raise/retract branches never
vary in the data yet.

`stage4_wrap8g` (`stage4_17_wrap`, worker `ae60d59`): the user's three
follow-ups. The jaw closed and reopened after the stop because the streamed
arm inherited the commander's cached hand targets (the previous round's
closed jaw); both hands are now set open at the takeover. The 2.2–3.0 s of
still hanging arms between the brake and the first hand motion is what a
policy learns as "stop at the table" (visible200), so the takeover now
happens the tick the gait ends (base < 0.10 m/s, legs < 0.50 rad/s, no
dwell; the planner never squares the feet anyway) and the reach profile
moves on its first tick; brake-to-hand is 1.0–1.3 s. Phase times per the
user's decision: raise 2.0, slide 2.5, entry 12 cm/s, close 1.0, lift 2.0,
recover 1.5 s; the arm motion is ~10 s and rounds take 35–41 s. Result
**6/8**: one closure-rise slip (pads 6 cm above the centre, the residual of
that family) and one late loss in the faster lift/recover after an 8.8 cm
lateral swing. Previews under `stage4_wrap8g/previews/`.

## Stage4.18–4.19: wrist cameras off the body — 2026-09-16

User review: both wrist cameras see the robot's own body during the walk and
the raise. A 9-ray cone test on the `stage4_wrap8g` traces
(`review_wrist_clearance.py`) put the nearest own-body hit at 2–6 cm (hip)
during the walk and 1–2 cm during the straight-up raise: the deploy's own
hang (shoulder yaw 0.79, elbow 1.08) points the camera, which sits on the
wrist-yaw link looking along the fingers, at the thigh. Changes (worker
`60796fd`, `ebb3770`): the walk ramps both arms (0.6 rad/s) to a configured
hang chosen by an offline sweep of the same ray test, and the raise ends
6 cm outboard and 5 cm ahead of the hanging tool point.

`stage4_wrap8h` (`stage4_18_wrap`, hang shoulder pitch 0.25 / elbow 0.8):
the cameras were clear (worst own-body distance ≥ 0.12 m in every round, most
rounds no hit at all) but the forward forearm changed the gait: every stop
overshot to 0.08–0.10 m from the edge (0.13–0.29 m in `8g` at the same
brake), the body met the table, **2/8**. `stage4_19_wrap` keeps the deploy's
arm mass distribution (elbow 1.2, shoulder pitch 0, roll 0.15 outboard) and
lets the wrist joints do the camera work (roll 1.2, pitch −0.6, yaw −0.4,
mirrored left): no own-body hit in the cone, camera looking straight down,
hand extent 0.13 m → `stage4_wrap8i`.

`stage4_wrap8i` (`stage4_19_wrap`): **5/8**, cameras clear in every round
(nearest own-body point in either wrist cone ≥ 21 cm, most rounds none), stops
back to 0.18–0.33 m; two table brushes (a 5 cm forward raise component from
6 cm behind the edge; a finger on the top as the tool sagged 5 cm during the
closure) → `stage4_20_wrap` (worker `d22e090`): vertical outboard raise, 8 cm
pull-back margin, no right-arm sweep during the closure. `stage4_wrap8j`:
**5/8**, 36–38 s per success; two fast arrivals (0.58–0.64 m/s at the brake
despite the crawl, stops at 0.11 m, the base creeping another 5 cm forward
during the reach until a hip met the table) and one closure without
bilateral contact. Previews under `stage4_wrap8j/previews/`. The fast-arrival
family is now the remaining stand limiter.

## Stage4.21–4.24: the takeover blip is the deploy's default clip — 2026-09-16

User review: the jaw still closed and reopened after the stop, a hand jump
with it, and the arms rolled outward during the walk. Traces
(`review_takeover_excursion.py`): the walk-time roll was the hang ramp still
running (moved to a standing `prepare_walk` before the first step, worker
`c25674f`). The blip is at `StreamedArm.start`: for 0.3–0.5 s both wrists
reset to the deploy's default, the elbow to 1.0 and all four finger joints
close, then the stream takes over. In `deploy.cpp` the command handler
answers `planner=False` by selecting the motion reader's clip
(`--default-motion neutral_kick_R_001__A543`), and only a pose chunk that
forces a merger catch-up (a session's first packet) assigns the streamed
motion again; chunks primed before the switch made the post-switch chunk an
ordinary window extension. Reordering `manager_state`/`command` (`b8420f3`)
changed nothing; priming a chunk between them (`936b32b`) cut the blip to
0.12–0.36 s; sending the session's first chunk there (`817e1a1`,
`stage4_24_wrap`) removes it: `stage4_wrap8n` round 1 jaw never below 0.015,
wrist dip 0.005 rad. Batches under a host load of 150–320 (`8k`–`8m`):
stand-up failures and deeper stops (0.09–0.17 m) are that load, not the
expert; `8l` 4/8, `8m` 5/8.

`stage4_wrap8n` (`stage4_24_wrap`): **7/8**, 38–41 s per success; the one
failure a deep stop (hip at the table). Takeover blip gone in every round
(jaw never below 0.015, wrist dip ≤ 0.01 rad); wrist cameras clear of the
body; arms taken to the walking hang standing (0.9–1.2 s) before the first
step, nothing moves during the walk. Previews under `stage4_wrap8n/previews/`.

## Stage4.25–4.27: untwisted hang, switch confirmed from the deploy's echo — 2026-09-16

User review: the walking hang must not twist the wrist at all; move the whole arm away from the body instead. Offline ray sweep on the `stage4_wrap8n` walks (25 rays per 80° wrist-camera cone, 35 gait phases × 4 rounds): shoulder roll 0.25 with a zero wrist hits the thigh at 12 cm, 0.35 leaves 23 cm, **0.45 leaves 35 cm right / 49 cm left** (hand 30 cm from the pelvis, fore-aft unchanged at 7 cm ahead), 0.55 is fully clear but visibly spread. Worker `a6c56cf` (root `stage4_26_wrap`): `hang_shoulder_roll_rad` 0.15 → 0.45, `hang_wrist_roll/pitch/yaw` → 0, `raise_outboard_m` 0.06 → 0; `hang_reached_rad` 0.15 (worker `67d19a3`) now takes `prepare_walk` to 0.8 s instead of the 3 s timeout.

`stage4_wrap8o` **7/8** (previews `outputs/bottle_dynamics_20260915/stage4_wrap8o/previews`): stops 0.12–0.33 m from the edge (same distribution as 8n), one deep stop (0.122 m, hip/table), camera cones ≥ 19 cm from own body in every round (was ≥ 18 cm with the twisted wrist). One round (30700:2) played the deploy's default clip for 0.25 s at the takeover although the first chunk went after the switch.

Read from the patched deploy (`zmq_manager.hpp` 228–300, `zmq_endpoint_interface.hpp` 290–372/470–520): the tick that sees `planner=False` runs a safety reset (the reference becomes a frozen copy of the reader's default clip) and then the ZMQ toggle, which sets `has_new_data_=false` and so drops a pose chunk that already arrived; the command and the chunk travel on two SUB sockets, so the chunk sent right after the command can land in that gap; the streamed motion is selected only when a later chunk is decoded with a catch-up. The deploy's `g1_debug.body_q_target` is built from the motion it is playing, so it is the only proof of the switch. Worker `a694e12` (root `stage4_27_wrap`): `StreamedArm._confirm_streamed` re-pushes the held frame every tick until a `RefTap.snapshot()` newer than the switch matches all 14 arm joints within 0.03 rad, re-issues the switch once after 1 s, gives up (recorded) after 2 s; result in `handoff_diagnostics["switch"]` and `stats.takeover.switch`. Unit test with a deploy that drops the first chunk.

`stage4_wrap8p` **6/8** (previews `outputs/bottle_dynamics_20260915/stage4_wrap8p/previews`): the switch was confirmed in 0.02–0.05 s in all 8 rounds (1–2 pushes; the second push was needed in 6/8, i.e. the first chunk is dropped most of the time). Residual: when the first chunk is dropped the default frame plays for 2–3 deploy ticks (30700:3: jaws to 0.009 for 0.06 s, wrist dip 0.16 rad), which is the deploy's floor — the reset and the toggle happen in one tick and the next chunk is decoded a tick later. Failures: 30800:0 deep stop (0.10 m, hip/table); 30800:3 the measured tool sagged 5 cm below the entry command over the last 0.8 s of the entry (grip 4.9 cm above the tool at arrival, 1–2 cm in the other rounds), the retreat ran at the sagged height and a finger touched the table. The entry law corrects the jaw-axis offset only; a height correction of the same form is the open item.
