# ScaleBFM 视觉抓瓶采集策略交接

> **2026-09-10事故后核查：outputs已空，ABCD进程不在运行，无法续跑。下文ACTIVE/2-of-120等都是删除前历史。**
> 核心源码/干预模块/tests/configs/docs仍在，3项ABCD单元测试通过；实验run.py、plan/progress、全部逐集结果及依赖的微调checkpoint/场景/清单已缺失。
> 最后对话核实2/120不代表删除时最终进度；目前无法验证ABCD最终结果。未启动恢复训练或GPU工作。
> 详细影响与恢复边界见docs/scalebfm_outputs_loss_20260910.md。不要把重建的计划或历史文字当作原始结果。


> ABCD正式运行已启动，driver3039355，初始进度2/120（A/B seed31001已完成），运行日志run_v2.log。
> 当前状态以outputs/experiments/scalebfm_pelvis_abcd_20260909/progress.json为准；异常看failure.json。
> 原始失败在failure_v1.json，并非当前运行故障。正式A31001与新离屏基线1200步逐项一致。


> Pelvis ABCD protocol v2 ACTIVE: old EGL baseline not repeatable (identical own state,1-level RGB differences).
>  OSMesa+deterministic math produced bit-identical full24s original/diagnosticA forseed31001.
>  Fresh30 A +90 B/C/D =120 formal cases, same30 seeds/interventions; VLA+ScaleBFM inferenceGPU0,
>  renderingCPU. Do not compare with oldEGL25/100 as paired baseline. PROTOCOL_REVISION.md documents
>  change before B/C/D outcomes; run_v2.log and progress.json track execution. Old failure_v1.json is archived evidence.
>  Root:outputs/experiments/scalebfm_pelvis_abcd_20260909/. No retraining or cancelledGoal restart.


> 最新授权：GPU0上骨盆ABCD对照，实验目录outputs/experiments/scalebfm_pelvis_abcd_20260909/。
> 30固定诊断场景、每层10，A复用原记录前先三场完整重演验证；B/C/D共90新rollout。
> B保持76cm；C低于72cm持续0.1s后恢复；D同触发暂停右手1s并0.5s渐放。参数见PLAN.md/plan.json。
> 新模块visual_pelvis_ablation.py和test_bfm_pelvis_ablation.py；原VLA/专家未修改。任务不涉及训练。
> run.py保存逐场完整标记；先检查progress/failure，不要重复已完成场次或并发启动driver。


> Failure analysis2026-09-09:docs/scalebfm_failure_analysis_20260909.md. Third-person export has
>  missing mj_camlight: light runtime direction remainszero; same-frame luma42.28->65.95 after update.
>  This is an export bug, not training RGB lighting change; current ZIP not rerendered yet.
>  38VLA cases descend below0.60m;29fall labels are solelyrootz<0.45 with0physics warnings.
>  Relative pelvis-z action is clamped bycheckpoint q99 to+1.836cm, restricting recovery from low poses.
>  7cases have>=3gripper switches;55never command near-full closure. No strategy or formal experiment changes.
>  Evidence:outputs/diagnostics/scalebfm_failure_analysis_20260909/. Read report before proposing fixes.


> 300视频小体积导出已完成：`outputs/videos/scalebfm_yplus200_eval100_small/scalebfm_200_collection_100_evaluation.zip`。
> 大文件夹内collection_200和evaluation_100，包含所有评估失败；480x400/12.5fps侧前方全身跟随，原速完整记录。
> ZIP53,748,107字节，每条86–375KB、中位163KB；300条编码帧数与ZIP完整性通过，14张代表帧已查看。
> 仅回放保存qpos，不重采集、不重推理。export.py可复用已有视频；manifest.json/verification.json为验收证据。


> 最新完成状态：yplus200实验已于2026-09-09 12:39:49UTC完成，200条新数据、3000步训练、100集闭环均完成。
> VLA成功25/100：高桌10/34、低桌近瓶10/33、低桌深瓶5/33；失败29超时、17闭合前碰瓶、29物理警告/跌倒。
> 最终报告 outputs/experiments/scalebfm_yplus200_s17_20260909/RESULTS.md；逐集证据evaluation/。
> 不得重复已完成实验；已知信息缺口仍按用户要求保留。下文12/3000及运行中PID仅为历史进度。


> 进度更新：新yplus200采集已完成，200次尝试全部接受，高67/低近67/低深66；导出和统计完成。
> 训练曾在模型初始化因强制offline阻断tokenizer元数据查询而失败，0步、无checkpoint；失败证据归档training_startup_failure_01/。
> 仅移除driver额外offline环境变量，配置/数据不变；恢复后GPU0已实际完成12/3000步。
> driver2458090、trainer2458112；100集闭环尚未开始。详情见实验RECOVERY.md与pipeline_v2_state.json，不要重复采集。


> 最新授权：当前专家采集200条，在GPU0训练VLA并做100集闭环测试；已记录的先验缺口暂不修正。
> 新实验 `outputs/experiments/scalebfm_yplus200_s17_20260909/`，先读PLAN.md及pipeline_v2_state.json。
> 训练seed17/3000步，最终checkpoint；100普通reset、y+、独立seed31001:31101，不额外加恢复扰动。
> 独立于已取消的ABC Goal；顺序自动执行collect200/export/stats/train/eval/report，失败停下待诊断。
> 运行中不得另启重复driver；完成阶段、逐集结果都可恢复。200条为通过审计的数据，失败尝试保留。


> 最新有界检查（2026-09-09）：完整专家审计见 `docs/scalebfm_policy_audit_20260909.md`。
> 已加 reset/1.60s/实际学生首帧头部可见性硬门，修复相机选择、entry朝向与固定瓶高抬升判定。
> 49项测试通过；三场正常CPU运行3/3抓取与审计，两场不可见注入2/2被拒绝并保留有效失败文件。
> 证据 `outputs/diagnostics/scalebfm_policy_audit_20260909/validation.json`；不得重复已完成五场。
> 修改文件：visual_visibility.py（新增）、visual_sensor.py、visual_planner.py、run_visual_bottle.py、
> audit_visual_case.py、collect_visual_cohort.py、export_visual_dataset.py；测试test_bfm_visual_visibility.py（新增）、
> test_bfm_visual_planner.py，及本交接/STATE/审计文档。源码均位于src/teleop_free/scalebfm/。
> 未解决：专家仍借助物体/桌面真值代理，尤其桌边无独立可见性门；固定路线与隐藏历史的VLA影响未验证。
> 下一步优先桌边可观测性与视觉不确定性验证，再做教师动作歧义检查；没有授权恢复旧Goal或自动训练。
> 旧10视频属修复前版本，未重收；固定视频检查命令会使用新的源码指纹。


> 固定功能（用户于2026-09-09授权）：**“检查采集视频”** 对应 `bash scripts/scalebfm_video_check.sh`。
> 按当前随机化挑选10组覆盖场景，真实执行专家并导出侧前方全身跟随视频，保留失败、参数、审计及打包下载。
> 只做这项CPU有界检查，不恢复旧Goal。规则见 [固定视频检查](docs/scalebfm_video_check.md)；
> 首批目录 `outputs/videos/scalebfm_check10_yplus_center_v1_26cc7ae54f7a/`，是否完成以manifest/progress为准。
> 首批现已完成：10条视频，专家任务与逐集审计10/10；40张成片解码代表帧已看，播放页与ZIP已生成。
> 这些是覆盖示范，不是VLA结果或随机成功率测试；没有重复物理尝试。

> 后续更新（2026-09-09）：用户另行授权将新采集限制为y+边中部。当前采集默认 `yplus_center_v1`，
> 沿边居中±8cm、近瓶向内12–18cm、深瓶18–24cm，机器人从y+侧接近。
> 42项回归测试和CPU三场完整专家冒烟3/3通过；详见 [新采集记录](docs/scalebfm_yplus_collection_20260909.md)。
> 未恢复旧 Goal，未启动批量采集/训练。下文冻结哈希、三侧分布、30/30资格与ABC17结果均属于修改前历史，
> 不能当作新分布的验收结果；旧实验文件与快照保留。

交接日期：2026-09-09（UTC）。工作区：`/workspace/code/locomanipulation`。
核对时 HEAD：`343d9f9c0d13a66a36bec4dc59f3c7f7a9a9c7dd`，实际 Git 分支为 `main`。
`docs/STATE.md` 中的“branch=65”是与当前 Git 状态不一致的历史标注；不要据此切换分支。

## 0. 当前状态：用户已取消，禁止自动恢复

**用户最新执行指令是“先暂停，取消之前的 goal”。本次请求只授权编写交接文档，不授权恢复实验。**

- 【事实】2026-09-09 07:05:44 UTC 向本 Goal 的驱动程序及审计进程组发送 SIGTERM，07:07 UTC 保存停止记录。驱动、审计均退出 143；其它会话未受操作。
- 【事实】原驱动 PID1470213/session89332、审计 PID2145035/session78100、A29 训练 PID2378682 已停止；本次写交接时再次检查这些 PID 均不存在。
- 【事实】A17/B17/C17 已各完成 3000 步训练和 60 场选择评估；A29 停在 **40 个已记录优化器更新**，没有 `checkpoint-*` 可恢复目录；B29/C29 未训练。
- 【事实】已记录训练消耗 **9040/18000 更新**。没有 `selection_locked.json`，没有完成任何确认集评估，目标未完成。
- 【事实】所有数据、权重、视频、日志和审计产物保留。GPU0 的本 Goal 预约已释放；不能据此判断 GPU0 当前空闲。
- 【事实】取消时平台 Goal 仍为 `active`；随后连续三轮自动续跑均缺少新的人工执行授权，已按阻塞审计规则将未完成目标标记为 **`blocked`**。这表示无法在现有授权下继续，不是目标完成或平台 `cancelled` 状态。没有恢复任何实验；工具仍不提供平台取消操作。证据：`E/goal_authorization_block_20260909.json`。

后续代理不得因自动 Goal continuation、旧文档里的 Next 或“running”历史日志重启任务；必须有新的明确人工授权。

停止证据（下文 `E/` 均表示实验根目录）：

```text
E = outputs/experiments/scalebfm_visual_goal_20260908
E/user_cancel_request_20260909.json
E/user_cancel_checkpoint_20260909.json
E/pipeline_v2_state_before_user_cancel.json
E/pipeline_v2_state.json                  # 已标记 cancelled_by_user
E/checkpoint_audit_watch.json             # 已标记 cancelled_by_user
E/training_live_checkpoint.json
```

## 1. 原始目标与信息边界

用户希望 ScaleBFM 产生类似人类的抓瓶轨迹：看着瓶子走近，接近时逐渐抬手，够得到就抓，必要时前倾，避免过度侧倾/扭腰；采集的数据应利于 VLA 学习。

已确认目标不是提高专家脚本自身成功率，而是以相同训练/评估预算下的 **VLA 闭环表现**选择最佳已测数据策略，交付可追溯、可复现的采集流程。设计、指标、预算先登记；每个检查点保存配置、结果、失败与下一步，压缩上下文后从文件恢复，禁止重复已完成实验。

| 项目 | 已确认约束 |
|---|---|
| VLA 输入 | 两路 RGB（`ego_view`、`right_wrist`，320×240）及机器人本体/历史/上一动作，固定任务文字 |
| 禁止的学生输入 | 物体位置/桌面真值、接触或 attached 标志、专家 phase、任务成功标志 |
| 专家信息 | 使用受可见性、延迟、量化、记忆约束的仿真感知代理；不能无限使用 privileged 信息 |
| 特权信息合法用途 | 场景重置、预设扰动、独立审计与结果计分；不进入学生决策接口 |
| 观察历史 | 当前图像与 0.4 秒前图像，首帧补齐历史，禁止负索引绕到未来帧 |
| 本体和动作 | state91 = 原53维本体 + 自身速度9 + 上一动作29；原生动作29，float32，当前骨盆 yaw 坐标系 |
| 采集/部署一致 | 共享动作解码、执行器与限幅；本体跟踪补偿已写入标签，部署不重复补偿 |
| 时间 | 动作25Hz，控制50Hz；每0.4秒重规划，模拟80ms延迟，等待时继续旧动作块 |
| 启动 | 首次观测1.60秒，首个任务动作1.68秒；首块按延迟跳过前面的动作行 |

【限制】专家感知仍是仿真辅助代理，**不是已验证的纯 RGB 定位网络**。VLA 接口不读真值与专家完全无需真值是两个不同命题。

## 2. 恢复阅读顺序与资源约束

1. [docs/STATE.md](docs/STATE.md)：共享资源和其它会话状态，优先于旧交接中的时间估计。
2. 本文第0节、[docs/scalebfm_goal_state.md](docs/scalebfm_goal_state.md)、`E/user_cancel_checkpoint_20260909.json`。
3. [设计](docs/scalebfm_visual_expert_design.md)、[实验计划及修订](docs/scalebfm_goal_plan.md)。
4. `E/seed17_interim_comparison.json`、`E/selection_{A,B,C}_s17_audit.json`，再读[结果报告](docs/scalebfm_vla_comparison_results.md)和[验收证据](docs/scalebfm_acceptance_evidence.md)。
5. 需要历史原因时再查 `docs/scalebfm_goal_history_20260908.md`、`docs/scalebfm_goal_history_20260908_c4.md`、`docs/scalebfm_goal_history_20260909_c5_044144.md`。其中 ACTIVE/Next 是历史状态。

部分旧报告正文仍有“C17评估进行中”或“以后审计C”的取消前描述；**停止记录和完整 C17 JSON 是当前事实**，不能因此重跑 C17。

- 不启动录制 stack，不使用其它会话端口，**永不执行 `stop_stack.sh`**。
- 停止时 STATE 记载其它 9d 工作占 GPU1、instance0/port5590，预计约07:30结束。预计时间过去不等于资源获释；恢复前重新核对，不擅自终止或挪动其它进程。
- 06:15曾观察到 GPU0 同时有外部 SONIC 部署（约1.3GiB）；见 `E/gpu0_shared_load_20260909_0615.json`。共享负载对训练波动的因果影响未验证，时间指标需要保留该背景。
- 正式解释器：`third_party/Isaac-GR00T/.venv/bin/python`；已用环境为 Python3.12.3、torch2.9.0+cu128、MuJoCo3.11、numpy1.26.4。不要切换系统 Python 做正式模型/物理实验。
- **实际冻结的 GR00T 源码在 `src/gr00t/`，不是 `third_party/Isaac-GR00T/gr00t/`。** 不修改共享 venv/库来临时修复问题。
- 本 Goal 没有提交代码。HEAD343d9f9 是另一会话提交；不得把其它 dirty 文件或历史提交算作本任务工作，不自动 commit/push/清理。
- 当前另有 `docs/ready_grasp_arms_20260907.md` 修改，属于其它工作。已有数据、退役数据、审计 JSON 和其它模型均须保护。

## 3. 已确认决策及理由

| 决策 | 理由与证据范围 |
|---|---|
| 同时平滑腕位置、旋转和切换偏置 | 旧脚本只平滑 pad 位置，约60°腕旋转跳变通过15.15cm手掌偏移产生约12cm腕目标跳变；单变量诊断确认其贡献于侧倾 |
| 保留 mode2，不直接换 mode6 | mode6也没有显式 torso 约束；已测肩肘参考的腕误差约4.9–8.4cm，大于 mode2 的3.1–3.4cm。不能推出所有 mode6 方法无效 |
| 动作由可见信息与实测接近程度驱动 | 避免学生看不到的秘密阶段/固定计时分支；低置信度暂停精细接近，视觉重新观测后再更新 |
| a16 用实测身体到达判据修正停早 | 内部参考已到位时，实际身体仍可能差>10cm；保留有界前进意图直到实际进入走廊，遮挡时停止 |
| 使用共同原生29维执行器 | 保留原生速度、避免依赖未来真实骨盆，减少录制与部署间偏差；同环境精确重放已有证据 |
| 比较 A/B/C 三组 | A是旧脚本经共同执行器生成的控制组；B是a16；C以恢复示范替换25%普通示范，用同样200条训练预算测试恢复数据价值 |
| a15 旧 B 队列退役 | 修正停早问题后采用a16；145次旧尝试仍计入采集成本，不能挑掉失败后“归零” |
| 只评最终3000步、先锁选择再确认 | 避免中间模型挑选及确认集泄漏；单次专家成功/验证MAE不决定模型优劣 |
| 保留失败和未满足项 | 姿态零违规不代表成功或自然；不能根据已看到的结果改选优规则以宣布通过 |

侧倾原始诊断在 Goal 之前，详见[单变量诊断](docs/scalebfm_sidelean_diagnosis_20260908.md)及 `outputs/diagnostics/scalebfm_sidelean_20260908/`：三个种子伸手胸部roll从18.40/17.20/19.93°降至8.90/8.80/9.70°，成功率仍均为1/3。它证明一个诱因，不能解释所有歪斜或证明稳定抓取改善。

## 4. 登记预算与验收门槛

完整修订以 `docs/scalebfm_goal_plan.md` 为准，以下是关键恢复参数。

- 每组200条训练、20条验证，最多880次计费采集尝试；保留失败和退役成本。
- A/B/C × 训练种子17、29 × 每模型3000更新：原上限18000。microbatch4 × accumulation16 = effective64；LR1e-4，AdamW8bit，weight decay1e-5，warmup0.05。
- 冻结视觉骨干和LLM，训练投影/扩散部分；state dropout0.2、全视野240/crop1、`allow_padding=True`。
- 每500步保存优化器/RNG，保留最新一个 checkpoint，另存最终根目录模型；只评最终3000步。
- 选择池每模型60场，种子21001..21060：48普通+12恢复（四类各3）。确认池每模型100场，31001..31100：88普通+12恢复。
- 推理4个去噪步、40行动作块；最长24秒；每场策略随机种子 `900000 + scene_seed`。
- 选择按两训练种子的平均任务成功率；差距在1/60以内，依次比较登记姿态违规数、实现简单性A/B/C。全部六次选择完成后先写锁定文件，再测优胜方案和A，各两个训练种子；若A胜不重复评A。
- 配对 McNemar 按训练种子报告；联合 bootstrap 按场景采样，不把同一场景的两个训练种子当两个独立场景。
- 基础快照：GR00T `2fc962b973bccdd5d8ce4f67cc63b264d6886495`，Cosmos `9ce19a195e423419c349abfc86fd07178b230561`，本地加载。

| 门槛 | 登记定义 |
|---|---|
| 抓取 | 瓶子升高≥8cm、倾角≤30°、稳定持有≥1秒，无提前碰动 |
| 专家资格/恢复 | ≥24/30资格成功，各桌高/深度层≥6/10；恢复≥8/12 |
| 物理 | 接受数据无非有限状态、MuJoCo不稳定警告、骨盆高度<0.45m |
| 腕连续 | 50Hz每步≤2cm、≤5° |
| 侧倾/扭腰/前倾 | 躯干roll峰值≤20°、>15°持续≤0.2秒；腰yaw≤25°；pitch≤40°；高桌合爪窗绝对pitch中位数≤15° |
| 接近与伸手重叠 | 成功普通接近中≥80%在身体速度仍≥0.05m/s时开始抬手；学生测量盲点见第7节 |
| 视觉 | 合爪前≥90%决策更新至少一个相机目标像素≥32、可见比例≥20% |
| 重放 | ≥9/10同状态重放成功，目标位置/旋转误差有严格一致性门槛 |

## 5. 已完成实现与开发实验

### 5.1 实现概要

连续位置/四元数参考、限制更新率/80ms延迟/5mm量化/1秒遮挡记忆的视觉代理、事件驱动 planner、a16连续接近/伸手/必要前倾/试提/持瓶、共享执行器、延迟动作块、真实图像历史、本体91维状态、隔离场景/恢复扰动、录制导出与训练评估、来源追踪、顺序驱动及自动检查点审计均已实现。

### 5.2 关键开发尝试（开发池，不是正式VLA对照）

| 尝试 | 记录结果及结论 |
|---|---|
| a00→a02 | 原始9/12；持续姿态滤波a01仅5/12，拒绝；连续切换a02恢复9/12，但仍有持续侧倾 |
| a03/a04 | grip kp8/kd0.15为7/12；更近站位与短carry的a04为10/12 |
| a05/a06 | 观测驱动版0/3卡在reach；实测预抓误差4–5cm超门槛；放宽预进入走廊后1/3，仍有接触问题 |
| a07/a08/a09 | 稳定夹爪PD后a07为6/12；可见抬升/carry修正a08为11/12；有界本体跟踪补偿a09低近场景12/12 |
| a10/a11 | 8/12、3/12，但高桌碰撞几何无效，不能当有效高桌资格结论 |
| a12→a15 | 修场景后9/12；固定接近边/直接参考仍有高桌超时；肩部对齐a15开发12/12，随后在正式集暴露停早，旧队列退役 |
| a16 | 实测到达修正后开发12/12，资格30/30且逐项审计通过，恢复11/12 |

逐版本日志/配置/失败原因在开发历史文档及 `E/dev_a*` 等目录；不要为恢复记忆重跑。

### 5.3 资格、一致性、测试

- a16资格高近/低近/低深各10/10；合爪窗绝对pitch中位数分别2.98°/25.23°/30.61°。
- 恢复11/12；失败1710为高近、可见物体移动后提前碰动。
- `E/qual_a16/startup_contract.json`：30/30，记录启动时刻、标签与上一动作一致。
- `E/qual_a16_chunk_replay10/summary.json`：10/10，同环境逐qpos和命令误差0。
- `E/qual_a16_perturbed_replay3/summary.json`：实际3cm腕偏移，原录制时长内3/3成功；不能解释为额外24秒恢复测试。
- `E/unittest_review_repair.log`：35项测试通过；独立规格/标准审查的问题已修复。不要再等待旧review agent。
- 实际GR00T加载/批次/处理器检查及冻结基础模型batch4前后向预检有限，预检0次优化器更新；见 `E/processor_batch_contract/result.json`、`E/dataset_B_contract.json`（6/6）。
- A17/B17/C17各60个实际验证点全部有限；验证MAE仅诊断，不用于选优。
- 18/36检查点审计完成，即种子17三个模型各6份；模型1030张量、优化器537状态、调度器步数、四类RNG、处理器均检查通过。**没有测试恢复后优化器更新逐位相同**。
- 实际处理器A/B/C17及A29已按各自范围检查；C17/A29覆盖全部91维state与29维action统计。`E/C17_pre_evaluation_source_check.json`：90项冻结记录匹配，另5个仿真依赖匹配基线。只证明检查时一致。

### 5.4 数据集与成本

| 组 | train/val条数 | train/val帧数 | 计费尝试/880 |
|---|---:|---:|---:|
| A |200/20|56522/5693|225|
| B（a16）|200/20|50572/5102|367，含退役145|
| C |200/20|47616/4758|225，含复用165|

B实际新执行222次。C实际新执行60次；训练150普通+50恢复，验证15普通+5恢复。C训练恢复分布为停早12、腕偏移15、遮挡12、物体移动11；验证恢复只有停早2、腕偏移3，没有遮挡/物体移动验证示范。正式闭环恢复池覆盖四类。

数据使用 `E/cohort_{A,B,C}_v2`、`E/dataset_{A,B,C}_{train,val}`；旧 `E/cohort_B` 仅审计。证据 `E/cohort_data_report.json`、各split inputs/attempts、导出清单。全部采集、导出、统计已完成，无需重复。

### 5.5 两次基础设施修复

- 训练启动最初因 OmegaConf 布尔resume字段序列化失败，0次更新。修正本任务launcher，六配置在首次实际更新前重新冻结。见 `E/training_startup_failure_01/`、training回归测试。
- A17初次评估验证loader共享模态dict，被pop动作项后触发KeyError。修正为独立deepcopy；额外消耗1次验证预测、0场闭环、0更新，未重训A17。真实20集/60点回归通过，见 `E/evaluation_startup_failure_01/`、validation测试。

## 6. 正式VLA结果：只有第一训练种子完成

`A17`表示数据组A、训练seed17；**a16专家对应B组，不是A17模型**。

| 指标 | A17 | B17（a16）| C17（恢复替换）|
|---|---:|---:|---:|
| 训练更新 |3000|3000|3000|
| 任务成功 |13/60（21.7%）|17/60（28.3%）|11/60（18.3%）|
| 登记姿态违规 |49/60|0/60|0/60|
| 成功且无姿态违规 |0/60|17/60|11/60|
| 高桌近瓶 |8/20|0/20|0/20|
| 低桌近瓶 |3/20|10/20|8/20|
| 低桌深瓶 |2/20|7/20|3/20|
| 普通场景 |11/48|14/48|10/48|
| 恢复场景 |2/12|3/12|1/12|
| 提前碰动失败 |24|15|11|
| 高度终止 |17|4|14|
| 超时 |6|24|24|
| 从未合爪 |27|15|12|
| 停住且未合爪 |14|7|1|

恢复分项：A只在遮挡成功2/3；B停早/腕偏移/物体移动各1/3、遮挡0/3；C只有腕偏移1/3，其余0/3。

训练完成时间及运行时间：A17 01:24 UTC、16832.36秒；B17 04:03 UTC、8923.33秒；C17 06:54 UTC、9647.08秒。末次loss分别0.0157/0.0189/0.0191，不能用它们替代闭环比较。

| 单种子配对比较 | 候选独赢/参考独赢 | 共同成功 | 双侧exact McNemar p |
|---|---:|---:|---:|
| B相对A |17/13|0|0.5847|
| C相对A |10/12|1|0.8318|
| C相对B |6/12|5|0.2379|

【事实】B在已测seed17点估计最高；C本次恢复替换没有带来点估计改善。以上是单种子探索性配对，未做多重比较校正，**不能宣称统计优势、普遍最优或已经选定B**。

证据：`E/selection_{A,B,C}_s17/{summary.json,validation.json,config.json}`、对应 `*_audit.json`、`*_failure_states.json`、`E/seed17_interim_comparison.json`、`E/model_{A,B,C}_s17_completion.json`。

## 7. 已确认限制与尚未验证的假设

### 已确认事实 / 尚缺证据的要求

1. B/C高桌均0/20；恢复低成功；学生自然性和鲁棒性没有全面通过。
2. A/B/C的17/4/14个 `physics warning/fall` 标签均对应骨盆z<0.45m，所有保存qpos有限且无MuJoCo警告。B的4例全低深，C为低深9+低近5。不能称为数值不稳定或已确认摔倒；A/B选定画面可见下蹲，C画面尚未人工检查。
3. “pre-close knock”是瓶子在夹爪命令开始减少>2mm前水平移动>5mm，称“提前碰动”，不等于瓶子倾覆。
4. 学生伸手检测复用评估器既有定义：相对1.68秒骨盆朝向坐标，向前≥3cm或向上≥1cm，速度取0.1秒历史。成功普通场景移动重叠A11/11、B12/14、C8/10，但检测时世界手部实际上升≥1cm仅A3/60、B3/60、C0/60；骨盆下降会触发局部向上阈值。因此不能据高重叠率宣称自然抬手通过，也不能推出检测时刻之后没有抬手。此补充诊断与专家阶段前参考指标不同，未改选优规则。
5. q01/q99归一化截断有损：腕位置误差>1cm的训练帧A2.39%、B1.74%、C2.00%；最大10.25/6.00/6.21cm，见 `E/normalization_tail_audit.json`。精确原生执行重放不证明截断后的标签也无损。
6. C是预先规定的合成恢复数据，**不是实际学生访问状态上的DAgger**。
7. 只覆盖登记桌高/深度及x-/y-/y+接近边，不能外推其它物体、相机或真机。
8. 环境145包清单是在A17第500步后采集，不是全时段环境证明或完整可安装锁文件。
9. 第二训练种子、选择锁定、独立确认、完整最终验收均未完成。取消前原计划不能作为当前运行授权。

### 假设 / 未来候选，尚未获得因果验证

- 学生访问状态补示范（DAgger）、重新设计不受下蹲混淆的协调指标、改进高桌示范分布，可能有帮助；尚未实施验证。
- 归一化尾部截断可能影响闭环，但没有单变量证据，不能用它解释当前失败。
- GPU共享负载可能影响吞吐/延迟，但当前记录不足以认定减速原因。
- 单种子B点估计最高不能保证第二种子保持相同排序。

## 8. 修改或新增文件

本任务的大部分文件目前为 **untracked**。以下为本 Goal 文件，不能靠 `git diff` 单独覆盖；`git status --short` 可核对。共享已跟踪修改是 `docs/STATE.md`；本次另新增根目录 `HANDOFF.md`。

### 文档与配置

```text
HANDOFF.md
docs/STATE.md
docs/scalebfm_visual_expert_design.md
docs/scalebfm_goal_plan.md
docs/scalebfm_goal_state.md
docs/scalebfm_acceptance_evidence.md
docs/scalebfm_visual_collection.md
docs/scalebfm_vla_comparison_results.md
docs/scalebfm_sidelean_diagnosis_20260908.md
docs/scalebfm_goal_history_20260908.md
docs/scalebfm_goal_history_20260908_c4.md
docs/scalebfm_goal_history_20260909_c5_044144.md
configs/gr00t_bfm_visual_2cam.py
```

### 实现文件（目录 `src/teleop_free/scalebfm/`）

| 功能 | 文件 |
|---|---|
| 感知、本体、相机 | `visual_observation.py`, `visual_sensor.py`, `visual_state.py`, `visual_camera.py` |
| 规划、连续动作、执行 | `visual_planner.py`, `visual_motion.py`, `visual_reference.py`, `visual_action.py`, `visual_executor.py`, `visual_chunks.py` |
| 场景、恢复、来源、基线 | `visual_scene.py`, `visual_recovery.py`, `visual_provenance.py`, `visual_baseline.py` |
| 录制、采集、导出 | `visual_recording.py`, `run_visual_bottle.py`, `collect_visual_cohort.py`, `export_visual_dataset.py`, `summarize_visual_cohorts.py` |
| 探针、重放、验收 | `probe_visual_expert.py`, `probe_visual_mode6.py`, `calibrate_visual_reach.py`, `replay_visual_actions.py`, `audit_visual_case.py`, `preflight_visual_vla.py`, `audit_visual_checkpoint.py` |
| 训练、闭环、比较驱动 | `train_visual_vla.py`, `eval_visual_vla.py`, `run_visual_comparison.py`, `report_visual_comparison.py` |

### 测试（目录 `tests/`）

`test_bfm_visual_action.py`, `test_bfm_visual_chunks.py`, `test_bfm_visual_executor.py`,
`test_bfm_visual_motion.py`, `test_bfm_visual_observation.py`, `test_bfm_visual_planner.py`,
`test_bfm_visual_reference.py`, `test_bfm_visual_report.py`, `test_bfm_visual_scene.py`,
`test_bfm_visual_state.py`, `test_bfm_visual_training.py`, `test_bfm_visual_validation.py`。

### 实验根目录中的辅助脚本与状态

`E/watch_registered_checkpoints.py`、`audit_selection_run.py`、`audit_selection_failure_states.py`、
`audit_student_extension.py`、`check_saved_processor.py`、`normalization_tail_audit.py`、
`render_expert_videos.py`、`render_vla_video.py`、`render_vla_selection_samples.py`、
`render_height_termination_samples.py`。

这些脚本及大量JSON/log/配置/权重/视频在outputs中，可能被git忽略；须按实际路径保留。
关键状态是 `pipeline_v2_state.json`、`training_live_checkpoint.json`、`checkpoint_audit_watch*`、
`model_*_completion.json`、`selection_*_audit.json`、`seed17_interim_comparison.json` 和第0节停止记录。

## 9. 已交付视频与尚未检查的画面

- [A16专家第三视角](outputs/videos/scalebfm_a16_delivery_20260909/a16_expert_seed4001_third_person.mp4)：高近seed4001，10.12秒，成功。
- [对应B17最终模型推理第三视角](outputs/videos/scalebfm_a16_delivery_20260909/vla_B17_seed21001_third_person.mp4)：固定首个held-out seed21001，22.4秒；该场超时24秒，视频省略1.6秒启动。
- 两段使用不同场景种子，不能当同重置配对或成功学生片段。均已交付、ffprobe通过，编码后代表帧已查看。来源见 `outputs/videos/scalebfm_a16_delivery_20260909/delivery.json`。
- `E/posture_ab_v1/`：A/B三个共同专家样例图片、B三个连续视频，已检查。
- `E/vla_selection_visuals_s17/A_first_three.png`、`B_first_three.png` 已查看；固定21001/21002/21003、2/4/6秒及最后一帧，共享 `cameras.json`。
- **`C_first_three.png` 已渲染完成（session88473 exit0），但用户取消前未做视觉检查，不要重复渲染或声称已看过。**
- 渲染只使用保存qpos做FK/绘制，不新增物理/策略运行。

## 10. 关键路径和命令

### 当前允许的交接核对（只读，不启动实验）

```bash
cd /workspace/code/locomanipulation
E=outputs/experiments/scalebfm_visual_goal_20260908
P=third_party/Isaac-GR00T/.venv/bin/python
cat docs/STATE.md
cat "$E/user_cancel_checkpoint_20260909.json"
cat "$E/pipeline_v2_state.json"
cat "$E/seed17_interim_comparison.json"
git status --short
```

权重路径是 `E/model_{A,B,C}_s17/checkpoint-3000`；数据路径是 `E/dataset_{A,B,C}_{train,val}`。
不要复制中间已清理checkpoint的旧命令；500..2500只剩审计记录。

### 仅供今后明确授权后的操作参考

下列命令**本次不要执行**。已完成审计通常拒绝覆盖，先判断是否已有产物。`ARM`/`SEED`必须由获授权的下一项任务指定，不能循环重做全部结果。

```bash
# 已完成ABC17；未来仅对新的完整选择运行使用
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$P" "$E/audit_selection_run.py" --arm "$ARM" --training-seed "$SEED"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$P" "$E/check_saved_processor.py" --arm "$ARM" --training-seed "$SEED"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$P" "$E/audit_student_extension.py" --arm "$ARM" --training-seed "$SEED"
```

`audit_selection_failure_states.py --arm X`、`render_vla_selection_samples.py --arm X` 当前固定seed17；ABC17均已执行，后续要支持seed29需先扩展接口，不能误审计旧seed17。

原完整流程入口（历史命令，不是可直接使用的取消恢复命令）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  "$P" src/teleop_free/scalebfm/run_visual_comparison.py --root "$E"
```

驱动有阶段/配置保护，但**不要把已标记cancelled的账本当成安全的一键恢复接口**，也不要删除状态文件绕过保护。审计watcher也不得重复启动；原实例已停止，其状态文件保留。

未来真正具备最新完整checkpoint时，训练入口支持 `--resume <该run最新checkpoint>`：

```text
train_visual_vla.py --dataset <dataset> --out <run> --seed <17或29> --resume <最新checkpoint>
```

**A29没有这个checkpoint，不能从40步精确恢复。** 重跑A29的3000步会令总花费变成18040更新；压到剩余2960步又改变原每模型3000步协议。应先明确处理40步中断成本和预算修订，再制定恢复方式，不能隐去40步或自行扩大预算。

正式GPU运行在重新获得资源后才设置 `CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0`；CPU渲染用 `CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。完整操作说明见 `docs/scalebfm_visual_collection.md`，其中“watcher已启动”等描述是取消前历史。

## 11. 下一步：按授权分开

**现在：仅交接，维持停止。** 不启动训练/评估/收集，不重渲染C、不继续旧自动Goal。

若用户以后明确恢复：

1. 先读停止记录、重新核对资源与平台Goal状态；明确A29的40步中断成本和剩余预算/协议，保留取消前账本，不覆盖。
2. 不重做ABC17、数据导出或18份审计。可先查看已经生成的C固定回放，补视觉review；将C完整结果同步到仍有旧措辞的比较报告/验收表。
3. 按获授权的恢复方案完成A29/B29/C29；B29/C29保存处理器后核对，A29先前处理器检查已通过；每500步检查点及最终3000按登记范围验证。
4. 第二训练种子各完成相同60场；检查逐场配对、失败/姿态/视觉/恢复分层；扩展目前固定seed17的辅助诊断前保留旧产物。
5. 全六个选择结果齐备后，按原规则锁定；随后做优胜方案和A各两训练种子的100场独立确认，报告不确定性和实际资源成本。
6. 按原始设计逐项最终验收，保留高桌失败、下蹲、抬手指标混淆、代理感知/DAgger未实现等限制。若未满足目标，报告未满足项及另行登记的下一实验，不把“脚本跑完/单种子B最高”说成目标完成。

建议技能：继续做失败因果定位时可使用 `diagnosing-bugs`（已用于启动问题）；有明确新实现规格时再使用 `implement`；只在用户要求审查时使用 `code-review`。这些建议不构成恢复运行或启动子代理的授权。
