# 当前任务交接：单臂瓶子抓取的行走准备与连续动作

更新日期：2026-09-14 UTC。工作目录：`/workspace/code/locomanipulation`。

**交接结论：下一版实现已推送 main，但“整体物理动作更平滑、数据可直接训练”尚未达成。**
本轮用户只要求保存交接；以下下一步尚未实施，本轮没有重新采集或训练。

标记约定：**事实**指代码、日志或保留轨迹支持的观察；**决策**指已采用的设计选择；
**假设**指需要实验区分的解释，不能作为已经确定的根因。
详细实验的权威记录是 [本版实现与验证](../handoff/bottle-direct-20260914.md)；
日常操作入口是 [单臂采集](../collection.md)，不要为继续任务重读整个仓库。

## 1. 目标与范围

- 用户希望随着机器人走近，手臂逐渐朝可见瓶子调整，直接接近并抓取；
  不希望站定后才开始准备，也不希望出现刻意编排的“上抬、后退、再前抓”。
- 用户已同意：行走时身体摆动较大，手臂不应高频追随相对物体位置。
  本版采用过滤后的低频规划和连续关节参考。
- 更长期目标是连续的 locomanipulation 示范：走近、弯腰伸手、抓取、抬升、
  恢复直立和搬离能衔接，供视觉与本体状态驱动的 VLA 学习。
- 专家允许使用仿真 privileged 几何辅助规划。不能靠学生观测无法解释的隐藏变量
  无限追踪物体，也不能靠瞬移、绑定或放宽成功条件伪造成功。
- 本轮针对单臂 bottle；不把历史双臂 crate 的可复现性作为本任务完成门槛。
  没有重新设计 crate，也没有做 VLA 训练或宣布学习效果改善。

## 2. 已确认的约束

- 使用 SONIC v1.1；本次比较使用同一份带 POSE/夹爪交接补丁的私有 deploy。
  不能只看 checkpoint 名称就假定二进制和配置相同，按产物中的身份记录核对。
- 候选入口为 `configs/tasks/bottle_whole_body.json`；默认
  `configs/tasks/bottle_visible.json` 没有切换。本版桌面仍为 62 cm。
- 固定 case bank、场景身份、目标/站位/出生扰动/外观/yaw；保留所有尝试，
  不通过补采成功样本或合并不同修订来提高报告的成功率。
- 保留碰撞、支撑、物体扰动、实际夹持、滑移、抬升、直立及负载稳持验收。
  不修改 gate 阈值来把本轮 FAIL 改成 PASS。
- 训练转换只去掉已标记的连续末端验收尾段，不删除任务中间停顿。
- 每次导出、提供查看或打包视频，必须生成 300–500 KB 左右的预览，
  **硬上限 500,000 bytes**；保留全时长、正常播放速度并核验大小与解码。
  原始视频保留。规则见 [AGENTS](../../AGENTS.md) 和 [视频导出](../rendering.md#video-export)。
- 本任务使用 **GPU 0 / launcher instance 0**，现已释放。GPU 1 属于其他工作，不使用、不停止。
  继续 GPU 工作前重新核查实时进程、实例锁及 [STATE](../STATE.md)，通过 launcher 获取资源。
- 不删除 outputs、数据集或权重，不追踪 `outputs/`、`third_party/`、
  `datasets/`，包括运行时软链接。保留其他会话文件。
- 参数集中在类型化配置和既有消费者；导航按 [bottle 修改地图](../reference/change-map-bottle.md)。
  最终验证选 commit hook 或 `make check` 一个入口，避免重复，详见
  [验证与完成标准](../agents/session-workflow.md)。

## 3. 版本与工作区事实

| 项目 | 当前记录 |
|---|---|
| 原基线 | `c0e994cf81d3d48dd8d6aae0ba24cf0a9b6091e1` |
| 本版代码提交 | `f7fc675`，17 个文件，包含代码、配置、测试和第一版报告 |
| 最终验证文档提交 | `fcf23d9` |
| 最近核实的 main / origin/main | 均为 `fcf23d94de76836f225542954072ca82f72ae366`，上一轮 push 成功 |
| 保留的隔离分支 | `fix/bottle-direct-20260914` |
| 隔离工作树 | `/workspace/code/locomanipulation-bottle-direct-20260914` |
| 本版产物根目录 | `/workspace/code/locomanipulation/outputs/bottle_direct_20260914/` |
| 最终原始 session | `/workspace/tmp/p03/outputs/2026-09-14-08-57-51` |

写交接前，main 无已跟踪文件改动；存在其他工作的未跟踪
`.serena/`、`node_modules/`、`package-lock.json`、`package.json`，不要清理或提交。
本次交接文档改动与上述已推送提交分开，不把尚未提交的交接内容说成已在 GitHub。

旧根 HANDOFF 已原样保存到
[旧 SONIC 动态回放交接](../archive/handoff-sonic-replay-20260910.md)，SHA-256：
`e86e0f7b98be367bff9f7e9ddbdfb620ce685f2354647d1e00ceb824246fe0a6`。
它的 GPU 分配、待办和当时结论不代表当前任务；历史判断仍需核对对应证据。

## 4. 已采用的决策及理由

| 决策 | 理由与证据边界 |
|---|---|
| 目标先低通，再以 5 Hz 更新手臂 IK；滤波时间常数 0.45 s | 降低步态摆动进入规划的幅度；CPU 测试覆盖注入摆动。并不等于所有物理抖动已消失 |
| 用过滤后的平面本体速度预测前进量 | 早期只过滤相对点会滞后，机器人继续向前时手可能越过瓶子；保留真实目标变化响应 |
| 行走时提前准备，关节命令持续限速/限加速度 | 从当前消费参考开始，避免一帧切到 READY；行走阶段上限 0.45 rad/s、1 rad/s²，下肢仍由 SONIC 控制 |
| 视觉处理独立线程，使用原图像时间戳控制缓存到期 | 中间批次实测一次同步观测耗时 3.579 s，随后状态间隔 3.40 s；迟到结果不能重新变“新鲜” |
| 同一目标和进入方向贯穿走近、减速及抓取 | 减少阶段之间重置姿势；短暂遮挡最多沿用配置的一秒缓存，隐藏真值不刷新目标 |
| 连续瓶盖避让通道代替 clear → descend → insert | 高处接近时先保证张开手指与瓶盖间隙，再连续收窄通道；直接穿过瓶盖的直线不是可执行路径 |
| 刹车名义深度 0.31 → 0.40 m，减速带 0.12 → 0.24 m | 给移动惯性和手部准备留余量；不是逐 case 的隐藏路点 |
| 支撑控制从实测/参考初始偏差建立反馈，躯干目标限速 0.25 rad/s | 零偏差初始化曾使保持参考在反馈启动时发生变化；更连续的参考仍未消除实际 POSE 接管回弹 |
| SupportedMotion 加强 COM 保持，WholeBodyIK 默认权重保持 5 | 中间批次出现 COM 超限；本控制器用权重 20，crate 的默认求解权重没有改变 |
| 保留 opt-in 和全部失败记录 | 最终样本较少，物理平滑度及数据准入仍有限制，不足以宣称全面升级默认策略 |

实现仍使用仿真几何，目标由实际 ego/右腕图像的新鲜度及近似相机视锥/ray 可见性准入；
**不是 RGB 位姿估计器，也没有证明 privileged 信息缺口已经完全消除。**

## 5. 已完成工作与修改文件

以下清单对应 `git diff --name-only c0e994c..fcf23d9`，不包含本次交接归档：

| 文件 | 修改用途 |
|---|---|
| `src/locomani/tasks/bottle/walking_reach.py`（新增） | 过滤目标、补偿前进、低频 IK、连续关节参考和过期处理 |
| `src/locomani/tasks/bottle/whole_body_config.py` | walk_plan_hz / walk_filter_seconds 类型化字段、默认值与边界验证 |
| `src/locomani/tasks/bottle/policy.py` | 由策略 owner 判断是否在导航期准备 |
| `src/locomani/tasks/bottle/round_phases.py` | 提前建立 whole_body 策略，保留当前参考，传入行走回调 |
| `src/locomani/backends/sonic/primitives_walk.py` | 回调拒绝时发送停止并中止导航 |
| `src/locomani/tasks/bottle/assisted_target.py` | 非阻塞视觉 worker，独立 ZMQ 与 MuJoCo MjData，沿用原时间戳 |
| `src/locomani/tasks/bottle/whole_body_policy.py` | 走近/刹车/全身抓取衔接、连续避瓶盖、捕获深度余量、躯干限速 |
| `src/locomani/primitives/supported_motion.py` | 初始反馈偏差、偏差滤波/边界和 COM 保持 |
| `src/locomani/primitives/whole_body_ik.py` | 可配置 COM 权重，保留原默认值 |
| `configs/tasks/bottle_whole_body.json` | 候选刹车深度与减速带 |
| `tests/test_bottle_walking_reach.py`（新增） | 摆动抑制、目标跟随、本体补偿、频率/速度/加速度、导航消费者与拒绝停止 |
| `tests/test_bottle_whole_body.py` | 参数、初始参考保持、连续瓶盖间隙、慢视觉及迟到缓存测试 |
| `docs/collection.md` | 当前用法、动作原则、候选限制 |
| `docs/reference/change-map-bottle.md` | 新模块和测试导航 |
| `docs/reference/config_fields.md` | 从配置定义生成的新字段 |
| `docs/handoff/bottle-direct-20260914.md`（新增） | 完整逐版实验、指标定义、失败和数据检查 |
| `docs/STATE.md` | 当前结果、限制和资源释放 |

本次交接另更新根 `HANDOFF.md`、归档索引 `docs/archive/INDEX.md`，
并新增旧交接的原样归档。没有在本次交接中改控制逻辑。

## 6. 测试和实验结果：事实

### 代码、标签与物理批次

- `f7fc675` 的已安装 commit hook：完整 lint + 默认 CPU 测试通过，
  记录 `commit.log`；`fcf23d9` 文档提交 hook 通过，记录 `docs_commit.log`。
  本版没有单独跑 crate GPU 回归，不能用其 CPU 测试推导实际行为等价。
- `final_actions.json`：最终 10 次录制全部通过 token/hand schema、wire
  及实际 consumed-hand receipt 检查，包括失败尝试。
- 最终 `final10` 为 **6/10**，成功 case：
  `31000:1,2,3,5,6,8`。旧版 `candidate10` 为 **3/10**。
- 本版中间运行单独保留：probe1/2/3 各 0/2，probe4/5 各 1/4；
  candidate10 与 candidate_balanced10 各 0/10；
  candidate_contact10 为 8/10。**8/10 不是最终版结果，不得合并或挑选它替代 6/10。**
- 最终四类失败：case 0 末端稳持滑移 3.5 cm；case 4 刹车时髋/桌间隙；
  case 7 腕/瓶间隙；case 9 物体/身体扰动。
- 最终没有 >0.5 s 的行走控制时钟失败；仍见约 0.22–0.24 s 的局部准备命令间隔，
  原因未确定。最后一次 observer 调用记录约 0.085–0.140 ms，不是延迟分位数。

### 动作指标

下表是不同成功子集的中位数（旧版 3 集、本版 6 集），不是所有 case 配对差值，
也不是统计显著性结论。数据在 `comparison.json`、`previous_motion.json`、
`final_motion.json`。

| 指标 | 旧版 | 最终版 |
|---|---:|---:|
| 原始帧数 / 名义 50 Hz 时长 | 1116 / 22.32 s | 931.5 / 18.63 s |
| 最后一个正向行走指令 → capture | 9.84 s | 6.68 s |
| 刹车后腕部路径（0.2 s 分块均值） | 0.970 m | 0.689 m |
| 路径长度 / 位移 | 3.522 | 3.109 |
| 最长相同 token 持续 | 0.38 s | 0.27 s |
| 实测身体关节步进 p99 | 0.0247 rad | **0.0342 rad** |
| WBC 关节指令步进 p99 | 0.0702 rad | **0.1085 rad** |
| 刹车后手部向上幅度 | 1.34 cm | **7.78 cm** |

**事实边界：**抓取更早、路径更短，但最后三项变大，视频仍有接管抬手回弹。
关节步进混合主动移动、步态和跟踪响应，不能直接当成 jerk 或纯抖动。
quiet 指标 0.26 → 0.03 s 也可能被抖动“打断静止”影响，不能证明所有停顿消失。

成功样本在 loaded_upright 验收时物体实际升高 12.7–16.6 cm、躯干倾角
1.6–4.7°，随后通过退离与稳持。行为等价 gate 仍为
**FAIL，10/13 通过**：刹车深度、失败类别和 reach 时长不等价。
见 `final_gate.json` / `final_gate.log`，阈值未放宽。

### 训练格式、世界坐标复核和视频

- `final_dataset/`：6 集、2464 帧、25 Hz、18 个相机视频，转换完成。
- `dataset_check.log`：**30 项通过、2 项失败，尚未通过完整训练数据准入**。
- 失败一：闭合后抬升检查 5/6。数据缺少 `observation.root_position`，
  检查器退回使用 pelvis-relative 高度；俯身/直立时它不能代表世界高度。
  独立 `world_lift_audit.json` 按原始 `priv_seq` 精确匹配物理 qpos，
  所有闭合后帧都匹配，**6/6 实际抬升通过**，峰值 13.2–18.4 cm。
  dataset episode 5 / source episode 8 的相对升高仅 0.000596 m，
  实际世界升高 0.167286 m。峰值与前述 loaded_upright 快照不是同一个指标。
  原始检查结果未被改成 PASS；通用检查器/转换链路尚未修复。
- 失败二：12 个腕部视频/状态相关性测量中，5 个 lag 0、6 个 lag +1、
  1 个 lag +2，最多 80 ms。**这是对齐疑点，不是已证实的统一曝光延迟**；
  不能直接按相关性峰值平移所有数据。
- action/body 的中位响应 lag +2 帧（80 ms）通过对应检查；
  不等于相机也已对齐。检查器“左臂从不运动”的通用提示已过时，不是本版测量。
- 完整预览 `previews/final_case01.mp4`：**404,108 bytes**，
  21.80 s（原片 21.84 s），正常速度、全程解码和大小检查通过；
  `previews/verification.json` 保存证据。已查看接触图及视频帧。

## 7. 关键路径和命令

以下命令以主工作区为当前目录。运行前按资源规则核查 GPU/实例；
复跑必须选新的输出目录，不能覆盖 final10、原始数据或已有分析。
这里给出复现入口，不表示本次保存交接又执行了它们。

固定样本：
`outputs/bottle_continuous_20260914/cases_fixed10.json`。
旧基线：
`outputs/bottle_continuous_20260914/candidate10/`。
本版代码快照与哈希：
`outputs/bottle_direct_20260914/{candidate_source,balanced_source,contact_source,final_source}/`。
本机 patched binary 仅是运行产物，其他开发者按 [安装指南](../setup.md) 构建已跟踪补丁。

```bash
cd /workspace/code/locomanipulation

# 固定 10 次尝试；少于 10 次成功时配额退出非零，保留实际失败。
SONIC_DEPLOY_BIN="$PWD/outputs/bottle_continuous_20260914/deploy/target/release/g1_deploy_onnx_ref" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
  -m locomani.pipeline.run_learn_pilot \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_whole_body.json \
  --scene outputs/scene/sim --out outputs/bottle_direct_repeat_NEW \
  --cases outputs/bottle_continuous_20260914/cases_fixed10.json \
  --set through=collect --set successes=10 --set batch_rounds=10 \
  --set max_batches=1 --set seed0=31000 \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0

# 重新审计现有录制时写新报告；无需 GPU。
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
  -m locomani.pipeline.audit_bottle_actions outputs/bottle_direct_20260914/final10 \
  --out outputs/bottle_direct_20260914/final_actions_recheck_NEW.json

# 重做转换到新目录。原始精确 argv 在 conversion_command.json。
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
  -m locomani.pipeline.convert_to_hf_format \
  --sessions /workspace/tmp/p03/outputs/2026-09-14-08-57-51 \
  --manifest outputs/bottle_direct_20260914/final10/collect/manifest_00.jsonl \
  --out outputs/bottle_direct_dataset_NEW --trim-verification-tail --jobs 2

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
  -m locomani.pipeline.check_dataset outputs/bottle_direct_20260914/final_dataset

# 有新代码改动时先运行相关测试；最终验证不要与 commit hook 重复。
make test TESTS="tests/test_bottle_walking_reach.py tests/test_bottle_whole_body.py"
```

`outputs/bottle_direct_20260914/motion_report.py` 和 `world_lift_audit.py`
保留离线计算；后者含固定输入/输出路径，先阅读再复用，避免覆盖原证据。
训练数据 owner 从 [数据转换指南](../dataset.md) 定位，初步入口：
`src/locomani/pipeline/convert_to_hf_format.py`、
`src/locomani/pipeline/check_dataset.py`。不要仅为保存交接重跑 GPU 批次。

## 8. 未解决问题、假设与下一步

| 未解决事实 | 待验证假设（不是结论） | 下一步与判定依据 |
|---|---|---|
| 减速/POSE 接管仍有回弹，身体与指令步进 p99 增大 | planner→POSE 的实际响应、初始偏差或姿态参考变化可能是来源 | 先按走近、刹车、POSE 三段对齐消费参考/实际 qpos，区分主动伸手与振荡；定位后做单因素固定样本比较 |
| 部分行走准备命令间隔约 0.22–0.24 s | 调度、IK 或其他同步计算可能导致；未证明具体调用是瓶颈 | 对控制段做耗时分解，先测量再改；不要臆测某个 reach-map 调用就是根因 |
| 髋/桌、腕/瓶碰撞及末端滑移仍失败 | 可达裕量、惯性、进入走廊或抓取余量可能不足 | 从保留失败轨迹定位首次接触/失稳，修复原因，保留原失败门槛 |
| 通用抬升检查用了错误坐标退路 | 已有 qpos 证据支持这一次误判，但其他数据集是否同样受影响未知 | 明确 world/root transform 的记录和使用契约，修复转换/检查消费者；以本例及姿态变化反例回归 |
| 图像/状态相关性偏移 1–2 帧 | 可能存在曝光/接收/重采样时序差异，也可能是相关性信号偏置 | 核对相机、state、priv_seq 和转换选帧时间戳；有时钟证据后再决定修复位置 |
| 尚无 VLA 训练证据 | 更直接且观测可解释的示范可能更易学习 | 动作与数据准入通过后，再做受控训练/闭环评估；不由视频观感推断学习收益 |

建议顺序：先离线定位接管回弹与数据时钟/坐标问题，再做有明确假设的最小修改，
完成相关 CPU 测试和必要的固定样本 GPU 验证。下一版仍报告全部尝试、同口径指标和
原始 gate；没有改进时如实保留，不通过新增等待、精心编排路点或裁掉停顿掩盖问题。

## 9. 建议技能及交接验证

- 排查接管回弹、控制耗时或对齐根因时，可按需使用
  `/home/yijie/.agents/skills/diagnosing-bugs/SKILL.md`。
- 只有需要改变模块接口/职责边界时，才考虑
  `/home/yijie/.agents/skills/codebase-design/SKILL.md`；普通局部修复直接找 owner。
- 本交接使用 `/home/yijie/.agents/skills/handoff/SKILL.md`。
  用户指定根目录 HANDOFF.md，优先于技能默认的系统临时目录；详细证据引用已有文档，
  不把历史实验全文复制成新的任务入口。没有启动子代理。
- 交接文档验证使用隔离工作树的 `make check` 文档分支，以免主工作区其他人的
  未跟踪 Node 文件触发无关全量测试；只检验本次文档/归档改动，不重跑物理实验。
