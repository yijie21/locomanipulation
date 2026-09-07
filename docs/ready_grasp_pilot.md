# 最新策略的小规模学习实验

目标：只使用明确指定的采集配置的成功示范，检查从 GR00T 基础模型训练出的 VLA 是否能完成正常起点的闭环抓取。默认是已完成固定案例验收的 `ready_grasp_coupled.json`；`ready_grasp_coupled_timing.json` 是独立时序实验，需要显式指定，不自动替换默认策略。

新采集在后退结束后标记 `training_tail.end_seq`，保留完整持握验收原始数据。pilot 转换启用 `--trim-verification-tail`，只去掉末尾纯验收等待，不删除中间静止帧；旧原始数据没有该标记时拒绝猜测裁剪位置。25 Hz 数据与 0.4 秒重规划对应约 10 步执行片段，40 步预测跨度为 1.6 秒。

固定预算：24 条成功示范，按 episode 隔离为 20 train + 4 val；训练 1,000 步；10 条独立固定案例，完整 90 秒动作预算，25 Hz，async 重规划间隔 0.4 秒。不改变原有默认采集策略。

产物：`outputs/experiments/ready_grasp_pilot_v1/`。`spec.json` 保存训练预算及源代码、配置、场景和评估案例指纹；`timings.jsonl` 保存每阶段的墙钟时间及退出码；采集 manifest 保留所有失败、重试、阶段事件和成功记录。`*.done.json` 只标记完整完成的阶段。

运行入口：

```sh
python src/teleop_free/tools/run_ready_pilot.py \
  --out outputs/experiments/ready_grasp_pilot_v1
```

首次使用前，将独立场景复制到输出目录的 `scene/`，再生成案例：

```sh
LOCOMANI_SIM_DIR="$PWD/outputs/experiments/ready_grasp_pilot_v1/scene" \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python src/teleop_free/grasp_gym.py \
  --reach-config configs/collection/ready_grasp_coupled.json \
  --targets water_bottle --rounds 10 --seed 16800 \
  --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 10 \
  --spawn-lateral .1 --edge-jitter .1 \
  --sample-only --write-cases outputs/experiments/ready_grasp_pilot_v1/eval_cases.json
```

案例文件保存完整精度的物体坐标、站位、朝向、起点偏移、外观种子和下蹲提示。`--cases` 跳过随机采样，校验场景元数据指纹；不能与 DART、按成功数量停止或评估的随机补采 resume 混用。物理执行仍有异步时序差异，不承诺轨迹逐帧复现。初始化维持最新采集策略的双臂准备姿态，VLA 接管后不由 expert 帮助执行任务。

采集最多四个 session，每个最多十轮（站立失败可能重试一次），达到 24 条成功才转换训练；不足即停止并报告 expert 失败原因。每个 session 和 manifest 成对保存。转换使用两个 worker，只处理本次原始数据；完整的数据检查通过后才训练。训练从 `nvidia/GR00T-N1.7-3B` 开始，继承现有训练脚本的动作空间和冻结视觉塔设置；本次只保存最终模型，不能作为精确恢复训练的 checkpoint。

同一输出目录的完整阶段可复用；输入指纹变化必须用新目录，部分转换和部分训练不自动覆盖。此入口面向当前双 GPU、instance 0 环境，应在无其他采集/评估占用时启动。跨实验的视频级增量缓存、优化器恢复和仿真并行加速不在本次实现范围内。

结果解释：

- expert 失败、数据审计失败：不启动训练。
- 训练/执行报错：实现问题，不计为策略效果。
- 闭环结果需同时查看初始化、到位、下蹲、闭手、提起的实际轨迹和失败数量。
- 10 条评估只能用于发现大问题及决定是否增加预算；1,000 步未学会也不能证明新数据最终不可学习。
- 后续候选使用同一案例文件和同预算基线，正式验收另外使用独立案例，不混用筛选成功率。

仅改变训练方法时，使用新的输出目录并显式复用已审计数据：

```sh
python src/teleop_free/tools/run_ready_pilot.py \
  --out outputs/experiments/ready_grasp_training_candidate \
  --reuse-data outputs/experiments/ready_grasp_pilot_v1
```

复用要求原实验已有 `convert.done.json`；重新计算数据文件 SHA-256，记录到新实验，拒绝审计完成后修改过的文件。场景和评估案例沿用原实验。复用入口不会再次采集、编码视频或计算统计。首个已启动试跑的执行器源码保存在实验 `source/` 中；后续入口增加了复用数据和预算不可达时停止的功能，未修改正在采集的动作策略。

## Fixed-budget paired runs

The runner now accepts `--collection-cases FILE`, `--eval-rounds 20`,
`--stand-retries 0`, and `--through collect|convert|train|eval`.
Fixed collection visits the provided cases in order, stops at the requested
success count, and fails if the finite bank cannot supply that count. Failures
remain in the manifest and raw session. The original random-batch path remains
available when no case bank is supplied.

`--through` is a stage boundary, so collection/conversion can be inspected
before any training starts. Later invocations reuse only fingerprint-matching
completed stages. Source/configuration changes require a new experiment
folder; a partial stage is not silently treated as complete.

Fixed-case collection also records source-timing receipts and verifies their
row-to-priv_seq correspondence before marking collection complete. These are
publication/receipt timestamps, not a proof of exact physical synchronization.
The explicit eval retry argument is forwarded into grasp_gym, so failed
initialization cannot be hidden by reset retry when set to zero.

For the unattended policy experiment, training remains gated on expert
regression, independent perturbations and data review. Preparing a command or
passing the stage-boundary unit test does not authorize skipping those gates.
