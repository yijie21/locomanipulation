# STATE — 2026-11-20（分支 `sonic-selfmodel`，推到 `origin/selfwm`；自身模型线，交接到另一台机器）

主线场景 RSI 的状态在 `main` 分支的 STATE.md（2026-10-23），这里不再记。

## 当前进度

- **已确认，已提交：**
  - 归因、感知避障、grasp_kit 都做完了；
  - 多路点自身模型的方案写好了（`docs/handoff/notebook-20261120/selfmodel_via_20261120.md`）。
- **还没开始：**新代码一行没写，没启动任何新运行。
- **先读：**完整经过、用户原话、每个选择和被否决的方案在 `docs/handoff/selfmodel-avoid-20261120.md`，决策在 DECISIONS 的 2026-11-20 一节。
- **改了什么：**用 `git log --oneline 4344165c..HEAD` 看。
  - 992427ab：grasp_kit；
  - d8993b35：`motor.reach` 的 `via_rot` 和 `simple_bench --avoid`；
  - ca6f66b9：TabletopGrasp 失败原因的文档更正；
  - ab3941e2：脚本从 Mac 临时目录收进仓库；
  - 1cd2de33：交接文档和笔记副本。

## 下一步（编号见 TODO 第 38 节）

1. **38.5.4 查新：**"学身体的动态，再用几何检查障碍"用在冻结的人形全身控制器上，是否有人做过。
2. **38.5.5：**`gen_reach.py` 加 `REACH_VIA` 和 `REACH_VIA_REPLAY`。同时做 38.5.6 覆盖检查和 38.5.7 全臂标签。
3. **38.5.8 试跑**（300 次开场 + 35 条回放），按方案里的判据决定要不要做 38.5.9 正式采集。

## 事实与假设

- **已确认：**
  - 现有自身模型数据每段按 10 Hz 存全身状态。正运动学重建手的位置，误差中位数 0.003 cm，95% 在 0.107 cm 以内（graspr_wrist_odo 200 段，`simple_g1_sonic` 场景）。
  - 重建时关节顺序按 `runtime.META` 的 `joint_names`，场景用 `simple_g1_sonic`。用 `walk_mars` 场景会有 0.7 cm 的偏差。
  - 现有数据里手偏离参考路线：站着 3.32 cm，蹲低 7.14 cm；蹲低、低目标、远伸手 9.66 cm（中位数，graspr_wrist_odo 28336 次伸手）。
  - 35 条真实 cuRobo 路线：路线长度约为直线的 2 倍，最远偏离直线 26 cm，手腕沿途累计转动 139°（中位数）。
  - 4090 的仓库停在 `4344165c`。两个没提交的文件（`motor.py`、`simple_bench.py`）和 `d8993b35` 里的 md5 相同。
  - `outputs/reachwm_20261030` 是指向 `loco_outputs/simple_selfmodel_20261119` 的符号链接。
  - 11-19 在 4090 上实测：10200 次开场用了 14.4 分钟（144 个进程）；两个种子的训练用了 8.2 分钟。
- **假设或待验证：**
  - 多路点采集 12000 次开场约 17–25 分钟。这是按路线长一倍估的。
  - 空地上，身体偏离参考的相当一部分是系统性的、能预测的（假设 H1，预测和判据在方案条目里）。
  - 上次手偏离规划路线 3.8–4.1 cm，其中多少来自碰桌子，要等空地回放才知道。
  - 现在的自身模型执行避障路线时预抓取点更差（6.5 cm，direct 是 3.4 cm），原因推测是它没见过多路点命令，没有验证过。

## 卡点和要注意的

- **4090 是共享机器：**GPU 0 常有别人的任务，11-20 还有别人的 `handover/lgrab_plan_test.py`。
  - `gen_part.sh` 写死了 `CUDA_VISIBLE_DEVICES=$((w % 4))`，会用到 GPU 0；开跑前看一眼 `nvidia-smi`，需要时改成只用 1–3。
  - 运行脚本发现工作区有未提交改动会退出（`dirty tree`），所以要先把 4090 的仓库更新干净。
- **实验笔记本只在 Mac 上：**Mac 上的 `~/research-notebook` 不在 git 里。新机器上按同样的纪律手写：启动前写假设、预测和判据；结果贴原始输出；要报告的结论做独立复核。
- **grasp_kit 的东西只在 Mac 上：**打包文件和示例都只在 Mac 的 `outputs/` 下，sha256 等细节见交接文档 1.5。
- **几件事还没定：**
  - 左右手各一半还是只采右手（38.5.12）；
  - curobo_avoid 复核留下的 E2 和 E3 还没跑（38.2.1、38.2.2）。

## 关键路径与命令

```bash
# 4090
ssh -p 27984 root@14.103.60.188
R=/wuji-vefps-D/wuji-il/yijie/ws/code/loco                                   # 仓库
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_selfmodel_20261119         # 自身模型数据、模型、运行脚本
C=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_bench_20261119/curobo      # 避障和归因的产出、35 条真实路线
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python                 # 仿真
SP=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE/.venv/bin/python     # SIMPLE 和 cuRobo，有 numpy

# 把 4090 的仓库更新到这次推送的版本（两个未提交文件和 d8993b35 相同，丢掉后快进）
cd $R && git status --short && git checkout -- src/locomani/backends/scalebfm/motor.py \
  scripts/experiments/simple_eval_20261112/simple_bench.py && git fetch origin selfwm && git merge --ff-only origin/selfwm

# 重建检查和偏差统计（在 $R 下跑）
$PY -I scripts/experiments/selfmodel_via_20261120/fk_check.py outputs/reachwm_20261030/data_graspr_wrist_odo 200 simple_g1_sonic
$SP -I scripts/experiments/selfmodel_via_20261120/dev_map.py outputs/reachwm_20261030/data_graspr_wrist_odo
$SP -I scripts/experiments/selfmodel_via_20261120/via_stats.py $C/via_tabletop $C/step2 $C/box/via $C/box/step2
```

- **现有抓取数据的采集方式**（新模式照这个加开关）：
  - 数据集定义：`$O/datasets_wrist.sh` 里的 `graspr_wrist_odo` 一行；
  - 环境变量：`REACH_GRASP=1 REACH_GRASP_ROOM=1 REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 REACH_GRASP_IKREF=1 REACH_WRIST=1`；
  - 分机器采集：`$O/gen_part.sh MACHINE NAME RESETS SEED ID0 [VAR=VAL ...]`；
  - 合并：`$PY scripts/experiments/reachwm_20261030/gen_reach.py merge`；
  - 整套流程：`$O/run_wrist_4090.sh`。
- **其他机器：**
  - vastgpu 只用 GPU 1；它的 SAM 3 服务、深度服务和 `/tmp/.tmp*` 都不是我们的。
  - a800 的 tmux `data_scaling_spin17` 不要碰。
