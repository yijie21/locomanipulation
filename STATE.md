# STATE — 2026-11-20（分支 `sonic-selfmodel`，推到 `origin/selfwm`；自身模型线，现在在 vastgpu 上接着做）

主线场景 RSI 的状态在 `main` 分支的 STATE.md（2026-10-23），这里不再记。

## 当前进度

- **已确认，已提交：**归因、感知避障、grasp_kit；多路点自身模型的方案（`docs/handoff/notebook-20261120/selfmodel_via_20261120.md`）。
- **11-20 在 vastgpu 上接着做的（已提交，`git log --oneline 453583f3..HEAD`）：**
  - 查新做完：这个分工本身不新，最接近 Dynamic Tube MPC；留给我们的是预测偏差的方向。用户定了两处修改，已写进方案、TODO 和 DECISIONS。
  - 代码写完并冒烟测试过：`gen_reach.py` 的 `REACH_VIA`、`REACH_VIA_REPLAY`；`scripts/experiments/selfmodel_via_20261120/` 下的 `via_gen.py`、`arm_spheres.py`、`via_dev.py`、`via_cover.py`；`tests/test_via_gen.py`。
- **还没开始：**试跑没启动。方案里的预测和判据随这次提交定下，第一次启动后只在"预测修订"一节追加。
- **先读：**完整经过、用户原话、每个选择和被否决的方案在 `docs/handoff/selfmodel-avoid-20261120.md`，决策在 DECISIONS 的 2026-11-20 一节和它的"续"。

## 下一步（编号见 TODO 第 38 节）

1. **把 35 条真实路线拷到跑试跑的机器**（用户 11-20 已同意）。它们只在 4090 上：`$C/via_tabletop`、`$C/step2`、`$C/box/via`、`$C/box/step2`。vastgpu 连不上 4090，要经 Mac 中转。
2. **38.5.8 试跑**（300 次开场 + 35 条路线各回放 4 遍），按方案里的判据决定要不要做 38.5.9 正式采集。先看两个数：回放时手偏离参考路线的中位数，和同一条路线各遍之间的方向稳定度。随机路线那 300 次不需要真实路线，可以先跑。

## 事实与假设

- **已确认：**
  - 现有自身模型数据每段按 10 Hz 存全身状态。正运动学重建手的位置，误差中位数 0.003 cm（graspr_wrist_odo 200 段）。关节顺序按 `runtime.META` 的 `joint_names`，场景用 `simple_g1_sonic`。
  - 现有数据里手偏离参考路线：站着 3.32 cm，蹲低 7.14 cm；蹲低、低目标、远伸手 9.66 cm（中位数，28336 次伸手）。
  - 35 条真实 cuRobo 路线：长度约为直线的 2 倍，最远偏离直线 26 cm，手腕沿途累计转动 139°（中位数）。
  - 4090 的仓库停在 `4344165c`，两个没提交的文件和 `d8993b35` 里的 md5 相同。`outputs/reachwm_20261030` 在 4090 上是指向 `loco_outputs/simple_selfmodel_20261119` 的符号链接。
  - vastgpu 上建了同一个空地场景：`outputs/reachwm_20261030/scene/simple_g1_sonic`。那里的 `outputs/scene` 是指向主仓库的链接，所以没建在它下面。
  - vastgpu 冒烟测试：`REACH_VIA` 8 次开场单进程 17 s，0 摔倒；标签脚本重建的手和记录的手相差中位数 0.016 cm。
  - 场景里左右手臂和手指的连杆坐标系互为镜像（零位下差 0.01 mm），左手用右手那组碰撞球的镜像。
- **假设或待验证：**
  - 空地上，身体偏离参考的相当一部分是系统性的、方向稳定的（假设 H1，预测和判据在方案条目里）。
  - 上次手偏离规划路线 3.8–4.1 cm，其中多少来自碰桌子，要等空地回放才知道。
  - vastgpu 上正式采集 12000 次开场：按每次约 2 s 折算，6 个进程约 70 分钟。6 个进程同时跑时单进程会不会变慢，没测。4090 上的估计是 17–25 分钟。
  - vastgpu 的场景和 4090 上那份是否逐字节相同，没有核对。
  - 参考位姿序列末端的手和命令相差中位数 0.6 cm，规划自己报的 IK 误差只有 0.015 cm，原因没查（TODO 38.5.7.2）。

## 卡点和要注意的

- **4090 是共享机器：**GPU 0 常有别人的任务。`gen_part.sh` 写死了 `CUDA_VISIBLE_DEVICES=$((w % 4))`，开跑前看 `nvidia-smi`，需要时改成只用 1–3。运行脚本发现工作区有未提交改动会退出，要先把仓库更新干净。
- **vastgpu：**只用 GPU 1，不超过 6 个进程；它的 SAM 3 服务、深度服务和 `/tmp/.tmp*` 都不是我们的。工作区里 `perceive.py` 的改动和 `selfcal.py` 是别的 session 的，不要动，也不要提交。
- **实验纪律要手动照做**（Mac 上的笔记本和启动拦截 hook 这里没有）：启动前写假设、预测和判据；结果贴原始输出；要报告的结论做独立复核。
- **`gen_reach.py` 的 worker 会吞掉 `ValueError`**，把那次开场记成 skipped 继续跑。改采集代码后先看 `w*_skipped.json` 是不是空的。
- **几件事还没定：**左右手各一半还是只采右手（38.5.12，开关 `REACH_VIA_SIDE`）；curobo_avoid 复核留下的 E2 和 E3（38.2.1、38.2.2）。
- a800 的 tmux `data_scaling_spin17` 不要碰。grasp_kit 的打包文件和示例只在 Mac 上。

## 关键路径与命令

- 4090 的登录、路径、更新仓库和现有抓取数据的采集方式：`docs/handoff/selfmodel-avoid-20261120.md` 第 3 节和 3.1。
- 多路点采集、标签、偏差分析和覆盖检查的命令：`docs/selfwm.md` 第 7 节末尾。
