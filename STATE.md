# STATE — 2026-09-22（会话：v1.1 小步微调探测 → 纸箱线回到物流房间）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。本会话改动**未提交**（只提交 handoff 两个文件）；
工作树里 codex 的改动（`probe_ghost_reach.py`、`ghost_reach_execution{,_cli}.py`、`test_ghost_reach_execution.py`、
`docs/bimanual_reachability.md`、`generate_joint_reach_bank.py`、`joint_reach_sampling.py`）**不是我的，不要动**。
没有运行中的栈（`find /workspace/tmp -maxdepth 3 -name stack.json` 为空）。GPU1 长期被别的项目训练占着；CPU 余量常被 JeVLA 的进程吃掉，
起栈前 `bash scripts/cpu_headroom.sh`，不够就用 scratch 里那种 60 s 轮询的重试器。

## 已确认（本会话）

- v1.1 步态实测（`outputs/diagnostics/gait_micro_v11_20260922{,_knee}/legs.json`，台账 `docs/ledger/crate.md` 同日第一节）：
  无原地踏步；前向最低档 0.20 → 0.05 m/s、5–10 cm/步；横向 0.32 才开；偏航漂移 +3…+16°/5 s；速度图与 09-11 不同（连续，release 样）。
- 合成桌子（`outputs/scene/box_table37`）上的四组纸箱 pilot **作废**（台账有作废节）。
- 房间 pilot `outputs/runs/box_room_20260922`（配置 `configs/experiments/box_room_20260922.yaml`）：0/20；起点放置正确
  （平台沿距 0.385–0.394，箱子偏航 ≈0）；站位深度中位 0.47 sd 0.07（计划 0.50）、横向 +0.01 sd 0.08；
  why：reach_stalled 8、摔倒 6、unsupported_assisted_reach 4、推箱 1、approach_unsupported 1。视频 `videos/`，`reel_failures.mp4` 前 6 局。
- 代码：`visual_approach.py` 新 `approach_heading/edge_distance/scene_object_yaw`，`initial_robot_xy` 按朝向；`expert.py` 放置按朝向、
  `prepare_stationary_ready(heading=)`；`visual_config.py` 新 `start_pose`/`start_advance`/`stand_lateral_gate`；
  `diag_shuffle.py` 新 `--march`；44 个 crate 视觉测试通过；`bimanual_design.md`、`config_fields.md` 同步。

## 下一步

1. 用户先看 `box_room_20260922/reel_failures.mp4`。然后按阶段拆 20 局：摔倒 6 局在哪个相位、reach_stalled 8 局的站位深度与手到抓点的距离
   （`visual_NN.log` 逐局 dict 里 `stand_attempts`、`control_reason`；`trace/*_state_trace.npz` 有 qpos）。
2. 再决定动哪个：`stand_off`（房间站位 0.27–0.66 两头都有）、`pregrasp_above`、或摔倒的原因。每次一个改动、20 局、同 seed 7000。
3. `stand_lateral_gate: turn` 在房间里没验证过；横向偏置在房间里不存在，先不用。

```bash
# 房间 pilot（两个终端；先查余量）
bash scripts/cpu_headroom.sh
bash scripts/start_stack.sh collect       configs/experiments/box_room_20260922.yaml --gpu 0
bash scripts/bimanual/run_agent_server.sh configs/experiments/box_room_20260922.yaml --gpu 0
# 步态实测（默认 sim 场景，不用纸箱）
bash scripts/experiments/gait_micro_20260922.sh          # GAIT_SPEEDS/GAIT_CRAB/GAIT_MARCH 可覆盖
```

## 未解决 / 待验证

- 待验证：摔倒 6/20 为什么比合成桌子多（起点离平台 0.4 m、几乎不走；假设是 reanchor/IDLE 期间贴着托盘垛）。
- 待验证：哪一项部署变更把 v1.1 速度图从台阶变成连续（候选：落座高度修复、部署二进制、栈起法）。
- 假设：`vla_rounds.py` 闭环评估在房间里会因写死偏航 0 而放错位置；没改。
- 待提交：上面列的所有代码/文档/配置改动（`git status` 里非 codex 的那些）。

## 关键路径

- 台账：`docs/ledger/crate.md` 末尾三节（步态实测 / 作废 / 房间 20 局）。
- 房间场景：`outputs/box_rim_execution_20260920/scene_reverse_camera`（`scene_meta.robot_start` (0, −1.205, 90°)）。
- 参考起点局：`outputs/box_rim_direct3_20260922/videos/dy0.225_r00.mp4`（request `robot_dxy [0, 0.225]`, yaw π/2）。
