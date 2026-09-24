# STATE — 2026-09-24（会话：离线伸手时钟修复；ready 原地抬升；对准/anti-windup/腰上限/前馈四轮试验；站位是剩下的杠杆）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，与 origin/main 同步（最新 `2401c38`）。
没有运行中的栈。全部结论、逐局表和复现命令在 [warehouse 交接](docs/handoff/warehouse-render-cost-20260923.md)；理由在 DECISIONS 2026-09-24。

## 工作树里未提交的改动（协作者的，不要动、不要提交）

- `docs/bimanual_reachability.md`、`scripts/probe_ghost_reach.py`、`src/locomani/tasks/crate/ghost_reach_execution{,_cli}.py`、
  `tests/test_ghost_reach_execution.py`；未跟踪的 `docs/handoff/g1-joint-reach-search-20260920.md`、`scripts/generate_joint_reach_bank.py`、
  `src/locomani/tasks/crate/joint_reach_sampling.py`、`tests/test_joint_reach_sampling.py`、`assets/crates/nor_cal_box/visual/`。
- 提交钩子会被协作者的测试挡住 → 先 `make check`（单独取退出码，别接 `| tail`）再 `--no-verify`，只按文件名 add。

## 当前进度（全部已确认，10 局/轮、seed 7000–7009、离线渲染）

- 09-23 两轮离线 pilot 的伸手段从未执行（占位帧戳晚于循环 `now`）；已修（`29d4357`）。修复后 m003b 0/10，三层几何限制。
- ready 原地抬升（`ready_pose lifted`，`ed5819d`）：抬手接触与箱子位移 0/10。基线配置 = **align**（+ 预抓点真值居中，`833ca23`）：试提 6/10。
- 四个负结果：抓深 0.03（推倒箱子）、沿下 anti-windup 单独（无效）、腰上限 0.35（试提 6→1）、送达差前馈（6→0）。
  送达差 = 每手 6–8 cm、前短上高，是控制器的**伸展饱和**：改命令不改送达。
- 五轮 50 局按站位带：0.45–0.52 试提 12/18，<0.45 3/12，0.53–0.56 1/11，≥0.57 2/9。
- 所有试提都是 1 cm 抓深夹不住（`probe_not_attached`/`lost_attachment`）。
- 手臂晃动（用户从视频看到）：是我们流出的参考在晃——IK 逐 tick 重解时肘/腕反相互换（参考 sd 9–15°），实测比参考还平。

## 下一步（编号见 `TODO.md`）

- **2.4.10.23** 站位收紧到 0.45–0.52：先离线量守线散布来源（预测步长 vs 实际），再定改标定还是站位准入丢弃（等用户确认）。
- **2.4.10.20** continuity 项去晃，单独一轮对照。
- **2.4.10.18** 站位收进后回抓深 0.03。

## 卡点 / 待验证

- 假设：站位散布来自守线的步长预测；未量。
- 待查（2.4.10.19）：align 后下落中跨壁误差再漂 3–5 cm、修正 +9…+15 未送达。
- 此前所有 warehouse 结论（守线越线、carry 摔倒）都在慢仿真里测的，需满速重测。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
bash scripts/cpu_headroom.sh && nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
find /workspace/tmp -maxdepth 3 -name stack.json     # 有输出 = 有栈在跑

# 基线（复制后改 name 和一个旋钮）
CFG=configs/experiments/box_room_grasp_offline_align_20260924.yaml
bash scripts/start_stack.sh collect $CFG --gpu 0
bash scripts/bimanual/run_agent_server.sh $CFG --gpu 0
bash scripts/stop_stack.sh --instance 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/render_trace_videos.py outputs/runs/<run>

# 逐局分析：手指到抓点 scripts/experiments/reach_gap_20260923.py；送达差/指垫位置 = trace 重放（脚本在交接文档里描述，未入库）
make test TESTS="tests/test_crate_descend_align.py tests/test_crate_lifted_ready.py" && make check
```

- 逐 tick 记录在 `outputs/runs/<run>/observations/<局>/observations.jsonl`（`low_reach.*`、`reference_joints`、`joints`）。
- 视频交付：单文件 ≤500,000 字节，各 run 的 `previews/`。
