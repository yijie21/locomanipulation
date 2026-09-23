# STATE — 2026-09-23（会话：walk 场景量 SONIC 行走 → 守线刹车 → 两段式伸手 → warehouse 真专家接入，0/10）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，**本地领先 origin/main 3 个提交未推送**
（`cabeb9b`、`eb9c4e3`、`20d52a3`，外加本 handoff 提交）。

## 工作树里的两类未提交改动

- **我的（TODO 2.4.5，待提交；默认全关，`make check` 已通过）**：
  `src/locomani/tasks/crate/{expert,visual_approach,visual_config,grasp_region}.py`、`src/locomani/backends/sonic/diag_brake.py`、
  `docs/reference/config_fields.md`；新文件 `src/locomani/primitives/foot_watch.py`、`src/locomani/tasks/crate/retreat.py`、
  `tests/test_crate_guard_retreat.py`、`configs/experiments/box_room_grasp_20260923.yaml`。
- **协作者的（不要动、不要提交）**：`docs/bimanual_reachability.md`、`scripts/probe_ghost_reach.py`、
  `src/locomani/tasks/crate/ghost_reach_execution{,_cli}.py`、`tests/test_ghost_reach_execution.py`；未跟踪的
  `docs/handoff/g1-joint-reach-search-20260920.md`、`scripts/generate_joint_reach_bank.py`、
  `src/locomani/tasks/crate/joint_reach_sampling.py`、`tests/test_joint_reach_sampling.py`。
  - `scripts/experiments/brake_then_reach_20260923.py` **依赖协作者未提交的** `original_reach_control_config(envelope=...)`。
  - 提交钩子仍被 `test_joint_reach_sampling.py` 挡住 → 先跑 `make check` 再 `--no-verify`（本会话一直这样）。

没有运行中的栈（`find /workspace/tmp -maxdepth 3 -name stack.json` 无输出）。

## 当前进度

- 已确认（walk 场景，数据在 `outputs/diagnostics/`，结论表在 `docs/walk.md`）：
  - SLOW_WALK 速度图：≤0.15 不走，0.20–0.50 不稳，≥0.55 线性（`speed_ramp_v11_20260923`）。
  - 守线刹车（extra 0.20 / margin 0.03 / stride0 0.14）：全程 0.25 50/50 不越线、身体最前点停在线前 15.5 ± 2.6 cm
    （`brake_guard025_valid_20260923`）；0.55→0.25 50/50（`brake_guard_valid{,2}_20260923`）。
  - 刹车后手比骨盆远 65–69 cm（`brake_reach_20260923`）。
  - 虚拟纸箱：staged + 抓深 1 cm + orientation_gain 1.0 → 4/5（`box_reach_fb_20260923`）。
- 已确认（warehouse 真专家，`outputs/runs/box_room_grasp_20260923`，10 局）：**0/10**；
  守线在 warehouse **6/10 越线**（最多 +5 cm）；停稳→伸手已缩到 ~1.4 s；后退段一局都没走到。
  逐局阶段表：`/tmp/claude-1002/.../scratchpad/grasp_anatomy.txt`（临时；可用 STATE 下方命令重算）。

## 假设 / 待验证

- **假设**：warehouse 越线是因为起点离托盘边只有 0.64 m，刹车落在起步冲刺里（刹车时骨盆 0.46–0.66 m/s、手臂前伸），
  与 walk 标定条件（0.5 m 助跑、0.12 m/s、手臂下垂）不同。起点后移后应恢复——**待验证**（TODO 2.4.6）。
- **假设**：伸手阶段骨盆后退 0.3–1.3 m / 2 摔来自扩包络 carry 块（`box_room_20260922` 同样 3 摔、回退后 0 摔）——
  与 walk 场景幽灵执行器不一致，**待验证**（TODO 2.4.7）。
- **待验证**：后退段 `tasks/crate/retreat.py` 只有单测，从未在仿真里执行过。

## 下一步（编号见 `TODO.md`）

- **2.4.5** 提交上述"我的"未提交改动（只按文件名 add）。
- **2.4.6** warehouse 起点后移 0.3–0.4 m，用真专家重标守线刹车后滑行，再验证不越线。
- **2.4.7** 伸手回到不摔的 carry 块（crate 值 + 0.7/0.04），另解"手悬在目标上方下不来"。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
bash scripts/cpu_headroom.sh && nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
find /workspace/tmp -maxdepth 3 -name stack.json     # 有输出 = 有栈在跑

# warehouse 真专家（新方案），两个终端或一个脚本串起来
bash scripts/start_stack.sh collect configs/experiments/box_room_grasp_20260923.yaml --gpu 0
bash scripts/bimanual/run_agent_server.sh configs/experiments/box_room_grasp_20260923.yaml --gpu 0
bash scripts/stop_stack.sh --instance 0

# walk 场景探针（各 .sh 自己起/收栈，instance 0；结果在 outputs/diagnostics/<name>）
bash scripts/experiments/brake_marker_20260923.sh --speeds 0.25 --marker 0.9 --guard --extra 0.20 --margin 0.03 --repeats 50
bash scripts/experiments/brake_then_reach_20260923.sh --box --reach-path staged --below-rim 0.010 --orientation-gain 1.0 --repeats 5
bash scripts/experiments/speed_ramp_20260923.sh --mode 1
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/box_contact_20260923.py outputs/diagnostics/<run>

make test TESTS="tests/test_crate_guard_retreat.py tests/test_crate_coordinated_reach.py" && make check
```

- 资源：一个栈约 1 核 CPU、1.5 GB 显存；**GPU0 上别的进程启动瞬间会拖断 deploy**（SimpleRoboJev vision worker 实测过），
  实验期间请把其它任务放 GPU1。
- 视频交付：单文件 ≤500,000 字节，超了拆成编号分段（见 `docs/rendering.md#video-export`）。
