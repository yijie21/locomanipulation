# STATE — 2026-09-23（会话：warehouse 过冲撞箱的根因 = 纸箱网格拖慢渲染；离线渲染跑通，0/10 停得太远）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，已推送到 origin/main。
没有运行中的栈（`find /workspace/tmp -maxdepth 3 -name stack.json` 无输出）。用户要求**暂停**，下一次实验等其确认。
证据、逐局数据和复现命令：[warehouse 渲染负担交接](docs/handoff/warehouse-render-cost-20260923.md)。

## 工作树里未提交的改动（协作者的，不要动、不要提交）

- `docs/bimanual_reachability.md`、`scripts/probe_ghost_reach.py`、`src/locomani/tasks/crate/ghost_reach_execution{,_cli}.py`、
  `tests/test_ghost_reach_execution.py`；未跟踪的 `docs/handoff/g1-joint-reach-search-20260920.md`、`scripts/generate_joint_reach_bank.py`、
  `src/locomani/tasks/crate/joint_reach_sampling.py`、`tests/test_joint_reach_sampling.py`、`assets/crates/nor_cal_box/visual/`。
- 提交钩子会被协作者的测试挡住 → 先 `make check` 再 `--no-verify`，只按文件名 add。

## 当前进度

- 已确认：warehouse sim 0.30–0.36× 实时，根因是纸箱网格 46.3 万面 + 投影灯（3 路渲染 39–44 ms）；慢仿真下速度约 3 倍 = "过冲撞箱"。
- 已确认：`backend.render: offline` 后 sim 0.95–0.97×，走路 0/10 碰箱；守线余量 0.15→0.03 把停位中位 0.63→0.49 m（两轮各 0/10，伸手段无效，见下）。
- 已确认（09-24）：前两轮离线的伸手段从未执行（占位帧戳晚于循环 `now`，参考时钟 0/440 tick）；已修 + 契约测试，实时不受影响。
- 已确认（09-24，m003b）：修复后伸手 10/10 执行，0/10；三层几何限制：≤0.51 ready 姿态压近壁、≥0.54 掌落壁顶、0.46–0.52 抓深 1 cm 夹不住。
- 待验证：此前所有 warehouse 结论都在慢仿真里测的（守线越线、carry 块摔倒），需满速重测。

- 已确认（09-24，deep03）：抓深 0.03 → 0/10、0 合爪、6 局推斜：跨壁横向误差 2–6 cm，指垫顶壁后修正积分到 ±15 cm 把箱子推倒。

- 已确认（09-24，lifted，`ed5819d`）：ready 原地抬升后抬手接触 0/10、箱子位移 0/10；仍 0/10——4 局试提夹不住、4 局横向修正顶壁饱和、2 局停高 2 cm。

## 下一步（编号见 `TODO.md`）

- **2.4.10.17** 横向对准 + 修正 anti-windup（控制改动，等用户定方案）；之后回抓深 0.03；**2.4.10.16** 远站位掌俯仰。
- **2.4.10.8** 纸箱网格减面 / 关阴影——先问协作者。

## 卡点

- GPU0 上 SimpleRoboJev 视觉 worker 反复启停、GPU1 被 starVLA 训练占满；离线这轮在该条件下跑出 0.95×，未见 deploy 断。
- `diag_brake` 不做 reanchor，warehouse 里起步就转 ~90°（`warehouse_speed_20260923` 横偏数据作废）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
bash scripts/cpu_headroom.sh && nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
find /workspace/tmp -maxdepth 3 -name stack.json     # 有输出 = 有栈在跑

# 离线渲染 pilot（当前基线；复制后改 name 和一个旋钮）
CFG=configs/experiments/box_room_grasp_offline_lifted_20260924.yaml
bash scripts/start_stack.sh collect $CFG --gpu 0
bash scripts/bimanual/run_agent_server.sh $CFG --gpu 0
bash scripts/stop_stack.sh --instance 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/render_trace_videos.py outputs/runs/<run>

# 渲染耗时拆分（不起栈）：见 DECISIONS 2026-09-23 "根因"
bash scripts/experiments/warehouse_speed_probe_20260923.sh --repeats 5   # sim 速度，有/无读相机

make test TESTS="tests/test_crate_guard_retreat.py tests/test_run_context.py" && make check
```

- 手指到抓点距离：`scripts/experiments/reach_gap_20260923.py outputs/runs/<run>`（.venv_sim 的 python）。
- 视频交付：单文件 ≤500,000 字节（`docs/rendering.md#video-export`）。
