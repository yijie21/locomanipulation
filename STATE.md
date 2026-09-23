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

- 已确认：warehouse sim 只有 0.30–0.36× 实时，根因是 `nor_cal_box` 显示网格 46.3 万面 + 2 盏投影灯（3 路渲染 39–44 ms，walk 1.3 ms）；
  慢仿真下实际速度约 3 倍 = 用户看到的"过冲撞箱"。
- 已确认：`backend.render: offline` 后 sim 0.95–0.97×，走路 0/10 碰箱；离线 10 局（`outputs/runs/box_room_grasp_offline_20260923`）
  0/10——停在 depth 0.53–0.72 m，手指离抓点 27–44 cm，7 局伸手超时、2 局准入拒绝、1 局 vision_lost（未查）。
- 假设：守线余量 0.15 → 0.03 可停近约 12 cm，伸手段仍需单独处理。**未验证**。
- 待验证：此前所有 warehouse 结论都在慢仿真里测的（守线越线、carry 块摔倒），需满速重测。

## 下一步（编号见 `TODO.md`）

- **2.4.10.6** 离线模式下守线余量 0.15 → 0.03，再跑 10 局（等用户确认再跑）。
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

# 离线渲染 pilot（复制 offline 配置改 name 与 guard_margin 再跑）
CFG=configs/experiments/box_room_grasp_offline_20260923.yaml
bash scripts/start_stack.sh collect $CFG --gpu 0
bash scripts/bimanual/run_agent_server.sh $CFG --gpu 0
bash scripts/stop_stack.sh --instance 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/render_trace_videos.py outputs/runs/<run>

# 渲染耗时拆分（不起栈）：见 DECISIONS 2026-09-23 "根因"
bash scripts/experiments/warehouse_speed_probe_20260923.sh --repeats 5   # sim 速度，有/无读相机

make test TESTS="tests/test_crate_guard_retreat.py tests/test_run_context.py" && make check
```

- 逐局接触/手指到抓点距离的分析脚本是会话临时文件，没有入库；重做需重放 trace 的 `mj_forward`。
- 视频交付：单文件 ≤500,000 字节（`docs/rendering.md#video-export`）。
