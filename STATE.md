# STATE — 2026-09-24（会话：夹持力→跨壁门→跟随→起身前向下限→骨盆系 hold→接触触发；warehouse 首批 6/20；Isaac 重渲与 ghost 规则）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。没有运行中的栈。理由与被否决方案在 DECISIONS 2026-09-24 两节；逐项数字在 `TODO.md` 2.4.10.23–2.4.10.30、3.12–3.15。

## 工作树里未提交的改动

- 无代码改动。2026-09-24 已全部提交并推送到 `origin/main`（`8ca2fc2` warehouse 抓取、`c548ee0` 三场景复现 + 离线录制、`7a76b37` warehouse runbook、`76c5196` 可达性分片/续跑 + 五维采样——原记为"协作者 WIP"，实为已提交的 `reach_envelope.sh` / `boundary_search.py` 依赖的代码）。
- 仍未跟踪、不属于任何提交：`assets/crates/nor_cal_box/visual/`（纸箱源网格，减面前先问维护者）、`assets/blender/{fonts,textures}/`、`package.json`、`uv.lock`。
- 提交仍用 `make check` + `--no-verify` + 按文件名 add。

## 当前进度（全部已确认，20 局/轮、seed 7000–7019、离线渲染、场景 warehouse_textured）

- **暂定基线** `configs/experiments/box_room_grasp_baseline_20260924.yaml`（= contact 版，**6/20**：提起 10 → 进后退 8 → 完成 6）。成功定义（用户）：抬起 + 直立 + 后退 0.3 m 箱子不掉。
- 今天的路径：0/10 → `close_target` −0.06（夹持力 2→16 N/指，箱子首次离地 7/10）→ 跨壁门 0.035 → 三个 lift 旋钮无效 → 跟随 0.6 → `stand_forward_min` 0.35 逐 tick（起身撞躯干是真因）→ `hold_frame` pelvis + 短 hold → 首批 6/20 → 接触触发 6/20（跨壁停滞消失、沿上停滞 7 留下）→ `contact_overdrive` 0.03 负结果 2/20。
- 场景：`warehouse` 的纸箱没贴图是场景早于贴图提交，`warehouse_textured` 已重装（物理逐项一致）。Isaac 并排视频 10/10 跑通（房间 USDC + 逐 tick 重放，曝光 ISO 30），`render_isaac_warehouse.sh`。
- ghost 夹爪规则已生效：每局写 `trace/<label>_ghost.npz`，第三视角自动画，ego/腕不画；`contact_20260924` 起的视频都带。
- walk 场景：守线 extra 0.17 只赚 3 cm；waypoint 过量下发不能替代守线；0.55 相位刹车残差 sd 3.6、最大 11 cm，不用（DECISIONS）。
- 夹持力公式（已确认）：F = kp 1.5 × gear 250 × (设定点 − 停住 q)；换物体先算这个。

## 下一步（编号见 `TODO.md`，等用户定顺序）

- **2.4.10.29** 沿上停滞 7/20：小步下探逐步查跨壁，或 align→descend 失败重做一次（不要盲推：overdrive 已证明会把指垫顶偏 4–8 cm）。
- **2.4.10.29.2** 后退剪切 2/20：`retreat_speed` 0.35→0.25，或后退时保留全身伸手控制器夹持。
- 提交本会话代码（先 `make check`）。

## 卡点 / 待验证

- 假设：接触触发后脱开从 1 涨到 5 是"合在壁顶几毫米"造成的；未逐局量入沿深度。
- 假设：后退剪切来自 FROZEN_UPPER_BODY 下两手随步态错开 4–6 cm（一局轨迹看到），未在多局验证。
- 待验证：`contact_along` 0.13 单独的效果（基线里显式写回 0.09）。
- 这条线 10 局方差极大（同代码 2–7/10 合爪），对照一律 20 局。
- 同机 JeVLA 任务会把 CPU 占满（评测 ~9 核）；runner 用 `scripts/cpu_headroom.sh` 门等余量。停栈后立刻重起会让 deploy 收不到 LowState、每局倒地（`initial_stance_unstable` ×16）——停后等 ≥20 s，确认无残留 sim/deploy 进程。
- `pkill -f` 会匹配到自己的命令行：杀进程用只在脚本文件里写目标名的脚本，命令行不提目标名。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
bash scripts/cpu_headroom.sh; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑

# 基线（复制后改 name 和一个旋钮；scene 用 name: warehouse_textured）
CFG=configs/experiments/box_room_grasp_baseline_20260924.yaml
bash scripts/start_stack.sh collect $CFG --gpu 0
bash scripts/bimanual/run_agent_server.sh $CFG --gpu 0
bash scripts/stop_stack.sh --instance 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/render_trace_videos.py outputs/runs/<run>   # 自动带 ghost
bash scripts/render_isaac_warehouse.sh outputs/runs/<run>        # Isaac 并排，previews_isaac/

# 场景重装 / Isaac 场景层
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/prepare_logistics_scene.py outputs/scene/<new_name>
python3 scripts/build_isaac_warehouse_scene.py   # 需要 pxr（pip install usd-core）

make test TESTS="tests/test_contact_close.py tests/test_lift_pitch_follow.py tests/test_crate_descend_align.py" && make check
```

- 逐 tick 记录：`outputs/runs/<run>/observations/<局>/observations.jsonl`（`low_reach.capture_error/observed_tilt/attachment_error/pitch_follow_*`、`base_position`）；轨迹 `trace/<局>_state_trace.npz`（qpos/seq）。
- 预览 ≤500 KB 在各 run 的 `previews/`；Isaac 并排在 `previews_isaac/`。
