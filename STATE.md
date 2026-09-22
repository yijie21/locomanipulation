# STATE — 2026-09-22（会话：包络 → 箱沿线 → 纸箱走篮子线）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，与 `origin/main` 一致（本会话 20 个提交已推送）。
工作树里 codex 的未提交改动（`probe_ghost_reach.py`、`ghost_reach_execution{,_cli}.py`、`test_ghost_reach_execution.py`、
`docs/bimanual_reachability.md` + 未跟踪的 `generate_joint_reach_bank.py`/`joint_reach_sampling.py`）**不是我的，不要动**。
没有运行中的栈或后台进程（`find /workspace/tmp -maxdepth 3 -name stack.json` 应为空）。

## 当前进度（已确认）

- 上次交接的 4 条全部完成：下蹲修复（三版 2/3/2 of 10，机制在台账）、真实边界校正表（仅矮桌一行）、
  y=±0.12 下蹲搜索（60 线 213 次，`query` 取最深过点）、推送。
- 篮子线基线 **17/20**（`outputs/runs/crate_pilot20_20260922`，reel `reel_2x.mp4`）。
- 箱沿线：0/70 深度扫描 → 朝向 bug（重锚定后按实测偏航保持，偏 20–40°）修好 → 逐闸门放开 → `direct_grasp` 模式，
  15 集合爪 0 次。结论：设计问题，退役。记录 `docs/handoff/box-rim-depth-sweep-20260922.md`。
- 纸箱走篮子线：`0abb131`，场景 `outputs/scene/box_table37`（`--preset box_bimanual --table-top 0.371`），
  两次 20 局 **0/20**（`outputs/runs/box_pilot_20260922`、`box_pilot_env_20260922`），端到端跑通。
  记录 `docs/handoff/box-via-basket-line-20260922.md`。
- 按实际站位查包络：进入 reach 的 30 局里 19 局 `reach`（余量 12–32 cm）、8 局横向偏出测量范围、2 局站太远。
  手最终离抓点 6–15 cm；3 局合爪抬起后脱手。

## 下一步（用户在问"要按这个顺序改吗"，未答）

三个失败源各一个改动，每个 20 局对照，模板 `configs/experiments/box_pilot_env_20260922.yaml`（GPU 0，instance 0）：
1. 走入落点：`visual.stand_off` 从 0.50 调到实际能停在 0.24–0.28 的值（实测过冲 10–20 cm）；横向容差按纸箱重标。
2. 落手：让 `pregrasp_above` 真正在壁顶之上生效，避免蹭壁顶推走箱子（10/30 局 `basket_displaced`）。
3. 停滞判据：expert 的 `reach_stalled`（3 s 无进展）与包络执行器（12 s、反馈上限 0.15）对齐。

```bash
# 建场景（30 MB 视觉 mesh 不入库，先复制）
cp outputs/scene/logistics_mujoco/mesh/nor_cal_box.obj assets/crates/nor_cal_box/visual/
PYTHONPATH=src "$VENV_SIM_PY" src/locomani/tasks/crate/bimanual_scene.py outputs/scene/box_table37 --preset box_bimanual --table-top 0.371
# 跑一组 20 局（两个终端；先查 bash scripts/cpu_headroom.sh 与 nvidia-smi，GPU1 常被别的会话占满）
bash scripts/collect_data.sh              configs/experiments/box_pilot_env_20260922.yaml --gpu 0
bash scripts/bimanual/run_agent_server.sh configs/experiments/box_pilot_env_20260922.yaml --gpu 0
# 按实际站位查包络（脚本在 scratchpad 里没入库；逻辑：observed_grips → 站位系 → reach_envelope.query）
```

## 未解决 / 卡点

- 待验证：`pregrasp_above` 在纸箱上是否真的把手先送到壁顶上方（篮子线有这一步，纸箱录像里手是直接落到壁顶）。
- 待验证：1 cm 薄壁在 dex1 指垫间能否保持（3 局合上后脱手）。
- 假设：走入横向偏 15–36 cm 是视觉停车点按篮子标定所致；未查 `visual_approach` 的停车逻辑。
- 已确认：篮子线的 `AssistedGuidance` 可见性采样网格按物体 box 缩放了，但 ego 相机门限未重标（视觉丢失 3/40）。
- 记忆 `MEMORY.md` 已接近 24.4 KB 上限（本会话加了两行），下次新增前要压缩。

## 关键路径

- 包络查询：`"$VENV_SIM_PY" -m locomani.tasks.crate.reach_envelope.query <5 个目录> -- x y h d yaw`（目录列表在 `docs/reach_envelope.md` §4）。
- 箱沿线扫描工具：`scripts/box_rim_sweep.py plan|run|table`（仅对照用）。
- 纸箱抓点：`src/locomani/tasks/crate/grasp_region.py::grasp_points_for`。
