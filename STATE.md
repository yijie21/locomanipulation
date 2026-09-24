# STATE — 2026-09-24（会话：三条采集线按 README 从零复现；warehouse 离线录制 + 数据集相机补渲；文档跟上代码；可达性代码补提交）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，与 `origin/main` 一致。没有运行中的栈。
理由与被否决方案在 DECISIONS 2026-09-24 最后两节；逐项在 `TODO.md` §4（4.1–4.7）。

## 工作树

- 无未提交的代码改动（今天的提交：`git log --oneline 5d93430^..HEAD`）。未跟踪：`assets/crates/nor_cal_box/visual/`（纸箱源网格，减面前先问维护者）、`assets/blender/{fonts,textures}/`、`package*.json`、`uv.lock`（来历不明）。

## 当前进度

- **复现（已确认）**：无上下文 agent（headless `claude -p`，关钩子 + auto-memory）按 README 跑三条线，
  报告 `outputs/docrepro_20260924/report.md`。50 局开局全部站稳，协作者说的"开局摔倒"未复现（用户：不追）。
  bottle 9/10、crate 17/20；warehouse 按旧文档走到 09-22 实时渲染配置，RTF 0.35，0/20。
- **warehouse（已确认）**：一律 `backend.render: offline`，入口基线 `box_room_grasp_baseline_20260924`，
  离线复跑 20 局 **6/20**（`docrepro_20260924_warehouse_offline`，与 contact 基线一致）。
- **离线录训练数据（已确认）**：exporter 离线时用占位帧，`scripts/render_dataset_cameras.sh` 转换前从 trace 补渲三路相机，
  convert 拒绝未补渲的离线运行。对照实时 crate 局 1000/1000 帧内容一致，同行差值中位数 5–6/255。
  端到端 `docrepro_20260924_warehouse_dataset`：8 局 2 成功 → 补渲 8 局（每局 6–10 s）→ convert → 审计 train/val 32/32。
- **修复与文档（已确认）**：holdout 校验、平铺 `videos/` 的第三人称录像、审计量到箱面、`docs/warehouse.md` 新页等，逐项见 TODO 4.3.x–4.4。
- **可达性代码**：原"协作者 WIP"已提交（`76c5196`，已确认 probe 接受分片参数）；包络搜索本身没起栈跑（待验证）。
- warehouse 抓取线本身（上一会话）：基线 6/20，失败以 `reach_stalled`（沿上停滞）和 `lost_attachment` 为主。

## 下一步（编号见 `TODO.md`，等用户定顺序）

- **4.6** 全新 clone 上按 README 从零验证三份 runbook（这次只在装好环境的本机跑过）。
- **2.4.10.29** 沿上停滞 7/20：小步下探逐步查跨壁，或 align→descend 失败重做一次（overdrive 已证明会把指垫顶偏 4–8 cm）。
- **2.4.10.29.2** 后退剪切 2/20：`retreat_speed` 0.35→0.25，或后退时保留全身伸手控制器夹持。

## 卡点 / 待验证

- 待验证（4.7）：补渲帧比实时帧"新" 1–2 行，训出的策略在实时闭环里是否受影响；warehouse 闭环要实时出图，被纸箱网格（2.4.10.8）卡住。
- 假设（上一会话，未变）：接触触发后脱开 1→5 是"合在壁顶几毫米"；后退剪切来自 FROZEN_UPPER_BODY 下两手随步态错开 4–6 cm。
- 这条线 10 局方差极大（同 seed 同代码一次成、一次败，本会话 7000–7002 也见到），对照一律 20 局。
- 停栈后等 ≥20 s 再起（否则每局 `initial_stance_unstable`）；`pkill -f` 会匹配自己的命令行，杀进程用脚本文件。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 本机：打过两个补丁的 deploy
bash scripts/cpu_headroom.sh; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑

# warehouse（完整跑法 docs/warehouse.md）：复制基线改 name 和一个旋钮
CFG=configs/experiments/box_room_grasp_baseline_20260924.yaml
bash scripts/start_stack.sh collect $CFG --gpu 0
bash scripts/bimanual/run_agent_server.sh $CFG --gpu 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/<run>
bash scripts/render_dataset_cameras.sh outputs/runs/<run> --gpu 0   # 只在 collect.record: true 时，convert 之前
bash scripts/convert_data.sh outputs/runs/<run>
bash scripts/render_isaac_warehouse.sh outputs/runs/<run>           # 可选：Isaac 并排

# 无上下文复现 agent（提示词 outputs/docrepro_20260924/prompt.md）
CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 claude -p --model claude-sonnet-5 --settings '{"disableAllHooks": true}' --permission-mode bypassPermissions --output-format stream-json --verbose "$(cat prompt.md)" < /dev/null > stream.jsonl
make test TESTS="tests/test_render_dataset_cameras.py tests/test_world_lift.py tests/test_check_dataset_proximity.py" && make check
```

- 复现产物 `outputs/docrepro_20260924/`，run `outputs/runs/docrepro_20260924_*`；提交：`make check` → `--no-verify` → 按文件名 add。
