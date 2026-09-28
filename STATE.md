# STATE — 2026-09-28（会话：四边线采集——场景、Isaac stage、每局抽边、pilot、首批 5 条、准备手臂 compact 定为默认）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。远端停在 `f5cbee3`，本地多出若干 handoff 提交。
**三批改动都未提交，混在一个工作树里**（提交等用户说，TODO 7.4 / 4.7.11.8 / 8.12）：
- 第一批 lockstep + Isaac 评测：实现、测试、文档，以及 115 份实验文件加了 `gap` / `camera_frames`。
- 第二批 Isaac 画质与预览（`blender_to_usd.py`、`render_isaac_videos.sh`、`pipeline/video_preview.py` 等）。
- 第三批四边线（本会话）：新文件 `tasks/crate/warehouse_box_scene.py`、`sim/scene_edges.py`、`tests/test_{warehouse_box_scene,four_edge_placement}.py`，
  以及 22 份 `configs/experiments/{box_platform_default,warehouse_edges_*}_20260928.yaml`。改动在 expert、visual_approach、visual_config、
  box_platform_scene（`resize_carton`）、collect（`--until-kept`）、两个评测 `run_round`、两个渲染入口、`scenes.yaml`、文档和台账。
- 与前两批混改、要按 hunk 拆：`build_isaac_warehouse_scene.py`、`mujoco_render.py`、`vla_{rounds,lockstep}.py`、`docs/{rendering,warehouse,single_arm}.md`、change-map ×2、两份测试。
- `src/teleop_free/lockstep_vendor` → `src/locomani/backends/sonic/lockstep_vendor` 是**已暂存**的 `git mv`。只提交别的文件时用`git commit -- <文件>`，否则会把它带进去。

已确认：没有运行中的栈、渲染或镜像，实例锁 0–4 全空，GPU0 空；GPU1 约 5.8 GB 在用（JeVLA 训练约 07:30 起已停）。

## 工作树

- 未跟踪、与本项目无关、不要提交：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`。
- cap120 运行的 `config.yaml` 被手改过（原件 `*.bak_20260927`）；`outputs/runs/{ls,wc}_*_20260927` 是 10 个只评测的对照目录。
- 本会话的运行：`outputs/runs/warehouse_edges_pilot_*_20260928`（7 个 pilot）、`warehouse_edges_prep_*_20260928`（3 组 A/B），
  以及 `warehouse_edges_<尺寸>_20260928`（quick 版，已暂停在 7/10/5/6 条，保留、不并入训练集）。
- 2026-09-29 后可删：`outputs/experiments/warehouse_edges_20260928/trial_45x20x18`（107 MB，TODO 8.11）。

## 当前进度（四边线，TODO 第 8 节；台账 `docs/ledger/crate.md` 2026-09-28 两节；DECISIONS 2026-09-28 两节）

- 已确认：四个场景（每边空地 ym 1.29 / yp 1.63 / xm 1.11 / xp 1.05 m）与 Isaac stage 轮廓差 ≤ 3 px（台账 `blender_scenes.md` 第 3 节）。
- 已确认：每局抽边的代码对旧配置不改变任何行为（4 份配置 × 200 个种子逐位相同）；12 局真实摆放骨盆离房间网格 ≥ 0.34 m。
- 已确认：pilot 29/70 = 0.41，按边 xm 6/15、xp 5/13、ym 11/19、yp 7/23，每局 RTF 0.99–1.00。
  假设（10 局分不开）：各边、各尺寸的成功率相同。
- 已确认：quick 版每种先采 ≥ 5 条，Isaac 视频 `outputs/videos/warehouse_edges_first5_isaac_20260928.zip`；pilot 录像 `..._pilots_20260928.zip`。
- 已确认：compact 17/40 vs quick 19/40（p = 0.83），准备 8.7 → 4.9 s；用户定为双臂采集默认，写在实验文件里，dataclass 仍是 full（`make check` 通过）。

## 卡点 / 待验证

- 待用户定：compact 版正式采集前，是否先每种 5 条并出 Isaac 视频；训练图像用 MuJoCo 还是 Isaac。
  现为 `images.source: mujoco`，采集一开始就写进运行目录的配置；GPU1 已空。
- 未做：四个尺寸的运行合成一个训练集的步骤还没有，convert 目前一次只处理一个运行（TODO 8.6.5）。
- 待验证：compact 在 10 cm 平台上没单独测过（envelope 的 24/60 是 quick）。
- 待验证（上一会话留下）：新 stage 下 checkpoint-5000 的 Isaac 评测是否偏离 81/83/88（TODO 7.2）。

## 下一步（编号见 `TODO.md`）

- **8.6.4**：用 compact 采集 `warehouse_edges_compact_<尺寸>_20260928`，每种 100 条，GPU0 实例 0 串行，约 8–9 小时（先问上面两个问题）。
- **8.12 / 7.4**：提交三批改动（等用户说）。
- **8.8**：采满后做数据规模对照（全量 400 条 vs 均衡抽 200 条，各 2 个训练种子）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export PYTHONPATH="$LOCOMANI_ROOT/src" TMPDIR=/workspace/tmp
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 这台机器上采集要用打过补丁的 deploy
timeout 30 bash scripts/cpu_headroom.sh 5; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# 一种尺寸的采集（录制只能用实例 0）；加 --until-kept 5 就先采 5 条暂停，去掉它再跑同一条命令即续采
E=configs/experiments/warehouse_edges_compact_45x32x33_20260928.yaml
bash scripts/start_stack.sh collect "$E" --gpu 0 && bash scripts/bimanual/run_agent_server.sh "$E" --gpu 0 --until-kept 5
# 首批 5 条 + Isaac 视频的整条流程脚本（改其中的配置名即可复用）：outputs/experiments/warehouse_edges_20260928/first5.sh
bash scripts/render_isaac_videos.sh outputs/runs/<run> --gpu 1 --insets [label...]   # 不给 label 就渲全部成功局
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/<run>   # 每局按边换机位
/venv/main/bin/python scripts/build_isaac_warehouse_scene.py --scene outputs/scene/<新场景>              # 新尺寸的 Isaac stage（要 pxr）
# 提交：make check → 按文件名 add（不要 -A）→ git commit -- <文件> → fetch 确认远端无新提交 → push
```
