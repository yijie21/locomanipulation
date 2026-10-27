# STATE — 2026-10-27 01:10 UTC（主线：场景 RSI，抓取线 grasp_atomic / grasp_prep）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，最新提交见 `git log --oneline -5`（没推送）。

## 提交与磁盘
- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`.playwright-mcp/`、
  `assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_*.md`、
  `run_engineer.out`、根目录的 `reports/`、`research_notes/`。`outputs/`、`third_party/`、`datasets/` 永不进仓库。只按路径 `git add`。
- 提交前确认 `git config --get core.bare` 是 false、`user.name` 是 yijie21。提交钩子对代码改动跑全套 CPU 检查（几分钟、多核）。
- **已确认（10-27 01:00）：`/workspace` 只剩 20 GB（99%）。** 10-26 新增约 33 GB：Kimodo venv 7 GB、Llama-3-8B 15 GB（HF 缓存）、
  33.3 的 8 个重放沙盒约 11 GB。删不删待用户定（TODO 33.3.1、38.10.1.1）。

## 另一个会话（已确认）
- `locomanipulation-ea`：视觉 RSI 线，worktree `/workspace/code/locomanipulation-vision-rsi`，分支 `design/vision-rsi-20261023`，
  用户的 herdr 里叫 loco_real。它也编辑进展报告（同一个本地文件），改之前互相发消息。
- 约定：主线开仿真 / RSI 运行 / Unreal 编辑器前先发消息；Unreal 编辑器开着时整机 CPU 不能打满（`editor_guard.sh` 到约 15 核就杀编辑器，
  10-26 被杀过两次：一次它的 10 个进程，一次我的 ffmpeg 默认多线程——ffmpeg 要加 `-threads 2`）。
- 视觉线现状（已确认，10-26 晚）：自主探索建图做完（L2 + 探索地图 28 / 50，L3d + 探索地图 19 / 50）；下一步是探索时的定位偏差、
  台面碎片合并、物体身份。它那边没有在跑的任务。

## 当前进度（已确认）
- 主任务 grasp_atomic 现任仍是 run C 的 `5cba2fa49d31`，封存 38 / 50；grasp_prep 31 / 40。
- 进展报告：https://claude.ai/artifact/5KrdvZBjGQikFjnEpj1oYY 第 13 版（本地 `outputs/scene_rsi_20261016/report/progress_20261026.html`），
  分享设置是“知道链接的人都能看”。按 mentor 视角重写过；7 段视频全是 Unreal 渲染（脚本 `progress_media/render_unreal_report.py`、
  `explore_unreal.py`）；视觉线建图一节有传感器 / 方法表和流程图，事实经视觉线核对。
- 10-26 连着做完的 TODO：39.1（39.2 按停止条件不做）、36.9 两条修复、38.10.2、38.10.1、33.4、33.3（含重放）。数字见各项。
- 38.10.1 的结论：Kimodo 生成 + ScaleBFM 跟踪，12 / 15 没摔，手离原局预抓点中位 12 cm；约束取自成功局（量的是复现能力）。
  文件 `outputs/scene_rsi_20261016/traj_prior_20261026/`（`README.md`、`track.jsonl`、`summary.json`）。

## 下一步（编号见 `TODO.md`）
- 用户还没回答的：38.10.5（报告要不要加回“生成轨迹”一节、配 3 段 Unreal 视频）。
- 等用户拍板后再做：比较方法（DECISIONS 10-26 38.2 收尾，“每种设置 ≥ 3 次”）、38.10.3（`follow` 原语接进循环）、33.2 / 37.6。
- 其余待用户定的列在 DECISIONS 2026-10-26“连着做 TODO”一节的“未决”里（36.9 四条、39.3、33.6、33.21、33.22、33.7、36.10、29.5）。

## 未解决 / 卡点
- 已确认：本机仿真确定（同代码同起点结局相同）；A800 和本机同设置 73 / 80 局相同，不配对。A800 只跑用户指定的实验。
- 已确认：Kimodo `num_samples=4` 只返回 1 个样本，没查原因；文字编码器加载时报 LoRA “UNEXPECTED” 键（文字条件可能偏弱，待验证）。
- 已确认：Mouse_1 在多数 Kimodo 跟踪局里挪了 2–6 cm，像是自己会滑（待验证：静止局里会不会动）。
- 假设：33.3 的 S 方向假设能不能被 Manager 选中、过验收门——重放量不到，要真运行。
- 规则：停进程用明确 PID 或进程组，不用 `pkill -f` / `pgrep -f` 匹配同一条命令里的文字；不改正在跑的 bash 脚本。

## 关键路径与命令
```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 && PY=$VENV_SIM_PY
# Unreal（默认工程 URLabSmoke，GPU 1）：开 / 关；渲染报告视频（bridge Python）
bash scripts/start_unreal.sh --gpu 1 ; bash scripts/stop_unreal.sh
PYTHONPATH=src MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$BRIDGE_PY" outputs/scene_rsi_20261016/report/progress_media/render_unreal_report.py [名字 ...]
# Kimodo：生成（自己的 venv）与跟踪（仿真 venv）
(cd third_party/kimodo && HF_HOME=/workspace/huggingface_cache .venv/bin/python ../../outputs/scene_rsi_20261016/traj_prior_20261026/kimodo_gen.py \
   ../../outputs/scene_rsi_20261016/traj_prior_20261026/jobs.json ../../outputs/scene_rsi_20261016/traj_prior_20261026/gen)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY outputs/scene_rsi_20261016/traj_prior_20261026/kimodo_check.py track
# 能力汇总与外环加权选择（33.4）
python3 scripts/rsi/capabilities_wanted.py
$PY scripts/rsi/outer.py select outputs/scene_rsi_20261016/outer/round2 --loops <沙盒...> --wanted outputs/scene_rsi_20261016/outer/capabilities_wanted.json --out /tmp/x.json
# 重放对照（33.3 用过）
python3 outputs/rsi_speed_20261021/ab/run_ab.py <plan.json> --jobs 4 ; python3 outputs/rsi_speed_20261021/ab/analyze.py <plan.json>
# 新运行：照 outputs/scene_rsi_20261016/grasp_prep_a/launch.sh（建沙盒、核对、loop、封存集），nohup setsid 后台跑
```
- 场景 `outputs/scene_discovery/adt_apartment_mars/`；沙盒都在 `/workspace/code/rsi-scene/`；外环登记表 `outputs/scene_rsi_20261016/outer/registry.json`。
