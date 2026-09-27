# STATE — 2026-09-27（会话：场景能力扫参 → 专家改进 ①–⑤ 全部 A/B → 双臂抓箱默认配置 `box_platform` + quick）

工作目录 `/workspace/code/locomanipulation`，分支 `main`；2026-09-27 起的代码全部提交推送：评测修复、`images.source`（`356d806`）、扫参会话代码 + box_platform 的 Isaac stage（6.9，本次提交）。
没有运行中的栈、训练或渲染；GPU1 上的进程是别的项目的。

## 工作树

- 未跟踪、与本项目无关、不要提交：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`；`third_party/GR00T-WholeBodyControl` 挂着 worktree `outputs/deploy_{docbuild_20260924,pinned_20260925}/`。

## 当前进度（均已确认，数字在 `docs/ledger/crate.md` 2026-09-26/27 与 `docs/box_grasp_envelope.md`）

- 扫参第 1–3 步跑完（835 + 60 局）：没有布置能确定 ≥ 0.5；中心格 H0.60 × L0.32 × W0.45 约 0.45；起点距离/朝向几乎无影响，箱宽/偏角/setback/横移（左侧）敏感；停位深度 0.36–0.45 m 是好区。
- 专家改进 ①–⑤ + 抓深 + 下探速度 + 壁厚 + 横向前馈（离线）全部无效或负结果；③ 附着反馈把站直期脱手 27 → 13%（总数不变）；`ready_after_brake quick` 每局省 5.4 s 成功率不变。
- 默认版本：`configs/experiments/box_platform_default_20260927.yaml`（场景名 `box_platform`，quick，起点安全范围随机，实测 24/60 = 0.40）；README / docs/README / warehouse §8 索引到 `docs/box_grasp_envelope.md`。
- 视频：`outputs/videos/box_range_quick_20260927.zip`（24 段，quick）、`outputs/videos/box_safe_range_20260927.zip`（11 段）。
- warehouse 200 集 checkpoint-5000：开环 MSE 0.00123；闭环评测器修了 4 处（摆放、物体名、初始 token、夹爪下限）后 10 局 **9/10** 抬起（台账 crate.md 2026-09-27 末节）；评测怎么跑见 `docs/warehouse.md` §6。遗留：`stand_off` 0.44 对照（2.10）、bottle 5.10。
- 图像来源是整次运行的 `images.source`（mujoco | isaac，DECISIONS 2026-09-27 末节）；`box_platform` 有 Isaac stage（ISO 20），冒烟 `box_platform_isaac_smoke_20260927` 走通采集→Isaac 补渲→convert→录像→镜像。全部已提交推送。

## 卡点 / 待验证

- 待验证：同几何在 warehouse 房间 26% vs 台上 43% 的差从哪来（台高 0.37 vs 0.10、房间）——搬回房间前先跑 30 局核。
- 待验证：箱子偏角几度就掉一半的原因（转正身体没用，⑤）；③ 的 27% 卡住是不是它带来的（需再 30 局）。
- 已确认：中心格上限由"下探前 1.4 s 整对手横向甩 3–5 cm"决定，专家层改不了；1 cm 捏夹蠕滑 0.2–0.4 cm/s 可用 ③ 或厚壁消掉但总数不变。

## 下一步（编号见 `TODO.md`，顺序等用户定）

- **6.9**：③ 附着反馈是否进默认配置；同几何房间 26% vs 台上 43% 的差异排查。
- **6.4.4.7 / 6.4.2**：箱子偏角敏感的根因；②b 位移闸门；方案 B。
- **4.7.6.1.2.6 / .8**："后退"一列怎么改（训练数据不含后退）；然后 100 局闭环（先挪走 run 的 `eval/`，`eval.rounds` 改回 100）。
- **4.7.11**：Isaac 渲图时冻结世界的 lockstep 评测，计划 `docs/handoff/lockstep-isaac-eval-plan-20260927.md`（先做 W0 校准）；过渡是 4.7.6.1.2.11 补帧。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export PYTHONPATH="$LOCOMANI_ROOT/src" TMPDIR=/workspace/tmp
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
bash scripts/cpu_headroom.sh; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑；停栈后等 ≥20 s 再起
# 默认版本采集（两个终端）；trace 事后渲第三视角
bash scripts/start_stack.sh collect       configs/experiments/box_platform_default_20260927.yaml --gpu 0
bash scripts/bimanual/run_agent_server.sh configs/experiments/box_platform_default_20260927.yaml --gpu 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/<run>
# 扫参（每实例一个循环）与报告：docs/scene_sweep.md §3；专家 A/B 对比：python3 outputs/ab_expert_20260927/compare.py arm=run1,run2 ...
# 多栈并行：实例 0–4 同放 GPU0（每栈 deploy 1.5 GB，RTF 0.97–0.98）；起栈错开 25 s，被拒 "not enough headroom" 就单独重起那一个
# 提交：make check（STATE ≤ 60 行）→ 按文件名 add → git commit --no-verify
```
