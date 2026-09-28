# STATE — 2026-09-28（会话：文档复查 → 只渲成功局 → lockstep + Isaac 评测实现与 100 局校准）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，已推送到 `f5cbee3`（只渲成功局）。
**未提交**：lockstep + Isaac 评测的全部实现、测试、文档，以及 115 份实验文件加 `gap` / `camera_frames`（TODO 4.7.11.8，待用户说提交）。
`src/teleop_free/lockstep_vendor` → `src/locomani/backends/sonic/lockstep_vendor` 是 **已暂存** 的 `git mv`：只提交别的文件时用 `git commit -- <文件>`，否则会把它带进去。
没有运行中的栈、镜像、训练或渲染；GPU1 上约 3.7 GB 是别的项目的。

## 工作树

- 未跟踪、与本项目无关、不要提交：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`。
- 运行 `box_room_grasp_200_cap120_20260925` 的 `config.yaml` 被手改（`eval.rounds: 10`、warehouse token），`summary.json` 手补 train 条目；原件 `*.bak_20260927`。
- 对照用的 10 个只评测运行目录 `outputs/runs/{ls,wc}_*_20260927`（原运行的 config + 改过的 eval 块，train 条目指向原 checkpoint），各约 120 MB。

## 当前进度（数字在台账 crate.md 2026-09-27 末节、计划 `docs/handoff/lockstep-isaac-eval-plan-20260927.md` 第 10 节）

- 已确认：双臂评测有两个时钟 `eval.client: blocking | lockstep`，间隙语义 `eval.gap: continue | hold`，lockstep 另有 `latency_frames` / `latency_dist` / `camera_frames`。
- 已确认：lockstep 下 Isaac 图按冻结时刻的 seq 渲，84/84 块 seq 相符；每周期 18 帧 = 36 tick，特权 seq 与 deploy 状态序号各进 36（请求之间世界不动）。
- 已确认（checkpoint-5000，Isaac 图，seed 6000–6099，各 100 局）：墙钟 hold **81**、墙钟 continue **83**、lockstep 复刻墙钟 **83**、lockstep 目标模型（L = 2）**88**。
- 已确认：lockstep 校准通过——墙钟 81 vs 复刻 83，不一致 14 : 16，p = 0.86，失败构成逐项差 ≤ 3；2026-09-02 的"停得远"签名没出现。
- 已确认：目标模型 vs 墙钟 88 vs 81，p = 0.27，100 局判不动；墙钟 continue 无差别（续发线程让观测 + 推理 70 → 95 ms）。
- 已确认：同 seed 的 lockstep 局不逐位复现，原因是 MuJoCo 默认 4× 多重采样（同一状态渲两次腕图差 1–2 灰度级）；局前 332 tick 和策略服务端（同观测同 seed）都逐位相同。
- 已确认：`render_isaac_videos.sh` 与两个数据集补渲都只渲成功局（已推送 `f5cbee3`）；converter 拒绝未补渲的局。
- 已确认：`make check` 通过（含新测试 `tests/test_crate_lockstep_eval.py` 14 个）。

## 卡点 / 待验证

- 待验证：88 与 83 之差来自"间隙续走"还是"延迟少 2 帧"（4.7.11.6，要加一臂 lockstep `hold` + L = 2）。
- 假设：相机延迟 C = 0；真实值取决于部署硬件。
- 待验证：按需渲与连续渲是否同图（4.7.9.8），lockstep 不消除这个问题。

## 下一步（编号见 `TODO.md`，顺序等用户定）

- **4.7.11.8**：提交并推送 lockstep 实现（等用户说）。
- **4.7.11.7**：默认 `eval.client` 是否改成 `lockstep`。
- **4.7.6.1.2.6**："后退"一列怎么改（训练数据不含后退段）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export PYTHONPATH="$LOCOMANI_ROOT/src" TMPDIR=/workspace/tmp
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
find /workspace/tmp -maxdepth 2 -name stack.json          # 有输出 = 有栈在跑；停栈后等 ≥ 20 s 再起
E=outputs/lockstep_eval_20260927                          # 对照工具：make_runs.py 建只评测的运行、queue.sh 串行跑、compare_arms.py 配对比较
"$VENV_SIM_PY" $E/make_runs.py NAME client=lockstep gap=hold latency_frames=2 rounds=100 images=isaac   # 新建一臂
bash $E/queue.sh 0 1 NAME                                 # GPU0 栈+策略、GPU1 镜像；INST=1 跑实例 1；建 $E/STOP 在当前臂后停
python3 $E/compare_arms.py wc_hold100_20260927 NAME       # 成功率、失败构成、跳帧、同 seed McNemar
"$VENV_SIM_PY" -m locomani.tasks.crate.vla_lockstep latency-dist <run>/eval/traces --out x.json [--inference-only]
# 提交：make check → 按文件名 add（不要 -A）→ git commit → fetch 确认远端无新提交 → push
```
