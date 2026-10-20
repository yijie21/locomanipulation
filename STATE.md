# STATE — 2026-10-20 13:10（主线：场景 RSI。在跑：TODO 31 抓取类从原子技能进化的内环，沙盒 `/workspace/code/rsi-scene/grasp-atomic-a`，12:47 UTC 起、墙钟上限 5 h）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-20 13:10）：已推送到 origin/main = `f5fb27c`（用户“先把目前的改动 push 到远端”；远端没有新提交，不用合并）。
已确认：henry-zhao 直接推 origin/main，可能自带检查失败；推之前先 `git fetch` 再合并。

## 提交与磁盘

- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_locomanipulation_discussion_2026-10-12.md`、`.playwright-mcp/`。按名字暂存，不用 `git add -A`。
- 已确认（10-20 12:00）：`/workspace` 剩 125 GiB，`/` 剩 14 GB。
- 删除预告（10-20 12:00 记，10-21 12:00 以后可删，TODO 29.9）：`/workspace/tmp/fresh_rsi_20261020/`（3.8 GB）、`/workspace/code/rsi-scene/pick-repro/`（1.4 GB）。删之前把 `report.md`、`progress.txt` 拷进 `outputs/`。

## 当前进度（全部已确认）

- 31（在跑）：任务族 `grasp_atomic`，10 个客厅物体的 `held(X)`，客厅里随机出生；技能目录从空种子起步，沙盒剥掉旧技能、记录和旧抓取程序（`c6a3139`）。内环 `loop.py run grasp_atomic`（`RSI_MAX_WALL_H=5`，日志 `/workspace/tmp/grasp_atomic_loop.log`）。参照线（封存集 50 局、0.5 m/s）：旧种子 2/50，当前目录 24/50（`outputs/scene_rsi_20261016/grasp_atomic_refs/`）。内环停后：沙盒里 `sealed.py grasp_atomic --all`，`evolution_media.py` 出视频和总图，做进化页面；用户要求约 8 小时内（21:00 UTC 前后）给结果。

- 27.14：台面高度改按面积、视频按实测手指画，已改并重测：8 族 188 → 187 / 632，标定 101 → 106 / 856，做成的任务 30 → 31。报告视频在新场景重跑、Unreal 重渲（10 局因 10-19 补碰撞变失败，用户定保留原起点）。报告网页第 4 版：https://claude.ai/artifact/1YY339VYjnSk8FTG1WCfBN （分享设置是“任何拿到链接的人”，原来就是）。
- 29：文档主线换成 RSI（`docs/rsi.md` 命令、`docs/rsi_design.md` 原理），VLA 文档原地归档；代码在 `scripts/rsi/`；复现材料 `rsi_kit/` 进了 git（用户推翻场景发现方案第 17 条），`git archive` 里没有它。
- 29.5 新读者测试：做完 0–4 节和 5.1，L1 第一次迭代的 Reviewer 跑到一半时按用户要求停了；7 处偏差已记进 TODO 29.5.1–29.5.8；现场见 `/workspace/tmp/fresh_rsi_20261020/PAUSED.md` 和 `report.md`。
- 30：动作层默认走路速度 0.25 → 0.5 m/s（`58d1dc5`，`rules.md` 第 7 版）。0.25 m/s 时 ScaleBFM 碎步蹭地（15 cm 步长、2 cm 抬脚、34–40% 滑行），0.5 m/s 正常迈步。对比视频页（私有）：https://claude.ai/artifact/Ss5X8HZdbX1f9RFZiqjfAe
- 28：AMO 不用，控制器还是 ScaleBFM（用户定）。第 2 轮外环仍暂停（用户 10-19）。

## 下一步（编号见 `TODO.md`；都等用户开口，用户让先暂停）

- 30.3：按 0.5 m/s 重测 8 族和 3 个标定样本，和 0.25 m/s 的记录逐局比（掉落、窄路、停步）。
- 30.4：重测后 `kit.py pack` 更新复现材料，改文档和报告里的数字，看搬运时长规则要不要重新拟合。
- 29.5.1–29.5.9：按新读者的偏差改文档，再从新克隆重跑新读者测试（30.5）。

## 未解决 / 卡点

- 已确认：现在代码默认 0.5 m/s，但所有记录（登记表、复现材料、报告、文档里的数字）都是 0.25 m/s 下测的；30.3 之前照 `docs/rsi.md` 第 4 节核对会对不上（文档已写明）。
- 已确认：同一台机器上用复现材料重测 pick_single 不是逐局相同（b 3 c 3，总数相同）。原因**待验证**：假设是 TorchScript 第一次调用的数值差或克隆的绝对路径不同（29.5.6）。
- 已确认：10-20 09:06 一个测试在工作树的提交钩子里把共享的 `.git/config` 改成 `bare = true`，已改回、测试已修；别的测试里有没有同样的问题**待查**（29.8）。
- 假设（没在场景里测过）：0.5 m/s 拿着东西走可能更容易掉、窄路可能更容易碰家具；30.3 的重测会回答。
- 规则：`pkill -f` 会杀到命令行含同样字样的 shell，停进程用明确的 PID 或会话号（`pkill -s <sid>`）；不改正在跑的 bash 脚本。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 && PY=$VENV_SIM_PY
# 30.3：全部任务族和标定样本按当前技能目录、当前速度重测（约 2.5 h，后台跑，日志看每族一行）
nohup setsid $PY scripts/rsi/regress.py run speed05_20261021 --procs 12 > /workspace/tmp/regress_speed05.log 2>&1 &
PYTHONPATH=src $PY scripts/rsi/successes.py            # 按任务统计做成 / 偶尔成 / 没成
PYTHONPATH=src $PY scripts/rsi/kit.py pack              # 30.4：重测后重打复现材料，再按名字提交 rsi_kit/
# 步态对比视频（空地直走 4 m，两种速度）：输出在 outputs/scene_rsi_20261016/gait/
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=src $PY outputs/scene_rsi_20261016/gait/gait_video.py 0.25 0.5
# 报告视频：先 bash scripts/start_unreal.sh --gpu 1（渲染时不跑仿真），脚本 /workspace/tmp/report_unreal_v4.sh；停：bash scripts/stop_unreal.sh
# 新读者测试：删掉 /workspace/tmp/fresh_rsi_20261020/repo 后 bash /workspace/tmp/fresh_rsi_20261020/launch.sh（提示在 prompt.md，监视 watch2.sh）
```

- 登记表 `outputs/scene_rsi_20261016/outer/registry.json`，各族 `last` = `surfaces_20261020`；备份 `/workspace/tmp/registry_before_surfaces.json`。
- 场景 `outputs/scene_discovery/adt_apartment_mars/`（台面修正前的备份 `adt_apartment_mars.bak_20261020_before_surface_heights/`）；沙盒都在 `/workspace/code/rsi-scene/`。
- 更早的线（warehouse / bottle 的 ScaleBFM RSI、SONIC 路线）都停着，状态在 TODO 25、26。
