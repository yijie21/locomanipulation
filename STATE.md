# STATE — 2026-10-22（主线：场景 RSI。32.12 跑完：新流程 1 轮做到 31/50；32.15 把三 Engineer + Composer + 经验库设为默认；10-22 用户：研究初期不做严格对照，35.2 等重测取消；下一步 33，D1–D5 待定）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-20 13:10）：已推送到 origin/main = `f5fb27c`（用户“先把目前的改动 push 到远端”；远端没有新提交，不用合并）。
已确认：henry-zhao 直接推 origin/main，可能自带检查失败；推之前先 `git fetch` 再合并。

## 提交与磁盘

- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_locomanipulation_discussion_2026-10-12.md`、`.playwright-mcp/`。按名字暂存，不用 `git add -A`。
- 已确认（10-21 12:10）：`/workspace` 剩 86 GB（之前 98%、23 GB）。78 个 A/B 重放沙盒（`/workspace/code/rsi-scene/ab-*`、`replay-ga-*`）删了 `assets/`、`papers/`（都在各自沙盒的 git 里，`git checkout` 能恢复），记录和会话没动；29.9 的两个沙盒已删，材料在 `outputs/fresh_rsi_20261020/`。
- 本地提交没推送：`24fc9f7` 起到最新（用户没让推）。

## 当前进度（全部已确认）

- 32（10-21）：仿真提速 S1 + S3 + 休眠，150 局 579 → 224 s；重放（`replay.py`）+ 五组对照（`outputs/rsi_speed_20261021/ab/`）：Reviewer 工具、分范围 Reviewer 没好处（关），Engineer 保持 xhigh，三候选 + Composer 和经验库倾向采用，进 32.12。
- 32.4.4（A800）：书本任务在 0.5 m/s 下变差是到位后身体还在晃（预抓误差 2.27 → 等 1 s 1.22 cm），成功只差 4–7 / 95、不显著；动作层不改，等稳留给技能层（32.4.5）。
- 33.1.1（A800）：身体包络 15120 格，5000 成功；前伸上限 0.6–0.7 m，0.7 m 以内的失败是控制器送不到；姿态原语骨盆最低 0.60 m，下蹲不帮忙（`outputs/scene_rsi_20261016/capability/envelope_remote/summary.txt`）。33 的 D1–D5 待用户定。
- 34（A800）：gr00t_wbc 走路更顺但场景任务 10 / 92 vs ScaleBFM 55 / 92，不换。
- 31（完成，2026-10-21）：任务族 `grasp_atomic`，空种子 → 2 次采纳；封存集 50 局：种子 0，第 1 版 18，第 2 版 24（= 当前目录 24，旧种子 2）。沙盒 `/workspace/code/rsi-scene/grasp-atomic-a`；参照和被拒候选的封存局在 `outputs/scene_rsi_20261016/grasp_atomic_refs/`；页面 https://claude.ai/artifact/Cm4CZTFfFp8tvid6GxyKKp （源 `outputs/scene_rsi_20261016/grasp_atomic_media/`）。
- 27.14：台面高度改按面积、视频按实测手指画，已改并重测：8 族 188 → 187 / 632，标定 101 → 106 / 856，做成的任务 30 → 31。报告视频在新场景重跑、Unreal 重渲（10 局因 10-19 补碰撞变失败，用户定保留原起点）。报告网页第 4 版：https://claude.ai/artifact/1YY339VYjnSk8FTG1WCfBN （分享设置是“任何拿到链接的人”，原来就是）。
- 29：文档主线换成 RSI（`docs/rsi.md` 命令、`docs/rsi_design.md` 原理），VLA 文档原地归档；代码在 `scripts/rsi/`；复现材料 `rsi_kit/` 进了 git（用户推翻场景发现方案第 17 条），`git archive` 里没有它。
- 29.5 新读者测试：做完 0–4 节和 5.1，L1 第一次迭代的 Reviewer 跑到一半时按用户要求停了；7 处偏差已记进 TODO 29.5.1–29.5.8；现场见 `/workspace/tmp/fresh_rsi_20261020/PAUSED.md` 和 `report.md`。
- 30：动作层默认走路速度 0.25 → 0.5 m/s（`58d1dc5`，`rules.md` 第 7 版）。0.25 m/s 时 ScaleBFM 碎步蹭地（15 cm 步长、2 cm 抬脚、34–40% 滑行），0.5 m/s 正常迈步。对比视频页（私有）：https://claude.ai/artifact/Ss5X8HZdbX1f9RFZiqjfAe
- 28：AMO 不用，控制器还是 ScaleBFM（用户定）。第 2 轮外环仍暂停（用户 10-19）。

## 下一步（10-22 整理的待办，编号见 `TODO.md`）

- 1. 33（D1–D8 等用户定；先不开始实验）：先 33.8 放宽验收门 + 候补、33.7 三个 Engineer 按范围分工（HarnessPro），再 33.1 身体模型、33.9 关键点约束抓取原语（RoboHarness）、33.2 定向检验（放宽门用过一次再定）、33.3 换做法、33.4 跨族汇总、33.5 准备动作层、33.6 总评。
- 2. 小项：29.5.1–29.5.8 新读者测试记下的文档偏差（29.5.6 只改措辞）、29.7 登记 `walk_mars` 的构建命令、29.8 查测试在提交钩子里跑 git。
- 3. 等用户：32.7 在账号里开 usage credits（角色用快速输出）；第 2 轮外环何时恢复（10-19 起暂停，27.5.3 / 27.5.4 / 27.8.5 / 27.10.8 跟着它）。
- 4. 可选 30.6 边走边转；推迟 27.6 对照（写论文前；10-22 加第 ③ 组 RoboHarness 式运行时大模型基线）。放弃（10-22 用户：研究初期不做严格对照）：35.2、30.4、30.5、29.5.9、32.4.5、27.2.1.2、27.7、27.13.3。
- 最近完成：32.12 / 35.3（新流程第 1 轮封存 31/50；同为 0.25 m/s 时，比老流程第 1 轮的 20 好（p 0.0096），和老流程五轮的 28 分不开），32.15（三 Engineer + Composer + 经验库设为默认，`rules.md` 第 11 版）。

## 未解决 / 卡点

- 已确认：登记表基准和复现材料是 0.5 m/s + 休眠的记录（`sleep_20261021`），动作层默认已是 0.25 m/s；按用户 10-22 不重测，下次外环的回归拿它比，结论只作参考。
- 已确认：同一台机器上用复现材料重测 pick_single 不是逐局相同（b 3 c 3，总数相同）。用户 10-22：不查原因，文档改成“总数相近”（29.5.6）。
- 已确认：10-20 09:06 一个测试在工作树的提交钩子里把共享的 `.git/config` 改成 `bare = true`，已改回、测试已修；别的测试里有没有同样的问题**待查**（29.8）。
- 已确认：A800 和本机同一设置的场景局只有 73 / 80 结局相同，远端的局不能和本机的配对。
- 规则：`pkill -f` 会杀到命令行含同样字样的 shell，停进程用明确的 PID 或会话号（`pkill -s <sid>`）；不改正在跑的 bash 脚本。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 && PY=$VENV_SIM_PY
# 全部任务族和标定样本按当前技能目录重测（后台跑，日志看每族一行）
nohup setsid $PY scripts/rsi/regress.py run <tag> --procs 12 > /workspace/tmp/regress_<tag>.log 2>&1 &
# 32.12 完整内环确认：launch.sh <提交> 从头起；resume.sh 从第 4 轮的 Prober 续跑
nohup setsid outputs/rsi_speed_20261021/confirm/resume.sh > outputs/rsi_speed_20261021/confirm/run_resume.log 2>&1 &
PYTHONPATH=src $PY scripts/rsi/successes.py            # 按任务统计做成 / 偶尔成 / 没成
PYTHONPATH=src $PY scripts/rsi/kit.py pack              # 登记表基准前移后重打复现材料，再按名字提交 rsi_kit/
# 步态对比视频（空地直走 4 m，两种速度）：输出在 outputs/scene_rsi_20261016/gait/
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=src $PY outputs/scene_rsi_20261016/gait/gait_video.py 0.25 0.5
# 报告视频：先 bash scripts/start_unreal.sh --gpu 1（渲染时不跑仿真），脚本 /workspace/tmp/report_unreal_v4.sh；停：bash scripts/stop_unreal.sh
# 新读者测试：删掉 /workspace/tmp/fresh_rsi_20261020/repo 后 bash /workspace/tmp/fresh_rsi_20261020/launch.sh（提示在 prompt.md，监视 watch2.sh）
```

- 登记表 `outputs/scene_rsi_20261016/outer/registry.json`，各族 `last` = `surfaces_20261020`；备份 `/workspace/tmp/registry_before_surfaces.json`。
- 场景 `outputs/scene_discovery/adt_apartment_mars/`（台面修正前的备份 `adt_apartment_mars.bak_20261020_before_surface_heights/`）；沙盒都在 `/workspace/code/rsi-scene/`。
- 更早的线（warehouse / bottle 的 ScaleBFM RSI、SONIC 路线）都停着，状态在 TODO 25、26。
