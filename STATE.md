# STATE — 2026-10-23（主线：场景 RSI。run C 跑完，封存集 36 → 38 / 50；TODO 37 内环提速 B–E 做完、没跑实验（用户：先不开 usage credits，B 到 E 按顺序做，先不开始实验）：37.1 `try.py`、37.2 压缩 history + 机制表、37.3 Engineer 软截止、37.4 验收门第二枪 + 草稿诊断池提前跑（规则第 16 版）；下一次 RSI 运行就是它们的检验；一轮 79 min 的分解见 `outputs/scene_rsi_20261016/speedup_20261023/analysis.md`；10-23 失败分析（`outputs/scene_rsi_20261016/failures_20261023/analysis.md`）：现任 19 / 90 败（手入手碰倒 7、木勺 7、走路 5、太宽 2、够不到 1），诊断池 30 / 30 饱和 → TODO 33.23 诊断池重抽（10-23 做了，用户“按这个顺序来，先做 33.23”：`diag_pool.py`、`loop.py redraw`、规则第 17 版；run C 沙盒驱动已同步，现任重抽完：60 个新起点、3 分钟，诊断池 0 → 15 / 30 失败，没续跑）；SuperMimic（arXiv 2610.28480）读完：是训法不是接口，不训控制器（用户），33.1.4 预测器做了（10-23：任务记录只有 41 个起点、按起点留出 AUC 0.5，改用 A800 15,120 格包络表 `body.reach()`，现行技能线预抓 AUC 0.78，IK 残差 0.60；看不到家具：0.3–0.4 m 那带的失败是花瓶和洗洁精）；33.12 挑选器做了（10-23：`choose.py` 站位、手、抓法、姿态一起打分，规则第 18 版；离线核对 150 局：成功局抓法种类全一致，木勺俯捏改斜 45° 失误概率 0.76 → 0.05）；33.5 准备动作层做了（10-23：`prep.py` 转 / 挪 / 清三种目标的接口和挑选器，起点 `placement` 选项，能力族 `grasp_prep` 三个任务、种子 `seeds/grasp_c11`，规则第 19 版；挑选器改为只站 `approach_edges`、算矮家具、走廊被挡算要准备）；都没接进技能、没开运行；36.10 未定）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-23）：合并了 origin/main 的 6 个提交（henry-zhao 在 SRA0889 上改的 LOCOMANI_WORKSPACE / LOCOMANI_VENV / RSI_GPUS / RSI_PROCS，合并提交 `366bb8f`），连同文档更新一起推送（用户 10-23“先把文档更新，让另一个开发者能够复现目前为止的 RSI 过程，然后 push 上去”）。
已确认：henry-zhao 直接推 origin/main，可能自带检查失败；推之前先 `git fetch` 再合并。

## 提交与磁盘

- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_locomanipulation_discussion_2026-10-12.md`、`.playwright-mcp/`。按名字暂存，不用 `git add -A`。
- 已确认（10-21 12:10）：`/workspace` 剩 86 GB（之前 98%、23 GB）。78 个 A/B 重放沙盒（`/workspace/code/rsi-scene/ab-*`、`replay-ga-*`）删了 `assets/`、`papers/`（都在各自沙盒的 git 里，`git checkout` 能恢复），记录和会话没动；29.9 的两个沙盒已删，材料在 `outputs/fresh_rsi_20261020/`。
- 10-23 已全部推送到 origin/main；之后的本地提交按用户要求再推。

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

- 1. 33（用户 10-23 定的顺序：33.23 → 33.1.4 预测器 + 33.12 → 33.21 → 33.5 → 37.5 / 37.6；33.23、33.1.4 预测器、33.12 挑选器、33.5 准备动作层做了；33.21 脆弱起点等 A800 能连上；下一步要么开 `grasp_prep` 的运行（`sandbox.py <沙盒> grasp_prep` 建沙盒再 `loop.py run grasp_prep --stop-rate 0.9`，约 8 小时），要么 37.5 / 37.6，用户定；33.8、33.13 结构、33.14 已做；选项 B 续跑完：沙盒 `ga-cont-b-20261022`，第 6、9 轮接受，封存集 31 → 36 / 50；再按 33.15–33.18 续跑 `ga-cont-c-20261022`，第 11 轮接受，封存集 38 / 50，见 TODO 33.18.1；33.18.2 预算 bug 10-23 修了，只数本次运行的验收门调用）：33.10 已做（`ab5403d`，6 / 10 次运行的第一个结论改判接受）；33.15–33.18 已做（10-22，规则第 15 版，D12–D14 按推荐，33.15.1 记录库没做）；接着从第 9 轮的现任接着跑一轮（`--stop-rate`），再 33.7 三人按范围分工、33.11，再 33.1 身体模型（含 10-22 A800 经验：33.19 设计探针、33.20 按层归因、33.21 脆弱起点、33.22 算力分工，D16–D20；A800 数据 `outputs/scene_rsi_20261016/a800_20261022/summary.md`）（含 33.1.2 动手前预测）、33.9 关键点约束原语、33.12 站位和抓法一起选、33.2、33.3、33.4、33.5、33.6。10-22 失败分析：32.12 第 1 轮那版 90 局失败 38（物体比爪口宽 9 + 5、侧抓够不到 13、木勺 5、没走到站位 7、摔倒 3）。
- 2. 小项：29.8 查测试在提交钩子里跑 git（29.5.1–29.5.8 文档偏差、29.7 `walk_mars` 登记 10-23 做完）。
- 3. 等用户：32.7 在账号里开 usage credits（角色用快速输出）；第 2 轮外环何时恢复（10-19 起暂停，27.5.3 / 27.5.4 / 27.8.5 / 27.10.8 跟着它）。
- 4. 可选 30.6 边走边转；推迟 27.6 对照（写论文前；10-22 加第 ③ 组 RoboHarness 式运行时大模型基线）。放弃（10-22 用户：研究初期不做严格对照）：35.2、30.4、30.5、29.5.9、32.4.5、27.2.1.2、27.7、27.13.3。
- 最近完成：29.10 / 29.5.x / 29.7（10-23：交接文档 `docs/handoff/grasp-atomic-line-20261023.md`——grasp_atomic 四次运行、每版改动、复现和续跑命令；runbook 8 处偏差、§5.2 工具和开关改列表；`walk_mars` 进 `configs/scenes.yaml`）、37.0 / 37.1（10-23：一轮时间分解；角色实验命令 `try.py`，走队列并行、240 局预算；修 `family.hidden_seeds` 在隐藏树锁住时崩）、33.18.1 / 33.18.2（run C 38 / 50；验收门预算只数本次运行）、33.8 / 33.13（结构）/ 33.14（多赢 − 少赢 ≥ 3 且少赢 ≤ 多赢 / 3，候补池留 2 个，`base` / `adopted_pool`，反复提出的优先，重开最多连两轮，技能代码不许写物体名字）、36.7 / 36.8（越界作废一轮、连续两轮才停；删“连续 3 次被拒”停止；Manager 第 5 轮上下文 93 → 42 KB；删 `library/`、`review_replay.py`；工具和展示脚本在 `scripts/rsi/report/`）、36（流程复查，约 60 条；32.12 第 1 轮被采用的草稿读过经验库里的旧抓取代码，31 vs 20 不干净）、32.12 / 35.3、32.15（三 Engineer + Composer + 经验库默认，规则第 11 版）。

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

- 登记表 `outputs/scene_rsi_20261016/outer/registry.json`，各族 `last` = `sleep_20261021`（0.5 m/s + 休眠；0.25 m/s 的上一批是 `surfaces_20261020`）；备份 `/workspace/tmp/registry_before_surfaces.json`。
- 场景 `outputs/scene_discovery/adt_apartment_mars/`（台面修正前的备份 `adt_apartment_mars.bak_20261020_before_surface_heights/`）；沙盒都在 `/workspace/code/rsi-scene/`。
- 更早的线（warehouse / bottle 的 ScaleBFM RSI、SONIC 路线）都停着，状态在 TODO 25、26。
