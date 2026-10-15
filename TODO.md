# TODO

树状任务账本。只改标记、不删行、不重新编号：`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成（带日期）· `[-]` 放弃（写原因）。

## 1. 平铺重构：bash 直接调功能脚本（`docs/handoff/flat-functional-scripts-20260922.md`）

- [x] 1.1 审查协作者报的 4 条重复，判定每条成立与否（2026-09-22）
- [-] 1.2 共享 driver `run_collect.py` / `run_eval.py` + `lines.py` 分派表 —— 用户否决：在功能代码上又加一层；已全部回退
- [x] 1.3 grilling 访谈，16 条决策写入 handoff 与 DECISIONS（2026-09-22）
- [x] 1.4 阶段 1：抽 `pipeline/batches.py`（2026-09-22）
- [x] 1.5 阶段 2：`collect_data.sh` 并入 `start_stack.sh collect|eval`（2026-09-22）
- [x] 1.6 阶段 3：`grasp_gym.run(options)` + 薄 `main(argv)`；`eval_closed_loop.gym_options()`（2026-09-22）
- [x] 1.7 阶段 4：`tasks/bottle/collect.py`，闸门 PASS 5/5（2026-09-22）
  - [x] 1.7.1 `privileged_pub` 端口改为按需解析（2026-09-22）
  - [x] 1.7.2 场景经 `GymOptions.scene_meta` 显式传入（闸门抓到的 import 期冻结）（2026-09-22）
  - [x] 1.7.3 `grasp_gym.run` finally 关 client；两批同进程泄漏采样 fds +2 / threads +1（2026-09-22）
- [x] 1.8 阶段 5：`tasks/crate/collect.py` + `expert.run(config)`，闸门 PASS 4/4（instance 2，顺带验证端口修复）（2026-09-22）
- [x] 1.9 阶段 6：两个评测器加 run-dir 模式，bash 直调；实跑一局 exit 0（2026-09-22）
  - [x] 1.9.1 评测准入改为先 attach 再要余量（2026-09-22）
  - [x] 1.9.2 还原 `bottle_simple_200_20260916` 的历史 eval 记录，冒烟结果留 `eval_smoke_20260922/`（2026-09-22）
- [x] 1.10 阶段 7：四个旧 driver 删除，测试与 dry-run 改形，`wall`/`carry` 进程内经 `main(argv)`（2026-09-22）
- [x] 1.11 阶段 8：`lint_repository.spawn_errors` + `interpreter_spawns.json` + 反向证明测试（2026-09-22）
- [x] 1.12 文档：README 快速开始、`standards.md` 规则、change map；README 进 `COMMAND_DOCUMENTS`（2026-09-22）
- [x] 1.13 提交并推送（`a8bd4e7`、`c175089`，`--no-verify`，协作者文件未碰）（2026-09-22）
- [x] 1.14 修 handoff 文档第 3 行的过时状态句（仍写"阶段 4–8 未开始"）（2026-10-01：改为"阶段 1–8 已落地，遗留见 TODO 1.14–1.20"）
- [x] 1.15 三处同 venv `sys.executable` spawn：`bimanual_scene.py`、`reach_envelope/floor_scene.py`、`reach_envelope/boundary_search.py`（诊断路径，谁动 reach_envelope 时顺手做）（2026-10-01：三处改进程内调用；`build_sim_scene.main(argv)` 退出时还原 `--seed`/`--table-top` 改过的 `scene_lib` 常量。新旧路径各建默认 / seeded / floor 三个场景逐字节一致，同进程先 seeded 后默认也一致；`_write_group` 的 bank 与脚本输出一致；`interpreter_spawns.json` 删 3 条 same-interpreter）
- [x] 1.16 `eval_crate_closed_loop → vla_rounds` 的同 venv spawn（需要双臂 checkpoint）（2026-10-01：`vla_rounds.main` 拆成解析 + `run(config, ...)`，每批在评测进程里先 `run_context.adopt_env(栈环境)` 再 import/调用，`run` 关掉 clean clock 与两个 client。第一次实跑按旧顺序先 import 后接环境，`adopt_sim_dir` 拒绝（IK 已持有 outputs/scene/sim），已改顺序并加新解释器回归测试（错顺序必 FAIL）。闸门：cap120 checkpoint-5000、实例 2、MuJoCo 实时出图、10 局 seed 6000、5 局一批：旧路径（HEAD 快照）4/10、新路径 5/10，逐 seed 不一致 1 : 2；两臂产物文件集合相同、traces 全局编号、`rounds_NN` rc 0、summary 写入。`interpreter_spawns.json` 删 venv_sim 那条）
- [x] 1.17 `wall`/`carry` 转成 `run(config, ...)`，去掉 `global DUMP_DIR`（可选；进程边界已没了）（2026-10-01：两者 `main` 只解析，`run(config, rounds, seed, manifest, ...)`；dump 目录与 ghost trace 走回合上下文（`WallRoundContext`/`CarryRoundContext.dump_dir`、`ghost`），不再是模块全局；`run` 收尾关 client。`tasks/crate/collect.py` 直接调 `run`，`prepare.expert_argv` 与 `diagnostic_argv` 删除。只有单元测试（三批同进程各关一次 client），没有实跑 wall/carry 一批）
  - [ ] 1.17.1 实跑一批 `wall` 或 `carry` 的 `run()`（拆分后只有单元测试；目前 100 份 crate 实验文件全是 `expert: visual`，等有人用到这两个诊断专家时做）
- [x] 1.18 两套评测器本身合并（协作者第 2、3 条的深层问题；需 checkpoint + 闸门）（2026-10-01：两条线的轮次循环不同（grasp_gym 自己的循环 vs `vla_rounds` 分批 + 批间重启 deploy），保留两个文件；循环之外的部分合成 `pipeline/closed_loop.py`——运行目录准入（`attach`：拒已完成的评测、checkpoint、deploy 补丁、本 run 的栈、余量）、`finish`（按状态文件收栈）、`record`（写 summary）、`score`（成功率 / Wilson / 失败分期）。合并时顺手修掉的漂移：双臂版不查 deploy 补丁、attach 之后才拒已完成的评测，Wilson 在 0 局时给 (0, 0)；单臂版用 attach 时的 pid 收栈（deploy 已按局重启）。验证：47 份历史 `closed_loop.manifest.jsonl` 新旧打分逐字段相同（只有 Wilson 1e-16 的浮点差）；双臂运行目录实跑 `gate118_crate_20261001` 4 局 3/4、rc 0、栈收干净、summary 写入。单臂的运行目录路径没能实跑：磁盘上已没有带权重的单臂 checkpoint（`bottle_simple_200_20260916/checkpoint-10000` 在 09-29 清理后只剩 config 和 index），只有单元测试）
  - [ ] 1.18.1 单臂运行目录评测经 `pipeline/closed_loop.py` 实跑一次（需要一个带权重的单臂 checkpoint；`bottle_simple_200_20260916/checkpoint-10000` 在 09-29 清理后只剩 config 和 index）
- [x] 1.19 `eval_crate_closed_loop` 的 run-dir 模式实跑一次（目前只 dry-run；磁盘上没有双臂 checkpoint）（2026-09-27：`run_overnight.sh --from eval` 以 run 目录跑通 10 局，`box_room_grasp_200_cap120_20260925`）
- [x] 1.20 重构审查（2026-09-23）发现的 7 项问题，全部修复；make check 绿，单臂 2×4 实跑闸门 PASS 5/5（`outputs/runs/gate8_fix_20260923`）（2026-09-23）
  - [x] 1.20.1 双臂 run-dir 评测：栈无 SONIC_AUTO_START 而 rounds 走 clean commander（deploy 永不启动）—— `Line.eval_backend`；未实跑，同 1.19（2026-09-23）
  - [x] 1.20.2 采集中断续跑：sessions/manifests 预先追加 → 重复、按位置配对错位（`crate_env2_so45` 已处于此状态）
  - [x] 1.20.3 进程内采集不再写 `collect_NN.log` / `visual_NN.log`，文档仍指向它们
  - [x] 1.20.4 双臂采集吞掉异常后 `timings.jsonl` 记 rc=0；`SystemExit` 不被接住；`main()` 返回值被忽略
  - [x] 1.20.5 `run_stack` 子进程不继承宿主环境（旧路径继承），且与 attach 后重启的子进程不一致
  - [x] 1.20.6 双臂评测 attach 模式不收栈，`run_overnight` 夜跑后栈残留
  - [x] 1.20.7 小项：第 0 批 exporter session 靠 mtime 猜；attach 不校验栈属于本 run；`run_overnight` 只看 `summary.json` 文件存在
  - [x] 1.20.8 提交前文档核对：runbook / change-map-runtime / single_arm_design / silent_failures 补齐（2026-09-23）
  - [x] 1.20.9 提交并推送 `278acdf`（`--no-verify`，rebase 解了两本 runbook 与协作者开环评测段的冲突，协作者在途文件未动）（2026-09-23）
  - [x] 1.20.10 实机演练一次中断续跑（采集跑到第 1 批中途 Ctrl-C，同名重跑，核对 `inputs.json` 与 `interrupted_manifest_*`）（2026-10-01，`drill_resume3_20261001`，bottle smoke、录制、2 局一批：第 01 批写了 1 行时给驱动 SIGINT → 退出码 130、栈被收掉；此时 `inputs.json` 记 batches 1、sessions 2、manifests 2（第 01 批已认领未完成）。同名重跑打印 "batch 01 did not finish; its manifest moved aside to interrupted_manifest_01_…"，第 01 批整批重来进新 session，最后 batches / sessions / manifests 3 / 3 / 3，每个 session 磁盘上的 episode 数都等于对应 manifest 的 produced 行数。前两次演练没中断成功：脚本里 `&` 起的后台作业 SIGINT 被设成忽略，python 继承了 SIG_IGN，KeyboardInterrupt 永远不来）
  - [-] 1.20.11 现存栈（本次之前起的）attach 会被拒——有长跑栈的会话需知会（无主 `stack.json`）（放弃 2026-10-01：09-23 之前起的栈已一个不剩，`/workspace/tmp/p03_i*/stack.json` 只有今天本会话起的）

## 2. 纸箱线（物流房间）——承接上一会话的 STATE

- [x] 2.1 房间 pilot `box_room_20260922` 0/20，代码与文档已提交（`704004d`、`f5ea2e9`）
- [-] 2.2 用户看 `outputs/runs/box_room_20260922/reel_failures.mp4`（放弃 2026-10-01：那是 09-22 旧 carry 专家 0/20 的录像，已被 2.3 的分阶段拆解和 09-24 基线取代；文件仍在原处）
- [x] 2.3 按阶段拆 20 局：摔倒 6 局在哪个相位；reach_stalled 8 局的站位深度与手到抓点距离（2026-09-23：`reach_anatomy_20260923.py`；伸手阶段反馈 1.0/0.15 顶满、手起点离抓点 ~20 cm）
- [x] 2.4 据 2.3 决定改 `stand_off` / `pregrasp_above` / 摔倒原因；每次一个改动、20 局、同 seed 7000（2026-10-01 子项全部关闭；单边基线复跑仍是 6/20，各单旋钮对照见台账 `docs/ledger/crate.md` 2026-10-01 节）
  - [x] 2.4.1 反馈 0.7/0.04（`box_room_fb07`）0/10；整个 carry 块回退（`box_room_carry`）0/10、摔倒 3→0（2026-09-23）
  - [x] 2.4.2 stand_off 0.40（`box_room_so40`）0/10，失败移到"碰动箱子"5/10（2026-09-23）
  - [x] 2.4.3 起点 advance 0 + 刹车余量 0.05（`box_room_approach`）0/10；`walk_steer: depth`（`box_room_steer`）0/10、停位离散 1.2 cm（2026-09-23）
  - [x] 2.4.4 真专家接入守线 + staged + orientation_gain + 1 cm 抓深 + 后退（`box_room_grasp_20260923`）0/10，守线 6/10 越线（2026-09-23）
  - [x] 2.4.5 提交 2.4.4 的代码（expert/visual_approach/visual_config/grasp_region/diag_brake 改动 + `primitives/foot_watch.py`、`tasks/crate/retreat.py`、`tests/test_crate_guard_retreat.py`、`box_room_grasp_20260923.yaml`；默认全关，`make check` 已过）（2026-09-23）
  - [x] 2.4.6 warehouse 起点后移 0.3–0.4 m，用真专家（手臂前伸姿态）重标守线的刹车后滑行，再验证不越线（2026-09-23：后移 0.25 m 到离线 0.90 m；根因其实是慢仿真，见 2.4.10）
  - [-] 2.4.7 伸手回到不摔的 carry 块（crate 值 + 0.7/0.04），另解"手悬在目标上方 5–10 cm 下不来"（放弃 2026-10-01：这条线 2.4.4 起换成 visual 专家 + 守线，carry 块不再用；"手悬在上方"后来就是 2.4.10.29 的沿上停滞，在那里跟）
  - [x] 2.4.8 跑通后退段（`retreat_distance 0.3`），出 10 局视频（2026-09-24 已跑通，2026-10-01 对账时勾上：2.4.10.28.2 后退 0.31–0.37 m、6/20 完成；带 ghost 的视频见 2.4.10.29.1，10 集视频见 2.9）
  - [x] 2.4.9 纸箱抓深常量 `GRASP_BELOW_RIM_M` 仍为 0.025；warehouse 靠配置 `box_grasp_below_rim 0.01`（决定是否改常量）（2026-10-01：常量不动——它是 crate 横杆的抓深；改的是 `CrateVisualConfig.box_grasp_below_rim` 的默认值 0.025 → 0.01：60 份纸箱实验文件都已写 0.01，0.03（2.4.10.15、2.4.10.25.1）和 0.02（09-27，10/30 vs 15/30）都输；没写这个字段的四份 `box_pilot_*_20260922` 钉成 0.025，保持可复现）
  - [x] 2.4.10 查"warehouse 过冲撞箱"：抬着手走 9/10 撞箱 → 改手臂下垂走、刹后抬手（`box_room_grasp_hanging*`）（2026-09-23）
    - [x] 2.4.10.1 站起后重放置 + 世界轴行走；守线线取箱子前面；`walk_body_past_line_max`（2026-09-23）
    - [x] 2.4.10.2 原样重跑排除资源抢占（`box_room_grasp_hanging2_rerun`，仍 0.35×）（2026-09-23）
    - [x] 2.4.10.3 定位慢仿真根因：纸箱 46 万面网格 + 阴影，3 路渲染 39 ms（`warehouse_speed_20260923` + 离线拆分）（2026-09-23）
    - [x] 2.4.10.4 `backend.render: offline` + 占位帧 + `scripts/render_trace_videos.py`；离线 10 局 0/10，停得太远（`box_room_grasp_offline_20260923`，视频在 previews/）（2026-09-23）
    - [x] 2.4.10.5 不录制的栈 CPU 预留 5 核（`NO_EXPORTER_CORES`）（2026-09-23）
    - [x] 2.4.10.6 离线模式下守线余量 0.15 → 0.03，再跑 10 局（2026-09-23：`box_room_grasp_offline_m003_20260923` 0/10；停位中位 0.63→0.49 m，手指差距 35→25 cm，手停在 pregrasp 前 15–23 cm）
    - [x] 2.4.10.7 伸手段诊断：不是包络/反馈——离线伸手时钟冻结（占位帧戳晚于循环 `now`，`reference_elapsed` 0/440 tick），实时不受影响；已修 + 契约测试（2026-09-24）
    - [x] 2.4.10.14 修复后同配置重跑 m003（`box_room_grasp_offline_m003b_20260924`）：0/10，伸手 10/10 执行；≤0.51 ready 姿态压近壁、≥0.54 掌落壁顶停滞、0.46–0.52 抓深 1 cm 夹不住（2026-09-24）
    - [x] 2.4.10.15 抓深 0.01 → 0.03（`box_room_grasp_offline_deep03_20260924`）：0/10、0 合爪、6 局推斜——横向误差 2–6 cm + 修正积分到上限推倒箱子；深抓本身几何可行（2026-09-24）
    - [x] 2.4.10.17 沿下横向 anti-windup（`lateral_hold_below_rim`）+ 预抓点真值居中（`descend_align_tolerance`）（`833ca23`）；hold 单独无效（试提 3），hold+align 试提 6/10（2026-09-24）
    - [-] 2.4.10.18 align 配置上回抓深 `box_grasp_below_rim` 0.03，10 局（放弃 2026-10-01：后来的基线上已经测过——2.4.10.25.1 在 close06 上 0.03 合爪 3、提起 2，无增益；09-27 的 2 cm 10/30 vs 1 cm 15/30；见 2.4.9）
    - [x] 2.4.10.19 下落过程中跨壁误差再漂 3–5 cm（align 后清零、修正 +9…+15 未送达）：查 SONIC 对横向指令的送达，不是接触（2026-10-01 查明：40 局（`s2_base` + `s2_adv015`）下落段，按骨盆系把命令与实测的手都用专家自己的 FK 算出来比——实测的手相对骨盆的跨壁移动跟命令走，corr 0.80、斜率 0.48，**SONIC 只送达约一半的跨壁手臂命令**；同时骨盆自己横漂 10–15 mm（中位），命令里那 25–41 mm 主要是在补骨盆漂移，送达一半就剩 15–21 mm 的世界系漂移。停滞局下落 5.2 s（合爪局 1.7 s），漂得更久，末端跨壁误差 21.8 vs 13.5 mm。跨壁前馈不再试：09-24 的送达差前馈（2.4.10.22）和 09-27 的共模横向修正都是负结果。脚本 `scripts/experiments/warehouse_s2_report_20261001.py`）
    - [x] 2.4.10.20 伸手/下落时手臂晃动（用户 2026-09-24 从视频看到）：是我们流出的参考在晃——逐 tick 重解 IK 时肘/腕俯仰反相互换（corr −0.95…−0.98，肘参考 sd 9–15°），手目标每 tick 只动 0.2 cm；实测关节比参考还平。修法候选：解算加 continuity 项（求解器已有、伸手传 0）或加大手臂 posture 权重；待用户定（2026-10-01 做了第一种，负结果：新旋钮 `visual.reach_continuity_weight`（默认 0），`s2_continuity005_20261001`（0.05）2/20 vs 6/20（不一致 5 : 1，p 0.22），伸手停滞 6 → 12、倾斜 3 → 6。晃动本身：align + descend 段肘参考 sd 15.8° vs 15.1°、肘腕相关 −0.86 vs −0.91，几乎没变；逐 tick 步长 sd 0.68° vs 0.85°（−20%），换向率 7% vs 10%。所以那 15° 是下落过程中肘和腕缓慢地此消彼长（零空间漂移），不是逐 tick 抖动；阻尼只压了本来就小的逐 tick 部分，还拖慢了纠偏。第二种也做了，同样负结果：新旋钮 `visual.reach_arm_posture_weight`（默认 0 = 全身 0.04），`s2_armposture02_20261001`（14 个手臂关节 0.2）**确实把晃动压了一半**——肘参考 sd 15.1° → 8.2°，肘腕相关 −0.91 → −0.85——但成功 3/20 vs 6/20（不一致 6 : 3，p 0.51），伸手停滞 6 → 8、多了 4 局 `evaluation_tilt`。把零空间钉住也拖了伸手的跟随。两个旋钮都不设默认；晃动目前是"看着不好、对成功率无害"的状态）
    - [x] 2.4.10.21 送达差量化 + 腰上限 0.35 试验（`box_room_grasp_offline_waist035_20260924`）：负结果——腰送达按比例约 2/3、不是上限，命令 20° 实测 13°，试提 6→1；基线仍是 align（2026-09-24）
    - [x] 2.4.10.22 送达差前馈（`box_room_grasp_offline_ff_20260924`）：负结果——命令多前伸 2.5 实测更短 1.5，z 修正饱和局 3→5，试提 6→0；送达差是伸展饱和不是偏移，字段保留默认 0（2026-09-24）
    - [x] 2.4.10.23 站位收紧到 0.45–0.52（≥0.55 全部停滞；0.46–0.54 在 align 里 6/7 到试提）：守线 `guard_extra`/`stand_off` 重标（2026-10-01 子项全部做完：站位能收紧——`guard_rewalk_depth 0.50` 让 20/20 局 ≤ 0.50 m——但 20 局里成功没变（7 vs 6），失败从停滞换成了脱手；`guard_extra` 不能再小（.1），交替步长模型反而站得更远（.3），起点平移只在固定起点下有效（.4）。都不设默认）
      - [x] 2.4.10.23.1 守线散布来源已量（2026-09-24）：刹车后最前点再前进 0.166 ± 0.011 m（仓库 60 局 + walk 80 局，最大 0.197，所以仓库里 `guard_extra` 不能低于 0.20）；0.25 步态是大步 ~0.21 + 并步 ≤0.07 交替，停位按 ~0.21 m 量子化；`stride_mode floor` 在大步后把下一步多估 0.12–0.17 m。落脚位置逐次 sd 2.5–5 cm、范围 10–20 cm
      - [x] 2.4.10.23.2 守线重走：站位 depth > 0.50 时后退 ~0.4 m 再走一次（每次独立抽样，P(≤0.50)≈30/60，三次 ≈87%）；需在 guard 路径放开 `_stance_converged`/`retry_closer`，10 局离线 pilot（待用户定）（2026-10-01 实现并跑 20 局：`visual.guard_rewalk_depth` / `guard_rewalk_back`（默认 0 = 关），guard 下静止站位深于阈值就沿走入轴后退 `guard_rewalk_back`（按目标的实测深度）再走一次，最多 `stand_depth_retries` 次，每次走入各有一份 `walk_timeout`。`s2_rewalk050_20261001`：**站位 20/20 ≤ 0.50**（基线 11/20；9 局重走、2 局走了三次，全部收敛），但成功 7/20 vs 6/20（不一致 4 : 5，p 1）——失败从伸手停滞（6 → 4）换成了脱手（3 → 7）。站得近解决了"够不到"，没解决"夹不住"。不设默认）
      - [x] 2.4.10.23.3 守线步长模型改交替式（大步后预测 0.08、并步后 0.23）：只让并步落在 ≥0.43 m 的局再多走一步，预计收益小；walk 场景 30 局验证不越线后再上仓库（待用户定）（2026-10-01：`GuardBrake(stride_mode="alternate")`，专家里是 `visual.guard_stride_mode`，`diag_brake --stride-mode alternate`。walk 场景 30 局（全程 0.25、线 0.9 m）：0 越线、0 摔，身体最前点最近离线 8.2 cm；但骨盆停在线前 37.9 ± 4.8 cm，比 floor 模型的 34.3 ± 3.4（09-23，50 局）**更远** 3.6 cm——并步后预测 0.23 比 floor 的 0.14 保守，刹得更早，抵掉了大步后多走的那一步。`outputs/diagnostics/brake_guard025_alternate_20261001`。仓库 20 局 `s2_altstride_20261001` 同样站得更远：站位中位 0.522 m（基线 0.4855），≤ 0.50 只有 6/20（基线 11/20），成功 2/20 vs 6/20（不一致 6 : 2，p 0.29），停滞 10。负结果，floor 保持默认）
      - [x] 2.4.10.23.5 0.25 守线站位（0.39–0.62）箱沿是否在可抓范围（用户 2026-09-24 问）：60 局逐 tick capture_error 最好时刻，两手都到抓线附近：跨壁 0–5 cm、沿上 0–5 cm、沿壁在 ±9 cm 允许滑动内；≤0.52 有 17/37 局真合爪。卡的不是够不够到，是最后 2–5 cm：跨壁要 ±1.8 cm（壁厚 1 cm）、要下过沿；≥0.53 沿壁向身体滑 11–19 cm 超出允许段，伸展饱和不再送达。结论：站位散布决定的是最后几厘米能不能送到，不是几何可达（2026-09-24）
      - [x] 2.4.10.23.4 `start_advance` 相位试探 −0.25 → −0.15 / −0.35 各 10 局离线：看 depth 分布的双峰（刹车 t 4.4 vs 5.6 s）是否随起点平移（待用户定）（2026-10-01，各 20 局，−0.15 另换 seed 7100 复制一次：**双峰随起点平移**——起点离平台 0.884 m（基线）时早刹（t < 4.8 s）的局站得深（中位 0.565–0.575 m）、晚刹的浅（0.465）；0.785 m（−0.15）时早刹变成多数（12–13/20）而且是浅的（0.453–0.487）；0.984 m（−0.35）全部晚刹（0.481）。是哪一步落脚触发刹车决定了站位，起点平移就换了一步。成功：−0.15 第一组 15/20 vs 6/20（不一致 10 : 1，p 0.012），复制组 6/20 vs 4/20（6 : 4，p 0.75），合并 21/40 vs 10/40（p 0.027）；−0.35 7/20 vs 6/20。站位 ≤ 0.50 合计 27/40 vs 23/40。**不改默认**：效应来自固定起点和步长的相位锁定，四边随机起点下不存在，复制组也弱；基线 `start_advance` 仍是 −0.25）
    - [x] 2.4.10.24 跨壁合爪门放宽到夹爪实际开口 `visual.capture_across` 0.035（`box_room_grasp_offline_across035_20260924`，align 基线 +1 旋钮，10 局）：0/10，但合爪 7/10（align 6），停滞 3；合爪时跨壁误差 ±0.7–3.2、竖向 ±1.6 cm；试提起点重放：四片指垫都贴壁两面（μ 1.2、无穿透）、爪口 1.05 cm = 壁厚，箱子 0 上升。根因是夹持力：仿真力律 F = kp 1.5 × gear 250 × (q_des − q)，expert 合爪设定点 −0.023 只比 1 cm 壁停住的 −0.0177 深 5 mm → 每指 2.0 N，四垫摩擦上限 9.5 N < 箱子 11.8 N；篮子横杆 1.6 cm 厚 → 每指 3 N、上限 14 N > 6 N，所以篮子能提。deploy 只按 Dex3 弧度限位裁剪（−1.05/−0.724），设定点可以推到 −0.06（每指 15.9 N，< 20 N 电机上限）。previews/ 已出（2026-09-24）
    - [x] 2.4.10.25 合爪设定点 `visual.close_target` −0.06（`box_room_grasp_offline_close06_20260924`，across035 基线 +1 旋钮，10 局）：0/10，但**箱子第一次离地：7/10 提起 11–16 cm**（此前 0）；5 局 `basket_tilt` 15.1–15.7°（门 15），2 局 `lost_attachment`（10–11° 时单手滑脱），3 局伸手停滞（含 1 局单手把箱子拎起 40°）。倾斜全是绕两手连线的俯仰、侧倾 ≈0；时间序列里箱子俯仰在 ±10° 来回摆、不跟手掌（掌俯仰漂 10–25°），抬升本身是 1 s 内 0→10 cm 的一跳再回落。机理：指垫捏 1 cm 壁顶是一个铰链（摩擦力矩 ≈0.4 N·m/手，重力回复力矩 15° 时 0.5 N·m），抬得急就摆成单摆，摆到 15° 触门或单手滑脱（2026-09-24）
      - [x] 2.4.10.25.1 三个候选各 10 局（场景 `warehouse_textured`，close06 基线 +1 旋钮，2026-09-24），全部 0/10：`max_tilt_deg` 30 → 合爪 2、提起 2、都在 16–18° 单手滑脱（门不响了，摆动改为甩掉握持）；`lift_seconds` 8 → 合爪 4、提起 4，到 5 cm 用 3.5–3.9 s（此前 2.0–2.4）但俯仰仍到 15–17°、1 局反向 −14°，摆动不只是起吊冲击，还有掌姿态漂移经指垫传给箱子；`box_grasp_below_rim` 0.03 → 合爪 3、提起 2，无增益。都不优于对照。伸手停滞主因：左手跨壁 4–5 cm 超 3.5 门、或停在沿上 3–4 cm（too_high）
        - [x] 2.4.10.25.2 场景对照 `box_room_grasp_close06tex_20260924`：合爪 5、提起 5（4 触门、1 滑脱）、停滞 5，介于 close06 的 7 与三候选的 2–4 之间。编译后模型逐项一致（option/质量/关节/执行器/44 碰撞体/qpos0），场景不是变量；这条线 10 局的伸手合爪率本身在 2–7/10 之间摆，站位中位 0.46 → 0.50–0.51 也解释一部分。结论：候选比较要 20 局起，或先把伸手段的停滞（跨壁/沿上）压下去再比 lift 段（2026-09-24）
    - [x] 2.4.10.27 抬升掌俯仰跟随箱俯仰（用户 2026-09-24 定，20 局；子项全部完成，增益 0.6 进入基线，2026-09-24）：`visual.lift_pitch_follow_gain` 1.5 / `lift_pitch_follow_cap_deg` 25，lift/stand/hold 段把命令掌姿态绕两手连线轴反向转 gain × 实测箱俯仰（相对合爪时刻，低通 0.2 s）；`box_room_grasp_follow15_20260924`，场景 warehouse_textured
      - [x] 2.4.10.27.1 增益 1.5 跑了 4 局就停：跟随生效，但命令滞后（低通 0.2 s + 送达延迟）下过校正——箱子 1.5 s 内从 −6° 甩到 +10°，2 局触 15° 门、1 局 5° 时滑脱。篮子横杆是刚夹住随手 1:1 转，1.5 合适；箱沿在爪里能滑，回路增益要 <1（2026-09-24）
      - [x] 2.4.10.27.2 增益 0.6、上限 12°（`box_room_grasp_follow06_20260924`，20 局）：0/20；合爪 15/20，8 局触 15° 门、7 局单手爪口 <3 mm 脱开（附着误差仅 1–5 cm）；全部在 stand 段 3.0–4.4 s、躯干已直到 2–12° 时发生。stand 目标把手从骨盆前 ~55 cm 拉回 15–25 cm（回 ready 位置），箱子近壁离骨盆只剩 6–9 cm，被顶进躯干才倾斜/滑出。抬升段本身箱子只倾 2–3°（2026-09-24）
      - [-] 2.4.10.27.3 stand_seconds 12 —— 放慢起身不解决撞箱，跑了 0 局就撤（2026-09-24）
      - [x] 2.4.10.27.4 `visual.stand_forward_min` 0.35 逐 tick（`box_room_grasp_follow06_fwd35_20260924`，20 局，2026-09-24）：0/20，但 **11 局合爪提起里 7 局走完 lift+stand 进入 hold、1 局走完 hold 进入后退**（此前 0 局过 stand）；触门 1、脱开 3、hold 段 6 局 `unstable_hold`、后退 1 局 25° 掉落、伸手停滞 9。hold 判据是箱子世界位置 2 s 内半径 ≤2.5 cm，而 hold 里整机向后漂 2.5–8 cm、侧漂到 5 cm（SONIC IDLE 带载漂移），箱子相对骨盆只动 ≤1.3 cm 前向 / 2–5 cm 侧向、附着 ≤2 cm、倾斜 ≤11°——箱子在手里是稳的，是身体在漂。第一版（合爪时刻一次性抬目标）9 局无效已归档 `_v1_world_fixed`
    - [x] 2.4.10.28 成功定义（用户 2026-09-24）：抬起 + 上半身直立 + 后退两步（`retreat_distance` 0.3）箱子不掉。hold 判据 `visual.hold_frame: pelvis` + 短 hold；6/20 成功（2026-09-24）
    - [x] 2.4.10.29 下一层：伸手停滞 7/20 与后退剪切 2/20（2026-10-01：子项做完。停滞的来源查到了（2.4.10.19：跨壁命令只送达约一半，骨盆横漂 10–15 mm）；后退放慢是负结果。后退时让伸手控制器继续夹持没做：要让 CoordinatedReach 在规划器后退时逐 tick 按实测手位重解上身，是结构改动，main 上的默认（四边、`box_platform`）已经不后退）。**暂定基线 = contact 版本**（用户 2026-09-24 定）：`configs/experiments/box_room_grasp_baseline_20260924.yaml`（= `box_room_grasp_contact_20260924`，6/20），之后每个实验复制它改一个旋钮
      - [x] 2.4.10.29.1 接触触发合爪 `close_trigger: contact`（`box_room_grasp_contact_20260924`，20 局，2026-09-24）：**6/20**（与 holdshort 持平）；合爪 13（此前 12）、提起 11、进后退 8、后退掉 2、脱开 5（此前 1）、停滞 7。跨壁类停滞消失了，剩下 7 局全是**手停在沿上方 1–3.7 cm 没下到沿下**（送达偏高，接触也没发生）；脱开变多是合爪精度放松的代价（合在壁顶几毫米处）。视频已带 ghost 夹爪
      - [x] 2.4.10.29.3 下落到接触为止 `contact_overdrive` 0.03 + `contact_along` 0.13（`box_room_grasp_contactdeep_20260924`，20 局，2026-09-24）：**负结果 2/20**（contact 基线 6/20）；停滞 14。轨迹：手确实下到抓线下 1–7 cm，但下落中跨壁误差从 ≤3 cm 跳到 4–8 cm（指垫沿壁面外侧滑下被顶开，沿下横向反馈又被 `lateral_hold_below_rim` 冻结，无法回正），爪口不再套壁。多下 3 cm 放大了偏转。基线回到 contact（overdrive 0）。剩余 7 局沿上停滞要换思路：小步下探 + 每步检查跨壁，或 align→descend 失败后重做一次
        - [x] 2.4.10.29.3a 首版跑 3 局停：手确实下到沿下 2–5 cm，但 `inside` 还卡在篮子时代的沿壁窗口 ±9 cm（手沿壁向身体滑 7–10 cm）和跨壁 4.5 边界；加 `contact_along` 0.13（壁长 0.32 − 指垫）。另见 round 1：指垫落在壁顶时继续下压把箱子横推 6–8 cm、抬 1.3 cm（`_stopped3`，2026-09-24）
      - [x] 2.4.10.29.2 后退剪切：`retreat_speed` 0.35→0.25 或后退时保留全身伸手控制器夹持（待用户定）（2026-10-01 只试了前者，负结果：`s2_retreat025_20261001` 2/20 vs 基线复跑 6/20（不一致 1 : 5，p 0.22）；进入后退 5 局里 2 掉 1 超时，基线 8 局里 2 掉——放慢没有减少后退段掉箱，还多了超时。后者（后退时保留伸手控制器）没做：要把后退改成由 CoordinatedReach 带着走，是结构改动；默认后退距离在 main 上已是 0，这条只影响 09-24 的成功定义）
    - [x] 2.4.10.30 ghost 夹爪规则（用户 2026-09-24 定，所有场景固定）：expert 把命令的抓取目标按 seq 记到 `trace/<label>_ghost.npz`，`render_trace_videos.py` 只在第三人称视角画半透明夹爪，ego/腕相机不画；Isaac 批渲同样转发。`docs/rendering.md` 有说明（2026-09-24）
      - [x] 2.4.10.28.1 `hold_frame pelvis` 单独（`_stopped4`，4 局）：round 2 到 hold 保持 5 s、倾斜 2°、附着 1.5 cm，但骨盆系下 2 s 窗内箱子仍漂 2.7 cm（IDLE 带载晃动）超 2.5 cm 半径，窗口反复重置，第 5 s 右指垫蠕变漏箱；4 局 3 脱开 1 unstable_hold。hold 拖得越久越掉（2026-09-24）
      - [x] 2.4.10.28.2 `hold_seconds` 1.0 + `hold_position_tolerance` 0.05 骨盆系（`box_room_grasp_holdshort_20260924`，20 局，2026-09-24）：**6/20 成功**（warehouse 线首批成功：抬起 13–21 cm、直立、后退 0.31–0.37 m、箱子在手、末态倾斜 0.9–9.6°）。漏斗：合爪提起 10 → 进后退 8 → 完成 6；后退段掉 2（倒走 0.35 起步 2 s 退 40 cm，两手前后错开 4–6 cm 剪切箱子，一侧爪口缩到 0 后 45° 翻落）；触门 2、脱开 1、空爪 1；伸手停滞 8（最大单项）。成功站位 0.39–0.46
    - [x] 2.4.10.16 远站位（≥0.54）下落时掌先落壁顶：收站位散布或补下落段掌俯仰（2026-10-01：走了前一条——守线重走（2.4.10.23.2）把 ≥ 0.50 的站位清零，`s2_rewalk050` 0/20 局在 0.50 以外、伸手停滞 6 → 4；补掌俯仰没做）
        - [x] 2.4.10.26 纸箱贴图丢失（用户 2026-09-24 问）：不是渲染 bug。`warehouse` 场景（`scene_reverse_camera`）09-20 从 `outputs/scene/logistics_mujoco` 做的，早于 09-22 的贴图提交 `0bf8eed`（纸箱拆 9 块材质 + 6 张实物照片）；所以纸箱只有一个 `nor_cal_box.obj` + 纯色 rgba。修：`prepare_logistics_scene.py` 从资产包重做运行时场景 `outputs/scene/warehouse_textured`（scenes.yaml 新名 `warehouse_textured`；机器人/工作面/物体/nq 全同，ngeom 121→129），reverse_room 与 head_camera 各渲一帧已见印字。之后的 warehouse 实验改用这个名字（2026-09-24）
- [-] 2.4.10.8 纸箱显示网格减面（46 万 → ~5k 面）或关阴影，让实时渲染/VLA 评测也跑满速；先问协作者 `assets/crates/nor_cal_box/visual/`（放弃 2026-09-24：被 4.7.1–4.7.6 的异步渲染替代；减面程度没有统一标准，治标不治本，用户定）
    - [x] 2.4.10.9 `diag_brake` 补 reanchor（非 0 起始朝向场景一起步就转 ~90°）（2026-10-01：`diag_brake --reanchor auto|on|off`，默认 auto——复位后起始偏航离 0 超过 5° 就先做一次 streamed 切换重锚 planner 朝向、再 IDLE 3 s、再读起始朝向。warehouse 场景（起始朝向 90°）实测：off 2/2 局 40 s 超时、沿朝向前进 −4…−6 cm；auto 3/3 走到 0.3 m 标记并刹住（停在 0.36–0.41 m，横向 ≤ 3 cm，偏航变化 ≤ 1.1°）。`outputs/diagnostics/reanchor_{off,auto}_20261001`。顺带：每局结果多记 `brake_phase`、`frontmost_*_past_marker`、`reanchored`，新增 `--phase-brake` 和 `--stride-mode alternate`）
    - [x] 2.4.10.12 ready 原地抬升：`visual.ready_pose lifted` + `elbow_min_deg` 旋钮（`ed5819d`，离线插值验证写成测试）；pilot `box_room_grasp_offline_lifted_20260924` 0/10，但抬手接触 0/10、箱子位移 0/10，4 局到试提（2026-09-24）
    - [x] 2.4.10.13 “腕相机始终看到箱子”只在下落段要求；抬手/平移段放开掌心姿态；从 assisted_guidance 的逐流可见性统计各阶段哪路相机看得到（用户提出 2026-09-24，待定）（2026-10-01：先统计——基线里两只腕相机在 reach / align / descend / lift / stand / hold 全程 100% 看得到箱子（≥ 4 个角点），ego 在 descend 只有 64%，walk 段腕相机 0%。再放开：新旋钮 `visual.reach_rotation_weight_transit`（默认 −1 = 沿用 `reach_rotation_weight` 0.6），只在手去预抓点的 reach 段用；0.2 跑两组：seed 7000 11/20 vs 6/20（不一致 7 : 2），seed 7100 8/20 vs 4/20（6 : 2），**合并 19/40 vs 10/40，不一致 13 : 4，p 0.049**，两组同向，停滞 6 → 4、11 → 7；腕相机在 reach 段仍 100% 看得到。这是这批里唯一复制过的正结果。用户 2026-10-01 定设成专家默认：dataclass 默认 −1 → 0.2；今天的 `s2_*` 对照臂钉回 −1（它们是按 −1 跑的），09-24 基线文件不设这个字段、此后按 0.2 跑（文件头已注明）。crate 篮子线和 `box_platform` 没在 0.2 下测过）
      - [ ] 2.4.10.13.1 `reach_rotation_weight_transit` 默认 0.2 在 crate 蓝篮子线（`crate_visual_low_reach`）和 `box_platform_default_20260927` 上各跑 20 局对照 −1，确认不退化（同一个 CoordinatedReach，只在单边 warehouse 纸箱上测过）
    - [x] 2.4.10.10 提交本会话代码（见 STATE 的文件清单），`make check` 已过（2026-09-23）
    - [x] 2.4.10.11 文档入口：删 `docs/STATE.md`，README/AGENTS 指向根目录 STATE/TODO/DECISIONS；`docs/handoff/warehouse-render-cost-20260923.md`；台账 crate 09-23、rendering 离线渲染、blender_scenes 已知问题（2026-09-23）
- [-] 2.5 待验证：摔倒 6/20 为何比合成桌子多（假设：reanchor/IDLE 期间贴着托盘垛）（放弃 2026-10-01：现行专家不再复现——09-24 基线 0/20 摔、10-01 复跑 `s2_base` 0/20、`s2_adv015` 0/20；6/20 是 09-22 旧 carry 专家的数，那条走法已不用）
- [x] 2.6 待验证：哪项部署变更把 v1.1 速度图从台阶变成连续（2026-10-01 查明：**不是部署，是 agent 侧的 commander**。同一个栈（5cab2fb、walk 场景）上 `diag_shuffle` 0.20–0.35 各 2 次：今天的代码连续（0.20 → 0.04–0.09 m/s，0.26 → 0.13–0.14，0.35 → 0.23–0.30）；换旧二进制 4add301、退回 09-11 之前的复位座高、换回 09-11 的桌面场景（每腿先转向空地）都还是连续；`--no-encoder` 的栈 SLOW_WALK 完全不动（0.000）。只有把 agent 侧换回 09-11 的代码（`aae3d72` 的 diag_shuffle + Primitives + commander，连同一个栈）才复现台阶：0.26 → 0.008–0.024，0.30 → 0.007–0.050，0.35 → 0.060–0.070，步频照旧 ~1.4/s、每步 0.1–7 cm，就是原地踏步。按代码对比，原因是 `dd36fd8`（2026-09-12）：之前每个非冻结上身的规划器帧都带 `vr_3pt_*` 字段，deploy 按字段存在与否切到 VR-3PT 编码器模式，手腕被拉向默认目标；之后普通帧两个字段都不带。没有用实验单独隔离这一行。`outputs/diagnostics/speedmap_v11_*_20261001`）
- [x] 2.7 `vla_rounds.py` 起点放置写死偏航 0 / 物体 π/2，房间里会放错（2026-09-27 核对属实并修复，另写死 `blue_crate`，见 4.7.6.1.2.2–3）
- [-] 2.8 `stand_lateral_gate: turn` 作为下一步 —— 它针对的横向偏置是合成场景现象，房间里没有
- [x] 2.9 warehouse 采 10 集并导出视频（用户 2026-09-25）：`box_room_grasp_10ep_20260925` 30 局保留 11；预览 `outputs/videos/box_room_grasp_10ep_20260925/` 30 个 322–493 KB + zip 11.8 MB（2026-09-25）
- [-] 2.10 `stand_off` 0.5 → 0.44 的 20 局对照（同 seed 7000，只改这一个）（放弃 2026-10-01：查调用方，`approach_stop: guard` 时专家根本不读 `stand_off`——`VisualApproach` 只在非 guard 分支用它，`ApproachPreparation` 在 guard 下是 None；guard 的站位由线 + `guard_extra` + 步态量子决定。按深度收紧站位改走 2.4.10.23.2–.4）：1013 局按停位深度分组 < 0.44 m 0.41、0.44–0.46 0.39、0.46–0.48 0.28、0.48–0.52 0.20、≥ 0.52 0.03；风险是躯干顶箱（box-stand-rams-torso）。待用户定（改专家参数）；分析脚本 `outputs/box_room_isaac_20260925/stand_depth_yield.py`

## 3. walk 场景：SONIC v1.1 行走能力（`docs/walk.md` 探针表）

- [x] 3.1 场景名 bottle / crate / warehouse / walk（`configs/scenes.yaml`，`50d3ec6`、`e8a0019`）（2026-09-23）
- [x] 3.2 小步小角度探针 `diag_micro`：转向可做准，平移量子化，`walk_to` 小目标不可信（2026-09-23）
- [x] 3.3 速度渐变视频 + 速度图（SLOW_WALK / mode 18 / mode 20）（2026-09-23）
- [x] 3.4 走到标记刹车 `diag_brake`；守线刹车全程 0.25 50/50 不越线、0.55→0.25 50/50（2026-09-23）
- [x] 3.5 planner 路点补丁 `patches/sonic_waypoint_target.patch`：停得齐但本体只走 35–55%（2026-09-23）
- [-] 3.6 停后收脚（B 切模式 / C 延迟刹车 / A 朝向微调）—— B、C 实测无效，A 用户否决
- [-] 3.7 守线收紧（extra 0.185 + recent2）—— 最近离线 1.7 cm、离散翻倍，不采纳
- [x] 3.8 刹车后伸手越线测量：手比骨盆远 65–69 cm（2026-09-23）
- [x] 3.9 虚拟纸箱伸手：staged + 1 cm 抓深 + orientation_gain 1.0 = 4/5（`box_reach_fb_20260923`）（2026-09-23）
- [-] 3.10 descend_speed / descend_z_guard 止下冲 —— 4/5、3/5，没能稳定避免手掌压箱沿，默认关
- [x] 3.12 守线 `guard_extra` 0.20 → 0.17 重标（全程 0.25，线 0.9 m，30 局）：骨盆 −34.3 ± 3.5 → −31.0 ± 4.2 cm，最前点最近 −7.4 → −5.7，0 越线 0 摔；只赚均值 3 cm，散布不变（`outputs/diagnostics/brake_guard025_extra017_20260924`，2026-09-24）
- [x] 3.13 waypoint 结尾（私有 deploy，SLOW_WALK 0.25，目标 0.9/1.2/1.5 m 各 10 局、不重发）：本体走目标的 0.64/0.64/0.71，散布 3.7/5.0/6.4 cm 随目标放大；相对 0.9 m 线骨盆 −32.4/−13.4/+16.4，1.2 m 时最前点已越线 +5.0 ± 2.7（10/10）；脚错开 10 cm，不是 09-23 的 3.9。过量下发不能替代守线（`outputs/diagnostics/waypoint_end_*_20260924`，2026-09-24）
- [x] 3.14 0.55 稳态走入 + 按相位提前刹车（用户 2026-09-24 提出）：标记 1.30–1.55 每 5 cm 各 7 局 + 09-23 的 10 局 = 52 局，0 摔。带距 21.5 ± 4.8 cm，随落地后相位周期变化（落地后 0.3 周期最短 15，摆动中段最长 27–29），相位一次谐波模型留一法 sd 3.6、最大误差 11 cm；同相位内仍有 5–13 cm 散布，脚位/速度特征不再降。只在落地瞬间刹车则 20.6 ± 1.5，但落地点按 ~27 cm 步长量子化，回到守线的问题。助跑 ≥1.0 m 才到 0.50 ± 0.02 稳态；托盘垛是 0–0.371 m 实心块，越线即碰。结论：相位刹车把停位均值从 −34 拉到可设的 −6…−10 cm、sd 3.5，代价是约 5% 局越线 ≤4 cm（`outputs/diagnostics/brake_055_mk*_20260924`，2026-09-24）
    - [x] 3.14.1 若接受少量越线：实现相位刹车（FootWatch 已有落地事件；带距 = 21.5 + 谐波(相位)），walk 场景 30 局对线验证，再上仓库离线 10 局（待用户定）（2026-10-01：实现为 `visual.guard_brake phase` + `guard_phase_band`（专家）和 `diag_brake --phase-brake`；带距按"刹车时骨盆 → 最终身体最前点"拟合（`scripts/experiments/fit_phase_band_20261001.py`），余量取留一法最差残差 + 3 cm，不接受越线。**稳态 0.55（助跑 1.4 m）有效**：40 局标定，残差 sd 2.4、最差 6.4 cm；30 局验证 0 越线 0 摔，最前点停在线前 10.3 ± 3.3 cm（守线 0.25 是 15.5 ± 2.6）。**warehouse 用不上**：这条边前只有 1.285 m 空地，起点最远 1.03 m，助跑 0.6 m 不到稳态；按同样助跑重新标定（残差 sd 5.6、最差 8.3 cm）后 20 局验证最前点 −12.4 ± 8.8 cm、1/20 越线 3.3 cm——均值和守线一样，散布大三倍；仓库 10 局 `s2_phasebrake_20261001` 成功 1/10，站位 0.37–0.58 m 更散，走入段身体离线最近 0.8 cm。不进默认；`docs/walk.md` 探针表有两行）
- [x] 3.15 warehouse 10 局 Isaac 重渲（`box_room_grasp_offline_close06_20260924`，2026-09-24）：房间用 Blender 直出的 USDC，`scripts/build_isaac_warehouse_scene.py` 生成 `assets/isaac/scene/warehouse_textured/`（scene.usda 引用 /root、停用 env_light 和静态纸箱；可动纸箱包装层也引用 /root 并停用其余 237 子 prim，否则丢材质绑定渲成素灰；sidecar 偏移 (0, 0.39, −0.371)），`scripts/render_isaac_warehouse.sh` 每局 mujoco_render（新加 `--azimuth 225 --distance 2.2`，默认机位在南隔断外只见墙）→ isaac_render → 并排。10/10 成功，~1 min/局，并排预览 `previews_isaac/`（130–175 KB），全分辨率 `isaac/<局>/side_by_side.mp4`。房间贴图、纸箱印字、机器人含头在 Isaac 里完整，机位逐帧一致；Isaac 略亮
    - [x] 3.15.1 Isaac 里 G1 发白（用户 2026-09-24 问）：颜色绑定没错（MJCF 0.7/0.2/0.3/0.25 四档都取到），是曝光。房间由 5 块 24828 强度天花板 RectLight + 自发光灯板照亮，dome/平行光改强度像素不变；默认自动曝光 + ACES 把 0.2 灰抬到 108、0.7 灰推到 212（头/躯干 0.51，MuJoCo 0.23）。`isaac_render` 新增 `--rtx-setting 路径=值`；自动曝光关 + `filmIso` 30 后躯干 139（MuJoCo 136）、头/躯干 0.36，已设为 `render_isaac_warehouse.sh` 默认并重渲 10 局。Clamp 全饱和、Linear 更平，不用（2026-09-24）
- [-] 3.11 VLA 视角：伸手阶段 latent 可学性复查（探针 B 在 box_room 数据上 reach R²≈0.16）；新专家跑通后重跑 `probe_visual_grounding_20260923.py`（放弃重跑 2026-10-01：探针是"策略能不能从画面学到伸手"的代理，这个问题已经被直接回答——新专家 200 集训出的 checkpoint-5000 在 Isaac 镜像闭环 81/100（`wc_hold100_20260927`），失败里伸手阶段只占 8/100（"到了桌边没合爪"），合爪不提起 11/100；探针要的 per-round 相机 mp4 在离线录制的运行里也不存在）

## 4. 三场景采集按 README 从零复现（协作者报 bottle 开局摔倒、文档没跟代码；用户 2026-09-24 定）

- [x] 4.1 提交本会话 warehouse 代码（`make check`，按文件名 add，`--no-verify`），给复现一个干净基点（`8ca2fc2`，make check 通过，2026-09-24）
- [x] 4.2 无上下文 agent（headless `claude -p`，关钩子 + auto-memory）按 README 复现 bottle → crate → warehouse 采集，各 10 局 `record: true`，run 名前缀 `docrepro_20260924_`；只报告不修，每步写信息来源；bottle 加 `convert_data`；报告 `outputs/docrepro_20260924/report.md`（2026-09-24 13:33–14:27 跑完，exit 0，无残留栈）：bottle 9/10 通过、crate 17/20 通过、warehouse 0/20（RTF 0.34–0.35）；三场景 50 局开局全部站稳，**开局摔倒未复现**；报告 `outputs/docrepro_20260924/report.md`
- [x] 4.3 读报告：文档问题直接改（grep 全部出现处）；代码 bug 先定位根因报给用户，点头再修，修后只重跑该场景 10 局；环境问题写进对应页常见问题表（2026-09-24）
  - [x] 4.3.1 文档：产物表 `videos_NN/`/`state_trace_NN/` 与实测平铺目录不符（`single_arm.md:171`、`bimanual.md:161`）（改为 `collect/videos/`、`collect/state_trace/` 与顶层 `videos/`，2026-09-24）
  - [x] 4.3.2 文档：`bimanual.md:63` "batch_rounds 真正决定跑几局"太绝对（实测 batch_rounds 10 → 20 局，受 successes/max_batches 约束）（表格改写 + 正好跑 N 局的设法；顺带更正 max_batches 行“每批重启一次栈”→栈跨批不重启，2026-09-24）
  - [x] 4.3.3 文档：双臂开局站稳判据 `placement.initial_stance_ready` 未写；`*_ghost.npz` 只在进入伸手的局写（warehouse 13/20 = 20 − 7 局 approach_unsupported_reach）未写（bimanual.md §7 trace 行、§8 排错行，2026-09-24）
  - [x] 4.3.4 文档：warehouse 入口——`scenes.yaml` 的 `example` 指向 09-22 旧配置（实时渲染 → RTF 0.35 → 0/20），现行基线 `box_room_grasp_baseline_20260924`（offline + record false）文档里找不到；录制要实时渲染，被 2.4.10.8 网格减面卡住（待用户定文档怎么写）——用户定：离线渲染 + 6/20 基线。文档已改（blender_scenes §6 新小节、bimanual 指路、README、scenes.yaml example、rendering.md）；离线 + record 实测只存 priv_seq；复跑 20 局验证中 `docrepro_20260924_warehouse_offline`；离线复跑 `docrepro_20260924_warehouse_offline` 20 局 6/20（= contact 基线 6/20），2026-09-24
  - [x] 4.3.5 代码（待用户点头）：`train.holdout ≥ collect.successes` 时 convert 在 stats_train 抛 `No objects to concatenate`，应在读配置时报清楚（`run_context.require_train_split`，单臂采集开始与 convert 开始时检查，dry-run 即报；bimanual 采集不查：26 份试跑配置 10/10 只采不转；`plan_holdout` 改 ≥，2026-09-24）
  - [x] 4.3.6 代码（次要）：单臂续批日志分母不更新（`round 2/1`）（不是分母 bug：原格式 `round 2/1 ok` 意为第 2 局/目标 1 成功，被误读；改成 `round 2, ok 0/1`，2026-09-24）
  - [-] 4.3.7 bottle 开局摔倒：10 局未复现，本机近 6 天 manifest 无开局摔倒记录；需协作者的命令/配置/时间用户 2026-09-24：复现没摔倒就不追
  - [x] 4.3.8 代码 bug（待用户点头）：`0becdd0`（09-21）起录像写平铺 `collect/videos/`，`third_person_source` 仍找 `videos_NN/` → `train_dataset/third_person/` 静默为空；修法：回落到 `videos/`，crate 按 label、bottle 按 `execution.started_wall` 匹配（用户 2026-09-24 点头；bottle 按 `execution.metrics.video_start_wall` 精确对到 `gym_<stamp>.mp4`；实测 bottle 12/12、crate 19/19，2026-09-24）
  - [x] 4.3.9 warehouse 离线录不出 LeRobot：要么事后从 trace 渲相机图写进数据集（新功能），要么纸箱网格减面改回 live（2.4.10.8）——待用户定——用户定 A（2026-09-24）：exporter 离线存占位帧（`sim/dataset_cameras.py`），`scripts/render_dataset_cameras.sh` 按 priv_seq 从 trace 渲三路相机覆盖，convert 拒绝未渲的离线运行；对照实时 crate 局 1000/1000 帧同行差值中位数 4.6–6.2；端到端 `docrepro_20260924_warehouse_dataset`（8 局 2 成功）：采集 → 渲 8 局（每局 6–10 s）→ convert 出 train/val，画面核对为真；**审计挂 1 项**：见 4.3.10；审计修后 train/val 均 32/32 通过（2026-09-24）
  - [x] 4.3.10 审计 `wrist within 30 cm of the object at close` 对纸箱不成立：量的是两腕中点到物体**原点**，`nor_cal_box` 原点在箱底（centre z 0.165），合爪时中点距原点 0.45、距中心 0.29、距箱面 0.11 m；改成量到 `scene_meta` 的 box 表面（待用户定）（用户 2026-09-24 定；`check_dataset.box_surface_distance`，有 `--run-dir` 且 scene_meta 有 box 时量到箱面，crate/bottle 原点即中心、不变严；复转 32/32，2026-09-24）
- [x] 4.4 warehouse 单独一页 runbook `docs/warehouse.md`（准备含建场景、两个补丁；README 功能表独立一行；blender_scenes / bimanual / rendering / docs 索引指过去；`SONIC_DEPLOY_ROOT` 那行注明仅原作者机器）（用户 2026-09-24 提出，2026-09-24）
- [x] 4.5 可达性"协作者 WIP"（分片 / 续跑 ghost-reach probe、五维联合采样 + 测试 + 09-20 handoff）提交推送：主线 `reach_envelope.sh` 依赖它（`76c5196`，2026-09-24）；纸箱源网格 `assets/crates/nor_cal_box/visual/` 不提交
- [~] 4.6 全新 clone 上按 README → setup → single_arm / bimanual / warehouse 从零验证（可再用无上下文 agent，这次三份 runbook 都只在装好环境的本机跑过）
  - [x] 4.6.1 2026-09-25 用户定：无上下文 agent（Opus 5.5，claude -p，hooks/记忆关）在沙盒 `/workspace/fresh_repro_20260925/` 从 SSH clone origin/main（b3b645f）起装第 0–4 步 + bottle/crate/warehouse 各 10 局录制到 LeRobot；只记录不修，挡路时仓库外最小绕过（记 Dn，≤30 min/问题），开局失败/摔倒不绕只收证据；只用 GPU0，/workspace 余量 <8 GB 停；期间机器不起别的栈；报告与视频拷到 `outputs/fresh_repro_20260925/`，沙盒等用户看完再定删——2026-09-25 完成（53 min）：安装 9 min，deploy 指纹 = c1360628…、四标记齐、mujoco 3.11.0；bottle 7/10、crate 8/10、warehouse 3/10，**30 局零开局站稳失败、零摔倒**（crate/warehouse `initial_stance_ready` 10/10）；三个场景的转换都卡在 holdout 与第 5 步 stats；问题 P1–P20、偏离 D0–D6 见 `outputs/fresh_repro_20260925/report.md`；沙盒 2026-09-25 已删（用户定），launch.sh 与证据留在 `outputs/fresh_repro_20260925/`
  - [~] 4.6.2 修复复现报告的问题（用户 2026-09-25：先 P7、P8、P19；P1 仓库私有不修）；证据与原文位置见 `outputs/fresh_repro_20260925/report.md` 第三节
    - [x] 4.6.2.1 2026-09-25 P7：holdout 准入再按 `batch_rounds × max_batches` 查，起栈（run_stack，领目录前）与两条线 collect 入口都查，不录制的运行跳过；三份 runbook 的 holdout 行统一、warehouse 排错行改成真实报错
    - [x] 4.6.2.2 2026-09-25 P8：convert 的 stats 改用 .venv_data_collection 跑 vendored `src/gr00t`（三数据集 stats/relative_stats 与训练 venv 逐字节相同），沙盒 0–4 安装上 bottle/crate convert 全程通过；README/setup/single_arm/bimanual 改成"采集+转换 0–4，训练评测 +5"
    - [x] 4.6.2.3 2026-09-25 P19：README/setup 前置与第 3 步第 0 条写明 CUDA 13 运行时（cuda_runtime.h + libcudart.so.13，`cuda-cudart-dev-13-0`，不含驱动；干净机器未验证），并警告 cmake 报错让跑的 install_deps.sh 不要照做
      - [ ] 4.6.2.3.1 待验证：没有 nvcc 的机器上 cmake 能否配置（strace 显示有 nvcc 时 `find_package(CUDAToolkit)` 执行 `nvcc -v`；setup.md 第 3 步第 0 条已改成这个说法，2026-09-25 推送）
    - [x] 4.6.2.4 2026-09-25 P16 single_arm.md:11-12"没有第三行就拒跑"与新机器"不要这行"矛盾（setup 已打 hand 补丁，已过时）——single_arm.md 改为"前两行每终端；第三行只在原作者机器"
    - [x] 4.6.2.5 2026-09-25 P15 补丁说明残留：warehouse.md:10-12"两个补丁都要打"、bimanual.md:12-13"并应用 POSE 补丁"、patches/sonic_pose_ownership.md:18-21 手工 apply 与 :29"separate DDS-domain patch"——warehouse/bimanual 改为"第 3 步已打全"；两份 patches/*.md 的手工块标成历史，DDS 指向 local_base
    - [x] 4.6.2.6 2026-09-25 P11 bimanual.md:19 不带 preset 的 install_scene.sh 把单臂 outputs/scene/sim 覆盖成多物体桌——删掉该行并写明勿跑；沙盒核对：bottle_only 的 sim 下建出的 crate 场景与原来逐字节相同
    - [x] 4.6.2.7 2026-09-25 P3 setup.md:185-189 第二次 download_from_hf.py 覆盖只读 planner 报 PermissionError，&& 链 exit 1（文件其实齐）——第二次加 `--no-planner`（沙盒核对链 exit 0、三文件与原来相同），重跑前 chmod u+w 写进注释
    - [ ] 4.6.2.8 P13 crate 开局距离采样区间 = 验收区间（expert.py 写死 1.2–1.5），重置漂移几 cm 即 invalid_initial_distance；P12 这类局 visual_NN.log 无输出（用户 2026-09-25：暂缓，之后再看）
    - [ ] 4.6.2.9 warehouse 基线 box_room_grasp_baseline_20260924 的 holdout 10 对 6/20 基线不够（改录制后 convert 必拒；文档已注明，配置未改）（用户 2026-09-25：暂缓，之后再看）
    - [ ] 4.6.2.10 小项：P5 场景包 README 要 setup_sonic.sh（第 6 步）、P14 USD 校验要 Blender、P9 `python`→`python3` 与采集/转换契约名、P18 p03_i<实例>（实例 0 是 p03）、P20 单臂开局判据只在源码、P4 磁盘预算、P2 上游脚本 sudo、P17 耗时/大小数字、G8 single_arm 缺"正好 N 局"写法（用户 2026-09-25：暂缓，之后再看）
  - [x] 4.6.3 2026-09-25 用户：严格按 README 自己跑一次 bottle 采集，查"当成理所应当、没写进文档"的东西。origin/main **d505880**（含 henry.zhao 两个提交），沙盒 `/workspace/readme_bottle_20260925/`，`env -i` 连 CUDA_HOME/LD_LIBRARY_PATH/HF_HOME/.env 都不给，sudo 换桩；20 局（max_batches 2）**20/20 站稳、零摔倒**（stand drift 0.02–0.16 m，骨盆最低 0.71 m），保留 10/20；唯一 [stand] 失败是腰 0.04 rad > 0.03 边界拒绝；convert 被 holdout 10 挡（P7 本地已修未推送）。新发现：每次起栈 sim 自己先摔一次（`Robot has fallen 0.199 m`，第一局 reset 前，所有 sim.log 都有，文档没提）；MuJoCo 走 NVIDIA EGL、deploy 链 /usr/local/cuda 的 cudart 13、convert 用系统 ffmpeg——都是本机提供、origin 文档没写。报告 `outputs/readme_bottle_20260925/report.md`；沙盒 2026-09-25 已删（用户定），证据留在 outputs/
  - [x] 4.6.4 2026-09-25 用户：用 strace 再跑一轮，逐个文件核实有没有用到隔离目录外的文件。新沙盒 `/workspace/readme_bottle_strace_20260925/`，同一套命令，每步 `strace -f --seccomp-bpf -y -e trace=%file,%process,connect,bind`；分析脚本 `outputs/readme_bottle_strace_20260925/analyze_traces.py`。观测副作用：跟踪下 RTF 0.90→0.71–0.80（deploy 每个控制周期读 /sys/fs/cgroup/memory.*、sim 反复 openat("/")，每次被 ptrace 拦停），重置后摔倒反复出现——采集成败不算 README 结果——完成：写入系统目录 0；沙盒外只写 /tmp/cdds.LOG（上游写死）和 git-lfs 的 /tmp socket；读的非系统文件只有 ~/.ssh（clone 与 README 的 git lfs pull，ssh 按 passwd 找家目录）；系统层用到 CUDA 13（cmake 还执行了 nvcc，与本地 P19 修复的"不需要 nvcc"矛盾）、NVIDIA EGL、apt 旧索引；我们的代码扫描全机 /proc（cmdline/stat）；uv 会读上级目录与 ~/.config 的 uv.toml。跟踪下 20 局 19 局开局失败（16 局重置后摔倒）。报告 `outputs/readme_bottle_strace_20260925/report.md`；沙盒 2026-09-25 已删（用户定）
  - [~] 4.6.5 2026-09-27 用户：以非本仓库开发者的眼光复查全部文档（起因：warehouse.md 的 `run_isaac_mirror.sh ... --gpu 1` 没有说明）。已做：warehouse.md §1/§5/§6 逐条命令加作用、参数、就绪行和停法，评测四终端分工表；rendering.md 新增“图像来源（images.source）”并改正 Isaac 只做离线、只有 warehouse_textured 有 stage、转换器不替换图像等过时说法；新增 `docs/reference/glossary.md`（栈、deploy、planner/POSE/token、exporter、session、trace、镜像等），README/索引/三份 runbook 链接它；setup 表补第 7 步；box_grasp_envelope/scene_sweep 的作者机器专用 export 标注；`make lint` 文档项通过
    - [ ] 4.6.5.1 未改：scalebfm.md §6 裸 `python -m`（没写 venv/PYTHONPATH）；walk.md 探针依赖协作者未提交的 `ghost_reach_execution*.py`；box_grasp_envelope.md 的样片 zip、`outputs/ab_expert_20260927/compare.py` 与 rendering.md 例子里的 `outputs/crate_collect_10/` 新机器上不存在；docs/README.md、reference/runbook.md 中英混写，索引末尾两段 G1 历史
    - [ ] 4.6.5.2 待用户定：README 100、docs/README 80、single_arm/bimanual 210、warehouse 150 行都顶在 `lint_repository.py` GUIDES 上限，再加说明要抬上限
    - [ ] 4.6.5.3 未细读：single_arm_design.md（1407 行）、bimanual_design.md、台账与 handoff/archive
- [x] 4.9 2026-09-25 清理 outputs（用户：已迭代新 expert，旧数据与 checkpoint 全部作废、要重采重训）：删 outputs 内 42,907 个文件 132 G + /workspace/tmp/p03* 31 G，/workspace 余量 20→184 G；留 <1 MB 文本骨架、运行时输入、deploy worktree 与进行中 TODO 的证据；清单/脚本/p03 小文件在 `outputs/cleanup_20260925/`
- [ ] 4.7 待验证：离线补渲的数据集帧比实时帧"新" 1–2 行，训出的策略在实时闭环评测里是否受影响（warehouse 闭环要实时出图，受 2.4.10.8 卡住）
  - [x] 4.7.1 测渲染速度：warehouse / crate / bottle 三路相机（640×480）串行 vs 每路一个进程并行（同 GPU），出单帧耗时 p50/p90、并行吞吐、CPU 占用；决定渲染进程拆几个（用户 2026-09-24 定：VLA 闭环要实时出图、不能漂移、不减面）（2026-09-24：`scripts/experiments/bench_camera_render_20260924.py`，产物 `outputs/diagnostics/render_speed_20260924/`。现行 `warehouse_textured` 三路串行 p50 3.2 / p90 6.9 ms，crate 1.7、bottle 1.6 ms；旧 `warehouse`（复现 agent 实时跑的 `scene_reverse_camera`）36.7 ms，三进程并行也还要 28.6 ms/帧。同状态 A/B：差别全在纸箱——旧场景是一整块 462,730 面网格，textured 把同样的面按材质拆成 9 块（最大 231k）。合成网格扫描：单块网格 3×面数 ≤ 2^20（349,525 面）0.31 ms，多 1 面（349,526）3.91 ms，是 MuJoCo 3.11/驱动的硬阈值，拆块无损即回到快路径）
  - [-] 4.7.2 渲染移出物理循环：物理循环每个 IMAGE_DT 只写状态快照（qpos/qvel/mocap + 墙钟时间）到共享内存；独立渲染进程取最新快照 → `mj_forward` → 渲染 → 发布，图像时间戳 = 快照时刻（放弃 2026-09-24：被 4.7.9 的按需 Isaac 镜像替代，用户定）
  - [-] 4.7.3 VLA 延迟补偿起点改为图像快照时刻（现为取观测时刻 `vla_client.py:204`）；每次推理记录图像年龄（图像 vs 本体状态差）（放弃 2026-09-24：被 4.7.9 的按需 Isaac 镜像替代，用户定）
  - [-] 4.7.4 验收：warehouse 专家实时渲染 20 局，RTF 0.35 → ≥0.9，图像年龄 p50/p90，成功率对照离线基线 6/20（放弃 2026-09-25：是 4.7.2 异步 MuJoCo 渲染的验收，4.7.2 已被镜像替代；RTF 实测并入 4.7.7）
  - [-] 4.7.5 回归：bottle 用现有 checkpoint 跑 VLA 闭环，异步渲染 vs 同步渲染不应变（该场景渲染 1.3 ms）（放弃 2026-09-25：对比的"异步 MuJoCo 渲染"不再实现；Isaac 路径的回归由 4.7.9.8 一致性闸门与 4.7.6 覆盖）
  - [~] 4.7.6 有 warehouse / 双臂 checkpoint 后（1.19）用实时出图跑闭环评测；策略服务与 sim 渲染分 GPU
    - [~] 4.7.6.1 `box_room_grasp_200_cap120_20260925`（用户 2026-09-25；`box_room_grasp_200_20260925` 上限 800 局在产出率 < 0.25 时不够，7/30 时停掉换名）：采 200 成功局 → Isaac 补渲三路数据集相机 → convert → train 10000 步 → 100 局闭环（Isaac 镜像）；编排 `run_overnight.sh --from convert --isaac-gpu 1`，日志 `<run>/pipeline.log`；2026-09-26 训练按用户要求停在第 5000 步（checkpoint-5000 完整），评测未跑，待用户定
      - 用户 2026-09-25：只用三路相机（ego + 双腕），第三视角不作为 VLA 输入
      - [x] 4.7.6.1.1 exporter 在批间被重启时丢掉了每批最后一集（2/94 批）：加 `batches.wait_saved`；本次运行的记录已更正，补采后共 204 集（2026-09-26）
      - [~] 4.7.6.1.2 用 checkpoint-5000 跑 100 局闭环（Isaac 镜像）：先在 `summary.json` 补 train 条目（checkpoint 指向 `model/checkpoint-5000`），再 `run_overnight.sh --from eval`；GPU1 被别的项目占时镜像约 23 GB 可能放不下（待用户定）（2026-09-27：评测器修复后 10 局冒烟 9/10，100 局待 .6/.8）（2026-09-28：100 局已跑，81/100，见 4.7.6.1.2.8；剩 .6 判据、.10 待定项）
        - [x] 4.7.6.1.2.1 开环（官方 `open_loop_eval.sh`，val 10 集，Isaac 渲的图）：平均 MSE 0.00123 / MAE 0.0217；对照 预测均值 0.0350 / 0.140、块首真值保持 16 帧 0.00179 / 0.0206；traj 4 最差 0.0035（约第 620 帧提前半合爪）；两手夹爪都在对的时刻合到 −0.06。产物 `eval/open_loop_ckpt5000/`（2026-09-27）
        - [x] 4.7.6.1.2.2 闭环程序性 bug：`vla_rounds.py` 写死 `blue_crate`，warehouse 的目标是 `nor_cal_box`——reset 静默跳过物体扰动，第一局 `p.object_pos` / `evaluate` 必 KeyError（即 2.7）；改为 `target_key(meta)`（2026-09-27）
        - [x] 4.7.6.1.2.3 （2026-09-27 改为直接调 `expert.start_placement`，IDLE 朝走入方向）闭环起点与训练分布不符：`place()` 机器人偏航 0 + IDLE facing +x、物体偏航 π/2、离台边 U(1.2, 1.5)；训练 205 局全是 (0, −1.455)、朝 +y（90°）、离台边 0.90 m、物体偏航 0±5°（`start_pose: scene`, `start_advance −0.25`）
        - [x] 4.7.6.1.2.4 （2026-09-27 生成 `configs/deployment/warehouse_initial_token.json`，snapped 均值 = medoid；run 的 config.yaml 已指向它）初始 token 不符（即 4.7.6.2，已核对）：`crate_initial_token.json` 与本数据 194 集首帧比，肩 yaw 差 0.56/0.59 rad、肩 pitch 0.34/0.36，token L2 1.54（数据内首帧离散 0.077）；需按同法从本数据集生成 warehouse 的 token 文件
        - [x] 4.7.6.1.2.5 （2026-09-27 双臂路径下限取 `visual.close_target`，单臂不动）夹爪指令被夹紧：`sonic_token.hand_from_gripper` 夹到 `GRIPPER_MIN −0.02`（bottle 量程），本数据合爪 −0.06；−0.023 每指约 2 N 提不起 1.2 kg 箱（`visual_config.close_target` 注释），闭环里策略合得再准也抬不起
        - [~] 4.7.6.1.2.6 闭环成功判据与采集不同（2026-09-27 加了 `RetreatWatch` 一列，但训练数据用 `--trim-verification-tail` 在 hold 开始处裁掉，后退段不在数据里，这一列对本 checkpoint 必然 timeout——待用户定：去掉，或改成"抬起后再持住 N 秒"）：`vla_rounds` 用 crate 的 `evaluate`（抬起 + 不倾 + 两手接触 持续 2 s），warehouse 采集成功还要求躯干直立、后退 0.3 m 不掉；`eval.task` "pick up the crate" / `eval.spawn_edge` 不被读取（prompt 实际取 `task.prompt` "pick up the box"，与数据一致）
        - [x] 4.7.6.1.2.7 10 局冒烟（Isaac 镜像，run 的 `eval.rounds` 临时改 10，`summary.json` 手补 train 条目，备份 `*.bak_20260927`）：**9/10 抬起**，1 局合爪后箱子被扯歪；9/9 成功后再持 10 s 不掉，3/9 结尾倾角 > 15°；延迟 p50 166 ms、图像状态年龄 11 ms；台账 crate.md 2026-09-27 末节，录像 `outputs/videos/warehouse_ckpt5000_closed_loop10_20260927.zip`（2026-09-27）
        - [x] 4.7.6.1.2.7.1 评测录像改用 Isaac 渲（用户 2026-09-27）：`render_isaac_warehouse.sh` 加 `--traces/--out/--insets`，10 局 Isaac 第三视角 + Isaac 小窗，`outputs/videos/warehouse_ckpt5000_closed_loop10_isaac_20260927.zip`（3.1 MB，每段 272–341 KB）；warehouse.md §6.2、rendering.md 改为 Isaac 命令；脚本与文档未提交（2026-09-27）
        - [x] 4.7.6.1.2.11 双臂 blocking 循环没有 gap feeder（单臂 2026-08-31 已修，台账 bottle.md「blocking 客户端三修复」）：每周期执行 chunk 第 4–19 帧（640 ms）后停发，取图 + 推理 p50 166 ms（Isaac 渲 95 ms）期间 deploy 钳在旧 chunk 第 19 帧，新 chunk 再从第 4 帧起（跳过 0–3），周期实测 810 ms；训练数据里参考是连续的。待用户定：移植 gap feeder（间隙里续发旧 chunk 第 20 帧起）或改 async；`eval.replan_interval` 这条线不读（2026-09-27）（2026-09-27 完成：`vla_rounds.GapFeeder`，`eval.gap: continue` 为默认，`hold` 保留旧行为；10 局与 100 局对照见 4.7.11.4）
        - [x] 4.7.6.1.2.8 100 局闭环：先把 run 目录现有的 `eval/closed_loop*`、`traces/`、`state_trace/`、`videos*` 挪到 `eval_smoke10_20260927/`（manifest 已存在会拒跑），`eval.rounds` 改回 100（待用户定，连同 4.7.6.1.2.6）（2026-09-28 完成，不动原运行目录：只评测的副本 `wc_hold100_20260927`（与原循环相同，`gap: hold`）**81/100**，另三臂见 4.7.11.4）
        - [x] 4.7.6.1.2.9 评测文档对齐（用户 2026-09-27：协作者训练完跑评测不出 bug）：warehouse.md 新 §6 训练与评测（核对表 5 项、命令、读结果、排错 3 行，reflow 后 150 行）；box_grasp_envelope / bimanual / bimanual_design §五 / rendering / 镜像 handoff 同步；实测 box_platform 默认配置 60 局开局手臂与 warehouse token 差 ≤ 0.03 rad（2026-09-27）
        - [~] 4.7.6.1.2.10 （2026-09-27 ① 已提交推送 `eb9549f`/`b1e6827`；② warehouse 四份实验文件已改并推送，`box_platform_default` 改在磁盘上、随 6.9 提交；③ 未做）文档管不到的：① 本会话的评测代码未提交，协作者拉 main 仍是旧 `vla_rounds`（blue_crate、错朝向、−0.02 夹爪）；② `box_platform_default_20260927.yaml` 的 eval 块是 `image_source mujoco` + `render offline`（必然每局 streams never ready）且 `crate_initial_token`；warehouse 基线等实验文件也仍是 crate token；③ 可选：加载时拒绝 mujoco+offline、token 与数据首帧差 > 0.3 rad（2026-09-27 续：③ 的 mujoco+offline 已被 4.7.10 的 `images.source` 在加载时化解；token 与数据首帧不符仍不报错，未做）
      - [ ] 4.7.6.1.3 或者从 5000 步训到 10000：没存优化器状态，只能从 checkpoint-5000 的权重重开一次 5000 步训练，不等于原训练的后半段（待用户定）
      - [ ] 4.7.6.1.4 可选：200 集的 Isaac 第三视角观看视频（`render_isaac_videos.sh`，约 1 分钟/集），用户未要求
      - [ ] 4.7.6.1.5 待用户定：`docs/warehouse.md` §5 的 Isaac 补渲耗时按 197 集更正（写的是最早 2 集的"每局约 90 s、三路 p50 76–84 ms"；全集中位 69.8 ms、含编码约 74 ms/次、约 68 s/集；闭环按需渲 p50 95.5 ms）（2026-09-27）
    - [x] 4.7.6.2 `eval.initial_token`（crate 的）与 warehouse 开局姿势不一致（2026-09-27 已核对并换成 warehouse 自己的，见 4.7.6.1.2.4）
    - [ ] 4.7.6.3 待验证：RTX 连续逐帧渲（数据集）与按需跳着渲（镜像）是否同图（即 4.7.9.8）；训练/评测同一 `IsaacView`，但时序不同
  - [ ] 4.7.7 实时渲染 RTF 实测：`warehouse_textured` + `backend.render: live` 起栈跑几局，量 sim 的 RTF（期望 ≥0.9，旧场景 0.35）；结果决定 4.7.2/4.7.3 是现在做还是作为结构性保险后置（用户定）
  - [ ] 4.7.8 建场景闸门：任何单块显示网格 3×面数 > 2^20（> 349,525 面）时无损拆块或报错；场景建完测三路相机一组渲染时间，超预算即失败（旧 `warehouse` 场景是否下线一并定）
  - [~] 4.7.9 VLA 闭环图像改走 isaac-sync 分支的按需 Isaac 镜像（worktree `/workspace/code/locomanipulation-isaac-sync-20260923`，其 TODO 3.4.6–3.4.8；评测时 MuJoCo 用 `backend.render: offline`，物理循环里不渲染）。定了则 4.7.2/4.7.3 被替代，4.7.7 只服务 MuJoCo 基线臂和采集——用户 2026-09-24 定：走 Isaac 镜像，Isaac 用 GPU1，在 main 开发，从分支参考/移植。**状态索引（做了 / 没做 / 分支对账）：`docs/handoff/isaac-vla-mirror-20260925.md`**
    - [x] 4.7.9.1 warehouse 三路相机的 Isaac 按需渲染耗时（`IsaacView` + main 的 warehouse USDC 场景，间隔 0.4 s）；分支只测过 bottle：ego p50 52 / p90 70 ms，ego+右腕 63 / 94 ms（2026-09-24，GPU1：间隔 0.4 s p50 89 / p90 108 / p99 130 / 最大 135 ms，连续 72 / 107 ms，300 帧无空帧；+ 推理 p50 72 ms ≈ 160 ms < 0.4 s。五状态三路与 MuJoCo 并排几何对齐。`outputs/diagnostics/render_speed_20260924/isaac_warehouse/`、`isaac_view_check/`；台账 crate.md 末节；文档 rendering/warehouse/blender_scenes 的"减面是根本解"已更正）
    - [~] 4.7.9.2 分支与 main 对账：分支从 `e8a0019` 拉出、落后 main 33 个提交；`CAMERA_CONFIGS` 分支挪到 `schema.py`、main 挪到 `sim/dataset_cameras.py`；main 的 MuJoCo 补渲 `render_dataset_cameras` 与分支的 Isaac 副本 `sim/isaac/render_dataset.py` 要定分工——2026-09-25：已移植 view / bench_render / world_lift；分支独有的 `render_dataset.py` 与台账 §38–39（编号与 main 冲突）未移植，对账表见 `docs/handoff/isaac-vla-mirror-20260925.md`
    - [x] 4.7.9.3 移植 `IsaacView`（分支 `sim/isaac/view.py`）+ `bench_render` 的 `--state-trace`/多 gap/多路相机到 main，相机表用 main 的 `sim/dataset_cameras.CAMERA_CONFIGS`（2026-09-24：`view.py`、`tests/test_isaac_view.py`、`bench_render` 已移植；`--rtx-setting` 解析从 `isaac_render` 挪到 `stage.apply_rtx_settings`，`IsaacView`/`bench_render` 共用；两个 venv 下 Isaac 测试全过；未提交）；完成 2026-09-24
    - [x] 4.7.9.4 移植分支的 `world_lift.RunTraces` 空 `collect/state_trace` 修复（main 仍有此 bug）（2026-09-24：只移植布局判断那一行 + 测试；新测试在修复前失败、修复后通过；未提交）
    - [x] 4.7.9.5 sim 发布最新 qpos + seq + 仿真时间（分支 3.4.6）（2026-09-24：不另开端口，`PrivilegedPublisher.state()` 加 `qpos`，与 StateTrace 同 seq 行逐位相同，`test_privileged_qpos.py`；未提交）
    - [x] 4.7.9.6 镜像服务（RENDER_PY、GPU1）：请求 → 渲最新状态全部 video_keys → 回图 + seq（分支 3.4.7）（2026-09-24：`sim/isaac/mirror.py` + `scripts/run_isaac_mirror.sh`，端口 `TF_ISAAC_MIRROR_PORT` 5558+10i，从 eval 栈状态文件取场景/端口，场景 Isaac stage 与曝光取 `scenes.yaml` 的 `isaac` 条目；协议单测 `test_isaac_mirror.py`；**未对真实栈跑过**；未提交）
    - [x] 4.7.9.7 `ObsBuilder` 图像源开关 mujoco / isaac，本体状态、t0、延迟补偿不动（分支 3.4.8）（2026-09-24：`VlaConfig.image_source` mujoco|isaac，lockstep+isaac 加载即拒；单臂 async/blocking 与双臂 `vla_rounds --image-source` 都接上，双臂 HONOURED 加入；`test_obs_builder_image_source.py`；未提交）
    - [ ] 4.7.9.8 在线/离线一致性闸门：镜像评测记每张图的 seq，离线同 seq 重渲对比，容差取同脚本两次的差（分支 3.4.10）
    - [x] 4.7.9.9 Isaac 数据集副本 `sim/isaac/render_dataset.py` 移植，与 main 的 MuJoCo 补渲 `render_dataset_cameras` 定分工（2026-09-25：不移植副本，改为转换前 Isaac 填 session 占位帧 `sim/isaac/render_dataset_cameras.py`；convert 按 `eval.image_source` 核对渲染器；`run_overnight.sh --isaac-gpu`）
    - [x] 4.7.9.10 镜像对真实 eval 栈冒烟：warehouse_textured 起 eval 栈（GPU0）+ 镜像（GPU1），用 MirrorClient 连续取图，核对 seq 递增、状态年龄、渲染耗时，并与同 qpos 的 MuJoCo 图并排；不需要 checkpoint（2026-09-24：栈与镜像都在 GPU1（GPU0 被别的项目占 72%）；50/50 应答、seq 严格递增、状态年龄 p50 11 / p90 20 ms、渲染 p50 98 / p90 120 ms；同 seq 的 MuJoCo 并排几何一致；SIGTERM 正常退出；`outputs/runs/isaac_mirror_smoke_20260924/eval/mirror_smoke/`，台账 crate.md 末节）
  - [x] 4.7.10 图像来源做成整条线的一个选项（用户 2026-09-27 定方案 B + box_platform 建 Isaac stage；DECISIONS 2026-09-27 末节）：现在的选择器是 `eval.image_source`，但它同时决定补渲与 convert；配套字段没在加载时核对
    - [x] 4.7.10.1 （加载即拒）漏洞：`image_source: isaac` + 采集 `backend.render: live` → Isaac 补渲打印"nothing to render"退出 0、convert 的渲染器核对也跳过，训练吃 MuJoCo 图、评测用 Isaac 图，无报错
    - [x] 4.7.10.2 （eval 渲染由来源推出；历史文件迁移成 live）漏洞：`image_source: mujoco` + `eval.backend.render: offline` → 每局 `camera/state streams never became ready`（已在文档，代码未拒）
    - [x] 4.7.10.3 （起栈前 `require_images`）漏洞：`image_source: isaac` 而场景没有 Isaac stage（目前只有 `warehouse_textured` 有）→ 采集跑完、到补渲才报错
    - [x] 4.7.10.4 （用户选 B：顶层 `images.source`，`run_context.migrate_images` / `evaluation_of` / `require_images`；四份模板改新格式）方案 A：保留 `eval.image_source`，起栈/加载时一次核对上面三条 + render venv；方案 B：挪成 run 级字段（如顶层 `images.source`），eval 渲染方式由它推出，带版本迁移
    - [x] 4.7.10.5 给 `box_platform` 建 Isaac stage 并登记（`scripts/build_isaac_primitive_scene.py`，ISO 20，`source.json` 钉 MJCF）；录像脚本改名 `render_isaac_videos.sh` 并按运行场景取 stage（2026-09-27）
    - [x] 4.7.10.6 冒烟 `box_platform_isaac_smoke_20260927`：采集 5 局 1 成功 → MuJoCo 补渲按来源拒绝 → Isaac 补渲 → convert（单集对齐审计 2 次测量 FAIL，样本太少）→ Isaac 并排录像 → 镜像 50/50、渲染 p50 89 ms、状态年龄 11 ms；warehouse 录像回归像素差 1.5–2.3/255（2026-09-27）
    - [x] 4.7.10.7 提交（用户 2026-09-27 定分两批，第二批随 6.9.1）：`images.source`、迁移、录像脚本改名、文档、三份模板已提交推送（`356d806`）；`scripts/build_isaac_primitive_scene.py`、`scenes.yaml` 的 box_platform `isaac` 条目、`box_platform_default` / 冒烟实验文件、文档里 box_platform 的 Isaac 提法随 6.9 一起提交（它们依赖扫参会话未提交的 `box_platform_scene.py`）
    - [ ] 4.7.10.8 可选：box_platform 用 `images.source: isaac` 真正采集 + 训练 + 闭环一次（目前只有 5 局冒烟，单集对齐审计因样本少 FAIL）
    - [x] 4.7.10.9 `docs/handoff/lockstep-isaac-eval-plan-20260927.md` 及 `docs/rendering.md`、`docs/handoff/isaac-vla-mirror-20260925.md` 的链接（2026-09-27 写完；随 4.6.5 文档复查一起提交推送，用户 2026-09-27）
    - [x] 4.7.10.10 Isaac 并排录像 `render_isaac_videos.sh` 默认只渲成功局（用户 2026-09-27：失败局也渲浪费时间）：成败读判这一局的 manifest（采集 `manifest_NN.jsonl` 的 `label`，评测旁边 `closed_loop.manifest.jsonl` 的 `video`），`--all` 全渲、点名 label 只渲那几局；`run_context.successful_traces` + 3 个测试；实测 cap120 采集 205/950、评测 9/10、box_platform 冒烟 1/5。数据集 Isaac 补渲本来就只渲成功局（cap120 渲 204/948，没改）
      - [x] 4.7.10.10.1 MuJoCo 补渲 `render_dataset_cameras.sh` 也只渲成功局（用户 2026-09-27）：`pending` 一律跳过 `discarded_episode_indices`（去掉 `kept_only` 开关）；为不让 `--include-discarded` 静默训黑帧，converter 用 `render_dataset_cameras.unrendered` 拒绝 marker 没列的局；`box_room_grasp_10ep_20260925` dry-run 30 → 11 局
  - [x] 4.7.11 闭环评测的时间模型对齐现实：Isaac 渲图时冻结世界（lockstep），推理延迟按帧模拟；计划 `docs/handoff/lockstep-isaac-eval-plan-20260927.md`（用户 2026-09-27 要求写成计划，未开工）（2026-09-28 完成，见子项与计划第 10 节）
    - [x] 4.7.11.0 W0 校准 lockstep 本身（2026-09-02 墙钟 46/100 vs lockstep 32/100 未过）：让 lockstep 复刻墙钟现有时间模型，看能否复现 checkpoint-5000 的成绩与失败构成（2026-09-28 完成：墙钟 hold 81/100 vs lockstep 复刻 83/100，不一致 14 : 16，McNemar p = 0.86，失败构成逐项差 ≤ 3——校准通过；2026-09-02 的"停得远"签名未出现）
    - [x] 4.7.11.1 W1 双臂 lockstep 客户端（`vla_rounds` 按 tick 推进、tick 版 prelude、放开 `client`）（2026-09-27 完成：`tasks/crate/vla_lockstep.py`；W1 验收见计划第 10 节；逐位复现被 MuJoCo 4× 多重采样的 1–2 灰度级噪声打破，已查明，未改渲染设置）
    - [x] 4.7.11.2 W2 Isaac 接入（去掉 lockstep + isaac 拒绝、offline 时 lockstep sim 不渲 MuJoCo、镜像按期望 seq 核对）（2026-09-27 完成：84/84 块的图 seq = 冻结时刻 seq；另修三处 lockstep 老化：`lockstep_vendor` 路径、`--planner-file`、offline sim 起栈死锁）
    - [x] 4.7.11.3 W3 延迟模型：推理 L 帧（实测约 2）、相机 C 帧，间隙里续发旧 chunk（2026-09-27 完成：`gap` / `latency_frames` / `latency_dist` / `camera_frames`；`vla_lockstep latency-dist` 从墙钟 traces 生成分布）
    - [x] 4.7.11.4 W4 同 checkpoint、同 seed：墙钟 vs lockstep 各 10 局，再各 100 局（2026-09-28 完成：墙钟 hold 81、墙钟 continue 83、lockstep 目标模型 88，对墙钟 p = 0.27 / 0.85，100 局判不动；台账 crate.md 2026-09-27 末节）
    - [x] 4.7.11.5 W5 文档与模板（2026-09-28 完成：warehouse.md §6.1、bimanual.md §6、bimanual_design §五、rendering.md、术语表、change map、全部实验文件加 `gap` / `camera_frames`；计划第 10 节是实施记录）
    - [ ] 4.7.11.6 可选：分开两个效应——lockstep `hold` + L = 2 一臂（目标模型 88 与复刻 83 之差是"间隙续走"还是"延迟少 2 帧"）；要逐位复现就得评测图和训练图都关 MuJoCo 多重采样
    - [x] 4.7.11.7 用户 2026-09-28 定：所有场景的 VLA 评测都走 lockstep；main `4b8180b` 已把默认 `eval.client` 改成 `lockstep`（见 §9；校准已过，目标模型 88 vs 墙钟 81 未显著）
    - [x] 4.7.11.8 已提交（`291cdeb`，三批合成一个提交，2026-09-28）：未提交（待用户说）：lockstep 实现（`vla_lockstep.py`、`GapFeeder`、镜像按 seq、三处 lockstep 老化修复）+ 测试 + 文档 + 115 份实验文件加 `gap` / `camera_frames`；`make check` 通过（2026-09-28）
    - [ ] 4.7.11.9 可选：10 个只评测的运行目录 `outputs/runs/{ls,wc}_*_20260927`（每个约 120 MB）用完可删，删前按规矩在 STATE 记 24 h
- [~] 4.8 协作者采集仍摔倒：核实用户的猜测"自编 SONIC deploy 二进制没同步到文档"（用户 2026-09-24）
  - [x] 4.8.1 静态核对（2026-09-24）：本机 9 个 deploy 二进制全部从只存在本机的 `lockstep-port` 分支编出（上游 1983e888 + 6 个未推送的本地提交 + 工作树补丁）；setup.md 的做法（clone 上游 main 不钉版本，现为 b042411，再打 4 个补丁）本机从没编过、跑过。文档版缺 SONIC_AUTO_START/`--default-motion`（每个 eval 栈都要，上游静默忽略未知参数）、DDS domain（`sonic_dds_domain.patch` 已打不上上游 main）、planner 初始化超时 10 s（上游 5 s）。已排除：SONIC 权重与 HF 逐字节一致；肩 roll 默认角两边编进去的都是 ±0.2；上游 1983e888 之后的 deploy 改动默认不改行为。MuJoCo 不钉版本，新装会是 3.14（本仓库在 3.11.0 上测）
  - [ ] 4.8.2 向协作者要证据：run 的 `deploy_receipts.jsonl`、上游 commit、`mujoco.__version__`、每局失败字符串和摔倒阶段、是采集还是闭环评测、同时跑几个实例、`scripts/cpu_headroom.sh`、deploy.log 里有没有 `Planner initialization timeout`——2026-09-25 协作者按新 setup 重编后仍 `initial_stance_unstable`；init token 假设已排除（文件已入库，双臂采集不读）；已给出要收的清单（deploy.log、sim.log、manifest、visual log、deploy_receipts、stack_env、mujoco 版本、cpu_headroom）——2026-09-25 本机从零按文档装一遍 30 局零开局失败（4.6.1），文档化的安装本身可排除
    - [ ] 4.8.2.1 待用户定：开局判定（`expert.py` 两处 quiet admission）失败时在 manifest 记卡在哪个条件、骨盆最低高度 / 最大倾角 / 末速度、仿真时间与墙钟各走多久（不改判定本身）
    - [ ] 4.8.2.2 问协作者的"一开始摔倒"是哪种：sim.log 里起栈时那行 `Robot has fallen 0.199 m`（本机每次都有、无害）还是 manifest/collect 日志里的逐局失败；两者要分开问
    - [ ] 4.8.2.3 待用户定：本机把 sim 强制走 Mesa EGL（`__EGL_VENDOR_LIBRARY_FILENAMES=.../50_mesa.json`）跑 10 局，看缺 NVIDIA 图形库的机器是否开局失败/摔倒、日志长什么样（用户 2026-09-25：严格按 README 不摔，这条线先到此为止；其余问题另行单独解决）
    - [x] 4.8.2.4 不用 strace、可控地把栈拖慢（CPU 限核 / 陪跑负载 / 只拖 sim 或只拖 deploy），测开局失败率随 RTF 的变化，拆开是 sim 慢还是 deploy 周期抖动——strace 轮 RTF 0.75 时 19/20 开局失败——2026-09-25 用户：做，分别只拖 sim 和只拖 deploy。方法：SIGSTOP/SIGCONT 占空比（每 100 ms 停 16 ms，单次 < 50 ms 不触发 deploy 掉线保护），只停目标进程、不加负载；三臂 baseline / sim / deploy 各 10 局，同 seed0 40000，沙盒 `/workspace/readme_bottle_20260925/`（d505880 未跟踪安装）；证据 `outputs/slowdown_20260925/`——完成：基线 7/10、只拖 sim（RTF 0.69）1/10、只拖 deploy（LowState p90 19 ms）8/10，**三臂开局失败都是 0**；sim 慢的失败是抓取时髋部碰桌 8/10。strace 轮的开局摔倒不是 sim 慢或 deploy 抖单独造成的。报告 `outputs/slowdown_20260925/report.md`
    - [ ] 4.8.2.5 待用户定：strace 轮开局摔倒的剩余候选——同样占空比只拖 exporter、只拖 agent、四个一起拖；四个一起拖也不摔，就是 ptrace 特有的停法，和协作者无关（用户 2026-09-25：严格按 README 不摔，这条线先到此为止；其余问题另行单独解决）
    - [ ] 4.8.2.6 待查：sim 慢（RTF 0.69）时抓取阶段髋部碰桌 8/10（基线 2/10）——假设抓取阶段按墙钟计时，动作相对身体提前；证据 `outputs/slowdown_20260925/`（用户 2026-09-25：另行单独解决）
  - [x] 4.8.3 本机复现（用户 2026-09-24 批准）：按 setup.md 编文档版二进制（上游 b042411 + 补丁），bottle 10 局对照现用二进制——完成 2026-09-24：上游 b042411 + setup.md 补丁编出 472ed2b，bottle 同配置同 seed（GPU1）：文档版 10/17、对照 10/12（GPU0）与 10/11（GPU1）；三臂 40 局 0 摔倒（骨盆最低 0.718 m）；文档版髋碰桌 5/17 对 1/23（p≈0.07）。台账 bottle.md §38，产物 `outputs/deploy_docbuild_20260924/`
  - [ ] 4.8.5 定位文档版抓取变差（髋碰桌）的源码差别：在文档版上逐个补回本地提交（先 lockstep 两个提交与 wbc_token，因为改动最大），每步 bottle 17 局；或先对两个二进制做同一 token 序列的开环回放，比 cmd_q
  - [~] 4.8.4 文档/构建对齐（待用户定方案）：钉上游 commit，把缺的本地提交（auto-start/default-motion、planner 超时、DDS domain）做成补丁或发布 fork；钉 mujoco 3.11.0；起栈前按模式检查二进制能力标记，缺了直接报错——2026-09-25 文档部分完成：setup.md 钉上游 1983e888 + 三个 deploy 补丁（新 `patches/sonic_deploy_local_base.patch` = 6 个本地提交，去掉肩 roll 那一处）+ 源码指纹 c1360628… 与四个能力标记核对；核对过在干净 checkout 上重建出 5cab2fb 的源码、cmake 编译通过、补丁循环可重复执行；另钉 mujoco==3.11.0、Isaac-GR00T 376ba89，补测试依赖。未做：起栈前按模式检查能力标记；本机 crate 仍跑 4add301（无夹爪补丁），与文档版不同，待 crate 20 局验证
    - [x] 4.8.4.1 补丁循环先查 commit 与源码是否干净，只对干净树一次打全；协作者 `g1_deploy_onnx_ref.cpp:46` 报错复现为旧 `sonic_dds_domain.patch` 残留；bash / zsh 四种状态实测（`b3b645f`，2026-09-25）
    - [ ] 4.8.4.2 起栈前按模式检查二进制能力标记（三条线都要 pose + hand + auto-start + DDS），缺了直接报错
    - [ ] 4.8.4.3 全线统一 5cab2fb（crate 目前 4add301，差夹爪补丁）：先 crate 同配置同 seed 20 局对照 17/20，过了再定用 `/workspace/.env` 还是本机 third_party 按新 setup 重编（用户 2026-09-25：暂不实测）
  - [x] 4.8.6 `bottle_simple_gate8` / `bottle_chain_smoke2` / `bottle_twobatch_probe` 的 `scene.cases` 指向 `outputs/bottle_dynamics_20260915/regression8_lowcom.json`（未入库，4.6 KB），runbook.md 与 single_arm.md 的示例用 gate8，新机器直接失败——2026-09-25 用户定入库：`configs/cases/bottle_regression8_lowcom.json`，新版本 `bottle_simple_gate8_20260925.yaml`，runbook / single_arm / single_arm_design / dry-run 测试改指新版；另修：bank 按 scene_meta.json 字节哈希，而 meta 里有绝对路径，入库也换机器必挂 → schema 2 用去掉仓库根前缀的规范 JSON 哈希（`portable_scene_hash`），schema 1 照读；`bottle_chain_smoke2` / `bottle_twobatch_probe` 仍指 outputs/（内部探针，未改）
  - [ ] 4.8.7 `configs/experiments/docrepro_20260924_bottle.yaml` 的 `train.holdout 10` = `collect.successes 10`，被 09-24 新加的准入检查拒绝（本次复现在临时副本里改成 0）；待定：出新版本还是保留为复现记录
  - [x] 4.8.8 推送 `b8f6c82` 与 handoff 提交（用户未答复是否推送）——2026-09-25 用户定推送：连同 P7/P8/P19/文档修复一起 rebase 到 origin d505880 上推到 `c52ffb8`


## 5. bottle 走近改成 warehouse 的方式（用户 2026-09-25 定）

步速 0.25 恒定、手臂交给步态自然下垂、按桌边守线刹车停得越近越好、停后双手原地竖直抬起、左手保持抬起、右手再去抓。腕部相机走路时拍到大腿，用户接受。先加开关，只在新实验文件里打开；试验通过、用户看完视频后，再定是否设为默认、删掉张开姿势 / 三档速度 / backoff。
基线：`bottle_simple_10ep_20260925` 试 11 局保留 10 集；刹车时骨盆离桌边 0.39–0.42 m，之后滑行 10–28 cm；停下时手离桌边 −3…+10 cm，10/11 局要先后退。

- [x] 5.1 走路：开关打开时不给上身参考，手臂交还给步态控制器（照 warehouse `HangingWalk`；视频确认自然下垂摆动）（2026-09-25）
- [x] 5.2 步速：simple 线从起步到刹车恒定 0.25；方向起步时定死、指向规划站位点（跳过 leg 1）（2026-09-25）
  - [x] 5.2.1 逐帧纠偏方向被否（2026-09-25）：方向定死后 10 局落脚间隔 0.24–1.18 s、无停顿，与"重规划造成停走"一致（两轮是不同 run，第一轮只 2 局）——deploy 对移动方向做精确比较，任何变化都触发重规划（10 Hz）；第一轮 2 局步态为 15 cm → 停 1.3–1.9 s → 34 cm，守线只能按 35 cm 预留，骨盆在离桌边 0.51–0.53 m 处停下，两局都 `[stand] arrival left the object out of reach`（`outputs/runs/bottle_simple_guard_20260925`，2026-09-25）
- [x] 5.3 刹车：`FootWatch` + `GuardBrake` 以桌边为线；`FootWatch.frontmost_band` 按网格顶点取桌面那一层（顶 −12 cm … +3 cm）最靠前的点，0.6 ms/次；逐局记录在 `simple.guard`（2026-09-25）。第一轮实测 realised_extra 0.192 / 0.192（标定值 0.20）；桌面层最靠前的是下垂的 Dex1 手指，在骨盆前 16 cm
- [x] 5.4 停稳后双手原地竖直抬到抬手高度（笛卡尔路径，手指朝下），停顿后右手滑向预抓点；左手保持抬起；去掉 backoff，接管时手越过桌边就判失败（2026-09-25）：9 局右手 0.57 → 0.78 m，水平漂移 2.4–4.1 cm，抬完误差 ≤3 cm
  - [x] 5.4.1 抬手改成 warehouse 的 `lifted_ready`（工具点骨盆前 16 cm、侧 16 cm、上 5 cm，掌心下倾 30°，肘下限 −60°，关节空间插值）——用户 2026-09-25 看视频指出抬手时手肘往外张。实测：现抬法肩外展 18°→68°（右）/20°→53°（左），肘比肩外 13–16 cm；warehouse 同起点肩外展 34°、肘外 9 cm。离线核对 10 个停位：关节插值中途手指最多前移 3 cm，最差一局离桌边 6.8 → 3.9 cm，不越线；与 5.6 收紧守线互相牵制——用户 2026-09-25 批准；实现：`BimanualIK.lifted_ready`（从 crate expert 挪出求解，warehouse 仍在解不准时 SystemExit）、`SimpleGraspPolicy._lift_ready`（接管后关节插值 2.5 s + 停 0.3 s，解不准判本局失败）；10 局对照 `bottle_simple_guard_lifted_20260925`（2026-09-25）：10/10，肩外展 18–23°、肘外 4.6–6.0 cm（上一轮 36–74°、9.4–17.5 cm），每局 39.5–44.3 s；预览 `outputs/videos/bottle_simple_guard_lifted_20260925/`
    - [ ] 5.4.1.1 抬手中途手指前扫 6–8 cm（参考轨迹约 3 cm + 跟踪滞后 4–5 cm：抬到 40% 时右肩后摆少 0.06–0.13 rad、肘少折至多 0.14 rad），停位最近两局抬手时离桌边只剩 6 mm / 1.2 cm，0 越线；收紧守线（5.6）前要先定：放慢插值、改插值形状（先折肘再摆肩），还是把前扫算进守线余量（待用户定）
- [x] 5.5 单元测试 + 新实验文件，seed 40000 跑 10 局，与基线逐局对比（刹车位置、手/骨盆离桌边、成功率、失败原因、耗时），导出视频——单测 `tests/test_bottle_edge_guard.py` 等通过；实验文件 `configs/experiments/bottle_simple_guard_20260925.yaml`；第二轮（方向定死）`bottle_simple_guard_straight_20260925`：9/10（失败 1 局 `[grasp] simple lost pad contact during the lift`，与基线失败同类、不同 case）；0 局越线；骨盆离桌边 0.24–0.40 m（中位 0.34，基线 0.12–0.31、中位 0.18）；桌面层最靠前是下垂手指，离桌边 0.07–0.22 m（基线 −0.03…+0.10，10/11 要后退）；realised_extra 0.127–0.174（设 0.20）；走路 8.2–12.1 s（基线 ~4.9 s），每局 39–44 s（基线 36–40 s）；预览 `outputs/videos/bottle_simple_guard_straight_20260925/`
- [-] 5.6 按 5.5 实测收紧 `guard_extra` / `guard_margin`——用户 2026-09-25：bottle expert 不再微调（实测刹后滑行 0.13–0.17 m，预留 0.20；收紧前还要先处理 5.4.1.1 的抬手前扫）
- [x] 5.7 文档（2026-09-25）：`single_arm.md` 首段与 gate 行、`single_arm_design.md` 新节"默认走近：守线刹车 + 原地抬手"、台账 §39、配置字段表重新生成
  - [x] 5.7.1 补齐（用户 2026-09-25 问"README 等要不要改"）：`single_arm_design.md` 图 ② 按新流程重画（行号换成函数名）、开头两张表改默认任务与动作过程、`change-map-bottle.md` 补 edge_guard / frontmost_band / lifted_ready、两份实验文件头部注释；README 不用改；推送 `4655254`（2026-09-25）
- [x] 5.8 设为 bottle 默认（用户 2026-09-25）：`SimpleGraspConfig.approach_stop` 默认 `guard`；主线 5 份（`bottle_simple_200`、`bottle_simple_smoke`、`bottle_smoke_norecord`、`bottle_eval_smoke3`、`agent_single_arm`）显式写 guard 全套参数，task/embodiment/scene/backend 与 10/10 试验逐字段相同；记录旧走法的 9 份（gate8 ×2、固定 bank 的 chain_smoke2 / twobatch_probe / doc_walkthrough_1ep、docrepro、single_arm_20ep ×2、`bottle_simple_10ep_20260925`）钉 `visible`；`make check` 通过；提交推送 `e3d49d9`（2026-09-25）
  - [ ] 5.8.1 gate8 闸门换成新默认：用 guard 在 8 局 bank 上录 `tests/gate/baseline_simple.json` 的新版（现基线是旧走法 stage4_wrap8p；guard 多一个 `ready` 阶段，旧基线必 FAIL）（待用户定）
  - [ ] 5.8.2 删除 visible 走法的代码（张开下垂 `hang_*`、三档速度与可见深度刹车、`retract_*` 后退）——钉 visible 的 9 份实验文件会跟着失效（待用户定）
- [x] 5.9 会话开头的任务：单臂 bottle 采集 10 集并导出视频（旧走法，`bottle_simple_10ep_20260925`，11 局保留 10；预览 `outputs/videos/bottle_simple_10ep_20260925.zip`，每段 ≤500 KB）（2026-09-25）
- [ ] 5.10 新默认跑一遍整条线：采集 → convert → train → 闭环评测；核对 `eval.initial_token` 与新数据开局姿势一致（假设一致，未验证；由用户安排）

## 6. 场景能力扫参：把 warehouse 布置限制在专家 + SONIC 抓起并保持 ≥ 50% 的范围（用户 2026-09-26 定；`docs/scene_sweep.md`）

- [x] 6.1 离线看已有 950 局（2026-09-26）：抓起并保持 247/950 = 0.26；停位深度 < 0.48 m 0.39–0.47、≥ 0.52 m 0.00–0.06；起始偏角实测 ~1°（p95 9°）、箱子偏角 ±5° 两端 0.20–0.24 中间 0.33，范围太窄，要扫
- [x] 6.2 工具（2026-09-26）：`box_platform_scene`（10 cm 实心台 + 1 cm 壁纸箱，宽/长/高可调）、visual 专家四个布局区间 + `start_edge` 放宽到 0.5–2.5 m、每局记 `real_time_factor`、`collect.stop_rule`（Wilson 95%，阈值 0.5，10–30 局，RTF < 0.9 不计）、`locomani.tasks.crate.sweep`（格子/链/剪枝/报告）；冒烟 2 局通过
- [x] 6.3 第 1 步（2026-09-26，215 局 52 分钟）：高度 [0.30, 0.45, 0.60, 0.75] × 长度 [0.20, 0.32, 0.44]，宽 0.45，起点 1.2 m 正对（`configs/sweeps/box_envelope_s1_20260926.yaml`；实例 0/1/2/3 同在 GPU0，RTF 0.97–0.98）。没有一格明确 ≥ 0.5：H0.45/0.60 × L0.20/0.32 四格 12–15/30（undecided），其余 8 格 below（H0.30、H0.75、L0.44 全部）；报告 `outputs/sweeps/box_envelope_s1_20260926/report.md`
  - [x] 6.3.1 用户 2026-09-26 定：第 2 步中心 H0.60 × L0.32（warehouse 纸箱几何），阈值降到 0.3
  - [-] 6.3.2 起点远近解释 warehouse 产出率低（假设）——第 2 步否定（2026-09-26）：起点 0.6–0.8 m 8/15、1.0–1.2 m 8/15，近起点并不差；扫参格 13/30 与 warehouse 247/950 的差距另有来源（台面 0.37 vs 0.10 m、房间，未查）
- [x] 6.4 第 2 步（2026-09-26 15:00–16:55，620 局；结果见 `docs/ledger/crate.md` 2026-09-26）（`configs/sweeps/box_envelope_s2_20260926.yaml`，30 格，实例 0–4 同在 GPU0）：中心 H0.60 × L0.32 × W0.45，阈值 0.3——宽 [0.30, 0.38, 0.55]；起点距离 0.6–2.0、朝向 ±30°、横移 ±0.25 m、setback 0–0.15、箱子偏角 ±20° 各分正负两侧成链（三档，内档 below 即剪掉外档）
- [ ] 6.5 横移方案 B：起步算一次斜线方向、全程不变的走法（用户 2026-09-26：先用 A 测现行专家的容忍，再做 B，现实部署不会总对准）
  - [x] 6.4.1 推荐布局定稿（用户 2026-09-27）：箱宽 0.45、长 ≤ 0.32、抓点高 0.54–0.69；偏角 0、setback 0、横移 −8..0 cm；起点 0.6–2.0 m、朝向 ±20° 随机 → `docs/box_grasp_envelope.md` §4
  - [ ] 6.4.2 待验证：箱子偏角几度就掉一半（0° 13/30 vs ≤3.5° 7/30）——专家的掌面/抓点是否没跟上箱子偏航（检查 grasp_points 与 palm 目标在偏航时的朝向）
  - [x] 6.4.3 逐帧复盘 835 局（2026-09-27，`docs/ledger/crate.md` 2026-09-27）：stall = 下探时整对手横向漂 3–5 cm 且修正被冻结；脱手 = 站直末段 1 cm 捏夹滑脱（97% 夹空）；翻箱 = 起抬不对称；全无重试
  - [~] 6.4.4 专家改进 ①–⑤ 依次实现并 A/B（用户 2026-09-27 定顺序；台账 `docs/ledger/crate.md` 2026-09-27）
    - [x] 6.4.4.1 ① `visual.lateral_hold_common_mode`：26/60 vs 基线 28/60，脱手 27→12% 但卡住 12→37%；共模修正把箱子推走（重试前已移 9 cm）。默认关
    - [x] 6.4.4.2 ② `visual.descend_retries` / `reseat_retries`（reopen 阶段、manifest `retries`）：12/30；①+② 25/60。重试触发但第二次几乎都再卡住（1/10、1/23）。默认 0
    - [ ] 6.4.4.3 ②b（待用户定）：descend 期加箱子位移闸门（reach 期有 `basket_displaced_before_capture`，descend 期没有），箱子一动就退回，再判断是否重来
    - [x] 6.4.4.4 ③（2026-09-27）：脱手实为竖直蠕滑（0.2–0.4 cm/s），改成 `visual.attachment_gain` 竖直附着反馈：站直期脱手 27 → 13%，成功 15/30 vs 基线 28/60（分不开）；抓深 2 cm 负结果（10/30、叠加 8/30）。默认关，待用户定是否设默认
    - [x] 6.4.4.5 ④ `visual.seat_depth/seat_skew`（2026-09-27）：7/30，18/21 首次检查不合格（交付深度中位 1.3 cm），重坐后 1/10 合格；不采用，默认关
    - [x] 6.4.4.6 ⑤ `visual.face_after_brake`（2026-09-27）：≤ 6° 转不动（死区），+7..14° 转到位（交付 86%）成功率仍 1/30；四格 2/30、3/30、8/30、1/30 vs 6/30、2/25、4/30、1/15。不采用，默认 none
    - [x] 6.4.4.8 横向前馈（2026-09-27）：离线否定——135 局下探末两手同向误差中位 −0.5 cm，固定预偏反而降低闭合窗内比例，未跑；改跑 `descend_speed` 0.05/0.08/0.11：14/30、11/30、8/30，沿上甩动不随速度变，负结果
    - [x] 6.4.4.9 壁厚 20/30 mm（用户 2026-09-27 提议）：14/30、27/60 vs 28/60；脱手 27 → 7–10%，卡住 12 → 22–30%，总数不变——抓持侧可消，落点侧是上限
    - [x] 6.4.4.10 `visual.ready_after_brake quick`（用户问刹车后为何停 10 s，2026-09-27）：prep→reach 13.3 → 7.9 s，一局 35.9 → 30 s，成功 26/60 vs 28/60（不变）。建议 warehouse/扫参配置改用 quick（待用户定是否改 dataclass 默认）
    - [x] 6.4.4.11 再压准备手臂到 5.8 s（blend 1.5 / hold 0.6 / reanchor 0.5，2026-09-27）：23/60 vs quick 26/60 vs 基线 28/60，单调下降但区间重叠；不建议默认，旋钮保留
    - [ ] 6.4.4.7 待用户定：③ 附着反馈是否设为默认（再跑 30 局确认卡住 27% 不是它带来的）；箱子偏角敏感的真正原因（转正身体没用）；⑤ 小角度可否用"先多转再回正"绕过死区
- [x] 6.6 第 3 步（2026-09-27）：安全范围内所有起点维度同时随机 + quick，4 几何 × 15 局 = 24/60 = 0.40（区间 0.29–0.53），叠加没有再掉；视频 `outputs/videos/box_range_quick_20260927.zip`（24 段）。warehouse 场景怎么落地待用户定
- [x] 6.7 刹车后 arm preparation 缩短（2026-09-27，见 6.4.4.10/11）：quick 7.9 s 成功率不变，设为默认路径
- [x] 6.8 双臂抓箱默认版本（用户 2026-09-27）：`configs/experiments/box_platform_default_20260927.yaml` + 场景名 `box_platform`（`outputs/scene/box_platform`），文档 `docs/box_grasp_envelope.md`，README / docs/README / warehouse §8 已索引
- [~] 6.9 待用户定：把本会话的代码（12 个旋钮、sweep 工具、场景生成器、文档）提交；③ 是否写进默认配置；同几何房间 26% vs 台上 43% 的差异排查
  - [x] 6.9.1 扫参会话代码（12 个旋钮、`sweep.py`、`box_platform_scene.py`、30 个实验文件、两份扫参配置、文档）与 box_platform 的 Isaac stage 一起提交推送（用户 2026-09-27）

## 7. Isaac 画质（用户 2026-09-28：Isaac 渲染不如 Blender，是设置还是版本？）

- [~] 7.1 诊断并修复 Isaac 与 Blender 的画质差距  ← 来源: 会话 2026-09-28；台账 `docs/ledger/blender_scenes.md` 第 2 节
  - [x] 7.1.1 两边配置对比（Cycles 1024 spp / AgX / 3840 vs Real-Time / FXAA / 640）、Blender 材质审计、Isaac 5.1 与 6.0 渲染改动调研 (2026-09-28)
  - [x] 7.1.2 A/B 6 组（Real-Time / PathTracing × 640 / 1280，外加屏蔽法线两组），`outputs/experiments/isaac_render_quality_ab_20260928/` (2026-09-28)
  - [x] 7.1.3 修法线：`blender_to_usd.py` 断开高度图；`build_isaac_warehouse_scene.py` 用 over 屏蔽单通道 normal；`warehouse_textured` stage 已重建 (2026-09-28)
  - [x] 7.1.4 `render_isaac_videos.sh` 默认 `--size 1280x960`，加 `--path-trace`；`mujoco_render` 按需放大离屏缓冲；GPU0 实跑验证 (2026-09-28)
  - [x] 7.1.5 核查 VLA 图像输入：任意渲染尺寸都进模型为 352×256、88 token，不改数据集相机 (2026-09-28)
  - [ ] 7.1.6 程序化材质（36 个 Bump、15 个程序化底色）烘焙成贴图：先要给 88% 没 UV 的物体展 UV；做不做等用户定
  - [ ] 7.1.7 可选：给 `--path-trace` 重调 filmIso（比 Real-Time 亮约 10/255）
  - [ ] 7.1.8 可选：评估 Isaac 6.0（RT 2.0），前提是渲染运行时能脱离 SIMPLE 的 4.5 锁定
- [ ] 7.2 待验证：新 stage（去掉噪点）下，checkpoint-5000 的 Isaac 评测是否偏离 81/83/88；数据集要不要按新 stage 重新补渲（问题：画面变了，策略受不受影响）
- [~] 7.3 side-by-side 预览压到 500 KB 以内
  - [x] 7.3.1 手工压两段验证视频，放在 `previews_share_20260928/`（1280 版 3 段，路径追踪版 1 段）(2026-09-28)
  - [x] 7.3.2 `locomani.pipeline.video_preview`：CRF 32→38 阶梯 + 按帧边界分段 + 自检；接进 `render_isaac_videos.sh`；5 个测试 (2026-09-28)
  - [ ] 7.3.3 重新生成 `box_room_grasp_200_cap120_20260925/eval/previews_isaac/` 里 3 个超限的旧预览（已问用户，未回复）
  - [ ] 7.3.4 待用户定：长局分段多时，阶梯里要不要加降宽（比如 960）
- [x] 7.4 已提交（`291cdeb`，与 4.7.11.8、8.12 合成一个提交，2026-09-28）。提交本会话改动（等用户说）。`docs/rendering.md`、`docs/warehouse.md` 里混有 lockstep 未提交的改动，提交时要和 4.7.11.8 一起处理

## 8. 四边线：warehouse 平台四条边随机走入，按 envelope ≥ 30% 的范围随机（用户 2026-09-28；台账 `crate.md` 2026-09-28 两节）

- [x] 8.1 第 1 步：四种尺寸纸箱场景 `warehouse_box_45x{32,20}x{33,18}`（`warehouse_box_scene.py`；每边 `clear_m` + `edge_<边>` 机位）（2026-09-28）
- [x] 8.2 第 2 步：四个 Isaac stage（`build_isaac_warehouse_scene.py --scene`），MuJoCo/Isaac 轮廓 ≤ 3 px（2026-09-28）
- [x] 8.3 第 3 步：每局抽边（`visual.start_edges` / `start_clearance` / `object_along_range`，`expert.facing_edge`），旧配置 A/B 逐位相同；录像按边换机位（`sim/scene_edges.py`）（2026-09-28）
- [x] 8.4 第 4 步：实验文件（按边 pilot ×4、按尺寸 pilot ×3、采集 ×4）（2026-09-28）
- [x] 8.5 第 5 步：pilot 29/70 = 0.41，无 0–1/10 的边或尺寸；录像 `outputs/videos/warehouse_edges_pilots_20260928.zip`（2026-09-28）
- [~] 8.6 正式采集，每种尺寸 100 条（用户 2026-09-28）
  - [x] 8.6.1 `collect --until-kept N`（暂停可续采）（2026-09-28）
  - [x] 8.6.2 quick 版每种先采 ≥ 5 条 + Isaac 并排视频 `outputs/videos/warehouse_edges_first5_isaac_20260928.zip`（7/10/5/6 条）（2026-09-28）
  - [-] 8.6.3 quick 版续采到 100 —— 改用 compact（8.7），quick 的 28 条保留不并入
  - [ ] 8.6.4 compact 版采集 `warehouse_edges_compact_<尺寸>_20260928`：待用户定是否先每种 5 条 + Isaac 视频，以及训练图像用 MuJoCo 还是 Isaac（现 mujoco；GPU1 已空）
  - [ ] 8.6.5 采满后：补渲数据集相机 → convert（四个运行合成一个数据集的步骤还没有）
- [x] 8.7 刹车后准备手臂缩短（用户：停得太久）：`ready_after_brake compact` 17/40 vs quick 19/40，准备 8.7 → 4.9 s；用户定为双臂采集默认（`box_platform_default_20260928`）（2026-09-28）
  - [-] 8.7.1 刹车后不重锚 `reanchor_after_brake: false` —— 14/40、卡住 7 → 13，不采用；要定需每组约 200 局
  - [ ] 8.7.2 可选：边停边抬手（刹车最后一步就抬），要改走近循环；09-23 走路时抬手 6/10 局扎进箱子
- [ ] 8.8 数据规模对照：400 条全量 vs 按尺寸均衡抽 200 条，各训 2 个种子，四边分布闭环各 100 局（用户认可的方案，采满后做）
- [ ] 8.9 可选：checkpoint-5000（只见过 ym）在四边分布上评测 100 局，看窄数据换边后掉多少
- [ ] 8.10 仿真实时录像（`SIM_VIDEO_DIR`）仍只用一个固定机位；四边线目前只按离线渲染跑
- [ ] 8.11 2026-09-29 之后删试建副本 `outputs/experiments/warehouse_edges_20260928/trial_45x20x18`（107 MB）
- [x] 8.12 提交第三批改动（四边线代码、配置、文档）：`291cdeb`（2026-09-28）。没有按 hunk 拆：第一批新增的 `vla_lockstep.py` 直接调用第三批的 `expert.facing_edge`，拆开的中间提交跑不起来

## 9. 文档按"只想跑通的人"重组；所有 VLA 评测默认 lockstep（用户 2026-09-28；DECISIONS 同日）

- [x] 9.1 先提交三批未提交的改动（`291cdeb`，见 8.12）（2026-09-28）
- [x] 9.2 代码：`paths.sh` 导出 `PYTHONPATH` 并读 `$LOCOMANI_WORKSPACE/locomani.local.sh`；`scripts/build_scene.sh <场景名> [--isaac]` 读 `configs/scenes.yaml` 的 `build`（`6783a9f`）；`eval.client` 默认 `lockstep` + `latency_frames` 2，bottle 评测栈按 client 起 lockstep（`4b8180b`）（2026-09-28）
- [ ] 9.3 bottle 的 lockstep 评测和墙钟对照：下一个 bottle checkpoint 出来后，同一 checkpoint 墙钟 vs lockstep 各 100 局。单臂的 lockstep 实现 2026-09-02 校准未过（46 vs 32），之后没重测
  - [ ] 9.3.1 如果仍不过：把单臂评测改用双臂那套已校准的 lockstep 实现（`tasks/crate/vla_lockstep.py`）
- [x] 9.4 实验文件：`bottle_simple_smoke` / `bottle_simple_200` / `crate_table30_record` 改 lockstep；新建 `crate_smoke`、`warehouse_smoke`、`warehouse_200`（`6712067`）（2026-09-28）
- [x] 9.5 三条线的冒烟按新 runbook 真跑一遍，把用时、产物、日志字样填进 runbook（2026-09-28；记录 `outputs/docs_smoke_20260928/`）：bottle 12 分钟、crate 8 分钟、warehouse 40 分钟（含 14 分钟评测录像），三条闭环都是 lockstep 0/2
  - [x] 9.5.1 冒烟抓出并修掉的问题：小数据集的视频对齐检查误判（`check_dataset`，`ffc3806`）；bottle 策略端口 5590 撞实例 1 的端口（改 5710，`ffc3806`）；单臂评测的 encoder 检查看全机的 deploy（`d290a36`）；评测结束后每局重启过的 deploy 成了孤儿进程（`Stack.down`，`a29924b`）；`crate_eval` 产物契约从没对过真运行（`7041ef0`）；长局的预览切不进 8 段（上限改 16）
- [x] 9.6 重写 README（文档清单）、三份 runbook（只写命令、做什么、得到什么）、新建 `docs/running.md` 和 `docs/warehouse_design.md`、`setup.md` 改成"命令 + 做什么 + 验证"且命令逐条不变、`docs/README.md` 改为历史与开发者索引、规则移到 `docs/agents/doc-ownership.md`、`ROADMAP.md` 归档、`reference/runbook.md` 并入 `running.md`；失效链接全修（2026-09-28）
- [x] 9.7 推送（用户 2026-09-28）：merge 了 henry-zhao 的两个提交（`--backbone-path` 移到 `running.md`），`4a3c0f3` 已推到 origin/main
- [ ] 9.8 下一轮（等用户认可新 runbook 的写法）：详解页、`rendering.md`、`scalebfm.md` 按同样的写法重写
- [x] 9.9 （2026-09-29 用户说提前删，已删：三个 `outputs/runs/*_smoke/model/` 和 `try2/run_bottle_simple_smoke/model/`，共 96 GB；try2 里其余约 0.3 GB 的日志、数据集留着作证据；清单、脚本、日志在 `outputs/cleanup_20260929/`；`/workspace` 空闲 77 → 170 GB）删冒烟的大产物：`/workspace` 只剩 26 GB（98%）。`outputs/runs/{bottle_simple_smoke,crate_smoke,warehouse_smoke}/model/`（各约 24 GB，10 步的 checkpoint，没有用处）和 `outputs/docs_smoke_20260928/try2/`（24 GB，第一次失败的 bottle 冒烟）。按 AGENTS 规则 2026-09-29 11:45 之后才能删，用户说可以就提前删；运行目录其余文件（manifest、日志、录像）保留作 runbook 数字的证据
  - [x] 9.9.1 （用户 2026-09-29 说"都删掉吧"，已删 69 GB，`/workspace` 空闲 170 → 239 GB；清单 `B_delete_manifest.txt`、脚本、日志在 `outputs/cleanup_20260929/`）B 类：① 65 个扫参和对照运行（`box_envelope_s1/s2_*`、`box_centre_*`、`box_yaw*`、`box_lat*`、`box_range_quick_*`、`box_room_grasp_10ep_20260925`）的 `observations/`，31.7 GB，manifest、trace、视频、日志留着；② `box_room_grasp_200_cap120_20260925/observations`，17.2 GB——这 200 条不能再补渲，要用新外观或 Unreal 重训就得重采，`train_dataset` 和 `checkpoint-5000` 留着；③ `third_party/EmbodiedGen/weights/sam-3d-objects`，12.3 GB，要用时重新下载；④ `datasets/` 旧专家数据 8 GB，只留 `g1_locomani_cover400_train/meta/`（`rtc/transition_events.py` 读它的 `info.json`）
- [ ] 9.10 `crate_table30_record`（crate 的完整规模）还没在 lockstep 下跑过；runbook 里它的数字是 09-15 墙钟 71/100
- [x] 9.11 （2026-09-30，`34fcd13`，用户问答定）三个 runbook 只留命令、做什么一句、得到什么一句、每步总耗时；移出的内容进详解页（`warehouse_design.md` 第 10 节）；行数上限降到实际加约 10%（warehouse 215、bimanual 125、single_arm 115）
  - [ ] 9.11.1 可选：隔离环境里的新读者照新 warehouse.md 跑一遍（这次没做，只做了 `--dry-run` 和改动部分的实跑）
- [x] 9.12 （用户 2026-10-13 批准报告里的"一"和"二"；STATE 10-13 15:34 UTC 写明，最早 2026-10-14 16:00 UTC 执行；10-14 03:30 UTC 用户说不用等，已执行，28 → 150 GB，记录 `outputs/cleanup_20261013/apply.log`）删约 120 GiB：`/workspace` 剩 40 GB（97%）。三个 10 步冒烟检查点 `outputs/runs/{warehouse_smoke_20260930,warehouse_smoke_unreal_20260930,warehouse_film_unreal_20260929}/model`（72 GiB）；两个 deploy worktree `outputs/deploy_{docbuild_20260924,pinned_20260925}` 和 `.claude/worktrees/` 四个已合并的 worktree（先存 diff）；10-08 及以前 run 和 `outputs/fresh_repro_20260925` 的 ≥ 1 MB 文件（约 48 GB；不动 model / train_dataset / val_dataset / skills / grasp_planned / manifest，代码按名引用的 run 整个跳过）；根目录 wandb、缓存。清单、规则、脚本在 `outputs/cleanup_20261013/`（`summary.md`、`apply.sh`）
- [ ] 9.13 可选（2026-10-14 提议，用户没定）：SONIC deploy 每次启动写单独的日志，不再往 `/workspace/tmp/p03/deploy.log` 追加（10-14 已涨到 2.3 GB，截到 20 MB）


## 10. warehouse 默认采集改为四边随机；采集前打印设置；固定起点作为选项（用户 2026-09-28；DECISIONS 同日）

- [x] 10.1 `warehouse_smoke` / `warehouse_200` 改为四边线 compact 版（场景 `warehouse_box_45x32x33`）；新建 `warehouse_{smoke,200}_fixed_start`（只改摆放）（2026-09-28）
- [x] 10.2 `collect.py` 在第一局前打印摆放设置（`expert.layout_lines`），终端里等 10 秒；`--dry-run` 也打印；测试在 `test_four_edge_placement.py`（2026-09-28）
- [x] 10.3 文档：`warehouse.md` §1–2、§9–10，`warehouse_design.md` §1、§4、§7，`box_grasp_envelope.md` §4，`rendering.md` 场景清单，README 表（2026-09-28）
- [x] 10.3.1 实跑核对（2026-09-28）：`warehouse_smoke` 以运行名 `warehouse_smoke_edges_20260928` 跑采集，第一局前打印的设置与 dry-run 相同；4 局抽到 yp、yp、xm、xm，保留 2 条（另两局 `lost_attachment`、`reach_stalled`），栈正常关闭；录像与 ≤500 KB 预览在该运行的 `videos/`、`previews/`。非终端运行（nohup），所以 10 秒等待这条分支没有实跑
- [ ] 10.4 本机的 `outputs/runs/warehouse_smoke` 是按旧配置跑的，照 runbook 用同名重跑会报 `config.yaml differs`。待用户定：删掉旧运行目录（9.9 已计划删它的 `model/`），还是在 runbook 里写明换 `--name`
- [ ] 10.5 `warehouse_200` 在四边随机下还没有按完整规模跑过（按 17/40 估计，约 470 局保留 200 条）
- [x] 10.6 提交并推送：`4dd5649`（与第 11 节同一个提交；推送时一并推了 `7135a83`）（2026-09-28）
- [ ] 10.7 待验证：终端里“打印后等 10 秒”这条分支没实跑过（两次实跑都是 nohup，非终端）；下次在终端里跑采集时看一眼

## 11. 双臂线采集默认并行（通道）；开跑前判断资源、打印通道数（用户 2026-09-28；DECISIONS 同日）

- [x] 11.1 实测一条 warehouse 通道的成本（2026-09-28，`outputs/warehouse_parallel_20260928`）：CPU 平均 +2.6 核、90 分位 +3.7 核，显存 +1.5 GB，线程数 +180 起、每录一局约 +130，4 局一批峰值 +715；RTF 0.993
- [x] 11.2 `pipeline/lanes.py`（计划、每项资源允许几条、`lanes.json`）、`stack.instance_busy`（栈、锁、端口被占）、`run_stack` 起多个栈、`collect.py` 每条通道一个进程、`batches` 按编号领取批次（文件锁）；扫参格子固定一条；`run_overnight.sh` 支持 `--collect-gpu` / `--lanes` 并在失败时关掉所有通道的栈（2026-09-28）
- [x] 11.2.1 实跑核对（2026-09-28）：`warehouse_smoke` 以运行名 `warehouse_smoke_lanes_20260928` 按 runbook 两条命令（不带 `--gpu`）跑，计划 2 条通道（实例 2 / GPU 0、实例 3 / GPU 1；实例 0 的端口被 Unreal 编辑器占着、实例 1 是 loco_agent 的栈，都被跳过），两批同时跑，8 局保留 5 条，每局 RTF 0.974–0.996，容器线程峰值 3279/3840；随后 Isaac 补渲 5 集、convert 4 训练 + 1 验证通过，两条通道的 session 与 manifest 配对正确
- [x] 11.3 测试 `tests/test_collection_lanes.py`（计算、分卡、领批、续跑、session 接管、端口占用）（2026-09-28）
- [x] 11.4 文档：`running.md` §6 重写，`warehouse.md` / `bimanual.md` / `box_grasp_envelope.md` 的采集命令去掉 `--gpu 0`，术语表加“通道”，change map（2026-09-28）
- [ ] 11.5 单臂线（bottle）的并行采集：驱动器不同，本次没做
- [ ] 11.6 实时渲染的通道（crate）成本没有实测，沿用旧单栈预算（10 核、6000 MiB、一张卡一条）；这台机器上 crate 目前只能开一条
- [ ] 11.7 两条以上通道的产出率是否与单条相同，只跑了一次冒烟（8 局保留 5 条，RTF 0.974–0.996）；完整规模下要按产出率和 RTF 再核一次
- [x] 11.8 提交并推送：`4dd5649`（2026-09-28）
- [ ] 11.9 本会话留下的小产物（共约 1 GB，可随 9.9 一起删，需先在 STATE 写明 24 小时）：`outputs/runs/{warehouse_smoke_edges,warehouse_smoke_lanes,warehouse_lane_cost,warehouse_lane_cost_i3,warehouse_lane_cost_i3b}_20260928`、`outputs/warehouse_parallel_20260928`（测量脚本与采样）、`outputs/warehouse_edges_default_20260928`；exporter session 在 `/workspace/tmp/p03_i2`、`p03_i3`（1.2 GB）。`warehouse_smoke_lanes_20260928` 的 5 集是并行通道的证据，删前留 manifest 与 `collect/`
- [ ] 11.10 旧资源闸门 `run_context.NEEDS["collect"]`（1800 线程、8 核）仍用于单臂线和 `--lanes` 之前起的栈；是否按 11.1 的实测改小，待定（单臂 exporter 没量过）

## 12. Unreal（URLab）替代 Isaac 渲染（分支 `loco_unreal`，2026-09-29 合入 main；用户 2026-09-28 定；DECISIONS 2026-09-28 URLab 一节）

- [x] 12.1 可行性（只替代 Isaac 角色）+ 安装 UE 5.7.4 到 `/workspace/ue_57`（2026-09-28）
  - [x] 12.1.1 `/workspace` A 类清理（+47 GB）、UE 压缩包校验后删除（2026-09-28）
- [x] 12.2 冒烟：headless 启动、导入 G1、puppet 回放（2026-09-28）
  - [x] 12.2.1 插件补丁 `include_cameras: "fresh"`：0/50 旧姿态、单相机 54–67 ms（`patches/urlab_fresh_capture.*`，2026-09-28）
  - [x] 12.2.2 机器人头部白带 / 白盘：近裁剪 1 mm → 2 cm 后 warehouse 头部相机不再出现（2026-09-28；纯机器人关卡未重测）
- [x] 12.3 warehouse 关卡：Blender 烘焙 → glTF → UE，6 盏 RectLight、固定曝光、强制 Lumen（2026-09-28）
  - [x] 12.3.1 锯齿：非 2 的幂贴图缩放 + 截图开 TSR + camera cut（2026-09-28）
  - [x] 12.3.2 （用户 2026-09-29 要求修；2026-09-29 已修：Interchange 的 glTF 母材质在截图里不渲自发光，关卡脚本 `emissive_materials` 换成自建 `M_LocoEmissive`，灯板 167 → 230，Blender 238；台账 blender_scenes.md 第 7 节）天花板灯板在 Unreal 里是暗的（Isaac 会发光），查自发光材质
  - [x] 12.3.3 （用户 2026-09-29 要求修；2026-09-29：曝光按物理推算的 −9.40 不用改，墙和地面与 Blender 差 0–2；天花板差在反弹光，关卡体积加 `lumen_diffuse_color_boost` 4，天花板 90 → 114，平均绝对差 23.1 → 17.4；台账第 7 节）对照 Cycles 参考图标定曝光和色调（`WAREHOUSE_EXPOSURE_OFFSET`；AgX 与 UE filmic 不同）
- [x] 12.4 G1 外观：`mujoco` 预设（MJCF 原色，sRGB→linear，非金属），用户确认（2026-09-28）
  - [-] 12.4.1 `metal` 预设（统一反光银色）：用户否决，不像真实 G1
  - [-] 12.4.2 可选：外壳粗糙度调到 0.2–0.25 或压暗底色，靠近 MuJoCo 的高光感——被 12.4.3 取代
  - [x] 12.4.3 改用 Unitree 官方 Isaac 资产（`unitree_sim_isaaclab_usds` G1 + Dex1）的材质：拉丝金属外壳、近黑的头/关节/夹爪；Isaac `stage.G1_LOOKS` 与 Unreal `unitree` 预设同值，另修 Isaac 的 Dex1 手指落到近白默认色（用户 2026-09-29，DECISIONS 2026-09-29）
  - [x] 12.4.4 warehouse 曝光（Isaac ISO 30、Unreal 曝光偏置）按新外观复核（2026-09-29：Isaac 见 .1/.2/.4，Unreal 见 12.3.3）
    - [x] 12.4.4.1 Isaac 曝光改按 Blender 标：现在的 ISO 30 是按 MuJoCo 躯干标的，整体比 Blender 暗（地面 39 对 80）；ISO 60–80 最接近（平均绝对差 59 → 17–24）。改了会改变策略看到的图像分布，要和 12.4.5 重训一起定——用户定改 ISO 80（2026-09-29，`configs/scenes.yaml`；台账 blender_scenes.md 第 4 节）
    - [x] 12.4.4.2 `build_isaac_warehouse_scene.py` 关 `env_light` 的理由（EXR）已过时，USDC 已修成纯色 dome；实测影响 0.1——用户定打开，新 stage `warehouse_textured_env`（2026-09-29）
    - [x] 12.4.4.4 天花板偏暗：间接光 ×2 + Hejl–Hable 色调映射，天花板 151 → 176（Blender 183），平均绝对差 11.1 → 6.9，耗时不变（2026-09-29，台账 blender_scenes.md 第 6 节）
    - [-] 12.4.4.5 把 Unreal 的烘焙贴图搬进 Isaac（放弃：640×480 下看不出程序化纹理的缺失，Isaac 多的是锯齿，归 12.4.4.3；2026-09-29）
    - [x] 12.4.4.3 （用户 2026-09-29 定：2× 超采样；已落地：`scenes.yaml` 的 `isaac.supersample: 2`（warehouse_textured、box_platform），镜像、数据集补渲、视频三处统一读，`stage.downsample`；实测代价 2.2 倍而非估的 1.4 倍，三路相机 95 → 209 ms；`outputs/isaac_ssaa_20260929/`）Isaac 抗锯齿：现在只有 FXAA（`/rtx/post/aa/op = 2`，1 spp），小字糊、货架网格闪。对 Blender 的结构相似度：FXAA 0.94，2× 超采样 + FXAA 0.96，TAA 0.96（但只在静止相机上攒了 24 帧），DLSS/DLAA 这样切换出来是噪点。建议 2× 超采样（逐帧确定，镜像和数据集同图），代价约 1.4×（待用户定；`outputs/isaac_blender_match_20260929/aa_crops.jpg`，2026-09-29）
  - [-] 12.4.5 新外观下重渲数据集并重训；旧 Isaac checkpoint 看的是发白的旧外观（放弃：用户 2026-09-29 定新外观不重训）
- [x] 12.5 一局闭环 VLA 用 Unreal 图（seed 6000，lockstep，1/1 ok）+ ego/腕部 MuJoCo 对照（2026-09-28）
  - [x] 12.5.1 ego/腕部对照视频一顿一顿（镜像只在策略取图时渲，约 1.4 张/秒）：改由 state trace 按仿真时间补渲，25 fps，1477 帧无重复帧（`vla_round/dataset_cams/`，用户 2026-09-28）
- [x] 12.6 同一局第三视角 Unreal / Isaac / MuJoCo 三路视频（`reverse_room`，2026-09-28）
- [~] 12.7 取图速度：3 相机截 8 张每次 679 ms（Isaac 95 ms）；扫 `urlab.FreshCaptureRepeats`（2–8）与画质的关系
  - [x] 12.7.1 剖析（2026-09-28，`outputs/unreal_render_speed_20260928/`）：每次 capture 约 23 ms，其中阴影 14 ms（6 盏 RectLight 各重画 1024² Nanite 立方体阴影，不缓存）；关灯光阴影 4.2 ms 但接触阴影全没；编辑器视口每 tick 自己渲一遍（关掉 fixed 等待少约 35 ms）
  - [x] 12.7.2 插件 `"stream"` 模式（连续序列：同 tick 取图、1 遍、不 camera cut）+ 三相机先渲后统一回读；trace 视频 642 → 139 ms/帧，与 8 遍差 2.1–2.2（同图两次 0.8–1.1、单遍 3.5），无拖影（`vla_round/dataset_cams_stream/`，2026-09-28）
  - [-] 12.7.3 虚拟阴影贴图（VSM）：开了第一次按需取图就 GPU 卡死，复现（2026-09-28）；运行时改 `r.ShadowQuality` 崩编辑器
  - [x] 12.7.4 阴影降本（待用户定）：只留 1–2 盏灯投影 / 机器人网格改非 Nanite / 硬件光追阴影（要开 `r.RayTracing` 重编着色器）——由 12.7.4.4/12.7.4.5 解决（无 Nanite 默认，阴影 14.3 → 0.29 ms，2026-09-28）；少灯投影 / 光追 / 自建 view family 不再需要，理由见 DECISIONS 2026-09-28 Unreal 出图提速
    - [x] 12.7.4.1 机器人非 Nanite（`unreal_no_nanite_20260928.py` SCOPE robot，原名 `unreal_robot_nanite_20260928.py`）：每次 capture 23–25 ms，同会话基线 22 ms，无收益；画面差 0.95–1.17（噪声内）。阴影 13.5 ms 里机器人只占 0.3 ms，13.3 ms 是房间静态 Nanite 几何的 36 个立方体面。根因：引擎 `ShadowSetup.cpp:4333` 只要有 scene capture 视图就不用阴影缓存（`!bAnyViewIsSceneCapture`），每次 capture 都整场重画（2026-09-28，`outputs/unreal_render_speed_20260928/g_robot_non_nanite/`）
    - [x] 12.7.4.2 方案 3（自建 view family 不走 scene capture）的前提检查：普通视图也不用阴影缓存——编辑器自己的视口，停掉 PIE、场景完全不动，6 个立方体阴影仍每帧重画（14.3 ms）；`r.Shadow.WholeSceneShadowCacheMb` 150→600 无变化；RHI 条件满足（VK_EXT_shader_viewport_index_layer 已加载）。原因未查明；同日较早一次空闲帧 3.4 ms 且无阴影 pass，是线索（2026-09-28）（结论有误，更正见 12.7.4.3）
    - [x] 12.7.4.3 查普通视图不缓存阴影的原因（限时约 1 h）；决定方案 3 能拿到的是「缓存 + 三相机共享」还是只有「三相机共享」（待用户定）——原因（2026-09-28）：普通视图其实用了缓存（profile 里是 `WholeScene MovablePrimitives`，每盏灯 0.04 ms），但引擎对每盏投影灯的 6 个面都跑一遍 Nanite 阴影管线（`ShadowDepthRendering.cpp` 1753–1865 行，只按 mobility 隐藏、不跳过），每面固定约 0.37 ms，6 盏灯合计 13.3 ms，与画没画东西无关。所以方案 3 的缓存省不掉这 13 ms，只剩三相机共享
    - [x] 12.7.4.4 阴影不走 Nanite：房间 + 纸箱 + 机器人全部非 Nanite，启动时 `r.Shadow.Nanite=0`，测每次 capture 耗时，并做同姿态画面 A/B（待用户定）——结果（2026-09-28，`outputs/unreal_render_speed_20260928/i_nanite_shadows/`）：阴影 14.3 → 0.29 ms，GPU 帧 18.3 → 3.8 ms；每次 capture 23.9 → 3.9 ms；三相机 fresh×8 640 → 138 ms，fresh×1 146 → 48 ms，stream 117 → 35 ms。画面与现状差 1.35–1.47（38 dB；同设置重复 0.39–0.48），只在边缘（地砖缝、箱边、字）；阴影形状一致。只设 `r.Shadow.Nanite 0`、网格仍 Nanite 不可用：Nanite 网格不投影，差 14–17
    - [x] 12.7.4.5 是否把「全部网格非 Nanite + `r.Shadow.Nanite=0`」设为默认：DefaultEngine.ini 加 cvar；房间资产存盘为非 Nanite 或每次开编辑器跑 `SCOPE=all`；之后的 Unreal 图（评测、补渲）都用它（用户 2026-09-28 定为默认）——已落地：`DefaultEngine.ini` 加 `r.Shadow.Nanite=0`；房间网格存盘为非 Nanite（备份 `/workspace/ue_57/backups/`），关卡脚本导入时即关 Nanite；每次导入后跑 `unreal_no_nanite_20260928.py`（原 `unreal_robot_nanite`，默认 SCOPE all）；镜像和 trace 视频启动时检查，有 Nanite 网格或 cvar 不为 0 就拒绝。两次全新会话验证：三相机 fresh×8 124 ms、stream 35 ms，与 allnn 同图（跨会话差 0.65–0.73）（`k_default_check/`）
  - [x] 12.7.5 待验证：`ShowFlag.Rendering 0` 关视口后取图差 1.1–1.3，略高于同图噪声 0.8–1.1；看图确认再决定是否默认关——已验证（2026-09-28，`outputs/unreal_render_speed_20260928/h_viewport_off_ab/`，12 个姿态 × ON/OFF/ON/OFF）：ON 与 OFF 差 0.74–1.30（最大 3.4），同设置重复只差 0.26–0.36；锐度不变（不是贴图 mip）；OFF 时间接光区域（箱内、阴影里的地面）偏暗约 2/255，直射高光不变，像 Lumen 全局光照收敛少了（机制未验证）
  - [x] 12.7.6 （用户 2026-09-29：对画质没影响就关；已关：`ShowFlag.*` 只作用于编辑器 / 游戏视口，不作用于截图；作弊类 cvar，工程 ini 设不了，由 `/workspace/ue_57/ue_cmd_server.py` 启动时设 0，`require_render_defaults` 检查；空闲 GPU 28 → 11 %）是否默认关编辑器视口（省约 35 ms/请求，画面按上条会变）：定下后所有 Unreal 出图统一用同一设置（待用户定）——质量已测（2026-09-28，`j_viewport_quality/`，各自对 64 遍收敛参考）：开 8 遍离参考 1.25，关 8 遍离参考 1.30，收敛质量相同；两个 64 遍参考之间差 0.86（腕部相机 0.9–1.1、头部 0.5），小于 8 遍本身离收敛的 1.25。全非 Nanite 下关视口不再省时间（131 vs 131 ms），Nanite 下省 20–35 ms；建议保持开（无 Nanite 下关了不省时间），用户未明确表态
  - [x] 12.7.7 （用户 2026-09-29 问：为什么不是渲一遍就够；已测：对 64 遍参考 1 遍 3.4–3.9 且整体偏暗 2–3、8 遍 1.2–1.4；1 遍 56–64 ms、8 遍 137 ms；handoff 文档"截图遍数"表，`outputs/unreal_repeats_20260929/`）无 Nanite 默认下扫
    - [x] 12.7.7.1 闭环和数据集用几遍（现默认 8）：用户 2026-09-29 定 8 遍（宿主工程 `urlab.FreshCaptureRepeats=8`，不改） `urlab.FreshCaptureRepeats`（1/2/4/8）：各对 64 遍参考（`unreal_pose_set_20260928.py`），定闭环遍数（1 遍 48 ms、8 遍 124 ms、Isaac 88–96）
- [ ] 12.8 用 Unreal 图跑 100 局 lockstep 评测，与 Isaac 88/100 比（checkpoint 是 Isaac 训练的，属于分布偏移）；用无 Nanite 默认（已跑的 1 局是 Nanite 版）
- [~] 12.9 正式接入 `images.source: unreal`：镜像进 src、run_overnight 接线、数据集用 Unreal 补渲、重新训练（用户 2026-09-29 定，main 会话转交本会话；做法与文档形式见 DECISIONS 2026-09-29「images.source: unreal」）
  - 用户 2026-09-29 定：由 loco_unreal 会话做（这个会话已发任务说明）。做法是 `images.source` 加第三个值 `unreal`，采集、转换、训练、评测命令都不变；`scenes.yaml` 给场景加 `unreal` 条目（与 `isaac` 并列）；文档不加 README 主线：`rendering.md` 图像来源表加一列、`setup.md` 加可选安装步骤、warehouse runbook 只加一句怎么改用
  - [x] 12.9.1 开关与校验：`images.source` 第三个值 `unreal`（`run_context.IMAGE_SOURCES`，客户端侧 `sim.mirror.MIRRORED_SOURCES`）；`scenes.yaml` 的 `unreal` 条目（关卡、`repeats`、build），无关卡的场景写 `unsupported`，加载实验文件时拒绝；`run_context.unreal_scene` 查关卡已建、且由本场景建（`source.json` 哈希）；也必须 `backend.render: offline`（2026-09-29）
  - [x] 12.9.2 代码进 src：`src/locomani/sim/unreal/`（editor、view、mirror、render_dataset_cameras、export_mjcf、build_level、in_editor/）；镜像协议提成 `sim/mirror.py`、按视图补渲提成 `sim/view_render.py`，Isaac 和 Unreal 共用；convert 的渲染器核对认 unreal（2026-09-29）
  - [x] 12.9.3 脚本：`start_unreal.sh` / `stop_unreal.sh`（看守 `scripts/unreal/editor_guard.sh`）、`run_unreal_mirror.sh`、`render_dataset_cameras_unreal.sh`、`build_scene.sh --unreal`、`setup_unreal.sh`（宿主工程模板 `scripts/unreal/host/`）；`run_overnight.sh --render-gpu`（旧名 `--isaac-gpu` 仍可用）按 images.source 选渲染器（2026-09-29）
  - [x] 12.9.4 关卡：warehouse_textured 与四个 warehouse_box_* 各建一个（房间关卡 save-as + 该场景导入），逐场景对 MuJoCo 出图核对，纸箱尺寸与位置一致，每次 121–148 ms（`outputs/unreal_images_source_20260929/`；2026-09-29）
  - [x] 12.9.5 端到端：warehouse_smoke_edges_20260928 的测试副本 Unreal 补渲 2 集（每集约 80 s，p50 139 ms）→ convert 通过；把来源改成 isaac 后 convert 拒绝；`run_overnight.sh --from eval` 3 局 lockstep 闭环 3/3（Isaac 训练的 checkpoint-5000，分布偏移；镜像 330 次 p50 137 ms，327 次观测的图像 seq 全等于边界 seq）（2026-09-29）
  - [x] 12.9.7 从零安装：`setup_unreal.sh` 装进一个空的 UNREAL_ROOT（引擎用本机已解压的，下载要 Epic 登录）8.5 分钟，起编辑器、建房间关卡和 warehouse_textured 关卡、出图核对都通过，出图与本机安装差 1.2–1.5（JPEG）。中途修了两处：URLab 导入要的网格预处理 Python 包与 `LocalUnrealRoboticsLab.ini`、重建房间时编辑器正开着它会崩（2026-09-29）
  - [x] 12.9.8 main 审核 30cb422 的阻塞项：URLab 原版端口 5555–5559（相机从 5558 往上）与实例 0 撞车。插件补丁加 `PortBase`（默认 5800，插件 ini 可改），所有端口由它派生；客户端 `editor.port_base` 读同一 ini；setup 写入 ini。实测：编辑器常驻、PIE 时只占 5800–5806 和 18801，实例 0 的 `run_overnight --from eval` 起栈、镜像在 :5558 服务，无端口冲突；2 局跑完 1/2（第 2 局伸手后没合爪，策略行为，Isaac checkpoint 分布偏移），250 次观测图像 seq 全对，出图 p50 142 ms（`outputs/runs/unreal_instance0_eval_20260929`；2026-09-29）
  - [x] 12.9.9 审核的非阻塞项：`start_unreal.sh` 查编辑器实际在哪张卡（nvidia-smi 按 pid 找显存最大的卡），已在跑的编辑器卡号不符就报错；`--gpu` 是 Vulkan 序号，本机与 nvidia-smi 序号一致（`--gpu 1` 时主要上下文 1.5 GB 在 GPU1）；`view.qpos_map` 改为精确匹配优先、否则取最长的后缀匹配（`tests/test_unreal_view.py`）（2026-09-29）
  - [x] 12.9.10 读者文档按规范补齐（用户 2026-09-29 问"有 follow 之前的规范吗"）：`running.md` 参数表和另外三处旧名 `--isaac-gpu` 改成 `--render-gpu`；warehouse runbook 的 Unreal 从"只有一张卡"里的一句话拆成第 10 节"改用 Unreal 出图"（代码块、做什么、得到什么），常见报错加两行，补渲和录像耗时标成估计；setup 第 8 步按第 7 步的格式写（验证块、Blender 烘焙和建房间关卡两条命令），安装排错加"Unreal 渲染环境"一节；术语表的镜像条目改成两种渲染器，新增 Unreal / URLab、编辑器、关卡、`$BRIDGE_PY`。页面上限 warehouse 150 → 170、setup 420 → 440（`scripts/lint_repository.py`），不为凑上限压缩。无上下文的子代理按新读者读过一遍，命令、参数、日志原文都与代码核对一致，它指出的 5 处卡点和 3 处写法已改
  - [x] 12.9.11 （2026-09-29 已改：默认 `$UNREAL_ROOT/levels`，paths.sh / paths.py 一致，有测试 `tests/test_unreal_paths.py`；本机 5 个目录挪到 `/workspace/ue_57/levels/`，用主 checkout 的场景核对 5 个关卡都通过；setup.md 排错那条已改）`$UNREAL_ASSETS` 默认在检出目录下（`assets/unreal/`），关卡却在全机共用的宿主工程里：关卡是 loco_unreal worktree 建的，主 checkout 的 `assets/unreal/` 为空，从主 checkout 起 `images.source: unreal` 会报 `source.json` 缺失（5 个场景的哈希与主 checkout 的场景一致）。建议默认改到 `$UNREAL_ROOT` 下；已转给 loco_unreal 会话
  - [x] 12.9.12 warehouse runbook 里 Isaac 和 Unreal 平行（用户 2026-09-29：Unreal 不该是排在第 10 节的附加功能，读者一开始就选渲染器）：开头加"先选渲染器"对照表；第 1、3 步和第 6 步终端 2 两种命令并排；原第 10 节删掉。新增 `warehouse_smoke_unreal.yaml`、`warehouse_200_unreal.yaml`，与默认只差 `images.source` 和 `name`（`tests/test_image_source.py`）；页面上限 170 → 195；两份文件加载、Unreal 前置检查、找到关卡都通过
  - [ ] 12.9.13 按 setup 第 8 步和 warehouse runbook 的 Unreal 列，从零实跑一次冒烟（无上下文的读者照文档做）；目前只有 loco_unreal 会话在本机从 `setup_unreal.sh` 起验证过，引擎下载、Unreal 数据的完整采集到评测没有照文档跑过
  - [x] 12.9.14 （2026-09-29 用户定：按渲染器统一）待用户定：Unreal 运行的第三视角观看录像仍由 Isaac 渲（`render_isaac_videos.sh`）；Unreal 的 trace 视频还是实验脚本 `scripts/experiments/unreal_trace_video_20260928.py`，要不要接成正式脚本
    - [x] 12.9.14.1 （2026-09-29 用户：采集导出的第三视角录像要和数据集相机同一个渲染器）补渲脚本接着用同一个 Isaac / Unreal 画每个成功局的第三视角录像到 `videos_<渲染器>/`，机位 = MuJoCo 录像的 `round_camera`，带 ghost，小窗取补渲好的数据集帧；convert 只从那里取、缺了拒绝。实跑：Isaac 8 局（warehouse_5ep_20260929）、Unreal 2 局（warehouse_film_unreal_20260929），均已 convert；未提交
    - [x] 12.9.14.2 （2026-09-29 用户：评测录像也按渲染器统一）`render_eval_videos_{isaac,unreal}.sh` → `view_render.film_eval`：先渲每局三路相机做小窗，再拍第三视角，到 `eval/videos_<渲染器>/` 与 `eval/previews_<渲染器>/`；runbook 第 6 步改用它们（`render_isaac_videos.sh` 留作 MuJoCo|Isaac 对照）
    - [~] 12.9.14.3 （2026-09-30 关卡部分已做：另 4 个关卡已重建，逐个验证相机图与第三视角；机位部分未做）Unreal 关卡只重建了 `warehouse_box_45x32x33`，另 4 个关卡渲第三视角时会报 rebuild；MuJoCo 实时录像（`SIM_VIDEO_DIR`，`mujoco` 运行的评测录像）四边线仍用 `video_camera`，不是每局的边机位
    - [-] 12.9.14.4 （2026-09-30 放弃：用户说协作者要的是一条独立命令，不是实验文件开关；见 12.9.14.6）（2026-09-30 用户：第三视角可选）实验文件 `images.third_person`（默认 true，`run_context.third_person`）：false 时补渲不画、convert 不要求、`mujoco` 运行的仿真不实时录；六个现行 warehouse 实验文件写明 true
    - [x] 12.9.14.5 （2026-09-30 两个渲染器整条实跑完，runbook 的规模和耗时按实测改写；计时 `outputs/warehouse_smoke_timing_20260930/timing.tsv`；另发现冒烟 20 局可能凑不够 10 条，Unreal 那次 8/10，`run_overnight.sh` 会停在采集；用户 2026-09-30 定 `max_batches` 2 → 3，三份冒烟文件一起改）（2026-09-30 已采 `warehouse_smoke_20260930` 11/20 条、Isaac 补渲 2/11 时用户叫停，等第三视角改完再接着跑；队列 `outputs/warehouse_smoke_timing_20260930/queue.sh`）`docs/warehouse.md` 的冒烟规模文字仍是旧的（2 条、每批 4 局），`warehouse_smoke.yaml` 已是 10 条、每批 10 局、最多 2 批（22c1fda），耗时估计要按新规模重测
    - [x] 12.9.14.6 （2026-09-30 用户：协作者要一条命令，跑了就独立渲第三视角）`scripts/render_third_person.sh <run> [--gpu N] [--eval] [--all] [--dry-run] [局名]` 按 `images.source` 选 venv（MuJoCo/Isaac/Unreal）→ `sim/render_third_person.py` → `view_render.film_rounds`；补渲不再画、convert 不再要求（有就链接）；`images.third_person` 撤掉（旧运行副本里的键读取时丢弃）；`render_eval_videos_*` 并入 `--eval`；新增 `sim/mujoco_view.py`。三种渲染器各实跑验证
    - [x] 12.9.14.7 （2026-09-30，`34fcd13`）warehouse runbook 每一步 Unreal 与 Isaac 平行：第 0 步 `RUN=`，共用命令写 `$RUN`；第 7 节只留 `render_third_person.sh`（两个渲染器同一条）和 MuJoCo 快速回放；新增 `--side-by-side` 写 `<局>_vs_mujoco.mp4`（左 MuJoCo 右渲染器，同机位逐帧对齐），两份冒烟实跑；`render_isaac_videos.sh` 移出 runbook
  - [ ] 12.9.6 用 Unreal 补渲的数据集重新训练，再用 Unreal 评测（训练集与评测同一渲染器；待用户安排）
- [x] 12.10 （2026-09-29 用户说提交，已随 12.14 一起提交）提交分支代码：12 个 `scripts/experiments/unreal_*` 和 `patches/urlab_fresh_capture.*`（等用户说）
- [x] 12.11 （用户 2026-09-29 要求修；已修：插件 `ImportXmlSync` 只在内存里建蓝图，`unreal_smoke_20260928.py setup` 现在把它存盘，`EditorStartupMap` 设为 warehouse；重启后不重新导入的图与刚导入时差 0.5–0.7，在跨会话噪声内）URLab 导入的蓝图不写盘，每次开编辑器都要重新 clear + setup；查能否保存
- [x] 12.12 （用户 2026-09-29 定：关掉；已关：宿主工程 `DefaultEditorSettings.ini` 的 `bSendUnattendedBugReports=False`、`bSendUsageData=False`，`GetIni` 读回合并值 False，之后两次崩溃都没生成崩溃报告）UE 崩溃时 CrashReportClient 自动把报告上传到 Epic（2026-09-28 编辑器崩溃时已传一份）；是否在宿主工程配置里关掉（待用户定）
- [x] 12.13 （2026-09-29 已提交到插件本地分支 `loco-fresh-capture` 的 `58c8a98`，未推送；与补丁逐字节一致）插件改动（fresh / stream / 统一回读）只在插件仓库工作区里，未提交到插件分支 `loco-fresh-capture`；`patches/urlab_fresh_capture.patch` 是唯一记录（随 12.10 一起定）
- [x] 12.14 （2026-09-29 用户说提交，已提交）提交 2026-09-29 会话的改动（等用户说）：`src/locomani/sim/isaac/stage.py`（G1_LOOKS、按 body 名查颜色）、`configs/scenes.yaml`（warehouse：新 stage + ISO 80 + 间接光 ×2 + Hejl–Hable）、`scripts/build_isaac_warehouse_scene.py`（取自 main 的版本再改 env_light，合并时以它为准）、`scripts/render_isaac_videos.sh` 注释、`scripts/experiments/unreal_robot_materials_20260928.py`（`unitree` 预设）、`tests/test_isaac_robot_colours.py`（新）、`tests/test_run_context.py`、`docs/rendering.md`、`docs/ledger/blender_scenes.md` 第 4–6 节
  - [x] 12.14.1 （已提交；仓库外那几项不进 git）同日第二轮（超采样、box_platform 颜色、Unreal 四项）加的文件：`src/locomani/sim/isaac/{stage,view,mirror,render_dataset_cameras,isaac_render,bench_render}.py`、`src/locomani/pipeline/run_context.py`、`scripts/build_isaac_primitive_scene.py`、`tests/test_isaac_primitive_scene.py`（新）、`tests/test_isaac_view.py`、`docs/warehouse.md`、`docs/handoff/unreal-urlab-20260928.md`、台账第 7–8 节；未跟踪的实验脚本 `unreal_smoke` / `unreal_warehouse_level` 也改了。仓库外（不进 git）：`/workspace/ue_57/ue_cmd_server.py`、宿主工程 `Config/DefaultEngine.ini`（`EditorStartupMap`）和 `Config/DefaultEditorSettings.ini`（崩溃上报）
- [x] 12.15 （用户 2026-09-29 要求修；已修：先把 sRGB 转成线性，新 stage `box_platform_srgb`，旧 `box_platform` 留在共享缓存；测试 `tests/test_isaac_primitive_scene.py`；台账第 8 节）`scripts/build_isaac_primitive_scene.py`（box_platform）同样把 MJCF rgba 直接当线性 `diffuseColor`，方块偏白；未改（2026-09-29 发现）
- [~] 12.16 工作流文档「用 grill-me + handoff 跑完一项工作」：在线版 https://claude.ai/code/artifact/659f6f44-4e6b-43ba-84c4-1bb8cad09618，离线版 `outputs/workflow_doc_20260929/`（2026-09-29）
  - [ ] 12.16.1 （用户 2026-09-29：现在不改）grill 那节用的是按实际决策重写的示范；保留还是换成 09-28 整理文档那次真实的 grill 记录（待用户定，在线文档里有评论）
  - [ ] 12.16.2 （用户 2026-09-29：现在不改）是否挪进仓库 `docs/` 并在 README 文档列表加一行（`outputs/` 被 git 忽略；待用户定）
- [ ] 12.17 URLab 插件段错误（2026-09-29 两次）：客户端 `close()` 切回 live 后 `UURLabZmqPublishTransport::PostStep` 空指针；另一次在前一个客户端不 `close()` 就退出、下一个客户端发 `hello` 时。两次之前都在 PIE 里改过相机后处理（已不这么做），因果未查；镜像和视频脚本结尾都调 `close()`
- [-] 12.18 （放弃：用户 2026-09-29 接受现在的天花板亮度）Unreal 天花板仍比 Blender 暗 60（114 对 174）：Lumen 的反弹只有 Cycles 的一半左右，增益再往上收益很小；可试色调曲线（Isaac 换 Hejl–Hable 有效）或接受（待用户定）
- [x] 12.19 （2026-09-29 完成：第一次合并 4caadb5，第二次 `images.source: unreal` 合并 b2fd13f，之后快进 5c12f26，均已推送）合入 main（分支会话 2026-09-29 请求；DECISIONS 2026-09-29 合并一节）：冲突 9 个文件已解；四边线换新外观、四个 `warehouse_box_<尺寸>_env` stage 已建；`make lint` 通过
  - [x] 12.19.1 （2026-09-29 用户说推送，已推送；origin/main 现为 fc28d1d）推送 origin/main：待用户说
  - [x] 12.19.2 视频是否对 ≥ 1280×960 的尺寸跳过超采样——用户 2026-09-29 定：不跳，视频照用场景的 2×。实测（`outputs/merge_check_loco_unreal_20260929/`，GPU0，`warehouse_smoke_edges_20260928` 的 `visual_crate_s7000_01`，22.5 s 画面、281 帧、带小窗）：整条 5.0 分钟，Isaac 286 s，约 1.0 s/帧；main 台账第 2 节 1× 时 66 s 画面 249 s、约 0.6 s/帧，约 1.7 倍。`docs/warehouse.md` 的录像 7 → 10 分钟、补渲 1 → 3 分钟、总计 40 → 50 分钟是按这组数和 209 ms/次推算的，下次冒烟实跑时核对
  - [x] 12.19.3 （2026-09-30 冒烟规模按新配置实跑，见 12.9.14.5） 下次 warehouse 冒烟实跑时核对 runbook 里按 2× 超采样估计的耗时：Isaac 补渲每局约 3 分钟、评测录像每局约 10 分钟、全程约 50 分钟（页面上标着"估计"，实测后改成实测值）
  - [ ] 12.19.4 待用户定：`outputs/merge_check_loco_unreal_20260929/`（66 MB，合并后 `_env` stage 的实渲核对）是否保留
- [~] 12.20 2026-09-29/30 会话留下的运行与产物（DECISIONS 2026-09-29/30 第三视角一节）
  - [x] 12.20.1 （2026-09-29）warehouse 默认设置采 5 集（`outputs/runs/warehouse_5ep_20260929`，20 局保留 8）；MuJoCo / Isaac / Unreal 三种第三视角、统一机位的三路并排预览（`outputs/videos/warehouse_5ep_20260929/`）
  - [x] 12.20.2 （2026-09-30）5 个 Unreal 关卡按带相机刚体和 ghost 的导入重建，逐个验证（`outputs/unreal_levels_rebuild_20260930/all_levels.jpg`）
  - [ ] 12.20.3 待用户定：删哪些（规矩：删前 STATE 写明 24 小时）。10 步冒烟模型三份各 24 GB：`outputs/runs/{warehouse_smoke_20260930,warehouse_smoke_unreal_20260930,warehouse_film_unreal_20260929}/model`；小的测试副本 `outputs/runs/film_{mujoco_test_20260930,eval_isaac_test_20260929}`（共 0.3 GB）；实验脚本目录 `outputs/{warehouse_5ep_20260929,warehouse_film_unreal_20260929}`（其中 `unreal_third_person.py`、`unreal_tracking_ghost.py` 已被 `render_third_person.sh` 取代）
  - [ ] 12.20.5 待用户定：09-30 下午新增的产物删不删（小）：两份冒烟运行里的 `*_vs_mujoco.mp4` 和预览、Isaac 冒烟多出的一局失败局第三视角 `visual_crate_s7000_00`、`outputs/docs_warehouse_rewrite_20260930/`
  - [ ] 12.20.4 可选：按 `max_batches: 3` 再跑一次冒烟采集，确认 20 局不够时第三批能补齐（第三批约 6 分钟；runbook 的耗时是 2 批时实测的）

## 13. 真机 URDF（saturn，cheng.zhao 2026-09-29 提供；DECISIONS 2026-09-30 回退一节）

- [-] 13.1 （2026-09-30 放弃：用户说协作者给错了 URDF；代码全部回退，改动存 `outputs/g1_urdf_compare_20260930/saturn_urdf_work.patch`）按 `assets/unitree_G1` 生成 saturn 模型、`build_sim_scene.py --robot saturn`、场景 `bottle_saturn`、SONIC + bottle 专家试跑
  - [x] 13.1.1 （2026-09-30）两份模型对比：29 个关节同名同序同轴同限位，腿完全一致；腰 rev 1.0、5010 腕、背包和云台 +0.86 kg、踝/腰力矩 50 → 35（`outputs/g1_urdf_compare_20260930/compare.txt`）
  - [x] 13.1.2 （2026-09-30）试跑：D435 视野 0/10（开局视野门）；saturn 身体 + stock 相机 6/10、不倒；stock 9/10
- [x] 13.2 （2026-10-02，用户定为默认机器人）待协作者：给正确的真机 URDF。来了以后先 `git apply --check` 那份 patch，再跑 `make_dex1_model.py`（它会拿 URDF 自检）
  - [x] 13.2.1 （2026-10-01）henry 推了无背包版 `c966ee1`：`assets/unitee_G1/urdf/g1_comp_new_hip_dex1_1.urdf`（目录名少个 r；robot.yaml 叫 mars）；旧 saturn 文件移到 `assets/unitree_G1_backpack`。MuJoCo 多角度渲染在 `outputs/g1_urdf_render_20261001/`（总重 34.85 kg，stock 35.87；云台上是 D455，D435 固定在脸部）
  - [x] 13.2.2 （2026-10-02，`62ed89f`、`81ef9cd`）用 henry 带腕部相机的 `cb3ae2f`（`assets/unitree_G1_with_wrist_camera`）：saturn patch 改名 mars，`make_dex1_model.py` 写出 `g1_29dof_with_dex1_mars.xml`，自检 FK 3e-7 mm、35.02 kg；7 个 `*_mars` 场景（DECISIONS 2026-10-02）
  - [x] 13.2.3 （2026-10-02，`cb3ae2f` 带了 D405 支架和位姿）mars URDF 没有腕部相机（robot.yaml 有两台 D405 的序列号，URDF 和 meshes 里都没有，calibration 只有 livox→D435）。仿真要加腕部相机，需 henry 给 D405 的安装位姿/外参、支架 CAD 或手腕照片（已向用户提议代写消息，用户未回复）
- [~] 13.3 （2026-10-02 用户：用云台上的 D455，俯仰锁 +4.71° 对齐原 52° 俯角；内参仓库里没有，先用规格值 D455 fovy 65、D405 fovy 58）待用户/协作者确认：云台上的 ego 相机是 D435 还是 D455，以及真机内参（patch 里按 D435 常用值 fx = fy = 615，fovy 42.6）
- [x] 13.4 （2026-10-02 用户：mars 全按 URDF，踝/腰 35、髋 roll 139、腕 13.4；stock 场景不变；同日用户改为腰 50，见 13.6.4）仿真踝关节力矩上限 50 N·m 和真机固件 35 N·m 要不要对齐（改了会动所有标定）
- [ ] 13.5 待用户定：saturn 产物删不删（约 180 MB：`outputs/scene/sim_lowcom_saturn*`、`outputs/runs/bottle_{saturn,saturn_stockcam,stock}_pilot_20260930`；patch 所在的 `outputs/g1_urdf_compare_20260930/` 要留）；删前在 STATE 写明 24 小时

- [~] 13.6 mars 设为 bottle / warehouse 默认机器人（用户 2026-10-02；DECISIONS 同日一节）
  - [x] 13.6.1 （2026-10-02）第 1 步：生成器、7 个 `*_mars` 场景、`scene_meta.json` 记 `robot`、力矩上限从场景执行器读、几何测试样例换 mars；全量测试过
  - [x] 13.6.2 （2026-10-02）第 2 步：配对 10 局试跑（`outputs/mars_pilots_20261002/`）。bottle stock 9/10、mars 9/10（修网格名后重跑）；warehouse stock 6/10、mars 0/10
  - [x] 13.6.3 （2026-10-02）warehouse 退步定位：9 组单变量对照，只有腰 roll/pitch 35 → 50 N·m 救回（5/10）；35 N·m 下腰俯仰塌到 ~35°（关节限位 29.8°），stock 抓箱姿势静力矩 p95 39 N·m
    - [x] 13.6.3.1 （2026-10-02，用户问手为何抖）手抖也是腰 35 N·m：warehouse mars 手相对躯干高频抖 9.4 mm RMS、约 4 Hz、主要在肩 pitch（stock 2.8）；腰俯仰高通速度 1.26 rad/s（stock 0.13），4.7% 的时间顶在限位上反弹。腰 50 → 2.9 mm、0.12 rad/s；腕 5、质量、运动学、踝 50 都不消除；bottle mars 3.1 mm（`outputs/mars_pilots_20261002/jitter/jitter.md`）
  - [x] 13.6.4 （2026-10-02 用户：腰改 50 再试）mars 腰 roll/pitch 改 50 N·m（`make_dex1_model.MARS_EFFORT_OVERRIDES`，URDF 和 Unitree 官方都是 35，踝仍 35），7 个场景重建。warehouse 重跑 `warehouse_mars_pilot_20261002c`：6/10（stock 6/10），手抖 2.8 mm（= stock）
    - [ ] 13.6.4.1 待查：mars 的 4 局失败全是 lost_attachment，且都在箱子已抬起 4–14 cm、倾 6–13° 后一只手脱开（3 局右手）；stock 的 2 局 lost_attachment 都在抬起前（rise 0）。可能是 D405 支架、手腕 13.4 N·m 或手臂运动学；需腰 50 下的单变量对照
  - [~] 13.6.5 第 3 步（用户看完第 2 步再做）：mars 的 Isaac 机器人资产（按机器人名分开存）、warehouse `_mars` 的 Isaac 场景与 Unreal 关卡（`scenes.yaml` 现写 unsupported 指向这里）、各实渲一局、补渲冒烟
    - [x] 13.6.5.1 （2026-10-03，用户：只改 warehouse，Unreal 作默认）`assets/isaac/robot/mars/g1_dex1.usda`（`build_isaac_assets.sh --robot mars`）、四个 `warehouse_box_*_mars_env` stage、四个 `warehouse_box_*_mars` Unreal 关卡已建；`warehouse_textured_mars` 不建关卡
    - [x] 13.6.5.2 （2026-10-03）Unreal 冒烟 `warehouse_mars_smoke` 照 runbook 跑第 1–4 步（`outputs/mars_assets_20261003/smoke/`）：采集 30 局保留 9 条，Unreal 补渲 9 局（标记 `robot: mars`），转换 8+1 条。用户：不训练、不评测，只要采集流程的代码和文档合规；第 5–7 步没跑
    - [ ] 13.6.5.4 待用户定：mars 冒烟保留 9/30，stock 09-30 是 11/20、8/20；要不要同种子配对重跑分清是机器人还是波动
    - [x] 13.6.5.5 （2026-10-03，用户）默认摆放改为只走 −y 边：`warehouse_mars_smoke`、`_200` 及 `_isaac` 的 `start_edges: [ym]`；stock 文件仍四边；`collect.layout_reminder` 改说三种摆放；四边那次冒烟的运行目录改名 `warehouse_mars_smoke_4edges_20261003`
    - [ ] 13.6.5.6 只走 ym 的 mars 采集还没实跑（只 `--dry-run` 核了摆放）；stock 的 ym 边 09-28 是 11/19
    - [ ] 13.6.5.7 runbook 第 5–7 步（训练、开环/闭环评测、第三视角）在 mars 上没跑（用户 10-03：暂不训练）；09-30 只在 stock 上跑过
    - [ ] 13.6.5.8 可选：runbook 常见报错补一行 `REFUSING TO START: a GPU already has`（别的项目占卡 > 4 GB 时 `train.sh` 拒绝，10-03 冒烟卡在这里 10 分钟）
    - [ ] 13.6.5.3 Isaac 版 mars 实渲一局（只建了资产）
  - [x] 13.6.6 （2026-10-03）第 3 步：Isaac/Unreal 的 `source.json` 与两种补渲标记加 `robot`（Q12），对不上拒绝（`run_context._require_robot`；旧文件无字段按 stock）
  - [~] 13.6.7 第 4 步：runbook（single_arm.md、warehouse.md）示例改用 `*_mars`、文档、推送（推送前问用户）
    - [x] 13.6.7.1 （2026-10-03）warehouse.md 改用 `warehouse_mars_smoke`（Unreal 默认）/`_isaac`；README、setup、glossary、rendering、warehouse_design §11 跟着改
    - [ ] 13.6.7.2 bottle（single_arm.md）、crate 还是 stock（用户 10-03 只要 warehouse；crate 还没有 mars 场景）
    - [x] 13.6.7.3 （2026-10-03，用户：提交并 push）`4db7deb` 连同之前 12 个本地提交推到 origin/main（`cb3ae2f..4db7deb`）
  - [x] 13.6.9 （2026-10-02，`ed26ed2`，用户：和 stock 相同的部位外观一致）逐部件比对颜色：除手腕（有意涂黑）外相同部位一致。改两处：云台俯仰支架涂黑（头部原有一根白横条；10-03 用户更正：那是头顶的 D455，真机银白，已改回）；骨盆保留的 stock 碰撞外壳改成骨盆色（MuJoCo 画碰撞体，白壳透出）。10-03 用户：骨盆（腿和上身之间的三角件）应为银色，`robot_look.SHELL_MESHES` 把 `pelvis_ver0529` 画成外壳色（URDF 和 Unitree 官方都是黑），碰撞外壳随之同色（对比图 `outputs/mars_look_compare_20261002/`）。10-03 用户先要 D405 本体黑，查明 Unreal 画面里的灰格子方块是 D405 的碰撞盒（mars 碰撞体 alpha 1 被导出，URLab 用 UE 默认 `BasicShapeMaterial` 画基本几何体）而非相机本体，改回银色；`export_mjcf.strip_to_dynamic` 现在丢掉有可视体的 body 上的碰撞 geom。同日用户：MuJoCo/Isaac 画面里支架后背有个黑框凹陷，也是这个碰撞盒；参照 `assets/unitree_G1_with_wrist_camera_new`（URDF 不给支架碰撞体、网格相同），`_mars_geoms` 把支架的碰撞基本体放到 group 3：仍碰撞，相机不画。之后 Unreal 里 D405 背面仍显内凹：是 URLab `clean_meshes.py` 导出 GLB 的平滑法线跨过 90° 棱（背面法线均值 0.84、离散 0.69），不是几何；补丁按 30° 拆分顶点后 0.998/0.13（`patches/urlab_fresh_capture.patch`）
  - [ ] 13.6.8 可选：`outputs/scene/hand_points_dex1.npy` 仍是 stock 手（mars 多了 D405 支架，最低点只在少数姿态差 >0.2 cm，最大 6 cm）；bottle 用它算手离桌高度
  - [ ] 13.6.10 待用户定：先做 13.6.4.1 的腰 50 单变量对照（支架不碰撞、腕 5 N·m，各 10 局约 15 分钟），还是先开始第 3 步（13.6.5）。10-02 已问，用户转去改外观，未答
  - [~] 13.6.11 上真机前：专家抱箱前倾要腰俯仰约 39 N·m，Unitree 官方 URDF 腰是 35；改专家少弯腰多屈膝，或向 henry 确认真机腰的固件上限（仿真现在用 50 掩盖了这点）
    - [x] 13.6.11.1 （2026-10-03，用户问真机遥操能弯 80° 且手不抖）静态腰俯仰负载（mars 模型）：前倾 80° 手垂向地面 36 N·m；30° 手前伸托 1.2 kg 箱 41；专家 10 局 p95 44（29% 的帧 > 35）（`outputs/mars_pilots_20261002/waist_load/`）
    - [x] 13.6.11.2 （2026-10-03）仿真弯腰测试 bend80（各 1 次）：腰 50 手垂地 76° 纹丝不动；腰 50 手前伸、腰 35 两种都 2–4 Hz 塌到限位、手抖 4–31 mm；无摔倒（`outputs/mars_pilots_20261002/bend80/bend80.md`）
    - [x] 13.6.11.3 （2026-10-03）抖动机制（`bend80/osc.py`）：髋压前倾 → 腰力矩到截断 → 截断后无刚度无阻尼 → 0.1 s 塌到限位、骨盆反踢 → 弹回负载 < 截断处 → 重复；腰 kp 只有 28.5 N·m/rad
    - [x] 13.6.11.4 （2026-10-03）50 N·m 的来源：SONIC 训练配置与训练 URDF（rev 1.0 几何）都是 50，即 2×5020（单台 25）；Unitree 所有 rev 1.0 文件写 35；deploy 不截断力矩，真机上限由固件定
    - [ ] 13.6.11.5 待协作者：真机保持弯腰（手垂地 80° 和前倾 50° 手前伸托箱）时 LowState 13/14 号电机的 `tau_est`，定腰的固件上限
    - [ ] 13.6.11.6 可选：腰 50、手水平前伸时能稳住的最大前倾（静态估算约 60°，未实测）
  - [ ] 13.6.12 待用户定：本次产物删不删（`outputs/mars_pilots_20261002` 4.1 GB，其中 `scenes_v1`–`v6` 旧版场景和 8 个 `ablate_*` 场景；14 个 `outputs/runs/*_20261002*` 共 7.5 GB；`outputs/g1_wristcam_render_20261002` 122 MB）；删前在 STATE 写明 24 小时
    - [ ] 13.6.12.1 10-03 新增产物：`outputs/runs/warehouse_mars_smoke_4edges_20261003` 712 MB、`outputs/runs/bend80_mars_w{50,35}_20261003` 共 314 MB、`outputs/mars_assets_20261003` 2 MB、`outputs/mars_pilots_20261002/bend80` 3 MB；删前在 STATE 写明 24 小时
  - [x] 13.6.13 （2026-10-04，用户：新 URDF 的手臂是黑的，旧的是银色，改成银色）三节 5010 手腕（小臂）从 `robot_look.DARK_MESHES` 去掉，按 URDF 的白色画外壳银；Dex1 底座、D405 支架仍黑。mars 模型、7 个 `*_mars` 场景、几何样例已重建（差别只有手腕 rgba 和 henry 的支架 group 3；旧场景在 `/workspace/tmp/mars_scenes_before_20261004`）；对比图 `outputs/mars_arm_silver_20261004/`
    - [x] 13.6.13.1 （2026-10-04，随 13.6.14 一起重建）4 个 `warehouse_box_*_mars` Unreal 关卡要重建（场景哈希变了，运行会拒绝旧关卡并给出重建命令）；Isaac 机器人 USD 按 MJCF 哈希缓存，下次用到时自动重建
  - [x] 13.6.14 （2026-10-04，用户：反光太强，按真机照片 `assets/mars_real_model.jpg` 调）外壳 metallic 0.75、roughness 0.6、底色 0.45，外观名 `g1_satin_20261004`（stock 与 mars 共用）；mars 腰部两个电机加入 `DARK_MESHES`，骨盆按 URDF 画黑（`SHELL_MESHES` 删除）。19 组对比 `outputs/mars_shell_match_20261004/`；7 个 mars 场景、几何样例已重建，本机 9 个 Unreal 关卡 10-04 重建（旧场景在 `/workspace/tmp/mars_scenes_silver_pelvis_20261004`）
    - [x] 13.6.14.1 （2026-10-05）Isaac 与 Unreal、MuJoCo 同机位对比（`outputs/robot_look_3renderers_20261005/sheet.png`）：Isaac 外壳与 Unreal 接近；4 个 mars Isaac stage 已重建
    - [x] 13.6.14.2 （2026-10-05，用户）MuJoCo 外观对齐：mars 外壳 rgba 0.7 → 0.62（DECISIONS 2026-10-05）；7 个 mars 场景、几何样例、4 个 mars Unreal 关卡与 Isaac stage 已重建。MuJoCo ego 手臂仍约 25% 过曝（头灯）
  - [x] 13.6.15 （2026-10-06，用户）第一视角可选脸上的 D435：`--ego-camera d435_head`，场景 `*_mars_d435`（bottle、warehouse_textured、warehouse_box_45x32x33），示例 `warehouse_mars_smoke_d435_unreal`、`_d435_isaac`、`_d435_mujoco`（10-06 用户：三个渲染器都有，Unreal 版由 `warehouse_mars_smoke_d435` 改名）；本机三个场景、Isaac stage、Unreal 关卡已建，三个渲染器对比 `outputs/ego_d435_20261006/`
    - [ ] 13.6.15.1 D435 上的专家产出率只跑过一次：2026-10-06 10 局保留 4（失败 basket_tilt 3、unstable_hold、reach_stalled、lost_attachment 各 1；`outputs/runs/warehouse_mars_d435_check_20261006`），没有同种子对照 D455；伸手/合爪阶段手臂出画
    - [ ] 13.6.15.2 待协作者：D435 真机内参、外参所用的 livox 坐标系定义（13.3）
  - [x] 13.6.16 （2026-10-06，用户）训练图加相机延迟：`images.camera_delay`，三个渲染器的离线补渲都读，审计按预期滞后检查；warehouse mars 线 9 个实验文件打开 `{fps: 30, latency_ms: [20, 30]}`（DECISIONS 2026-10-06）。验证 `outputs/camera_delay_check_20261006/`
    - [ ] 13.6.16.1 待协作者：真机三台相机的帧率和"画面比关节状态晚多少"，测到后替换 30 fps / 20–30 ms
    - [x] 13.6.16.2 闭环评测的观测图也按同一模型变旧：2026-10-07 随 16.7 实现（lockstep；`camera_frames` 与它同设被拒）
    - [ ] 13.6.16.3 训一对有/无延迟的模型比较（未做，效果未知）
  - [ ] 13.6.17 待用户定：10-04–10-07 产物删不删（删前在 STATE 写明 24 小时）：`outputs/runs/camdelay_check_{a_nodelay,b_delay}` 各 345 MB（延迟检查用的副本）、`outputs/t7_lockstep50_smoke_20261007` 566 MB、`outputs/runs/t7_mirror_check_{isaac,unreal}_20261007` 各 216 MB、`/workspace/tmp/mars_scenes_{before_20261004,silver_pelvis_20261004,shell070_20261005}` 各 535 MB（换外观前的旧场景）。`/workspace` 剩 97 GB（91%）
  - [ ] 13.6.18 待用户定：warehouse mars 的 11 个实验文件顶层 `backend.dataset_fps: 30.0` 与 `dataset.fps`（25/50）不一致；采集不读它（评测块里那份才被读），改成和 `dataset.fps` 相同还是删掉

## 14. G1 外观追 URLab hero 图（用户 2026-10-01；DECISIONS 2026-10-01 材质一节）

- [x] 14.1 （2026-10-01）对比 hero 与我们 Isaac / Unreal 的机器人：主因是外壳 roughness 0.75、深色件参数、无贴图、白房间平光和分辨率（`outputs/g1_look_vs_hero_20261001/`）
- [x] 14.2 （2026-10-01）Unreal 里一次改一项的 A/B：v1 粗糙度 0.35/0.45、v2 黑塑料、v3 侧光 40k–400k cd、v4a/v4b 组合；另测光滑法线、32 次采样（`outputs/g1_look_ab_20261001/ab_sheet.png`）
- [x] 14.3 （2026-10-01，`f7eadec` 已推送）用户选 v4a，四路画面（ego、左右腕、第三视角）和 Isaac 都用：`src/locomani/sim/robot_look.py` 一处定义，关卡 `source.json` 与两种补渲标记记 `robot_look`
  - [x] 14.3.1 （2026-10-01）本机 5 个 Unreal 关卡按新外观和 1280×960 重建
  - [x] 14.3.2 （2026-10-01）Isaac、Unreal 各实渲一局第三视角带小窗（`outputs/g1_look_v4a_check_20261001/`）
- [ ] 14.4 待用户定：Isaac 深色件比 Unreal 亮（腕部相机夹爪顶板 136/255，Unreal 近黑；底色 0 也只到 108，间接光 ×1 是 135）。要不要只给 Isaac 单独压暗（目前两边数值相同，按用户"同一套设定"）
- [ ] 14.5 待验证：新外观下的数据集补渲（`render_dataset_cameras_{isaac,unreal}.sh`）和评测镜像没单独实跑；只跑了第三视角（它的小窗走同一个 view）
- [ ] 14.6 待用户安排：09-30 及以前的数据集和 checkpoint 都是旧外观；要用新外观就得重新补渲/重训（与 12.9.6 一起排）
- [ ] 14.7 待用户定：本次产物删不删（共约 220 MB：`outputs/g1_look_ab_20261001` 46 MB、`g1_look_v4a_check_20261001` 35 MB、`g1_look_vs_hero_20261001`、`g1_urdf_render_20261001` 134 MB）；删前在 STATE 写明 24 小时

## 15. 协作者（henry-zhao）的推送

- [x] 15.1 （2026-10-01）origin/main 上 henry 的 6 个提交（`5e9c613`…`c966ee1`）快进合入 main；用户 2026-10-01 说"merge 掉 henry 的修改"时已无可合并内容（PR #1–#3 的提交也都在 main 里）
- [x] 15.2 （2026-10-01，随 `f7eadec`）`assets/unitee_G1/README.md` 的 Python 代码块按 ruff 格式化：origin/main 原本过不了 `make check`
- [x] 15.4 （2026-10-02，`577ce78`）合入 henry 的 `cb3ae2f`（新增 `assets/unitree_G1_with_wrist_camera`，backpack 目录改名 `unitree_G1_with_backpack`）。他的 `mjmodel.mjb` 是 MuJoCo 3.10 编的旧模型（无相机、无 D405），`mjmodel.xml` 是空文件；README 的文件名和序列号与 robot.yaml 对不上；README 代码块按 ruff 格式化
- [x] 15.5 （2026-10-07，`a0f05af`）rebase 到 henry 10-06 的 5 个提交（`1094808`…`eb984fd`，含 `dataset.fps` 与 `src/real_robot/`）：`view_render`、`render_dataset_cameras`、STATE、TODO、change map 冲突，两边都保留；`camera_delay` 的预期滞后改按运行的 `dataset.fps` 算；他新加的 `_d435_{isaac,mujoco}` 也开相机延迟
- [x] 15.6 （2026-10-07，`e6d7e80`）合入 `henry-dev` 的 `dc3b909`（官方式 `run_gr00t_server.py` 与按官方逐函数重写的真机客户端），无冲突；同一提交把还写着 `serve_policy` / `get_action_raw` 的设计文档、DECISIONS 追加条、TODO 16.2、change map、术语表、running.md、README 更新
- [ ] 15.3 可选：GitHub 上 PR #1–#3 若仍显示打开，在网页上关掉（本机没有 `gh`，看不到状态）

## 16. 仿真数据零样本上真机：GR00T + SONIC（用户 2026-10-06，逐项问答定；DECISIONS 同日；设计 `docs/handoff/real-robot-sim2real-plan-20261006.md`）

- [x] 16.1 （2026-10-06）T1：`dataset.fps` 字段（25 | 50）贯穿 `run_context.migrate_dataset` → `run_convert --fps` → `convert_to_hf_format` → `world_lift.resample_selection`；Isaac/Unreal 补渲按它决定渲哪些行，标记记 `dataset_fps`，`render_dataset_cameras.serves` 决定渲过的 session 能否服务某个转换（50 服务 25，25 不服务 50）；`eval_closed_loop` 遇到非 25 的运行拒绝（lockstep 的 `RATE = 25` 写死，16.7）
- [x] 16.2 （2026-10-06）T2：`src/real_robot/` 推理客户端（`real_robot.inference`，官方结构；`observation`、`chunk`、`gripper`、`policy_client`），`scripts/run_real_robot_inference.sh`，runbook `src/real_robot/README.md`，`UPSTREAM.md` 钉 WBC v1.1 a0732b6；测试 `tests/test_real_robot_client.py`。只做了本机冒烟（真实 pinocchio 模型拼 60 维观测、protocol v4 打包），没连过机器人或策略服务器。2026-10-07 henry `dc3b909`（`henry-dev`，同日合入）按官方逐函数重写客户端、调官方 `get_action`，策略服务器换成官方式 `src/real_robot/run_gr00t_server.py`（`scripts/run_gr00t_server.sh`）；仍没加载过真实 checkpoint
    - [ ] 16.2.1 用真实 checkpoint 起 `scripts/run_gr00t_server.sh`，让 `run_real_robot_inference.sh` 在本机连上它（不上真机）：核对 `identity`、嵌套观测、40 步 chunk 和 Dex1 换算；henry 只用占位策略试过
- [x] 16.2.2 （2026-10-10）夹爪单位改为启动时核对：服务器 `identity` 带 checkpoint `statistics.json` 里夹爪与手的 q01/q99（`real_robot.statistics`），客户端 `--gripper-units` 默认 `auto` 按范围判 `sim_m`（±0.1 内）或 `dex1`（0–5.5），显式值矛盾或判不出就拒绝启动（`policy_client.resolve_gripper_units`）。起因：checkpoint-30000 是米制，默认 `dex1` 会让两只夹爪一启动合死并把 5.39 喂给模型
- [ ] 16.3 T3：转换器写 Dex1 单位：`teleop.*_hand_joints` 第 0 维按 `real_robot.gripper.sim_m_to_dex1` 映射（命令低于 −0.02 截到 0），`observation.state` 手 7 维广播映射后的指爪位置（第 4、5 维均值）；与客户端共用一个映射函数
    - [ ] 16.3.1 `dex1_to_sim_m` 闭端截在 −0.02 m，warehouse 线闭合目标是 −0.06（−0.02 拿不起箱子）：Dex1 0 要换回线自己的闭合目标，16.5 的仿真桥同样（DECISIONS 2026-10-08；方向维持 0 闭 5.5 开）
- [ ] 16.4 T4：新运行 `warehouse_mars_200_50hz`（复制实验文件加 `dataset: {fps: 50}`，运行目录复制 collect 条目、manifest、trace、场景；删 74 个 session 的 `meta/offline_render.json`）→ Unreal 补渲（按 2026-10-07 决策 1 带 `camera_delay`，约 3/5 的行出新图，约 15 h 单卡；推翻 D2）→ 转换 → 训练；同源数据留 25 fps 版本作对照
    - [ ] 16.4.1 本机没有 `outputs/runs/warehouse_mars_200`，要在采集它的机器上做；步骤见 `docs/migrate_old_data.md` 第 3 节，代码要含 `6a6bd10`（否则补渲第一局 `KeyError: 'dataset_fps'`）
- [ ] 16.5 T5：sim2sim：官方结构客户端对接本仓库 MuJoCo 栈（`--hand-wire sim`），核对延迟补偿与跳帧行为；仿真桥若改读 Dex1 单位则用 `dex1_to_sim_m`
- [ ] 16.6 T6（需真机）：核对相机服务器三路 key 与 `g1_debug` 手字段；`]` + `i` 手动验 Dex1 方向（5.5 开、0 闭）；按 D13 跑 10 局（起点 0.6–1.0 m，30 s，骨盆到箱心 ≤ 0.6 m 且站稳 2 s）
- [~] 16.7 T7（2026-10-07 实现，DECISIONS 同日）：50 Hz 的 lockstep 闭环评测按官方客户端：节拍从 `dataset.fps` 读（50：一步 1 拍，动作块 40，执行 20 拍再推理，块尾保持，第一块不混合；25 不变）；评测图按 `camera_delay` 变旧（仿真保留状态历史，MuJoCo 与镜像按序号渲旧状态）；推理延迟按 `latency_dist` 每块抽，先用 lockstep 实测的 `serve_policy` 耗时；`eval_closed_loop` 放行 50 fps 的双臂 lockstep；warehouse mars smoke 系列实验文件改 50 fps。已验证（`outputs/t7_lockstep50_smoke_20261007/`，25 fps 旧 checkpoint 只验机制）：24 拍一周期、图晚 40–60 ms、实测延迟分布 `configs/eval/latency_serve_policy_50fps_20261007.json`
    - [x] 16.7.1 （2026-10-07）Isaac/Unreal 镜像按携带状态渲：真实镜像各 2 局通过，取图 seq 全在镜像渲染记录里、无一等于冻结时刻（`outputs/runs/t7_mirror_check_{isaac,unreal}_20261007/`，DECISIONS 2026-10-07）
    - [ ] 16.7.2 有了 50 fps checkpoint（16.4）后跑策略层面的评测；T5 后把延迟分布换成真实链路的
- [x] 16.9 （2026-10-06，用户）客户端从 `src_real_robot/real_robot/` 移到 `src/real_robot/`（`src/` 已在每个 PYTHONPATH 上，和 `src/gr00t` 同级），启动脚本改为 `scripts/run_real_robot_inference.sh`，`src_real_robot/` 删除
- [x] 16.8 （2026-10-07 本机 `make check` 全量通过，三条只在 henry 的机器上失败：没有 `outputs/scene/`、`.venv_sim` 缺 pyarrow；`warehouse_mars_200` 两个文件 `eb984fd` 后又一致）预先存在、与本次无关的测试失败（henry 的机器）：`test_experiment_files::test_paths_the_file_names_exist` 的 7 个 `*_d435` / `warehouse_edges_*` 场景目录不在 `outputs/scene/`；`test_image_source::test_the_unreal_and_isaac_warehouse_runs_differ_in_the_image_source_only[warehouse_mars_200_isaac-warehouse_mars_200]`（两个文件的 collect/train 块已不同）；`test_check_dataset_proximity` 在 `.venv_sim` 缺 pyarrow
- [ ] 16.10 告诉 henry：2026-10-07 推翻了他设计文档的 D2（50 fps 数据集改按 30 fps 相机加延迟出图，T4 补渲带 `camera_delay`），见 DECISIONS 2026-10-07 与提交 `804f962`
- [ ] 16.11 提交钩子在合并提交 `e6d7e80` 的第一次提交时失败一次（`make: *** test Error 1`），重跑通过，前后两次全量 `make check` 也都过；没抓到是哪个测试（输出被截断）。再出现时保留完整日志，查是不是不稳定的测试
- [x] 16.12 （2026-10-07，用户）SONIC 两条线 state 改回官方 46 维，去掉两腕位姿 14 维（DECISIONS 同日）；旧 53/60 维 checkpoint 不能再服务或评测，16.4 按 46 维训练
    - [ ] 16.12.1 告诉 henry：真机客户端的 state 改成 46 维，`real_robot.observation` 不再做腕部 FK
- [ ] 16.13 核对真机 C++ 是否按官方 `b042411` 的 `vla_inference.md` 把踝关节俯仰电机 Kp、Kd 放大 1.5 倍（`--motor-kp-scale 4,10=1.5`、`--motor-kd-scale 4,10=1.5`）；本地仿真 deploy 分支 `lockstep-port` 没有这个参数（`docs/official_comparison.md`）
- [x] 16.14 （2026-10-07，用户）与官方教程的对照文档 `docs/official_comparison.md`，README"通用"给入口；取代 08-28 审计，B1 颜色增强核对与教程相同（`84f0fe7`，DECISIONS 2026-10-08）
    - [ ] 16.14.1 待用户：训练配置向教程靠——批 32（现 64）、20000 步（现 10000），只改 `train` 块；两项都没做过对照
    - [ ] 16.14.2 `docs/archive/INDEX.md`（3 个）和 `docs/archive/official_parity_audit.md`（1 个）有既有坏链接，修好后加"已被取代"注
- [~] 16.15 （2026-10-07，用户）`warehouse_mars_50`：照 runbook 第 1–4 步采 50 条（120 局保留 50，Unreal，50 fps，带相机延迟）；训练集 45 条、验证集 5 条，审计全过（台账 crate.md 2026-10-07，`6a6bd10`）
    - [x] 16.15.1 （2026-10-07）补渲 `KeyError: 'dataset_fps'`：`view_render.plan` 返回 look，入口交给 `fill`，加测试（`6a6bd10`）
    - [x] 16.15.2 （2026-10-07）runbook 第 1 步补"场景已建过会拒绝覆盖"，常见报错补 `basket_tilt`
    - [ ] 16.15.3 训练（`bash scripts/train.sh outputs/runs/warehouse_mars_50 --gpu 0`，1 万步）和闭环评测（100 局，Unreal 镜像）；这会是第一个 46 维、50 fps、带相机延迟的 checkpoint，可供 16.7.2、16.2.1
      - 2026-10-08 起跑（GPU0，1.89 s/步，预计 5.2 h），约 500 步时用户叫停（21.3.1 前置检查取消），没有 checkpoint；仍待做
    - [ ] 16.15.4 待用户：`check_dataset.bimanual` 按任务文本含 `crate` 判断双臂，warehouse（`pick up the box`）的 3cam_bimanual 档没被审计；本次手工核对过 state 46、action 66、三路相机
- [x] 16.16 （2026-10-07，用户）旧数据迁移文档 `docs/migrate_old_data.md`，README 顶部提示；`running.md` 的 `dataset.fps` 行改为 crate/warehouse lockstep 接受 50（`b8f5e9c`，DECISIONS 2026-10-08）
    - [ ] 16.16.1 待验证：实时录制运行按 50 fps 转换，审计 "action vs state" 中位 +0 帧不过（`crate_smoke` 只 1 条；25 fps 时 +3）；在完整数据上复查
    - [ ] 16.16.2 可选：一条命令从旧运行派生 50 fps 新运行的脚本（现在是文档里的手工命令，要同时改 `summary.json` 和 `collect/inputs.json` 时容易漏）
    - [ ] 16.16.3 告诉同事（henry）：旧数据按迁移文档处理，不用重新采集；代码要含 `6a6bd10`
    - [ ] 16.16.4 `docs/migrate_old_data.md` 第 1 节写实时录制的图"约 33 Hz"（按代码推的），实测 50 fps 数据集里不同画面约 30.3 Hz；改成实测值，原因待查

## 17. agent 可拼接的技能（skills）

分支 `scene_discovery` 上原编号 7（2026-10-08 合入 main 时改为 17，避开 main 同期新开的第 7 节）：本节条目号 17.x 即原来的 7.x，DECISIONS、台账、交接文档和提交信息里写的 7.x 都指这里。

- [x] 17 agent 可拼接的 primitive（`src/locomani/skills/`，设计 `docs/handoff/primitive-contracts-design-20260928.md`，2026-09-28；第二批见 §9b；标定项 7.7.1 / 7.8.1 / 7.8.2 于 2026-09-28 傍晚全部关闭）
  - [x] 17.1 session / contract / profiles / scene_model / executor + 目录 7 个适配器 + 占位；`tests/test_skills_*.py` 41 个通过；未提交
  - [x] 17.2 包出四个黑盒（2026-09-28）：approach 按 profile 分派、grasp_single、grasp_bimanual、place；双臂每拍逻辑抽到 `tasks/crate/attempt.py`；78 个测试通过，未上真栈
  - [x] 17.3 回归（2026-09-28）：bottle plan 10 局 9/10（台账 bottle §40）；pick_place plan 冒烟 1/3（pick_place §7，`simple`+place 未标定）；warehouse plan 冒烟 1/3（crate 台账末节）；三条线管路通，warehouse 的 10 局对照 4/10 已跑（7.11）
  - [x] 17.4 前置带宽回放（2026-09-28）：`scripts/replay_preconditions.py` + `STANCE_BANDS` / `stance_violations`；932 条已录站位（skills trace 43、bottle 采集 286、warehouse 603）上任何深度 / 横向带都不分成功与失败（bottle 深度中位 0.324 vs 0.325，warehouse 0.436 vs 0.454），带保持全开；报告 `outputs/analysis/precondition_replay_20260928/report.md`
  - [x] 17.5 `carry_to`（2026-09-28）：冻结上身 + walk_to + 持物守卫；持瓶直走/倒退 1–2 m 漂移 < 1 cm（6/6），走动中急转 3 cm 掉（0/2），180° 转 IDLE 0/2、SLOW_WALK 4/4 → 默认 turn_first + slow_walk；台账 bottle §41。持箱未量
  - [x] 17.6 `scripts/plan_episode.py` 已写（起栈/prime/spawn/plan/判据/manifest/trace，2026-09-28）；claude -p 规划器 `skills/planner.py` + `--agent` 接线（2026-09-28 晚：JSON schema 结构化输出、每局一个会话、`agent_io_NN.jsonl`；真栈冒烟见 pick_place 台账 §8）
  - [x] 17.7 跨位置搬运回归（2026-09-28 晚）：`walk_to` 持物走 `carry_to` 守卫；单臂 `approach` 持物 = carry 到放置带中心站位 + 原地转向 + 残差修正 + 写站位；默认 plan 加 `approach(goal)`。真栈三轮（pick_place 台账 §8）：容器进带 6/6，place 3/6；对照直接 place 0/5。place 前站位深度不是主因——`simple` 抓罐头后 place 的 carry-over / 下降伺服（7–12 cm 偏、掌系滑 3 cm）是下一个标定对象（7.10 流程），未调
    - [x] 17.7.1 `simple` 抓罐头 + place 标定（2026-09-28 下午→傍晚，pick_place 台账 §11–12；结论见 7.7.1.1 / 7.7.1.2）：按 7.10 流程跑了三个单旋钮 pilot（`entry_lateral_tolerance` 0.015、`v_descend` 0.05 + 8 s、`entry_overshoot` 0）各 10 局，grasp 7/10、6/10、4/9 与 place 3/6、2/6、0/4 均不优于基线 6/10、3/6；trace 的 stats 表明合爪前已对准、罐心在工具点前 0.7–1.1 cm，失败在 6.1 cm 罐身与 Dex1 指垫的闭合几何和 PlannerArm 物体伺服的跟踪，旋钮够不着。`coupled` 策略（§4 10/10）没被 skills 包装（approach 抛 `walking` 缺失，现已在前置拒绝）
      - [x] 17.7.1.1 包装 `coupled` 策略（用户定走 A，2026-09-28 下午，pick_place 台账 §12）：三处修补（`walking` 用 getattr、`walk_tick` 收关键字、approach 放行 coupled），profile `pick_place_coupled`；全程 2/10（grasp 5/10、place 2/5，持罐走动掌系滑 3.2–3.7 cm）；容差 0.05 对照 1/10（grasp 2/10，转向中掉罐 ×3）。coupled 在 skills 里带来的是"停位挑剔 + 握持转向滑"两族，不是罐头的解
        - [x] 17.7.1.2（用户定走 A，2026-09-28 傍晚）：§4 的抓取率本来就是 10/19，coupled 差在放置侧——它从不持罐走路。coupled 原地放置 3/4（持罐走近后 2/5）；`default_plan` 对 coupled 不再插 `approach(goal)`。抓取侧 ~50% 是 coupled 自己的失败族，与 §4 一致，不属于本项
  - [x] 17.8 `put_down_bimanual`（`catalogue/crate_put_down.py`，2026-09-28；7.8.1 / 7.8.2 傍晚关闭：站直不回拉后 8/8 放稳）：真栈 7 个样本 6 次把箱子放稳（≤ 0.7 cm、≤ 2.5°）；持箱 carry 180° 转 + 1 m 3/6（crate 台账末节）
    - [x] 17.8.1 put_down 前送（用户定走 A，2026-09-28 下午，crate 台账末节）：新 primitive `nudge`（定时 SLOW_WALK 脉冲 + 刹停 + 量残差，持物带守卫）；三个变体 60 局：nudge 到位 13/29（0.5 s 从不冲出但单脉冲 0–12 cm；0.7 s 触发整步 21–77 cm，掉 2 摔 1），**到位后 put_down 12/13 放稳**（z ≤ 0.3 cm、倾 ≤ 0.8°）。"手仍触箱"改记指标不判失败，`back_m` 0.08。臂前送（`forward_m`）1/3 保留为参数。剩下的厘米级平移是 v1.1 planner 的量化，不在旋钮里
      - [x] 17.8.1.1（用户定走 B，2026-09-28 傍晚）：`stand_forward_min` 0.35 → 0.50（站直时手目标留在骨盆前 0.50 m，箱子不回拉），不挪步直接放：20 局里到 put_down 的 8 局 **8/8 放稳并过判据**，抓取率 8/20 与基线同。skills `warehouse_box` profile 改读 `skills_box_fwd50_20260928.yaml`；采集线配置不动
    - [x] 17.8.2 持箱 carry 默认走法（2026-09-28 下午）：倒退不转 8/12 vs 转 180° 3/6，但两种走法箱子都在走动中以约 2 cm/s 下滑（1 m 掉 3–9.5 cm），瓶颈是冻结上身下的捏持蠕变而非转向；默认不改（`turn_first=True`）；持箱超过 1 m 要中途重抬或带反馈的 POSE 流搬运（未立项）
  - [x] 17.9 `navigate_to`（2026-09-28 晚）：沿 `Navigator.waypoint()` 分段走，空手朝路点、持物先 slow_walk 转再守线走；假栈 7 个测试；真栈持罐 1 局 7 段 2.57 m 漂移 0.8 cm（pick_place 台账 §10）；空手真栈未跑
  - [x] 17.10 新物体标定流程（2026-09-28）：写在 `docs/skills.md` §5（新 profile 登记 `PROFILE_SOURCES` → `plan_episode.py --profile` 10 局 pilot → 按失败族改一个旋钮 → gate 对照 → 回放定带）；按几何自动落位仍无机制，文档里明说没有
  - [x] 17.11 warehouse plan 10 局对照（2026-09-28 晚）：`skills_warehouse_pilot10_20260928` 4/10 vs 采集入口 6/20，同一噪声带（crate 台账末节）；pick_place 运行的预览视频 `outputs/videos/skills_batch2_20260928/`（4 段 ≤ 436 KB）；warehouse plan 运行无视频（render offline）
  - [x] 17.12 提交本会话代码（2026-09-28 晚：skills 包 + 第二批 primitive、attempt.py 提取、plan_episode.py `--agent`/`--profile`、回放脚本、docs/skills.md、台账、plan 文件；`make check` 通过）
  - [x] 17.13 目录去重（用户 2026-10-01 定）：`carry_to` 并入 `walk_to`（持物时同名走冻结上身 + 持物守卫，持物参数空手传入拒绝），删除 `retreat`（= `hand_back` + `walk_to(relative=[-d, 0], turn_first=false)`）；目录 15 → 13，10 个 plan 文件改名并写明原 `carry_to` 的 `stop_dist` 0.08；理由见 DECISIONS 2026-10-01 第二节。`take_over` / `hand_back` 是否藏进 executor 待用户定
  - [ ] 17.14 `navigate_to(target=物体)` 名字像"走近物体"，但走完不写站位、不能直接抓（agent 曾用它贴桌沿停、随后转身扫掉罐子）：改名，或在目录说明里写清"之后还要 approach"；用户 2026-10-03 未答，待用户定

## 18. Q11 验收：失败注入配对比较

分支 `scene_discovery` 上原编号 8（2026-10-08 合入 main 时改为 18，避开 main 同期新开的第 8 节）：本节条目号 18.x 即原来的 8.x，DECISIONS、台账、交接文档和提交信息里写的 8.x 都指这里。

- [x] 18 Q11 验收：失败注入 paired 比较（agent 唯一不可替代的贡献是失败后的决策；2026-09-29 立项）
  - [x] 18.1 注入层（2026-09-29）：`skills/injection.py`（`move_target` / `drop_held`，某 primitive 第一次成功后钉物体 0.5 s 再松开，一局一次，agent 不被告知）、`plan_episode.py --inject`、`Executor.note`、manifest 记 `seed` / `injection` / `injected`；`scripts/injection_report.py` 按 seed 配对、精确符号检验、恢复段按第一步归类；`tests/test_skills_injection.py` 6 个；注入文件 `configs/injections/pick_place_{move_target,drop_held}.json`；docs/skills.md §7
  - [x] 18.2 注入有效性（2026-09-29）：横向 −0.17 m **无效**（`grasp_single` 开始时按当前位姿重规划，臂包络内，1/3 且失败是基线族）；沿桌沿 +0.60 m 有效（plan 臂 0/3，全在 `[grasp]`）；pick_place 台账 §13
  - [x] 18.3 配对 10 局 `move_target` +0.60 m（2026-09-29，pick_place 台账 §13，报告 `outputs/analysis/injection_20260929/report.md`）：plan **0/10**（全 `[grasp]`）vs agent **0/10**，p = 1.0；但 agent **5/10 重新抓起**（决策链 approach! → navigate_to → turn_to → approach → grasp_single，release 清旧 held 也对），重抓后 5/5 在持罐 `approach(basket)` 转向时掉罐，3/10 贴桌沿 `turn_to` 把罐头扫下桌，2/10 用完 16 次调用。Q11 的"显著高于脚本"没出现，卡在 primitive 不在决策
    - [x] 18.3.1 恢复路径的 primitive（2026-09-29，用户定全做）：持物 `approach(goal)` 同桌沿 + 1.2 m 内 → `carry_to(turn_first=False)` 横移不转向；`navigate_to` 点目标推到离台面 ≥ 0.30 m（`clear_of_surfaces`）；`turn_to` 离台面 < 0.25 m 拒绝（`edge_clearance`，收臂没有已验证的姿态，改成拒绝让 agent 先退）；`nudge` 用 `remeasured_stance` 重量 stance 不清；`tests/test_skills_recovery.py`
    - [x] 18.3.2（2026-09-29）：空手 `approach` 站位偏离走近线 > 0.12 m 时先 `walk_to` 站位后 0.6 m 的引导点（face_yaw 锁桌法线）再跑原相位；`metrics.prewalk`
    - [x] 18.3.3 第二轮配对（2026-09-29；第三轮 agent 臂跑 3 局确认蠕变机制后停掉，drop_held agent 臂未跑——8.3.5 未定前不再烧调用）：`move_target` plan 0/11（10 触发）vs agent 0/10（10 触发），9 对，agent 重抓 3/9——恢复链缩到 2 次调用（`approach` 自带横向预走 + 提示规则让它不再浪费一次抓空），但 4 次重抓后 agent 都直接 `place`，`place` 自己的搬运转向滑罐 4/4；报告 `outputs/analysis/injection2_move_20260929/`。`drop_held` plan 臂跑完即停（agent 臂改用 8.3.4 后的代码重跑，队列 `queue_round3.sh`：`inj3_move_agent_20260929c`、`inj3_drop_agent_20260929c`）
    - [x] 18.3.5 待用户定（决定性）：重抓后的罐头在 agent 想下一步的 5–8 s 里以 0.4–0.5 cm/s 滑出冻结握持（第三轮 round 0：`place` 被拒、想 8 s、罐头无动作掉桌），固定 plan < 1 s 接手所以不受影响。三选一：(a) sim 指垫接触调硬（唯一治蠕变，改物理）；(b) 藏延迟（Q5 提前规划 / haiku 3–4 s，仍 > 1 s）；(c) executor 两次调用间跑"保持"（更深合爪 / 周期重抬，未测）。定了才值得再跑配对
      - [x] 18.3.5.1 用户选 (a)（2026-09-29 晚）；先量蠕变就推翻了前提（pick_place 台账 §14）：掉罐是 deploy 的 1 s 规划器超时（`zmq_manager.hpp` `PLANNER_TIMEOUT`，释放冻结上身和手指命令），runner 在 agent 想的时候不发帧。`plan_episode.py --think-s 8`：静默 4/4 在 ~1 s 掉罐，保活 0/14 掉、蠕变 0.10 cm/s；§13 三轮 agent 掉罐都对得上
      - [x] 18.3.5.2 (a) 本身：`crate` 档指垫 0.06 vs `object` 0.10 cm/s，0/12 vs 0/14 掉，任务 1/10 vs 2/10（噪声内）；不改默认场景。solref 在 200 Hz sim 里被 refsafe 钳到 0.01，两档只差 solimp；离线回放台对更硬一侧不可信
      - [x] 18.3.5.3（用户定做，2026-09-29 晚）：`Executor.idle` 保活，`agent_round` 在 `claude -p` 回答期间、注入 settle 期间都走它；Q11 第四轮（pick_place 台账 §15）：move_target plan 0/10 vs agent 3/10（p 0.25，重抓 5/10），drop_held 配对 0/5 vs 1/5（重抓 3/9），合计不一致对 4 : 0、p 0.125；重抓后的罐头每次思考后都还在手里（1 例浅抓除外）
      - [x] 18.3.5.4 用户定保留（2026-09-29 晚）：8.3.1 持物横移、8.3.4 `STRAFE_YAW` / `placing_band`；第四轮 agent 重抓后都是 `placing_band` 拒 → approach(basket) → place（台账 §15）
      - [x] 18.3.5.5 定位（2026-09-29 晚，台账 §15a）：不是 nudge / walk_to / turn_to / navigate_to，是空手 `approach` 内两段不发帧——刹车后 `settle_for_admission`（0.49–0.87 s）和起步前建 `SimpleGraspPolicy` + 首次 edge_guard（0.45–0.73 s）；负载下被拉过 1 s。两段都在手里没东西时，超时无后果
        - [x] 18.3.5.5.1 用户定不修（2026-09-30）：两段都在空手 `approach` 里，超时无后果；不改 bottle 采集专家与 commander。若以后持物路径出现同类静默，用 `scripts/experiments/command_gap_probe_20260929.py` 定位
    - [x] 18.3.4（2026-09-29）：`STRAFE_YAW` 0.6 rad（抓完身体 wrap 0.35 rad 时仍横移不转向；drop plan 臂 round 2 因此走了转向）；`place` 前置 `placing_band`（容器在 `band_x × band_y` ± 0.10 m 内，否则先 `approach(container)`；coupled 免检）
  - [x] 18.4 `drop_held` 两臂 10 局（2026-09-29，台账 §13 末，报告 `outputs/analysis/injection_drop_20260929/`）：注入只在抓起后触发，基线抓取 ~50% → 各臂 5 个注入、3 个配对；plan 0/10（注入后 5/5 `[approach] load lost while turning`），agent 0/10、注入后重抓 2/5，第一反应 5/5 直接 `place`（信 `held` 不信场景），重抓后又在持罐搬运里滑 3.3 cm
    - [x] 18.4.1（2026-09-29）：`--until-injected N`（`--rounds` 为上限），`more_rounds()`；manifest 总结行印触发数
    - [x] 18.4.2（2026-09-29，用户定加）：系统提示加"session 快照是本体感觉、场景是真值，冲突先 release"；从此配对比较测的是"有提示下的恢复"，第一轮（无提示）结果留在台账 §13 作对照
  - [x] 18.6 待用户定：Q11 下一步——加配对数（每种注入 20 对以上）、先提高 `grasp_single` / `place` 成功率，还是就此收尾写结论（台账 §15：方向一致 4 : 0，p 0.125）（用户 2026-10-02 定：收尾，不再跑；结论写在 pick_place 台账 §16）
  - [-] 18.5 未做：推机器人（走偏）/ 对负载施力需要 sim 侧命令（`EpisodeControlServer` 没有 xfrc 通道）；放置点被占需要第三个物体（pick_place 场景只有罐头和篮子）（搁置 2026-10-02：Q11 线收尾，用户定）

## 19. scene_discovery Demo 1：场景 → 自动发现任务 → 技能采集 → VLA

分支 `scene_discovery` 上原编号 9（2026-10-08 合入 main 时改为 19，避开 main 同期新开的第 9 节）：本节条目号 19.x 即原来的 9.x，DECISIONS、台账、交接文档和提交信息里写的 9.x 都指这里。

- [~] 19 scene_discovery Demo 1（纯仿真）：场景文件 → 自动发现任务 → 技能组装采集 → VLA（用户 2026-10-02 定；决策与节奏见 `docs/handoff/scene-discovery-plan-20261002.md`，DECISIONS 2026-10-02）
  - [x] 19.1 Q11 收尾（`023a31a`，pick_place 台账 §16）
  - [x] 19.2 worktree `../locomanipulation-scene_discovery`，分支 `scene_discovery`（基于 `023a31a`）
  - [x] 19.3 ADT 下载：3 条公寓序列的 `main_groundtruth` + 356 个物体模型（4.6 GB）到 `/workspace/datasets/adt/`，sha1 全部校验（`scripts/adt_download.py`）；链接文件移到 `/workspace/datasets/adt/links/`（不进 git）
  - [~] 19.4 `plan_episode.py --record`：每局随机一句说法作语言指令，元数据记谓词 / plan / trace，只有谓词判成功的局进 convert；验收 = 抓瓶子固定 plan 20 局与原 bottle 采集线逐字段比对 + 各训一个小模型（2026-10-03 实现：`Recording` 类，`--record` 拒绝 `--agent` / `--inject` / `--think-s`、要求 instance 0；每局 spawn 稳定后 c 开录，判据成立 c 存、否则 x 弃；成功局没有自己的 `training_tail` 时取 plan 结束处；行写进 `collect/manifest_00.jsonl` 并在 summary 记 `collect`，`convert_data.sh` 照常转；测试 `tests/test_plan_episode_record.py` 6 个。实跑 20 局与对照见 9.4.1）
    - [~] 19.4.1 验收：`sd_record_bottle_plan_20261003`（bottle 默认 plan，20 局，GPU1 / instance 0）与 `bottle_simple_guard_lifted_20260925` 用同一个转换器转出来逐字段比（`scripts/experiments/record_parity_20261003.py`）。2026-10-03：**字段一致**——37 个字段名 / dtype / shape、25 fps、指令"pick up the water bottle"、每个成功局都按 training_tail 截了；20 局 15 存 5 弃，与 manifest 逐行对上；成功率 15/20（抓取 3 败：抬起丢接触 2、合爪前碰倒 1；走近 2 败：腰没回中、停位够不着）。**内容差一处**：采集线的局多一段后退两步（`teleop.planner_movement` x 到 −1.0，plan 的不低于 0），中位时长 25.4 vs 23.0 s；plan 的 spawn 无横向 / 朝向抖动。"各训一个小模型"要占两张卡几小时，待用户定
      - [ ] 19.4.1.1 "各训一个小模型"对照（plan 录的 15 局 vs bottle 采集线），要占两张卡几小时：待用户定是否做
  - [~] 19.9 第一轮 C 方案（用户 2026-10-03 定，并同意先跑下列五个抓取试跑）：范围缺口按维度挑代表任务试跑（香蕉 / 0.90 岛台、小花瓶 / 0.39 架底、咖啡罐 / 0.70 推车 0.6 kg、钱包 / 1.12 顶层、梨 / 0.90 岛台深 0.25 m），结果写回 capability.yaml；放置与搬运的代表等抓取结果出来再挑；技能缺口 `place_on_surface.single` 进造技能排序。前置：ADT 场景跑通 plan_episode（实验文件在 outputs 下、出生点取场景文件的 spawn 区）
    - [x] 19.9.1 ADT 场景接上 plan_episode（2026-10-03）：仿真端读物体 / 台面 body 时跳过只是外观的静态物体（`run_sim_locomani`、`privileged_pub`）；台面 body = `adt_env`；骨盆初始位置放到 `robot_start`；守线刹车的桌沿（`edge_guard.facing_edge`）与离桌距离（`surfaces_edge_distance`）读台面朝向；`SceneModel` 用 Claude 的物体名；`plan_episode` 在场景文件房间里按占用图选出生距离（1.2 → 0.6 m，出生点要站得下、走入直线不穿家具轮廓）。试跑实验文件在 `outputs/scene_discovery/adt_apartment/experiments/`
    - [x] 19.9.2 **抓取高度是技能硬限制，不是试跑能测的范围**（2026-10-03）：bottle 抓取的 `sample_stand` 只规划下蹲 0.03–0.13 m 的模板，物体中心世界高度只接受 0.66–0.89 m，之外技能在机器人动之前就拒绝。原定五个代表里香蕉 0.93、梨 0.98、钱包 1.14、小花瓶 0.47 都在外面，只有咖啡罐 0.79 在内。能力说明加 `grip_height_accepted`，检查器把它记成技能缺口 `grasp_single.grip_height`：可试跑任务 326 → 72，1606 个任务要先扩抓取高度（与"放到台面"1664 个并列为第一轮造技能的候选）。试跑改为范围内的四个：咖啡罐（推车 0.70 m，0.6 kg）、中号白花瓶（展示柜格子 0.73 m，上有隔板）、青苹果（餐桌 0.75 m，离边 26.5 cm）、小盆栽（同格子）
    - [x] 19.9.3 可走近的边要有助跑（2026-10-03）：原判定只往外探 0.31 m，推车那一侧 0.5 m 外就是扶手椅（家具在 MuJoCo 里无碰撞，机器人会穿过去）；现要求站位后还有 0.6 m 通道（`approach_edges(runup=0.6)`，`adt_to_scene.py refresh` 原地重算）：可走近台面 53 → 35，模板任务 3747 → 2890（第 2 桶 1241、第 3 桶 1649）。第一次试跑队列（`sd_pilot_p3_coffee_canister_20261003`）因出生点检查拿膨胀图查站位全部秒败，作废
    - [x] 19.9.4 **物体原点要在包围盒中心**（2026-10-03）：第二次试跑（`*_20261003b`）咖啡罐 / 花瓶 / 盆栽 20 局的抓取全停在 `simple clearance ... adt_env`，失败记录里抓取目标点 z 0.701 = 推车台面——抓取代码按 `init_pos` z 算抓点（`box` 的中心偏移读的是 `center`，场景里写 `centre`，恒为 0），手工场景的物体原点本来就在中心所以看不出来，ADT 物体原点在底部，抓点落到台面上。改转换器而不是 bottle 代码：body 原点放在包围盒中心（`adt.AdtObject.body`，网格与 Isaac 自由体 USD 一起平移），`refresh` 原地更新位姿，测试 `tests/test_scene_discovery_adt.py`。修后咖啡罐第 1 局即成功。苹果 8 局"停位够不着"（物体在骨盆前 0.51–0.62 m）不是这个原因，但站位规划也读抓取高度，所以四个试跑全部重跑（`*_20261003c`）。Isaac 自由体 USD 待重导（`scripts/adt_to_usd.py`）
    - [x] 19.9.5 四个范围试跑结果（2026-10-03，`*_20261003c`，各 10 局，GPU1 / instance 1，结果写进 capability.yaml `pilots_adt_20261003`）：**咖啡罐 4/10**（推车顶层 0.70 m、开放台面、离边 4 cm、0.6 kg；抓起 4/8，败于抬起丢接触 2、合爪无双侧接触 1、接管时身体过线 1，走近 2 局被预检挡）；**花瓶 0/10、盆栽 1/10**（展示柜格子 0.73 m，上方有隔板：入手时手指扫到柜体 8 / 7 次）；**青苹果 0/10**（餐桌 0.75 m、离边 26.5 cm：走近 10 局全败，停位离物体 0.51–0.62 m 够不着 5、腰没回中 5）。结论：开放台面、离边近的物体在 0.70–0.79 m 抓取中心范围里能抓，成功率与罐头在 0.62 m 桌上的 6/10 同量级；有顶的格子和离边 > 0.2 m 都是现有技能做不到的，分别对应"从有顶格子里取"和"深够"两个新技能需求
    - [x] 19.9.6 Isaac 自由体 USD 按新原点重导（2026-10-03，`scripts/adt_to_usd.py`，64 个）
    - [x] 19.9.7 第一个新技能选哪个（决策 11）（用户 2026-10-03 定 A："单手放到台面"，由 Claude 造，见 9.12）：我推荐"单手把东西放到台面上"（`place_on_surface.single`，挡住 1169 个第 2 桶任务中的大多数；现有 `place` 已有抬 / 移 / 降 / 松，改判据与松手高度，成本最低），并作为 Claude 自己造技能的那一个；备选"扩抓取高度"（下蹲抓 2/10，贵）、"从有顶格子里取"、"深够"。待用户定
    - [-] 19.9.8 咖啡罐（4/10，过 ≥3/10 门槛）是否现在就作为第一轮第一个任务采集，还是等更多任务（用户 2026-10-03 定：等可采任务多了再一起采，现在不采）
    - [x] 19.9.9 "从有顶格子里取"与"深够"两个新缺口进检查器（现在只在试跑结果里，`capability.yaml pilots_adt_20261003`），重算它们各挡多少任务（2026-10-05：不是"有顶格子"——录像前的 trace 显示花瓶 / 盆栽都是手指 / 腕撞**托着它的那块板**（`adt_env` 只有台面板有碰撞），工具点离板 2–9 cm、伸进 15–16 cm，上方隔板 0.32 m 高；记成失败区域 `grasp_failures`：`deep_entry`（inboard ≥ 0.14 且抓点离板 ≤ 0.08 m）、`deep_reach`（inboard ≥ 0.26；阈值取整到让失败试跑本身落在区域里），按技能缺口处理（不再试跑）；另加头顶检查 `overhead.open_above` 0.60 m（规则，未测）：抓 / 放点上方 0.60 m 内有别的台面板记 `*.covered`。重算：`deep_entry` 433、`deep_reach` 88、`grasp_single.covered` 455、`place_on_surface.covered` 443 个任务；可试跑 264 → 85）
  - [x] 19.5 场景文件格式（扩展 `scene_meta.json`：物体字段带出处、台面、房间、随机化规格）与目标谓词库（2026-10-03：`docs/handoff/scene-file-format-20261003.md`，用户过目，三处按推荐：台面存 yaw 并改导航/站位、每局只把任务物体和干扰物做自由体、规则切房间 + Claude 命名；`scene_discovery.scene_file.validate/load` + `scene_discovery.predicates.compile_goal`（on/in/held/upright/near/in_room），`tests/test_scene_discovery.py` 7 个）
  - [x] 19.6 ADT → 场景转换：核实 `ApartmentEnv` 是否房间外壳（轴向、俯视图）；物体 glb → MuJoCo（碰撞体凸分解）+ Isaac/Unreal；Claude 估质量 / 摩擦 + 掉落测试；渲染器二选一（plan 决策 16）
    - [x] 19.6.1–9.6.3 坐标轴核实、墙与房间、台面（2026-10-03，`scripts/adt_to_scene.py geometry`，跑在 /venv/main；纯几何 `scene_discovery.adt/floorplan/surfaces` 在 sim venv 测试 9 个）：2 个房间（开放式客厅 / 餐厅 / 厨房 room_1、卧室 room_2，门洞 0.9 m 分开）；71 个台面（53 个有可走近的边），含外壳里的厨房台面 0.92 m × 3.1 m，楼梯按"≥3 级逐级一个台阶高"剔除；356 个物体（可移动 215，其中单手可抓 112、双手 1，131 个落在台面上）；校验 0 问题。产物 `outputs/scene_discovery/adt_apartment/{scene_file.json,floorplan.png}`
    - [x] 19.6.4 Claude 估属性（2026-10-03，`adt_to_scene.py estimate`，Opus 8 次调用 12.4 min / 4.24 USD，`scene_discovery.properties`，测试 4 个）：356 个物体的英文名、质量、摩擦、是否容器，71 个台面名，2 个房间名（`living_dining_kitchen`、`bedroom`）；规则只改 3 处（开口 < 6 cm 的"容器"否掉），校验 0 问题。3 个物体在 ADT 目录里没有模型（BookDeepLearning、DinoToy、Flask），几何阶段改为跳过，这份产物里删掉（`source.pruned`）
    - [x] 19.6.5a MuJoCo 构建与跌落测试（2026-10-03，`adt_to_scene.py mujoco` + `scripts/scene_drop_test.py`，`scene_discovery.mjcf/drop_test`，测试 3 个）：外壳与家具只有外观，台面是 2 cm 碰撞板，可动物体用凸包（容器用底 + 四壁），惯量按 Claude 质量的实心盒；默认自由体 = 能抓且落在台面上的 64 个（不在台面上的 48 个试过：47 个掉下去，没有碰撞体托着）；2 s 跌落 48/64 停住，不稳的 16 个是会滚的水果、靠不住的立书、展示柜上的两个花瓶，`semantic.drop_test.ok` 记录，任务发现只用通过的
    - [x] 19.6.5b 渲染器二选一（plan 决策 16）。**Isaac 通过**（2026-10-03，`scripts/adt_to_usd.py`（Blender 把 glb 连 PBR 贴图导出：静态 290 个进 `apartment.usdc`、64 个自由体各一份 USD，4.5 GB 在 `$ISAAC_ASSETS/scene/adt_apartment`）+ `scene_discovery.usd_scene`（每个房间天花板下 2.5 m 网格 12 盏球形灯：公寓封闭，外部穹顶光进不来，第一次全黑）+ `scripts/experiments/adt_render_check_20261003.py`）：三个站位三路相机，场景构建 20 s，进程显存 7.7 → 12.8 GB（GPU1 另有别的项目 10 GB，合计 ~23 GB 放得下），三路一起渲染 p50 0.148 / p90 0.194 s，贴图还原见 `outputs/scene_discovery/adt_apartment/render_check/isaac_only.png`（待用户看）。**Unreal 未测**：`images.source unreal` 只在 main / loco_unreal 上，本分支（基于 loco_agent）没有。**用户 2026-10-03 定：用 Isaac，不测 Unreal**（不合并 main，避免把 mars 机器人带进冻结的栈，决策 12）
    - [x] 19.6.6 导航与站位读场景文件（2026-10-03）：`OccupancyMap` 读 `room.bounds/walls/obstacles` 与台面 `yaw`（手工场景不变）；`plan.nearest_edge` 在台面自身坐标系里只在 `approach_edges` 中选边，`stand_plan` 的朝向加上台面 yaw；`SceneModel` 的 put-on 判据与 `front_line` 带 yaw；`round_loop` 遇到转角台面直接报错（它沿世界轴滑动目标）。`tests/test_scene_discovery_consumers.py` 4 个，现有 skills / bottle CLI 测试照常通过
  - [x] 19.7 能力说明文件（从已测包络整理：伸手高度与距离、桌面高度、下蹲 0.28 m、夹爪 9.5 cm / ~20 N、双手箱宽、行走限制）+ 程序可行性检查（2026-10-03：`configs/scene_discovery/capability.yaml` 写完，每条带 measured / measured_one / untested / unavailable 与出处文档；核对时改正三处出处，并发现流式全身姿态能下蹲（骨盆 0.57 m），不止 IDLE_SQUAT；程序检查 2026-10-03 完成，见 9.7.1）
    - [x] 19.7.1 程序可行性检查（2026-10-03，`scene_discovery.feasibility` + `scripts/scene_feasibility.py`，测试 8 个）：每个目标谓词展开成抓 / 搬 / 放三步，每步按 capability.yaml 判 ok / untested（记缺口名）/ blocked（记硬限制）；路线用场景导航图的真实绕行距离。ADT 公寓模板空间（48 个跌落测试通过的可抓物体 × 容器 / 台面 / 拿起）3747 个任务：**严格规则第 1 桶 0 个**，第 2 桶 1990，第 3 桶 1757（够不着的深度 935、台面前没地方站 316、容器在墙架 / 挂钩上 264、放不进 191、容器沿低于手能到的最低处 144、走不到 88）；按"试跑规则"（未测的也去试，试跑定生死）第 1 桶 1990。最简单的缺口组合是"物体尺寸 + 台面高度"（拿起香蕉等 5 个），没有哪个缺口单独补上能解锁任务。修了检查器两处：起点站位开逃生通道、架子上的容器在它所在台面的边上站；场景里门不再算障碍（第一轮按开着，MuJoCo 里门本来无碰撞），新增 `adt_to_scene.py obstacles` 只重写障碍不动 Claude 的字段（这份文件此前没有家具障碍，现 117 个）。产物 `outputs/scene_discovery/adt_apartment/feasibility/`
  - [x] 19.8 查重（scoop_check + 文献）：RoboGen、GenSim 及之后"LLM 自动生成任务 / 技能 / 数据"的工作，人形方向重点（2026-10-08：**Level 3 中度重叠**，7 条检索、238 篇、精读 7 篇，过程在仓库根 `step1.md`–`step7.md`，PDF 在 `papers/`（都未跟踪）。"缺口驱动的技能增长"已被 Agentic Skill Discovery（arXiv 2405.15019）写明，AutoRT（2401.12963）已有 LLM 把提议任务分给可用采集策略或按本体拒绝；站得住的差异 = 实测能力包络 + 程序检查 + 三桶 + 按解锁数 / 成本排序 + 人形全身移动操作数据训 VLA。决策 #1 的措辞待用户定）
    - [ ] 19.8.1 决策 #1 改写措辞（待用户定）：差异从"缺口驱动的技能增长"收窄到"实测包络 + 程序检查 + 成本排序 + 人形"
    - [ ] 19.8.2 让差异站稳的对照：只让 Claude 判可行（ASD / AutoRT 做法）vs 程序检查，量"判可行的任务实际能执行几个"
    - [ ] 19.8.3 待用户定：查重留下的仓库根 `step1.md`–`step7.md`（未跟踪）和 `papers/`（7 篇 PDF + 文本，183 MB，有版权，未跟踪）挪到 `outputs/scoop_check_20261008/` 还是删
  - [~] 19.10 Claude 提任务 + 程序检查反馈循环（决策 8；第 2 周）：现在第 2 桶是模板枚举出来的（`feasibility.templates`），Claude 读场景文件 + capability.yaml 提任务、检查结果回给它修改的那一环还没写（2026-10-05 循环已写并实跑一次：`scene_discovery.proposals` + `scripts/scene_discover_tasks.py`，Opus 3 次调用、5.32 USD、20.5 min，`outputs/scene_discovery/adt_apartment/discovery/opus_20261005b/`：49 个任务（客餐厨 32、卧室 17），三轮桶 2/25/13 → 2/32/11 → 2/38/9，修订把第 3 桶往第 2 桶推、没造出新的第 1 桶；第 1 桶就是模板空间里那 2 个（都找到了），49 个里只有 2 个在模板空间之外（带 near）；可试跑 6 个，5 个是咖啡罐放到别处。第一次运行 `opus_20261005_upright_bug` 作废：检查器把"搬过去并立着"里的 upright 判成"已经立着"无效（15/39），已修）
    - [ ] 19.10.1 Claude 给第 1 桶任务写固定 plan（决策 10：试跑 3 局验谓词、10 局 ≥ 3/10）：现在第 1 桶的 2 个任务 plan_episode 的 `default_plan` 已覆盖且实跑过（拿起 4/10、放到圆桌 12/20），等第 1 桶变多再做
    - [ ] 19.10.2 任务文件 → 运行：`plan_episode` 现在按实验文件的 `target` + `goal`（一个物体、一个台面或容器）判成败，谓词库 `compile_goal` 没接进来；`tasks.jsonl` 的一行（谓词合取、说法、`upright` 约束）还不能直接跑或录（9.14.1 之前要有）
    - [ ] 19.10.3 试跑提议里可试跑的 6 个（按外推距离：罐子 → 厨房岛台 0.90 m、→ 垃圾桶、→ 餐桌 0.75 m、→ 门厅长凳、→ 玻璃柜一格，白色大花瓶 → 马桶水箱），各 10 局，结果写回 capability（C 方案的做法，待用户定是否现在跑，约 1 h GPU）
    - [ ] 19.10.4 主结论 #1 的"任务广度"：Claude 提的任务 47/49 落在程序模板空间里，增量在挑选、说法和缺口说明，不在发现新任务形态；要么扩谓词 / 模板（near、多物体），要么把这点写进查重（9.8）的对比
  - [ ] 19.11 用户本人：给 projectaria@meta.com 发邮件问公开视频 / 衍生场景的许可（决策 17；2026-10-02 提过，未见回复记录）
  - [x] 19.12 第一个新技能 `put_down_single`（单手放到台面，用户 2026-10-03 定、由 Claude 造；规格与验收门槛在跑之前写定：`docs/handoff/put-down-single-20261003.md`）
    - [x] 19.12.1 实现（2026-10-03）：持物 `approach(台面)` 选空位并把放置带对准它（`surface_spot` / `surface_occupants`）；`put_down_single` 复用 `place_phase`（`_SurfaceGeometry`：静止高度、低处先蹲、落稳判定、站在台面上的判据）；`plan_episode` 对 `on(X, 台面)` 默认 plan 与判据；检查器在 `capability.yaml place.on_surface_single` 有实测后按范围判；测试 `test_skills_put_down_single.py` 9 个 + `test_place_phase.py` 5 个 + 检查器 1 个
    - [x] 19.12.2 验收：咖啡罐 → 小圆桌（0.52 m），抓取成功的前 20 局里走近台面 + 放下都成功 ≥ 10；预算 2 天 / 200 局（到 2026-10-05 18:15 UTC）（2026-10-03：第六版 **12/20 通过**，门槛外 2 局另计共 13/22；六版、171 局、2.3 h，逐版改动与证据在记录 §4）
    - [x] 19.12.3 验收后写回 `capability.yaml place.on_surface_single`（measured_one），重跑检查器与缺口排序（2026-10-03：可试跑 72 → 264，剩下的技能缺口只有 `grasp_single.grip_height` 882 个；第 3 桶 1630 → 1684；第 1 桶仍 0）
    - [ ] 19.12.4 蹲会让底盘后移 7–13 cm（第二版 2 局、第六版 2 局），降落够不着：蹲完、降落前按当前位姿再选一次点，或按后移量预先把手送远
    - [ ] 19.12.5 `approach(台面)` 持物原地转身掉罐 3/20（第六版）：复用的现有持物转身（`turn_with_load`，slow_walk）；看转角与掉的时刻，或改成不转身的横移
    - [ ] 19.12.6 交接后底盘仍会动（第六版 2 局脉冲拉不回、1 局转动中滑）：脉冲记录里加当时位姿定位；横向脉冲每次只动 0–3 cm
    - [ ] 19.12.7 "腰没回中"（`[stand] neutral waist not delivered`）在验收里频繁出现（第三版 3 次、第六版 ≥ 2 次），前一局成功、失败、抓取败之后都有，原因未明；不计入门槛但耗预算
    - [x] 19.12.8 检查器第 1 桶仍是 0：它只认 0.62 m 桌上的抓取实测，罐子在 0.70 m 推车上的抓取试跑（`pilots_adt_20261003`）没有进判据；接上后罐子 → 圆桌这类任务才会进第 1 桶（2026-10-05：capability.yaml 单手抓取改成实测设定列表 `grasp_settings`（瓶子线 + 咖啡罐推车试跑 4/10），检查器按设定**整体**匹配（各维分别测过不等于组合测过）；咖啡罐 inboard 按检查器口径是中心到边 0.084 m（试跑记录的 4 cm 是侧面到边）。第 1 桶 0 → 2：拿起咖啡罐、放到圆桌；放到书柜一格被 9.9.9 的头顶检查挡回第 2 桶）
    - [x] 19.12.9 成功局预览（2026-10-04）：第六版 13 局成功全部压成预览（30–44 万字节、全长、逐个解码与时长核对），`outputs/scene_discovery/adt_apartment/put_down_v6_success_previews_20261003/` 与同名 `.zip`（4.5 MB）
    - [x] 19.12.10 放进容器那条线回归：`place_phase` 的改动（几何对象分支、`_servo` 的 `until` / `level` / 日志）对 `place` 只有单元测试，pick_place 跑 10 局对照 §8 的 3/6（2026-10-05：`sd_place_regress_20261005`，同 §8 第三轮的 plan / 配置 / seed 300–309：抓取 6/10、place **3/6**、端到端 3/10，与 §8 的 6/10、3/6 相同；place 失败 3 局都是 `descent timeout` xy 6.5 / 7.4 / 9.0 cm，§8 是 6.0 / 9.0 cm，同族；没有新失败形态）
    - [x] 19.12.11 现有抓取对罐子朝向很敏感（长边与走近方向夹角 |a| ≤ 20° 抓住 6/9，20–40° 2/5，> 40° 0/18）：咖啡罐采集（9.9.8）前要么限制罐子朝向，要么抓取按窄边选入手方向；capability 里还没记这条（2026-10-05：选了限制朝向——`plan_episode.spawn_object_yaw`：底面只有一边放得进夹爪的物体，长边限制在走近方向 ±20° 内（两侧各一个窗），其余物体照旧均匀随机，原有 seed 的局不变；capability.yaml 记 `grasp_yaw`。抓取按窄边选入手方向没做，是技能改动，未实跑验证新出生朝向下的抓取率）
  - [ ] 19.13 第一轮采集吞吐（2026-10-05 补记，估算未实测）：`put_down_single` 第六版 80 局抓住 22、端到端成功 13（约 16 %），每局约 48 s；按决策 14（≤ 25 个任务 × 每任务 100 局成功）一个任务约 600 局、单实例约 8 h，25 个任务约 200 h；`--record` 目前要求 instance 0，Isaac 补渲另算（warehouse 实测约 68 s/集）。里程碑 1（第 3–4 周）前要有实测和对策
    - [ ] 19.13.1 实测：第一批第 1 桶任务各录 10 局，记成功率、每局墙钟、补渲耗时，按实测重算一轮总时长
    - [ ] 19.13.2 提吞吐的选项（待用户定）：`--record` 放开到其他实例（栈已有 `RECORD_INSTANCE=1`，plan_episode 没接）两卡并行、先提高抓取成功率（9.12.11）、或减每任务局数（改决策 14）
  - [ ] 19.14 第一轮数据 → VLA-1 的后半段在 ADT 公寓里一步都没跑过（2026-10-05 补记；决策 2 / 13 / 15 / 16；`--record` 只在 bottle 场景做过逐字段比对，9.4.1）
    - [ ] 19.14.1 冒烟先行：取一个任务约 10 局走通 record → Isaac 补渲 → convert → 短训 → 闭环几局，在正式采集前暴露 ADT 特有的问题
    - [ ] 19.14.2 Isaac 补渲 ADT 公寓的数据集三路相机（ego + 双腕；`ISAAC_ASSETS` 指主目录），核对与 9.6.5b 渲染检查同图、只渲成功局
    - [ ] 19.14.3 convert：多任务合一个数据集，每局随机说法进指令字段；留出说法与留出组合在采集前定好、不进训练
    - [ ] 19.14.4 训练 VLA-1 两个种子（决策 15，从基础模型）
    - [ ] 19.14.5 ADT 公寓 lockstep + Isaac 闭环评测：同场景随机摆放，"见过的组合 / 留出组合 / 留出说法"三格分开报，每格 ≥ 20 局（决策 2）
  - [ ] 19.15 `upright_axis` 与场景实际姿态对不上（2026-10-05 发现）：64 个自由体里 26 个按 `upright_axis` 算倾斜 75–125°，但跌落测试里都静止（倾角变化 0.3–11.6°）。书是 ADT 的"正立"= 平放、书在格子里竖着放（90°）；盆栽 2 个、香蕉、钥匙、遥控器、收纳盒等原因未核。后果：检查器给这些物体的 `upright(X)` 记 `reorient` 缺口（Opus 任务里 3 个盆栽任务因此带 `reorient`），运行时 `upright(X)` 判据在成功局也会判失败。待定：按初始姿态重定义"正立"（初始就静止的姿态算正立），或逐类核对后修 `adt.AdtObject.body`；先看一张盆栽 / 书的渲染再定
  - [ ] 19.16 第二个新技能选哪个（决策 11，待用户定）：49 个提议任务里 32 个缺技能，几乎都是抓取。按挡住的任务数：抓取高度 28（台面高度 0.91–0.99 m 8 个、肩高 1.11–1.15 m 8 个、低于 0.66 m 且多在有顶格子里 12 个；模板空间 882）、深处低位入手 13（模板 433）、深够 1（模板 88）、放到指定物体旁 1。我推荐先做台面高度抓取（不蹲、不钻格子，对现有抓取改动最小），再延到肩高
    - [ ] 19.16.1 扁平物体（钱包、车钥匙、木勺、平躺香蕉，厚 1.5–3.8 cm，8 个任务）实际要"从上往下捏"，检查器只记成 `grasp_single.object_size`（范围缺口，会被当成可试跑）；是否加一条按厚度判的技能缺口，待定
  - [x] 19.17 报告 doc「公寓场景任务发现」（2026-10-05，用户要求）：https://claude.ai/code/artifact/132adcd3-de5d-4ff1-99ba-1bc683f47549 ，两页（正文七节 + 流程图；任务清单 49 行）；私有，含 ADT 衍生任务清单，Meta 答复（9.11）前不分享到团队外
  - [~] 19.18 合入 main 并改用 mars + Unreal（用户 2026-10-08；DECISIONS 2026-10-08 两节）：做在分支 `scene_discovery-sync`（worktree 已删），合入前的分支打 tag `scene_discovery-pre-sync`
    - [x] 19.18.1 合并 main（`10b3895`）：16 个冲突；分支编号 7/8/9 改为 17/18/19；`make check` 全过
    - [x] 19.18.2 mars 场景 `adt_apartment_mars`（`adt_to_scene.py mujoco --robot mars`；64 个自由体换成带贴图的扫描网格，凸包只管碰撞）；`check_mars` 过；MuJoCo 渲染核对 `outputs/renders/adt_mars_check_20261008/`
    - [x] 19.18.3 Unreal：`export_mjcf` 导出全部自由体；`build_level --room-level`；`adt_to_usd.py --unreal` 烘焙房间；关卡 `adt_apartment_mars` 建成，与 MuJoCo 同状态三路并排核对（`outputs/renders/adt_mars_check_20261008/`，`0fcceaf`）
      - [ ] 19.18.3.1 曝光沿用 warehouse 的数值、没有参照图校准，木地板在灯下偏亮；要不要按 Isaac 那张渲染检查图或照片调，待用户看图定
    - [x] 19.18.4 回归（pick_place 台账 §18）：同机同 seed，合并后 / 合并前 bottle 7/10 vs 6/10、pick_place 2/10 vs 4/10、p4 3/10 vs 3/10，看不出差别；mars 上 p4 1/10
    - [ ] 19.18.5 `capability.yaml` 的实测都是 stock 机器人上的；mars 的伸手 / 抓取包络要重测后检查器才可信（19.7）
    - [ ] 19.18.8 mars 上单手抓罐子 5/10 在合爪前被间隙检查拦下：右腕 `right_wrist_yaw_link`（带 D405 支架）碰到罐子（stock 上没有）。要在 mars 上重标 `bottle_simple` 的抓取几何，或查支架碰撞体是否比实物大
    - [ ] 19.18.6 冒烟：mars + Unreal 上录几局 → Unreal 补渲 → 转换（19.14.1 的前半段）。没跑：`plan_episode --record` 录的运行现在补渲不了（Isaac 也一样，分支上从没走通过），缺三样：
      - [ ] 19.18.6.1 `plan_episode` 起栈不设 `SIM_STATE_TRACE_DIR`（采集栈在 `pipeline/run_stack.py` 设），离线出图没有状态轨迹可补
      - [ ] 19.18.6.2 不写 `collect/inputs.json`（`world_lift.RunTraces` 和 `view_render.finished_sessions` 读它），只写 `summary.json`
      - [ ] 19.18.6.3 `RunTraces` 要运行目录下恰好一份场景副本、带 `target_object`；plan 运行不复制场景，ADT 场景文件的 `target_object` 是空的（目标在实验文件的 `task.target`）
      - 实验文件已备好：`outputs/scene_discovery/adt_apartment_mars/experiments/p4_canister_on_round_table_unreal.yaml`（`images.source: unreal`、`backend.render: offline`、50 fps、相机延迟）
    - [ ] 19.18.7 双臂技能（`skills/catalogue/crate.py`）不读 `visual.ready_after_brake`，自带 ready；main 09-28 起采集默认 compact、10-03 起 warehouse 在 mars 上只从 ym 边走入。技能的双臂线没在 mars / 新默认上跑过（原 17.11 的 4/10 是 stock、`warehouse_textured`）
    - [x] 19.18.9 快进到 main（2026-10-08，用户定；`6f4d645`，之后 STATE 两个提交），没推送
    - [x] 19.18.10 推送 origin/main（用户 2026-10-08：全部推送，43 个提交，`b8f5e9c..d6a8208`；推前查过：没有 outputs / third_party / datasets 路径、没有大于 5 MB 的文件、没有 ADT 签名链接）
    - [x] 19.18.11 删三个旧 worktree（2026-10-08，用户）：先 unlink 数据软链接再 `git worktree remove --force`；主目录 `outputs/` 核对完好
    - [ ] 19.18.12 可选：删已合入的分支 `scene_discovery-sync`、`scene_discovery`、`loco_agent`（`git branch -d`；tag `scene_discovery-pre-sync` 保留）——待用户定
    - [ ] 19.18.13 loco_agent 那个 Claude 会话（pid 962611）还开着，工作目录已删：用户在那个终端退出它

## 20. warehouse 手臂规划：CuRobo 与 mink 的对照（用户 2026-10-08；SIMPLE 论文调研之后；同日用户叫停，DECISIONS 2026-10-08 最后一节）

脚本 `scripts/experiments/curobo_warehouse_20261008/`、`scripts/experiments/mink_warehouse_20261008/`；输出 `outputs/curobo_warehouse_20261008/`、`outputs/mink_warehouse_20261008/`；数据是 `warehouse_mars_50` 的 120 局（119 局有伸手段）。

- [x] 20.1 专家自己的路线查碰撞（2026-10-08，`cases.py` / `hand_pose.py` / `ghost_path.py`）：标称抓取位姿（沿下 1 cm）离箱壁 11.4 mm，119 局都不碰；抓深只在手绕夹爪轴外偏时可行（2 cm 偏 −10～−40°，3 cm 偏 −20～−30°）。专家的预抓路线（ghost，未加跟踪修正）伸手、对齐段 0/119 碰壁，下降段 33/119 碰（手指或掌压进壁顶，多在下降开头，抓点跨壁偏 4–6 cm）；碰壁的局成功 8/33，不碰 42/86（相关，未证因果）
- [x] 20.2 距离判据：MuJoCo 3.11 的 `mj_geomDistance` 对分开的网格偶尔返回 0.0（11520 对里 100 对，平移 20 µm 即得真值），还出现过 +11.4 mm 报成 −11.4 mm；改用表面采样到盒子的符号距离（`cases.Surfaces`，约 2 mm 精度，1 cm 壁穿透读数饱和在 −5 mm）
- [-] 20.3 CuRobo（0.7.4，SIMPLE 的 venv）对照：mars 模型自拟合内切球 722 个（CuRobo 自带拟合漏 14–30 mm；0.7.4 自碰撞上限 1024 球），FK 与 MuJoCo 逐关节一致。我们的 IK 跑完 238 个请求（119 局 × 两种起点 × 3 种抓法）；CuRobo 跑了 61 个后用户叫停（每次成功规划约 27 s：722 球走慢速自碰撞核、关了 CUDA graph、整条轨迹精调）。已完成的 25 局里两边可解的局几乎重合（CuRobo 多 0–3 局，没有 IK 能解而 CuRobo 不能的），失败主要是够不着；部分日志 `curobo_plan_stopped_partial.log`
- [x] 20.4 mink（1.3.0，装在 `outputs/mink_warehouse_20261008/pylib`，没动 `.venv_sim`）离线跟踪专家的对齐 + 下降预抓目标，只动腰 + 双臂 17 个关节，手用盒子代理几何（比真实外形保守 0.4–11 mm）：
  - mink 1.3.0 的 `CollisionAvoidanceLimit` 上界单位错（QP 变量是位移 Δq，上界按速度写，dt 0.03 s 时放宽 33 倍，实测一步从 +12.5 mm 走到 −15.4 mm），改成乘 dt 的子类 `displacement_collision_limit`
  - 世界静止（身体和箱子停在对齐开始时）：不加约束 101/119 碰壁，加约束 16/119；到位（< 1 cm）115 → 108；既不碰又到位 94/119
  - 身体和箱子按实测轨迹每帧写入：加约束仍 40/119 碰壁（身体晃动和被撞动的箱子把手推进壁里，只动手臂来不及），既不碰又到位 59/119
  - 每帧（4 次 QP）2.1 ms（加约束）/ 1.0 ms（不加），够 50 Hz
- [x] 20.5 （用户 2026-10-08）mink 碰撞约束接进 `CoordinatedReach` 的对齐 + 下降段（`visual.collision_guard: mink`，`src/locomani/tasks/crate/collision_guard.py`，默认 off），真栈 20 局配对：关 9/20、mink 7/20（只有关成功 6、只有 mink 成功 4，p 0.75）；卡箱沿 6 → 3、进入合爪 14 → 17，脱手 0 → 4，合爪时跨壁误差 24 → 34 mm。同配置同 seed 的 `warehouse_mars_50` 是 3/20，20 局量不出这个差别。实测的手两组都只有 2/20 压进箱壁 > 1 mm。台账 `docs/ledger/crate.md` 2026-10-08 最后一节
  - [-] 20.5.1 （用户 2026-10-08：这条线先停）可选：只约束掌和 Dex1 底座、不约束手指（检验"手指被推开 → 跨壁误差变大 → 脱手"的猜测），或加局数到每组 60 以上
- [-] 20.6 （用户 2026-10-08：这条线先停）可选：向 mink 报 `CollisionAvoidanceLimit` 上界单位问题（对外发布，需用户同意）

## 21. SIMPLE（arXiv 2606.08278）调研与可借鉴项（用户 2026-10-08；DECISIONS 2026-10-08 最后两节）

- [x] 21.1 读论文与公开代码（2026-10-08）：公开的自动采集 = 手写 `decompose()` + CuRobo 手臂 + 开环底盘指令，AMO 驱动，数据是 43 维关节动作、不是 SONIC token；走路类运动规划任务、BoDex 抓取合成、CoACD / 跌落求稳定姿态的代码都没放；SONIC 线只有 VR 遥操作和固定动作片段。记忆 `simple-paper-covers-infra-layer`
- [x] 21.2 跑 SIMPLE 的 `G1WholebodyTabletopGraspMP` 一集（2026-10-08，GPU0）：第 5 次规划才成功（前 4 次 `IK_FAIL`，这个物体只缓存了 2 个右手抓取），3.7 s，不走路；`outputs/simple_tabletop_grasp_mp_20261008/`，预览 `simple_tabletop_grasp_mp_ep0_grid_preview.mp4`（158 KB；原来交付的 1.4 MB 版超了 500 KB 上限，2026-10-08 补做预览）
- [~] 21.3 待用户选：从 SIMPLE 借鉴的采集改进（2026-10-08 列出，未定；CuRobo / mink 那条见第 20 节，已停）
  - [ ] 21.3.1 补渲时按局随机外观（材质、光照、只渲不碰的干扰物），一条轨迹渲 K 份；SIMPLE 表 3 的消融每格只有 10 局，要自己按 2 个种子对照（用户 2026-10-08：做，先写计划之后再测——[计划](docs/handoff/appearance-randomization-plan-20261008.md)：属性表、实现 6 步、A/B 各 2 种子 × 原样 / 留出外观各 100 局；待用户定留出目标）
    - [-] 21.3.1.1 （用户 2026-10-08）前置检查：基线 checkpoint 换光照还能不能做（用户同日叫停：外观泛化本来就是要克服的问题，21.3.1 不论前置结果都做，DECISIONS 2026-10-08 末节）。留下的：Unreal 镜像 `--light-scale` / `--light-temperature`（`in_editor/lighting.py`，只改 PIE，未提交）；候选档 `outputs/light_shift_20261008/`，×0.4 + 3200 K 第一视角均值 105 → 51、箱沿仍清楚。16.15.3 的训练跑到约 500/10000 步停掉，没有 checkpoint
  - [-] 21.3.2 跌落测试列出物体所有稳定姿态，按稳定姿态生成物体、判"正立"（对上 19.15、19.12.11、19.16.1）（用户 2026-10-08：没必要，物体直接摆在桌上；接 21.3.4 物体库并随机翻面时再说，DECISIONS 2026-10-08 末节）
  - [-] 21.3.3 开局前用站位规划 + `ArmIK` 判可行、不可行就重抽布局（要记录拒绝比例，防止数据分布变窄）（用户 2026-10-08：不做。离线拿 `outputs/curobo_warehouse_20261008/eval_baseline_ik.json` 对 `warehouse_mars_50` 119 局结局：实际站位上 IK 误差预测成败 AUC 0.56；最好的切点 ≤ 10 mm 良品率 0.42 → 0.49、丢 19/50 成功局；开局前只知道计划站位，只会更弱。记忆 ik-feasibility-does-not-predict-warehouse-success）
  - [~] 21.3.4 物体 / 场景库：SIMPLE 已转好的约 75 个 GraspNet 物体（CoACD 碰撞体 + USD + 稳定姿态）和 HSSD 场景（本地 3.9 GB）；用之前先查许可
    - 用户 2026-10-08 定用途：把 GraspNet 物体摆到 bottle 场景的桌上，用 GraspNet 方法合成抓取位姿，交给 bottle 线 SONIC 驱动的同一套专家去抓——证明接上抓取模块后同一套专家逻辑能抓起这些物体。
    - 事实（2026-10-08 查）：专家的抓取只有"抓点 + 水平进入方向的 yaw、手指水平"（`simple_policy.grasp_rotation`，抓点来自 `TabletopGrasp.grip`）；实测过的只有直立圆柱 4.5–6.1 cm 宽、14–16 cm 高。按稳定姿态算，75 个里 33 个有"窄边 ≤ 9.5 cm 且高 ≥ 8 cm"的姿态（现有侧抓大致够得着），32 个够窄但矮（< 5 cm 8 个、5–8 cm 24 个，要斜抓或从上往下），10 个任何姿态都比夹爪开口宽。`dex_grasp` 是 Dex3 三指手的，用不上；GraspNet-1B 的平行夹爪标注不在本机。
    - 用户 2026-10-08 定：抓取用 graspnet-baseline 从深度点云预测（B）；机器人直接初始化在桌边、不走路，用我们的 IK 执行模块给的抓取。
    - 事实（2026-10-08，`scripts/experiments/graspnet_bottle_20261008/orientation_reach.py`，输出 `outputs/graspnet_bottle_20261008/`）：站在桌边、抓点在桌面上 7 cm 时，右臂 + 腰（腰按模型真实限位）的 IK 能到 37 个接近方向里的 27–33 个，且 Dex1 手不碰桌；只有指尖朝机器人、贴桌水平的几个方向到不了。抓点只高 3 cm 时只剩 30° 斜下 10 个方向。没查前臂 / 肘和物体的碰撞，也没查 SONIC 实际能跟多准。
    - 用户 2026-10-08 加：再跑 SIMPLE 的 CuRobo planner（`simple.mp.curobo.CuRoboPlanner`，0.7.4，SIMPLE 的 venv）和我们的 IK planner 对比。位置在"抓取筛选"之后、"执行"之前：同一批 graspnet 抓取、同一个起始姿态，两个 planner 各出一条到预抓 / 抓取的路线。先离线比（规划成功率、耗时、离桌面和物体的最小间隙、路线长度，沿用 `scripts/experiments/curobo_warehouse_20261008/` 的 `evaluate.py` 格式），再真栈配对比（同物体同 seed）。要做：给 bottle 用的机器人建 CuRobo 配置（stock G1 + Dex1；20.3 只做过 mars），世界用桌面方块 + 物体（网格或点云体素）。
    - [x] 21.3.4.1 （用户 2026-10-08："直接执行完"）65 个物体各跑 IK 一局、CuRobo 一局，左右对比视频 66 个（≤ 500 KB）在 `outputs/graspnet_bottle_20261008/videos/`。抬起 IK 16/65、CuRobo 15/65；规划耗时中位 1.5 s 对 17 s；失败主要在执行（合爪没夹住、抬起脱手、碰倒）。[记录](docs/handoff/graspnet-bottle-planner-comparison-20261008.md)
    - [ ] 21.3.4.2 待用户：要比较两个规划器，每个物体多跑几个朝向；执行层的三种失败（合爪未同时接触、抬起脱手、进入碰倒）要不要接着修
      - 2026-10-10：执行层三种失败的方案列在第 23 节（23.2 位姿缓存 + 验证、23.4 接触）
    - [ ] 21.3.4.3 待用户（2026-10-09 提出）：ScaleBFM（mode 1/2）与 SONIC POSE 流 + 我们的 IK，在同一组手部目标（如 65 个物体的预抓位姿，含倾斜姿态）上开环执行，比手的到达误差；09-08 的 84 格对比里 SONIC 走的是 planner 通道，不可比。接入代码 `src/locomani/backends/scalebfm/`，约半天到一天
    - [~] 21.3.4.4 可选：CuRobo 候选补齐到固定批量并预热后重测耗时（17 s 里约 10 s 是新批量形状的准备，对比时没补齐，SIMPLE 自己会补齐；估计约 6 s，未实测；`profile_curobo.py`）
      - 2026-10-09：主线服务（22.2）固定批量 2 + 预热后，单目标请求中位 2.9 s（119 个已存请求重放）；多候选批量没测
    - [x] 21.3.4.5 （用户 2026-10-12 commit and push，已提交并推送）提交 21.3.4 的实验代码 `scripts/experiments/graspnet_bottle_20261008/`、`docs/handoff/graspnet-bottle-planner-comparison-20261008.md` 和 `docs/README.md` 的索引行（都未提交；README 那行指向未提交的文档）
    - [ ] 21.3.4.6 待用户：调试运行删不删——`outputs/runs/gnb_obj_*_smoke*`（40 个，682 MB）、`gnb_obj_*_batch1`（10 个，154 MB，代码修正前的第一轮批跑）、`outputs/graspnet_bottle_20261008/videos_full/`（778 MB 全分辨率原片）；`third_party/graspnet-baseline` 8.3 GB 是运行环境，保留
      - 2026-10-10 新增待定：`ghost_rounds/`（740 MB）、`videos_single_full/`（390 MB）、`fixed/`（1.7 GB，含 `videos_full/` 965 MB、`ghost_rounds/` 763 MB）、`outputs/runs/gnb_obj_*_fixed`（1.3 GB）、`*_fixed_oldstance`（284 MB）
    - [x] 21.3.4.7 （用户 2026-10-09）对比视频补品红 ghost，不重跑：`ghost_overlay.py` 离线配准录像相机再叠 ghost + 右上 2.5 倍放大框；66 个预览重出（2026-10-09）
      - [x] 21.3.4.7.1 修 GraspNet CuRobo 服务"单候选成功也报错"：10-08 那轮 053、060 的"找不到路线"是它（2026-10-09）
    - [x] 21.3.4.8 （用户 2026-10-09）固定同一个抓取重比：`run_grasp.py --fixed-grasp`，IK 11/65、CuRobo 11/65；开环到预抓误差 3.3 对 4.1 cm（p 0.002）；结果 `outputs/graspnet_bottle_20261008/fixed/`，记录第 5b 节（2026-10-09）
      - [ ] 21.3.4.8.1 待用户：局序抵消重跑——第 1 局（CuRobo）朝向在 58/58 个物体上更顺时针（中位 4.8°），一半物体先跑 CuRobo 可去掉，约 1.2 小时
    - [x] 21.3.4.9 （用户 2026-10-10）并排视频拆成单局、文件名带 `success` / `fail`：`split_videos.py`，`videos_single/` 与 `fixed/videos_single/` 各 130 局，`index.csv` 列失败原因（2026-10-10）
  - [ ] 21.3.5 待用户（2026-10-12 读 SIMPLE 源码时列出，为导师问"可学之处"）：SIMPLE agent 层值得借的三点，DECISIONS 2026-10-12 有出处
    - [ ] 21.3.5.1 抓取缓存按物体稳定姿态分键：每个稳定姿态各建一份缓存，在线按实测姿态取最近的一份（现在只有 make_scenes 选的那一个静止姿态，物体被碰倒后候选被桌面检查滤光）
    - [ ] 21.3.5.2 计划步改成带类型的子任务（参数有名字、可校验），替代现在的原语名字符串 + dict；要回答：对 `skills/contract.py` 的 preconditions 有没有增量
    - [ ] 21.3.5.3 双臂时第二只手的抓取按第一只手已选的抓取过滤（SIMPLE `store_grasp_key` / 按接近方向筛）；bimanual 线目前没有这一步

## 22. 单臂主线接 CuRobo 抬手路线（用户 2026-10-09；[计划与验证](docs/handoff/single-arm-curobo-route-plan-20261009.md)；DECISIONS 2026-10-10）

- [x] 22.1 计划与验证方法：每条"没 bug / 有效 / 不明显变坏"三关、A/A 定误报率、40 对配对（2026-10-09）
- [x] 22.2 第 4 条：CuRobo 服务进 `src/`（`primitives/curobo_service.py`、`arm_route.py`），固定批量 ≥ 2、重置种子、预热、探针 + 重建自愈；119 个请求重放成败 117/119 一致，单目标 54/57 与旧服务实跑相同（2026-10-09）
- [~] 22.3 第 1 + 2 条：开关 `reach.simple.arm_route: curobo`（默认 lifted_ready），左手下垂
  - [x] 22.3.1 左臂按关节姿态保持（`loop_prior`），否则闭环把左肘折到 0.1 rad（2026-10-09）
  - [-] 22.3.2 路线直达预抓（B / B' 33、29；B' 4 局手指贴桌被闸门拦下；腰横滚中位 0.48 rad）——否决，DECISIONS 2026-10-10
  - [-] 22.3.3 直达预抓 + 腰横滚限 ±0.05（12 个停位 0/12 能规划）——否决
  - [x] 22.3.4 方案 (b) 抬到原地抬手的抬起位姿（手腕位姿目标）：C / C' 34、34，抬手 0 接触；进入阶段闸门拦下 3 局（2026-10-09）
  - [x] 22.3.5 改关节空间目标（协议 2）：D / D' 36、35（A / A' 35、34），抬手 76 局 0 接触、0 局 < 1 cm，进入闸门 0 + 1，慢 2 s（2026-10-10）
  - [ ] 22.3.6 待用户：默认改不改成 curobo；改的话同步 `docs/single_arm.md`、change map、`make config-doc`
  - [ ] 22.3.7 待验证：`[stand] neutral waist not delivered` 只出现在跑 curobo 的实例 3 / 4（专家开始前），在实例 1 / 2 跑一组 curobo 才能和开关分开
  - [ ] 22.3.8 可选：`route_joint_speed` 0.6 → 约 1.0，把多出的 2 s 拿回来（要再跑 2 × 40 看跟踪误差）
- [-] 22.4 第 3 条离线抓取库（用户 2026-10-09：先不做，主线瓶子 / 罐头本来就是已知物体 + 几何抓取模板）
  - 2026-10-10：GraspNet 物体那条线上重新提出，见 23.2（主线瓶子 / 罐头不受影响）
- [ ] 22.5 待用户：单臂主线 `comfort_waist_weight` / `comfort_elbow_wrist_weight` 从未生效（`simple_policy` 不调用 `blend_posture`），修不修
- [x] 22.6 （用户 2026-10-12 commit and push，已提交并推送）提交主线改动（`git status` 里 9 个已改文件 + `primitives/arm_route.py`、`curobo_service.py`、`tests/test_arm_route.py`、`scripts/experiments/curobo_route_20261009/`、`configs/experiments/bottle_route_*`、计划文档）；`make check` 已过（2026-10-09）
- [ ] 22.7 （用户 2026-10-13：先不删，留着）本线产物删不删——`outputs/runs/bottle_route_*`（14 个，5.2 GB）、`outputs/curobo_route_20261009/`（4 MB）


## 23. 仿真专家的抓取来源：graspnet 感知 → 特权几何 + 物理验证（用户 2026-10-10 提出；2026-10-13 用户定：抓取位姿用物体表面对抓缓存）

- 定了（用户 2026-10-13）：抓取位姿来自物体表面采样的对抓 + 物理验证缓存（23.2 的 `grasp_cache.py`，2026-10-12 起已是默认 `--grasp-source cache`）；graspnet 只留作对照选项，不再往下做（23.7.4 关掉）。23.3 / 23.5 / 23.7.5 / 23.7.6 是缓存框架内的后续，照旧待用户。

- 背景（2026-10-10 会话）：用户判断"仿真里有物体全部信息，graspnet 对任意物体合成抓取没必要"。支持的事实：21.3.4 每局重新感知时 graspnet 输出 65/65 不同（站位只差 3 cm）；graspnet 不认识 Dex1，候选质量实际由我们手写的转换 + 过滤层决定；专家本来就是特权教师；SIMPLE 公开代码也只走 BoDex 离线缓存（`detect_grasps` 只有 `pass`）。两轮失败都在执行层（合爪未同时接触 15/19、抬起脱手 13/9、进入碰倒 12/8），但位姿是这些失败的上游。
- 讨论结论：机器人侧"最方便抓"和物体侧"抓得住"是两个独立判据，只按前者选会选到边缘 / 单指垫接触的位姿；"按实际停下位置动态选"在缓存框架里免费（缓存在物体系，站位变只是换被选中的候选），不需要每局从头合成。领域通用解法 = 离线采样 + 仿真验证（ACRONYM、Dex-Net）+ 在线按可达性选（SIMPLE 缓存、CuRobo goalset）。没有解析法能保证物理可行，仿真本身当裁判是领域答案。
- [x] 23.1 用户 2026-10-10 定：graspnet 感知抓取暂作备选项，不删；先做方案 A 到验收（2026-10-10）。原题：定 graspnet-baseline 感知抓取是否退出专家线（退出后 `graspnet_server.py` 专用 venv 和深度渲染一并退出；21.3.4 的场景转换、Dex1 转换 + 碰撞过滤、三层排序、IK / CuRobo 规划、闭环执行、ghost 视频都与来源无关，保留）
- [x] 23.2 方案 A（推荐，用户 2026-10-10 开工，同日跑完验收）：ACRONYM 式离线抓取缓存 + 在线按可达性选
  - [x] 23.2.1 离线采样：在物体网格上采反向法线点对（Dex1 开口 ≤ 9.5 cm、指垫覆盖 ≥ 1.5 cm），接近方向限定在专家标定过的三层姿态内，过现有 `hand_cloud.py` 碰撞过滤——复用 `grasp_planned.py` `candidates` 的过滤层，只换输入
    - 2026-10-10 做法（`grasp_cache.py` `sample`）：可视网格面积加权采 20000 面点 + 法线，3000 个起点沿内法线找对侧最近点（离线 3 mm 内、法线夹角 ≤ 30°），闭合轴 = 掌心 y，绕它 12 个接近方向 × 3 个深度；桌面 / 指垫包络 / 手点云净空 / 直线进入净空同 `candidates`；1 cm + 15° 去重。物体 000：1623 对 → 364 个；采样 21 s（最近邻查询加距离上界前 336 s）。没有限定三层姿态——层级在线按朝向算（`_tier`），缓存保留全部
  - [x] 23.2.2 物理验证：MuJoCo 里把夹爪摆到候选位姿、合爪、抬 10 cm、晃，按保持时间和滑移打分，掉的丢。必须用专家同一套夹爪控制器和接触参数（kp、20 N 钳位、指垫 solref），否则"验证通过"和"专家抓得住"仍是两回事
    - 2026-10-10 做法（`grasp_cache.py` `HandModel`）：MjSpec 从场景 XML 里把右腕子树（Dex1 底座 + 两指滑动关节 + 指垫接触参数）、物体 body、桌面盒拼成只有手的模型；手指力学照抄仿真：tau = 1.5 (q_des − q) − 0.1 dq，钳 20，gear 250，关节 actuatorfrcrange ±20（deploy `dex3_hands.hpp` sizeCommand 的 kp 1.5 / kd 0.1）；200 Hz；合爪 1.0 s → 0.8 s 内双指接触 → 抬 10 cm 2 s（minjerk）→ 停 1.5 s → 晃 ±2 cm 1.5 Hz 各 2 s（x、y）；score = held ? 1 − slip/2 cm : 0
    - 踩的坑：手直接挂在 mocap body 下时 mocap 几何体瞬移、速度为零，摩擦带不动物体（0/40）；改成自由腕 + weld 到 mocap；weld solimp 0.95/0.99 时手自重下垂 3 cm、带物 6 cm，吃掉抬升，改 0.9999 后抬起 8–9 cm，和真实仿真成功局 7.4–12.4 cm 一致
    - 物体 000：300 验证 → 200 held（52 没抬起、41 晃滑、5 晃掉、2 抬掉），每个候选 0.06 s
  - [x] 23.2.3 缓存按（物体，稳定姿态）存，复用 `make_scenes.py` 的稳定姿态重建；要回答：65 个物体各有多少验证通过的候选、10 个比夹爪宽的物体是否仍无候选
    - 2026-10-10：`cache_all.sh 12` 65 个场景 5 分钟；每物体 300 个验证（物体 005 只有 26 个）；64/65 有通过的，中位 243/300；009（2.2 cm 高、9.3 cm 宽，超开口）0 个候选。验证失败合计：晃掉 1090、抬掉 897、晃滑 320、没抬起 62、合爪无双指接触 22。场景只有一个稳定姿态（make_scenes 选的），所以缓存按场景存：`scenes/obj_XXX/grasp_cache.npz` + `grasp_cache_summary.json`
  - [x] 23.2.4 在线选择：真值位姿变换缓存，按 IK 可达 + 关节限位余量 / 可操作度代价选第一个（按实际站位动态选，不重新合成）；CuRobo 侧用 goalset 对整组候选一次批量规划
    - 2026-10-10：`grasp_planned.cached_candidates`（`run_grasp.py --grasp-source cache`）：只取 held 的，按测得物体位姿变换，重查桌面净空，同一套 `_tier` 分层，层内按验证分；IK 规划器按顺序取第一个能解的（和 graspnet 来源一样，没加关节余量代价——那是 23.3）。冒烟物体 000：80 个合适、46 个第 0 层、第 2 个候选可解、规划 1.2 s、开环到预抓误差 3.1 cm、抬起 9.1 cm（graspnet 线这个物体没抬起）。CuRobo goalset 没做（验收只跑 IK）
  - [x] 23.2.5 验收（2026-10-10 04:38–05:34 跑完：`ROUNDS=3 EGL_GPU=1 bash run_batch.sh 4 1 0 _cache --grasp-source cache --planners ik --rounds 3 --yaws 3`，运行名 `gnb_obj_XXX_cache`，第 0 局 = 21.3.4 的朝向；汇总 `cache_summary.py` → `outputs/graspnet_bottle_20261008/cache/summary.md`）：同一 65 个物体重跑（每物体多个朝向，见 21.3.4.2），和 21.3.4.1 的 IK 16/65 对比；把失败拆成"验证通过仍失败 = 执行问题"和"位姿问题"两类，决定 23.4 值不值得做
    - 结果（handoff 第 5 节）：第 0 局（同朝向）缓存 25/65 对 graspnet 16/65，配对只缓存 18 / 只 graspnet 9（p 0.061 单侧）；三局至少一局 45/65 物体；193 局里执行失败 103（碰倒 49、无双指接触 30、抬掉 18）、规划够不到 17、没候选 8。位姿问题已基本排除，剩下在执行层 → 23.4 值得做。副作用：规划耗时中位 20 s（候选 ≤ 80 逐个解），要排序或 goalset（23.2.4 未做部分）
- [ ] 23.3 方案 B：机器人侧"方便 / 省力"代价的定义与检验——关节限位余量、可操作度、静力矩（腰 + 右臂）；先离线在 21.3.4 的 130 局上算代价和成败的相关性，再决定进不进 23.2.4 的选择代价
- [ ] 23.4 方案 C：执行侧接触问题（与位姿独立；23.2.5 拆出执行失败占 53% 的局，优先级升到最前）
  - 2026-10-10 解剖（handoff 第 9 节，260 局）：碰倒 67 = 进入时手相对指令横向偏 3–5 cm（−y 侧、2 号指先碰），低 / 斜抓取为主；无双指接触 47 = 薄（开口 1–2.3 cm → 0.75–2.6 N）+ 矮（指垫擦顶）；抬掉 29 = 轻薄物体；IK 无路 17 = 最矮物体
  - [x] 23.4.5 （用户 2026-10-10 开工，同日跑完两批）解耦右臂：专家的关节目标照旧给 SONIC 当参考，同时每 tick 送仿真侧 arm pin（`run_grasp.py --arm-pin arm`，`grasp_planned.DECOUPLE`），右臂电机按位置环跟 IK，平衡仍归 SONIC
    - 事实（2026-10-10）：第一版 pin 让手差 9 cm——三个原因依次修掉：pin 的 PD 没有重力前馈（kp 120 下伸直的臂垂 4° ≈ 3–5 cm，加 `qfrc_bias`）；腰一起 pin 时 IK 的 0.52 rad 前倾被完整交付、躯干 32°、支撑闸门跳（腰 pin 改用控制器增益也不行，21 cm，弃用）；**IK 的右臂构型会把肘和肩偏航连杆压进躯干 8–9 cm**（右肩横滚指令 +0.3..+0.9 rad，策略从来只交付 0.1），闭环的 WholeBodyIK 又不吃 `ik.lo/hi`——加 `RIGHT_SHOULDER_ROLL_MAX` 0.05 并把规划器的关节上限传进 `SupportedMotion(joint_limits=)` 后，pin 的关节跟踪到 0.01–0.04 rad，000 抬起、013 两局 1/2（不 pin 的横向误差 8–11 cm，pin 后 4.6–5.4）
    - 推论：之前归到 SONIC 跟踪的 3–5 cm 偏差，有一部分是 IK 给了自碰撞的参考、控制器拒绝执行
    - [x] 23.4.5.1 两批 65 局 `_pinA` / `_pinB`（presort 开、pin 右臂）对 `_presort` / `_presort_b`（2026-10-10 09:45–10:26）：抬起 22、26 对 16、24（合计 48 对 40 / 130，Fisher p 0.19）；碰倒 22 对 35 局（p 0.04）、无双指接触 28 对 25、抬掉 17 对 15；低抓取（离桌 < 6 cm）20/59 对 13/58。进入段"实测手 − 指令"横向误差不变（2.9 对 2.8 cm），高度误差 3.4 → 2.0 cm；pin 后关节跟踪 0.01–0.04 rad、IK 自身手误差 1 cm——剩下的横向偏差来自骨盆 / 躯干在进入时的晃动（IK 按实测骨盆解、手到位前骨盆已动），不在手臂。汇总 `cache/pin_compare.txt`
    - [x] 23.4.5.2 用户 2026-10-10 定：pin 设为默认（`DECOUPLE on`，`--arm-pin arm`）；`_pinC` 批 26/65
  - [x] 23.4.6 （用户 2026-10-10）拆骨盆晃动 + 三根杠杆（handoff 第 12 节，三条都否定；基线 `_pinD` 26/65 完好）：轨迹加骨盆位姿（46 列），`sway_decompose.py` 逐 tick 分解 `_pinC` 58 局——骨盆晃动只占 0.2 cm；横向 2.9 cm 的来源是闭环 IK 指望骨盆前移 / 俯仰 / 下沉而 SONIC 没交付（~3.2 cm）、腰偏航没跟上（2.0 cm，0.08 rad）、pin 的 PD 稳态差（1.2 cm，0.04–0.065 rad）；handoff 第 11 节
    - [-] 23.4.6.1 冻结闭环 IK 的骨盆自由度（用户 2026-10-10 定做）——做了三版都不行，放弃：只收范围时虚拟基座仍锚在构造时的骨盆，等于没冻；每 tick 重锚到实测骨盆 + 实测腿 + 重取脚 / 质心参考后，解算器在腰限位内够不到低目标，SupportedMotion 的跟踪反馈和进入段的横向 shift 积分跑飞（013 目标跑到 108 cm 外）；把实测航向流回 SONIC 还会让偏航正反馈跑 33°。**更正**：分解里的"ik 项"大部分是 SupportedMotion 加在目标上的积分反馈偏置（我离线没减掉），不是骨盆没交付；轨迹已加反馈列（49 列），`sway_decompose.py` 现在把它单独拆出。代码留着（`WholeBodyIK(base_range=)`、`anchor`、`SupportedMotion.anchor_base`、`--base-freeze`，默认 off）
    - [-] 23.4.6.2 腰偏航 + 横滚 pin（`--arm-pin arm+yaw`，控制器增益，俯仰留给策略）——有害，第一批就停：`_yawA` 11/65（arm pin 三批 22、26、26），碰倒 20 局，骨盆横向晃动 3.9 cm / 13 cm/s（arm pin 1.2 cm / 7 cm/s），反馈偏置 6.8 cm、解算器残差 7 cm；腰偏航 / 横滚是策略用来平衡的自由度，pin 住它们和 SONIC 打架。冒烟里关节项 2.8 → 2.0 cm 的改善没有传到成功率。选项保留，默认 `arm`
    - [-] 23.4.6.3 pin 增益 120 → 300 / kd 6（`ARM_PIN_KP` / `ARM_PIN_KD` 环境变量，`run_sim_locomani` 读，已进 lint 白名单）——有害，第一批就停：`_kpA` **0/65**（碰倒 23、抬掉 16、无双指接触 19）；关节误差没降（0.054 对 0.049 rad，这是 50 Hz 台阶指令的滞后，不是稳态差），tick 间抖动 +50%，手臂变硬后合爪 / 抬起时指垫失去顺应性。4 局 kp 300 复核也 0/4
    - 基线核实：`_pinD`（kp 120，当前代码）26/65，arm pin 四批 22、26、26、26；分解（反馈拆出）：解算器目标与实测骨盆 FK 之差 6.2 cm，反馈补 4.5，关节 2.8，骨盆瞬时 0.2
    - 事实（2026-10-10 12:40–13:04）：追"回归"追了一小时是白费——用"三批都成功的 11 个物体"做快速对照，kp 120 只有 2/6、3/8、7/16、8/16，连把 pinC 之后的改动全回退也 8/16。这是对成功结果做选择后的均值回归（这些物体的真实成功率约 0.6），不是回归；核实基线只能跑整批 65 局（`_pinD`，13:04 起）
    - 旧 23.4.5.2 原题：pin 要不要设为默认（`--arm-pin arm`，现在默认 off）；下一层是骨盆晃动——轨迹里加骨盆位姿列再解剖，或进入段按实测横向误差闸门（23.4.4）
  - [x] 23.4.4 （用户 2026-10-10 开工，同日做完：闸门成立、高度优先否定）进入段按实测横向跟踪误差闸门或放慢、横向修正速度 / 上限（3 cm/s / 4 cm）加大；候选排序里优先离桌面高的抓取——要两批起测（每批 ±5 个物体噪声）
    - [x] 23.4.4.1 进入段闸门 `grasp_planned.ENTRY_GATE`（`--entry-gate on`）：实测掌面横向 / 垫高偏差 > 1.5 cm 时推进停住等横向修正追上，每局最多等 2 s；两批 `_gateA` / `_gateB` 32/64、33/65 = 65/129 对 arm pin 四批 100/260（p 0.017），碰倒 14/129 对 22/130；合爪时偏差不变，变的是到合爪的局数和碰倒数（handoff 第 13 节，`cache/entry_compare.txt`）（2026-10-10）
      - [x] 23.4.4.1.1 闸门设为默认（用户 2026-10-12：`ENTRY_GATE["on"]` True、`--entry-gate` 默认 on）；修正上限 4 → 6 cm 没试，用户叫停尝试（2026-10-12）
    - [-] 23.4.4.2 候选优先离桌高 `grasp_planned.PREFER_HIGH`（`--prefer-high on`）：层级内按工具点离桌高度 2 cm 分桶降序，桶内按验证分；两批 `_highA` / `_highB` 24、24/65 = 48/130 对基线 100/260（p 0.83）；只有一半物体换了抓取、中位只高 1.4 cm，换了的反而略差（27 对 36）（放弃：没收益，2026-10-10）
    - 事实（sway 分解 `_pinD`）：离桌 0–6 cm 抬起 12/30，6–10 cm 5/7，≥ 10 cm 9/18——高度和物体高度混在一起，不是单调
  - [ ] 23.4.1 CoACD 凸块脊缝导致指垫单点接触：检查"合爪未同时接触"失败局的接触点是否落在凸块交界；可选指垫对原始网格或更细分解
  - [ ] 23.4.2 抬起脱手（偏重物体，中位 0.14 kg）：按质量和摩擦算需要的夹持力，对照 F = 1.5 × 250 × (setpoint − q) 与 20 N 钳位，定 close_target / 钳位要不要调
  - [ ] 23.4.3 摩擦与滑移：condim / noslip_iterations / 子步对抬起滑移的影响；指垫蠕变已知 0.1 cm/s（硬化后）
  - [x] 23.2.6 （用户 2026-10-10 开工，同日做完）规划耗时——缓存候选 ≤ 80 个逐个解，中位 20 s（graspnet 1.5 s）；按 IK 可达先排序
    - [x] 23.2.6.1 预检排序 `grasp_planned._presort`（`PRESORT`）：每个候选从挂手姿态解抬手点、预抓上方、4 步下降、预抓、抓取；可行（≤ 1.5 cm / 10°）优先于层级，层内保持验证分顺序。离线重放 168 局（`replay_plan.py` / `replay_summary.py` → `cache/replay/summary.md`）：关 9.6 s 中位 / p90 43.6 / 最长 84；直接预检 4.0 / 40.3 / 80；分段 4.5 / 41.3 / 82；分段 + 可行优先 3.6 / 5.6 / 24，选的抓取 167/168 不变（2026-10-10）
    - 事实：尾部 41 局是第 0 层候选全部在 1 cm 步进路径里失败（抬手段跳分支 15、下降段差 1.7–2.1 cm 14、上方 7），粗预检判它们可行；只能靠"可行优先于层级"绕开。关节余量排序（margin）无收益
    - [x] 23.2.6.2 在线核实 65 物体 × 1 局 `_presort`：规划中位 15.4 → 4.1 s，p90 60 → 7.7，最长 129 → 10；预检本身 1.9 s；选同一抓取 47/57；抬起 16/65 对 25/65——同一抓取的 47 个物体里 9 个翻成失败、2 个翻成成功（p ≈ 0.07），批内抬起率与规划时长无关（42/46/28/44%），漂移和预抓误差反而更小（2026-10-10）
    - [x] 23.2.6.3 A/A 复跑定噪声底（2026-10-10 06:59–07:47）：关 25、18 / 65，开 16、24 / 65；合计 43 对 40 / 130（Fisher p 0.40）；关/关之间翻转 13 + 6，和关/开之间一个量级；任意两批选同一抓取 47–52 / 57。结论：预检排序不改变成功率，规划中位 15.4 / 19.4 s → 4.1 / 4.1 s，p90 60 / 63 → 7.7 / 6.6，最长 129 / 127 → 10 / 9.4；已设为默认（`PRESORT` on，`--presort off` 可关）。汇总 `outputs/graspnet_bottle_20261008/cache/presort_compare.txt`
  - [x] 23.2.9 （用户 2026-10-10）最佳配置（缓存 + 预检 IK，`_presort_b` 第 0 局）65 局带 ghost 夹爪的完整视频：`ghost_overlay.py --tag _presort_b`（58 局有 ghost，7 局没选中抓取）→ `best_videos.py` → `cache/videos_best_full/`（466 MB）+ `cache/videos_best/`（≤ 500 KB，index.csv）（2026-10-10）
  - [~] 23.2.7 运行目录已删（用户 2026-10-10：10-10 新增的 749 个 `gnb_obj_*` 运行目录，`outputs/runs` 158 → 150 GB）；剩下待用户——`outputs/runs/gnb_obj_*_cache`（65 个，2.1 GB）、`gnb_obj_000_cache_smoke`、23.2.6 的 `gnb_obj_*_{presort,cache_b,presort_b}`（195 个，约 2.3 GB）、`cache/replay/`（小）、`presort_b/ghost_rounds/`（配准录像，约 0.5 GB）、`cache/videos_best_full/`（466 MB）；缓存本身 `scenes/obj_*/grasp_cache.npz` 很小，留
    - 事实（2026-10-12 复核磁盘）：上一行列的 `_cache`、`_presort*` 等运行目录已在那 749 个里删掉；现在还在的是 `gnb_obj_*_{gateA,gateB,highA,highB}`（260 个，2.2 GB）、`gnb_obj_040_demo*`（3 个，20 MB）、`presort_b/ghost_rounds/`（392 MB）、`cache/videos_best_full/`（202 MB）、`demo/` 与 `cache/videos_demo*`（10 MB）；以及 10-08/09 的 `gnb_obj_*`、`_fixed*`、`_batch1`、`_smoke*`（21.3.4.6）
  - [x] 23.2.8 （用户 2026-10-12 commit and push，已提交并推送）提交 23.2 的代码（`grasp_cache.py`、`cache_all.sh`、`cache_summary.py`、`replay_plan.py`、`replay_summary.py`、`presort_compare.py`、`grasp_planned.py` / `run_grasp.py` / `run_batch.sh` 改动）、`docs/handoff/grasp-cache-20261010.md`、`docs/README.md` 索引行
- [ ] 23.5 方案 D（实验开关，不默认开）：判定抓住后把物体焊到夹爪（weld），用来把位姿问题和接触问题分开看；代价是抬起 / 搬运动力学失真、带载行走的创滑看不到，不进正式采集
- [-] 23.6 方案 E：每次停下后在线从头优化新抓取（MPC / 采样）——放弃：和 23.2.4 相比没有收益，每局新位姿无法预先验证，接触型 MPC 仍是研究问题
- [x] 23.7 文档（用户 2026-10-12 要求，文档和接口同日做完；剩下的 23.7.4 于 2026-10-13 关掉：读者页 `docs/object_grasp.md`、详解 `docs/object_grasp_design.md`、README 支线一行、术语表三条、代码导航一行、两份 handoff 补默认值变更）：23.1 定下后把结论写进 `docs/handoff/graspnet-bottle-planner-comparison-20261008.md` 或新 handoff，DECISIONS 记"感知式抓取退出专家线"的决定和理由
  - [x] 23.7.1 抓取来源接口（2026-10-12）：`GraspPlanned(source=)` 显式参数，计划步可用 `source` 覆盖，`record.json` 记 `source`；`run_grasp.py` 默认改为 `--grasp-source cache --planners ik --rounds 1`，`run_batch.sh` 的 `ROUNDS` 默认 1；动手臂前的就绪检查（`grasp_cache` / `graspnet_server` / `grasp_source`），`graspnet_server.py` 加 ping 和 `--graspnet-dir`；`cache_summary.py` / `best_videos.py` 按 tag 分开输出、字幕写来源；`run_one.sh` 尊重 `paths.sh` 选的 deploy
  - [x] 23.7.2 `install_graspnet.sh`（2026-10-12）：克隆固定提交、扩展补丁、钉版本的 venv、编译 pointnet2 / knn、graspnetAPI、权重、自检；在空目录从零装通（104 s，安装包已在 uv 缓存里，自检 110 grasps），补上了原环境后装的 graspnetAPI 依赖（matplotlib 等），绕开镜像里不可写的 `/.uv` 和 CUDA 13 的 `CUDA_HOME`
  - [x] 23.7.3 冒烟（2026-10-12，实例 1，物体 040）：缓存来源抬起（74 s）；graspnet 服务没开时动手臂前被拒（`graspnet_server`）；graspnet 来源在现在的默认配置（预检、arm pin、闸门）下抬起（80 s，服务 1.6 GB 显存）
  - [-] 23.7.4 （2026-10-13 关掉：用户定抓取来源用缓存，graspnet 的当前对比不再补）原题：graspnet 来源在现在的默认配置下两批 65 局（约 40 分钟），给详解第 3 节补一组同执行逻辑的当前对比；现在的对比是 10-10 之前的执行逻辑
  - [ ] 23.7.5 待用户：`make_scenes.py` 只读 SIMPLE 的 GraspNet 物体集；加一个读任意网格（obj / stl + 质量）的入口，缓存和 runbook 才对任意物体成立（详解第 6 节局限）
  - [ ] 23.7.6 待用户：缓存只在一个静止姿态下验证（同 21.3.5.1）；物体出生被碰歪时 `no cached grasp fits here`

## 24. SONIC 对 ScaleBFM：65 物体抓取线上的控制器对比（用户 2026-10-12 提出；[计划与第 1 步](docs/handoff/wbc-compare-20261012.md)；DECISIONS 2026-10-12 末节）

只换全身控制器，手程序、抓取、站距、指标两边相同；受控臂 + 一条放开姿态层的 ScaleBFM 臂；结果是诊断和论文资产，不承诺换线。

- [x] 24.1 手指律统一 + 场景适配器（2026-10-12）：`ScaleBfmBackendConfig.grip_clip`；`scripts/experiments/wbc_compare_20261012/bfm_object_scene.py`（SONIC 律配置、`verify_hand`、`stance`、`check`）；65 场景 0 摔 0 合不上，右掌垫垂边偏置 4.2 cm；`tests/test_scalebfm_grip_law.py`。
- [x] 24.2 开环探针（2026-10-12）：27 格（前 0.30/0.38/0.46 × 左 −0.20/−0.12/−0.04 × z 0.67/0.74/0.81，接近下俯 30°，格间回家），`probe_cells.py` / `bfm_probe.py` / `sonic_probe.py` / `probe_summary.py`；偏置 ScaleBFM (−0.4, +0.3, +1.8) cm、SONIC (−1.2, −4.1, +1.0) cm，减常数后残差 p90 3.0 对 5.2 cm；SONIC 的误差是腰 yaw / roll 只交付一半（handoff §4）。
  - [x] 24.2.1 用户 2026-10-12：SONIC 不补腰，按决定 2 原样只减常数。
  - [x] 24.2.2 用户 2026-10-12 定修：`grasp_planned.apply_waist_limits(ik)` 在 `StreamedArm` 接管后再套一次（探针同）；SONIC 探针重跑 23/27 可达、偏置 (−0.5, −2.8, +0.7) cm（宽腰 (−1.2, −4.1, +1.0)），yaw 仍只交付一半；基线 `_wbc2` 48/130 对 `_wbc` 48/129 配对 21 : 21（p = 1）——进入撞倒 19 → 4、预抓误差 2.9 → 2.3 cm，但 IK 无路径 12 → 18、tier 2 选中 19 → 32、合爪无接触 19 → 35：交付精度换可达性，净效零（handoff §8，`compare/sonic_wbc_vs_wbc2.md`）。
- [x] 24.3 手程序移植（2026-10-12）：`bfm_grasp.py`，接近开环减偏置、settle 起 `SupportedMotion` 同款跟踪反馈 + 同一套 gate / 修正 / 合爪 / 抬 / 保持，放开臂沿用规划器的路径清障拒绝；SONIC 侧 `grasp_planned.BIAS` + `HOLD_S`、`run_grasp.py --bias-file/--hold`；配对冒烟 040 抬起（handoff §5）。
- [x] 24.4 批次（2026-10-12）：SONIC `_wbc` 48/129（偏置 + 2 s 保持）与对照 `_wbc_nobias` 53/130（p 0.54，偏置对 SONIC 无效）；ScaleBFM 受控臂 `bfm_grasp/paired` 77/130，放开臂 `bfm_grasp/free` 55/130。
- [x] 24.5 汇总（2026-10-12）：`compare_summary.py` → `compare/{paired,free}.md`（配对表、失败类、三个稳定量）；`compare_videos.py` → `videos/paired/*.mp4` + `reel.mp4`、`videos/free/`。受控臂 SONIC 48 对 ScaleBFM 77，不一致 10 : 39，p < 0.001；见 handoff §7。
- [x] 24.6 ScaleBFM 姿态交付包络（用户 2026-10-12 定做；handoff §8）：`bfm_attitude_probe.py`（仰角 7 × 偏航 12 × 滚转 2 = 156 姿态 × 2 高度，5 min 不录像）→ `attitude_summary.py` → `probe/scalebfm_attitude/envelope.{json,md}`：可交付 31/156（≤ 5 cm 且 ≤ 15°），全在虎口朝左、偏航 −60..+90°；虎口朝右（滚转 180）一个都没有。受控臂 `paired2`（按 `_wbc2` 配对）SONIC 48 对 ScaleBFM 68 / 129（13 : 33，p = 0.005；`_wbc2` 多选的 tier 2 让 ScaleBFM 从 77 掉到 68）；包络事后分 SONIC 选的抓取：包络内 92% 抬起、外 37%；放开臂 `free2`（`bfm_grasp.py --envelope`）SONIC 48 对 ScaleBFM 104 / 129（3 : 59，p < 0.001；没过滤时 55），预抓误差 p90 14.9 → 5.3 cm，合爪无接触 0。
  - [x] 24.6.1 （用户 2026-10-13 定做，handoff §10）按姿态前馈 `bfm_grasp.py --bias-attitude`：放开臂开环预抓误差中位 3.3 → 2.8 cm、受控臂 2.9 → 3.2 cm，抬起都不变（105 → 105、68 → 67）；不比全局常数好，默认不变。原题：按包络给每个姿态单独一个前馈常数（`envelope.json` 里已存 `bias_heading_cm`）；包络只在一个物体、两个高度上探的，位置 × 姿态的交叉没扫。
- [x] 24.7 终点线刹车，SONIC 对 ScaleBFM（用户 2026-10-13 提出；handoff §9）：空地走到 1.5 m 的线再停，0.55 / 0.35 / 0.25 × 10 局；`bfm_brake.py --program marker|plan`、`brake_batch.sh`、`brake_summary.py`，`runtime.Sim` 允许无物体场景，渲染复用 `render_brake_marker_20260923.py`（按行线位置 + 字幕标签）；`tests/test_wbc_brake.py`。越线刹车惯性 ScaleBFM 4.5 ± 0.6 / 2.5 ± 0.6 / 0.5 ± 0.4 cm 对 SONIC 24.7 ± 3.3 / 10.4 ± 4.9 / 4.0 ± 1.4（横向 sd ≤ 1.6 对 4–5 cm，yaw 0.2° 对 2–3°，0.5 s 站稳）；不越线程序两边 0/10 越线，ScaleBFM 脚尖停线前 4–8 cm（sd ≤ 2）对 SONIC 守线 13–22 cm（sd 3.5–12，最差 −2.3）。表 `outputs/wbc_compare_20261012/brake/brake.md`，视频 `brake/videos/`。
  - [x] 24.7.1 （用户 2026-10-13 定做，handoff §10）`bfm_brake.py --ff speed` 按速度补停短量：误差 −1.6 / −3.9 / −4.5 → −0.2 / −0.2 / +0.5 cm，0/30 越线；常数 3 cm 在 0.55 m/s 越线 3/10，不用。原题：ScaleBFM plan 停稳比参考终点短 1.6–4.5 cm（0.25 有两局 8–9 cm），加 3 cm 前馈再验；要不要在 65 物体线让 ScaleBFM 真走进去（决策 4 站位不动，是另一条线）。

## 25. SONIC → ScaleBFM 替换评测（用户 2026-10-13 提出；[计划与冻结判据](docs/handoff/controller-compare-20261013.md)；DECISIONS 2026-10-13（续））

替换范围是仿真采集、仿真 VLA 评测和真机部署三处；VLA 学习效果不进判据。L1 原语层过关才移植专家跑 L2；判据开跑前冻结（计划 §2–§4）。

- [x] 25.1 否决关：W0（mars 上站 30 s、走 1.5 m、弯腰到 0.18 / 0.33 m）、a（mars 标称：腰 50、踝 35 N·m；任一项 10 次摔倒 ≥ 3 次才否决）、h（本机同 TensorRT 版本测推理 p99 下限）；任一不过就停下查原因并汇报。
  结果（2026-10-13，计划 §9）：W0 + a 80 局 0 摔（站 10、走 30、弯腰 40；mode 2 开环躯干少弯 6–7°、腕差 8–15 cm，mode 1 不弯腰；站立时脚外滑 6 cm，stock 同样）；h 本机 TensorRT FP32 p99 0.90 ms、eager CUDA 8.6、CPU 12.4 ms，FP16 偏 0.5 rad 会摔、不能用。
  - [x] 25.1.1 `foot_slip` 换成按着地段的净位移（2026-10-13：逐帧接触点积分会把抖动积成 7 cm；`bottle_measure.foot_slip` 和 W0 的在线统计同一规则）；W0 重跑见 25.2.4。站立脚外滑（23.7 → 40 cm / 30 s）仍待和 SONIC 对比。
- [x] 25.2 bottle L1（stock）：B1–B8，生产档 + 外推档，每格 10 次。结论见计划 §14：无致命短板，ScaleBFM 停位和正手可达明显更好，虎口朝右和伸手时躯干姿态不如 SONIC。
  生产档结果（2026-10-13，计划 §10–11）：SONIC 生产 10 局（6 成功）对 ScaleBFM 走近（余量 3 / 6 cm）+ 回放 SONIC 实际轨迹（mode 1/2/6/7）。B1 桌边余量 19.3 ± 4.8 对 3.4 ± 1.4 cm（6 cm 余量，0 越线；3 cm 余量 5/10 越线，手甩 2–5 cm），横向 sd 7.7 对 0.9 cm、yaw 7.3° 对 0.4°；B2 2.6 对 0.8 s；回放 mode 6 在 SONIC 抓起的 6 局全部抓起、拿到最后，SONIC 失败的 3 局回放也失败；ScaleBFM 躯干多俯 5–9°。
  - [x] 25.2.1 用户 10-13 选 A：SONIC 实时率作废门槛 0.9 → 0.85（空闲机器上只有 0.85–0.88），报告注明实测 RTF；L2 前再考虑 lockstep。
  - [x] 25.2.2 B3 可达包络（10-13，计划 §13）：各自走进去的站位，108 格，SONIC 41 对 ScaleBFM 63；虎口朝右 11 对 4（ScaleBFM 的短板）；SONIC 第一次探针因换姿态 bug 作废后重跑。
  - [x] 25.2.3 外推档（10-13，计划 §13）：0.7 m/s 两边都越线（SONIC 6/9、ScaleBFM 3/10，6 cm 余量不够）；±40° 都没影响；携带四种，ScaleBFM 24/24 拿到最后，SONIC 22/23，后退 2 m 掉 1 次。SONIC 携带计划的实时率未核对。
  - [x] 25.2.4 W0 重跑（10-13，`w0/run3`）：80 局 0 摔；站立脚滑 6.6 cm 是真滑。
- [x] 25.3 warehouse L1（mars）：W1–W8，生产档 ym 边，外推档另三条边。结论见计划 §17：无致命短板（0 摔；SONIC 成功的 5 局 ScaleBFM 都抬起），双手扣箱容易斜（平抬 3–4/8 对 6/8），停位散布小 2–17 倍，带载走路两边都差。
  - [x] 25.3.1 回放双手扣箱要夹爪中心 + 姿态伺服（`bfm_warehouse.py --servo pad`）：开环只抬起 3/8（手腕姿态差 10–30°，夹爪偏 2–9 cm）；L2 移植 warehouse 专家必须带这个闭环（计划 §15、DECISIONS 25.3）。
  - [x] 25.3.2 skills profile `warehouse_box_mars`（mars 生产参数 + stand_forward_min 0.50）：stock profile 在 mars 上 10 局只 1 局进携带；换后携带 3–6 局走到，但 SONIC 全掉（倒退 0/3、转身 0/6）。
  - [x] 25.3.3 （10-14）C6 地面搬箱 mode 6 / 7 关键帧都 0/40，mode 3 才 10/10；warehouse 移植专家按 L1 种子选了 mode 2（计划 §20 / §22）。原题：L2 前：mode 6 和 mode 7 平抬率相同（4/8）、mode 7 箱子更平，在 25.4 的 mode 扫描里再比；合爪判据要看两指都夹住。
- [x] 25.4 （10-14，计划 §18 / §22）复杂动作 C1–C7（mars）：ScaleBFM 选对 mode 后 C1 / C4 / C6 / C7 10/10，SONIC 带关键帧的片段都 0/10；两边都做不到 C2 / C3 / C5；mode 3 / 4 比 mode 2 好得多。原题：先下载 G1 重定向 LAFAN1、写 IK 关键帧合成；mode 扫描（§5）；S1 切换瞬态。
  - [x] 25.4.1 （10-13）片段库 16 个（`clips.py`，计划 §18）：LAFAN1 镜像 `datasets/lafan1_g1_retarget/`（Unitree 原仓库 404），IK 关键帧在 mars 上合成；C1 最低 0.164 m、C5 前倾 65°（双脚平踩、质心在脚上方时的上限，DECISIONS）。
  - [x] 25.4.2 （10-13）ScaleBFM 全 mode 扫描 940 局（+S0 50 局）0 摔；SONIC 流式改发完整骨盆姿态（只发偏航那批 C1 连摔 6/6，作废）。
  - [x] 25.4.3 （10-13）S1：ScaleBFM 加关键点的切换（2→4/6/7）平滑，手腕移 1.6–2.5 cm；去掉肩肘的切换（6/7→2）手腕掉 21 cm；SONIC 交接 1–3 cm（`switch/s1.md`）。L2 每个任务只用一个 mode，没有切换。
  - [ ] 25.4.4 可选：SONIC 用 planner 通道（movement / speed / facing）跑 C4 和 L0，和流式通道对比——流式不收骨盆位置，G-MPKPE 17–69 cm 是接口造成的（DECISIONS 10-14）。
- [x] 25.5 （10-14，计划 §19 / §24）SONIC 怕延迟（40 ms 下 bottle 摔 9/10，warehouse 站不稳），ScaleBFM 不受影响；估计位姿臂：ScaleBFM 用腿运动学 + IMU 后刹车 30/30 越线，两个移植专家都失败——真机世界位姿来源待用户。原题：扰动 a–g 按计划 §4.6 套用；估计位姿臂（腿运动学 + IMU 估计器，§4.5）。
  - [x] 25.5.1 （10-13）`src/locomani/sim/perturb.py` 两边共用（b c d f g；e 用 `payload_scenes.py` 的场景副本）；腿运动学 + IMU 里程计 `backends/scalebfm/odometry.py`（站立漂移 < 0.2 cm，ScaleBFM 后退滑脚 → 60–90 cm）。计划 §19。
- [x] 25.6 （10-13，用户指示连续做完，这条线按 §3 自判，计划 §21）L1 关口通过：0 摔、a / h 不否决、§3.4 不成立（warehouse 移植专家弱是专家和 SONIC 一起调出来的问题，记为 L2 风险）。原题：L1 关口判定（计划 §3），结果交用户。
- [x] 25.7 （10-13，计划 §20）移植专家：bottle simple 专家 → ScaleBFM（主臂）；warehouse 四边专家 → ScaleBFM（各段用选出的 mode）；bottle_visual 参考臂有余力再跑（没跑）。
  - [x] 25.7.1 （10-13）`bfm_expert_bottle.py`（simple_policy guard 分支）L1 种子 mode 4 10/10 对 SONIC 6/10；`bfm_expert_warehouse.py`（CoordinatedReach 原样）最好 mode 2 + tracking_gain 0.5 2/10 对 SONIC 5/10——专家和 SONIC 一起调出来的（计划 §20，DECISIONS）。bottle_visual 参考臂没跑。
- [x] 25.8 （10-14，计划 §25）L2：bottle 标称 40 对 39 / 45、组合 2 对 35 / 46；warehouse 标称 20 对 6 / 40（p 0.003）、组合 1 对 4 / 40。主判据不满足（warehouse 专家和 SONIC 一起调出来的），组合档不否决，bottle 标称掉物护栏不过。原题：L2：每场景 ≥ 40 局配对，标称档 + 真机组合档（b 20 ms + c + f，力矩按标称）；SONIC 每局记 RTF，< 0.9 作废重跑。
  - [x] 25.8.1 （10-13 起跑，10-14 跑完；补跑 `_b` / `_c`，看门狗停过一局，DECISIONS 10-14）SONIC 四批 `controller_compare_l2_{bottle,warehouse}_sonic[_combo]_20261013`（种子 41000 / 47000 起各 40 局），队列 `sonic_l2_p25_queue.sh`；之后 `after_sonic_queue.sh` 自动接 ScaleBFM（`bfm_l2_batch.sh`、`bfm_p25_batch.sh`）和汇总（`p25_l2_post.sh`、`l2_measure.py`）。
  - [ ] 25.8.2 并排视频只做了 warehouse L2 两局（`l2/videos/`）；C1–C7 片段、bottle L2、扰动格都没做（计划 §8 要求每项都有）。
  - [ ] 25.8.3 待查：SONIC bottle L2 第 30 局挂住 12 分钟的原因（站在离瓶子 1.4 m 处、sim 在跑、采集进程不动；py-spy 没有权限，没拿到栈）。
- [ ] 25.9 最终判定（计划 §2），结果交用户。
  - [~] 25.9.1 L2 主判据不满足——要不要把 warehouse 专家按 ScaleBFM 重新标定（下蹲 / 前倾上限、站位、反馈增益）后再比，还是按"专家原样"出结论。2026-10-14 用户：重标，由 agent 闭环来做（RSI 第一个实例）；方案 `docs/handoff/scalebfm-warehouse-rsi-20261014.md`
    - [x] 25.9.1.1 方案 §10 的 D1–D7（用户 10-14：都按建议；token 上限暂定 400 USD）
    - [x] 25.9.1.2 第 0 阶段（10-14，方案 §12）：验收池 60 局（SONIC 27/60）；分叉改为重跑前缀；找到并修掉进程首局不确定（TorchScript 首次调用，`warm_up`）；空分叉 / 换分片 / 倒序逐拍一致；25.7 增益 0.5 的 L1 2/10、L2 6/40 结局全同；交接记录、包络、gate、驱动、rules.md 都在
      - [ ] 25.9.1.2.1 增益 1.0 的 25.7 参考只复现了计数（0/10），逐局原因 5 局不同，最早在 reach 那一拍差 1e-8；那组是 /workspace/tmp 里的临时搜索，没查
    - [x] 25.9.1.3 第 1 阶段（10-14，方案 §13–14）：诊断池上骨盆不低于 0.74 m 8→29、重试 2 次 8→25、两者合起来 8→34（SONIC 25）；假设成立，主因是 ScaleBFM 深蹲时手下不去；我读轨迹的预测把主因判反了
    - [x] 25.9.1.4 第 2 阶段（10-14，方案 §15）：一轮就停；h1（0.74 + 重试 + 重夹）验收池 45/60 对 incumbent 10/60（SONIC 27/60），10.19 USD
    - [x] 25.9.1.5 第 3 阶段（10-14，方案 §16）：封存集 80 局 ScaleBFM final 48 对 SONIC 30（p 0.013），护栏全过；start 9；phase1_best 46（闭环的重夹只多 2 局，p 0.5）
    - [x] 25.9.1.6 第 4 阶段（方案 §17）——冷启动两臂（本方案闭环 / RoboRSI 式扁平闭环），量无人闭环自己能做到多少（用户 2026-10-15：A、B 两臂都跑；实现 §17.1，结果 §19）（2026-10-15）
      - [x] 25.9.1.6.1 隔离副本 `phase4_sandbox.py`（e9f0530，剔除第 1–3 阶段答案，数据复制）+ 驱动 `--arm/--gpu`、臂 B 扁平模式、禁区审计；单测 `tests/test_scalebfm_rsi_phase4.py`（2026-10-15）
      - [x] 25.9.1.6.2 两臂开跑（A GPU 1、B GPU 0）到 §9 的停止条件：A 06:52（3 次验收，验收池 52），B 07:44（3 次验收，45）（2026-10-15）
      - [x] 25.9.1.6.3 臂 B 的预测补跑打分（诊断池，不回传）：精确度 A 17%、B 15%，`phase4_score.py`，`sealed/phase4_predictions.md`（2026-10-15）
      - [x] 25.9.1.6.4 两臂最后的 incumbent 上封存集（第五次使用），和 SONIC 30 / start 9 / final 48 配对；写 §19
        - [x] 25.9.1.6.4.1 臂 A（10-15 06:52 停，3 次验收，验收池 52/60，20.73 USD）：封存集 68/80 对 SONIC 30（4 / 42），对 final 48 是 25 / 5（p 0.0003），护栏全过，ego p95 0.359；`phase4_sealed.py`，`sealed/phase4.md`（2026-10-15）
        - [x] 25.9.1.6.4.2 臂 B：封存集 57/80（8 / 35 对 SONIC），对 A 7 / 18（p 0.043），对 final 21 / 12（p 0.16）；第 6 轮 Engineer 自己在诊断池跑了两个变体各 50 局才选出交的版本（§19）（2026-10-15）
      - [ ] 25.9.1.6.5 待用户：每臂再跑一次，量 LLM 随机性（约 4 h、20–60 USD / 次）
      - [ ] 25.9.1.6.6 待用户：臂 A 的配置叠加 CuRobo 路线（25.9.1.7），封存集上看是否高于 68
      - [ ] 25.9.1.6.7 待用户：按 RoboRSI 代码行为的臂 C（§17.2：2 种子门、读单局轨迹、12 次工具调用、持久记忆）
      - [ ] 25.9.1.6.8 提示词缺口：Engineer 只说跑一局冒烟、没禁止多跑（臂 B 第 6 轮跑了 102 局）；Reviewer 没规定局名格式
    - [x] 25.9.1.7 （用户 2026-10-14：要自然的动作）伸手接 CuRobo 路线，从下垂直达预抓（`visual.reach_route planned`，方案 §18）：验收池 45 → 54、封存集 48 → 68，途中碰撞 18 → 0 局，一气呵成 11 → 100 / 140 局，走停到预抓 4.9 → 3.7 s
      - [ ] 25.9.1.7.1 带路线时箱子最后举得低（封存集抬升 12.0 → 8.0 cm），原因没查
      - [ ] 25.9.1.7.2 为什么带路线夹得更稳（倾斜 11 → 1）没隔离验证
      - [ ] 25.9.1.7.3 SONIC 生产线（`expert.py` / `attempt.reach_step`）没接：实时栈要等 6 s 规划，腰部交付打折，要另外验证
      - [ ] 25.9.1.7.4 规划 5.5–6.5 s，真机前要降（cuRoboV2 或减球）
      - [ ] 25.9.1.7.5 走停时下垂手指已贴箱壁（诊断池 1/50）：走近停得太近，路线回退；看是不是站位或刹车的问题
  - [ ] 25.9.2 待用户：ScaleBFM 上真机的世界位姿来源（腿运动学 + IMU 在它滑脚时漂 15–40%，计划 §24）：外部定位、视觉里程计，还是把世界锚定的参考改成相对机身。
  - [ ] 25.9.3 待用户（2026-10-14 建议）：bottle 专家也按 ScaleBFM 重调（同 25.9.1 的干预归因 + 验收门），再比 bottle 标称（现 SONIC 40 对 ScaleBFM 39 / 45，掉物 2 对 6）
- [ ] 25.10 ScaleBFM 部署程序（Python + TensorRT，接口同 SONIC deploy）→ sim2sim。
- [ ] 25.11 真机冒烟（计划 §7 的六步，用户团队操作，SONIC 同日对照）。
  - [ ] 25.11.1 待用户：机载电脑型号、实验室有没有外部定位（现按没有处理）。
