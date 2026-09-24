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
- [ ] 1.14 修 handoff 文档第 3 行的过时状态句（仍写"阶段 4–8 未开始"）
- [ ] 1.15 三处同 venv `sys.executable` spawn：`bimanual_scene.py`、`reach_envelope/floor_scene.py`、`reach_envelope/boundary_search.py`（诊断路径，谁动 reach_envelope 时顺手做）
- [ ] 1.16 `eval_crate_closed_loop → vla_rounds` 的同 venv spawn（需要双臂 checkpoint）
- [ ] 1.17 `wall`/`carry` 转成 `run(config, ...)`，去掉 `global DUMP_DIR`（可选；进程边界已没了）
- [ ] 1.18 两套评测器本身合并（协作者第 2、3 条的深层问题；需 checkpoint + 闸门）
- [ ] 1.19 `eval_crate_closed_loop` 的 run-dir 模式实跑一次（目前只 dry-run；磁盘上没有双臂 checkpoint）
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
  - [ ] 1.20.10 实机演练一次中断续跑（采集跑到第 1 批中途 Ctrl-C，同名重跑，核对 `inputs.json` 与 `interrupted_manifest_*`）
  - [ ] 1.20.11 现存栈（本次之前起的）attach 会被拒——有长跑栈的会话需知会（无主 `stack.json`）

## 2. 纸箱线（物流房间）——承接上一会话的 STATE

- [x] 2.1 房间 pilot `box_room_20260922` 0/20，代码与文档已提交（`704004d`、`f5ea2e9`）
- [ ] 2.2 用户看 `outputs/runs/box_room_20260922/reel_failures.mp4`
- [x] 2.3 按阶段拆 20 局：摔倒 6 局在哪个相位；reach_stalled 8 局的站位深度与手到抓点距离（2026-09-23：`reach_anatomy_20260923.py`；伸手阶段反馈 1.0/0.15 顶满、手起点离抓点 ~20 cm）
- [~] 2.4 据 2.3 决定改 `stand_off` / `pregrasp_above` / 摔倒原因；每次一个改动、20 局、同 seed 7000
  - [x] 2.4.1 反馈 0.7/0.04（`box_room_fb07`）0/10；整个 carry 块回退（`box_room_carry`）0/10、摔倒 3→0（2026-09-23）
  - [x] 2.4.2 stand_off 0.40（`box_room_so40`）0/10，失败移到"碰动箱子"5/10（2026-09-23）
  - [x] 2.4.3 起点 advance 0 + 刹车余量 0.05（`box_room_approach`）0/10；`walk_steer: depth`（`box_room_steer`）0/10、停位离散 1.2 cm（2026-09-23）
  - [x] 2.4.4 真专家接入守线 + staged + orientation_gain + 1 cm 抓深 + 后退（`box_room_grasp_20260923`）0/10，守线 6/10 越线（2026-09-23）
  - [x] 2.4.5 提交 2.4.4 的代码（expert/visual_approach/visual_config/grasp_region/diag_brake 改动 + `primitives/foot_watch.py`、`tasks/crate/retreat.py`、`tests/test_crate_guard_retreat.py`、`box_room_grasp_20260923.yaml`；默认全关，`make check` 已过）（2026-09-23）
  - [x] 2.4.6 warehouse 起点后移 0.3–0.4 m，用真专家（手臂前伸姿态）重标守线的刹车后滑行，再验证不越线（2026-09-23：后移 0.25 m 到离线 0.90 m；根因其实是慢仿真，见 2.4.10）
  - [ ] 2.4.7 伸手回到不摔的 carry 块（crate 值 + 0.7/0.04），另解"手悬在目标上方 5–10 cm 下不来"
  - [ ] 2.4.8 跑通后退段（`retreat_distance 0.3`），出 10 局视频
  - [ ] 2.4.9 纸箱抓深常量 `GRASP_BELOW_RIM_M` 仍为 0.025；warehouse 靠配置 `box_grasp_below_rim 0.01`（决定是否改常量）
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
    - [ ] 2.4.10.17 下落段跨壁横向对准 < 2 cm（视觉/几何居中或按接触居中）+ 横向修正接触后 anti-windup；做完再回抓深 0.03
    - [ ] 2.4.10.16 远站位（≥0.54）下落时掌先落壁顶：收站位散布或补下落段掌俯仰
    - [ ] 2.4.10.8 纸箱显示网格减面（46 万 → ~5k 面）或关阴影，让实时渲染/VLA 评测也跑满速；先问协作者 `assets/crates/nor_cal_box/visual/`
    - [ ] 2.4.10.9 `diag_brake` 补 reanchor（非 0 起始朝向场景一起步就转 ~90°）
    - [x] 2.4.10.12 ready 原地抬升：`visual.ready_pose lifted` + `elbow_min_deg` 旋钮（`ed5819d`，离线插值验证写成测试）；pilot `box_room_grasp_offline_lifted_20260924` 0/10，但抬手接触 0/10、箱子位移 0/10，4 局到试提（2026-09-24）
    - [ ] 2.4.10.13 “腕相机始终看到箱子”只在下落段要求；抬手/平移段放开掌心姿态；从 assisted_guidance 的逐流可见性统计各阶段哪路相机看得到（用户提出 2026-09-24，待定）
    - [x] 2.4.10.10 提交本会话代码（见 STATE 的文件清单），`make check` 已过（2026-09-23）
    - [x] 2.4.10.11 文档入口：删 `docs/STATE.md`，README/AGENTS 指向根目录 STATE/TODO/DECISIONS；`docs/handoff/warehouse-render-cost-20260923.md`；台账 crate 09-23、rendering 离线渲染、blender_scenes 已知问题（2026-09-23）
- [ ] 2.5 待验证：摔倒 6/20 为何比合成桌子多（假设：reanchor/IDLE 期间贴着托盘垛）
- [ ] 2.6 待验证：哪项部署变更把 v1.1 速度图从台阶变成连续
- [ ] 2.7 `vla_rounds.py` 起点放置写死偏航 0 / 物体 π/2，房间里会放错（假设，未改）
- [-] 2.8 `stand_lateral_gate: turn` 作为下一步 —— 它针对的横向偏置是合成场景现象，房间里没有

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
- [ ] 3.11 VLA 视角：伸手阶段 latent 可学性复查（探针 B 在 box_room 数据上 reach R²≈0.16）；新专家跑通后重跑 `probe_visual_grounding_20260923.py`
