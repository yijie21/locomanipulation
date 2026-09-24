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
    - [x] 2.4.10.17 沿下横向 anti-windup（`lateral_hold_below_rim`）+ 预抓点真值居中（`descend_align_tolerance`）（`833ca23`）；hold 单独无效（试提 3），hold+align 试提 6/10（2026-09-24）
    - [ ] 2.4.10.18 align 配置上回抓深 `box_grasp_below_rim` 0.03，10 局
    - [ ] 2.4.10.19 下落过程中跨壁误差再漂 3–5 cm（align 后清零、修正 +9…+15 未送达）：查 SONIC 对横向指令的送达，不是接触
    - [ ] 2.4.10.20 伸手/下落时手臂晃动（用户 2026-09-24 从视频看到）：是我们流出的参考在晃——逐 tick 重解 IK 时肘/腕俯仰反相互换（corr −0.95…−0.98，肘参考 sd 9–15°），手目标每 tick 只动 0.2 cm；实测关节比参考还平。修法候选：解算加 continuity 项（求解器已有、伸手传 0）或加大手臂 posture 权重；待用户定
    - [x] 2.4.10.21 送达差量化 + 腰上限 0.35 试验（`box_room_grasp_offline_waist035_20260924`）：负结果——腰送达按比例约 2/3、不是上限，命令 20° 实测 13°，试提 6→1；基线仍是 align（2026-09-24）
    - [x] 2.4.10.22 送达差前馈（`box_room_grasp_offline_ff_20260924`）：负结果——命令多前伸 2.5 实测更短 1.5，z 修正饱和局 3→5，试提 6→0；送达差是伸展饱和不是偏移，字段保留默认 0（2026-09-24）
    - [ ] 2.4.10.23 站位收紧到 0.45–0.52（≥0.55 全部停滞；0.46–0.54 在 align 里 6/7 到试提）：守线 `guard_extra`/`stand_off` 重标
      - [x] 2.4.10.23.1 守线散布来源已量（2026-09-24）：刹车后最前点再前进 0.166 ± 0.011 m（仓库 60 局 + walk 80 局，最大 0.197，所以仓库里 `guard_extra` 不能低于 0.20）；0.25 步态是大步 ~0.21 + 并步 ≤0.07 交替，停位按 ~0.21 m 量子化；`stride_mode floor` 在大步后把下一步多估 0.12–0.17 m。落脚位置逐次 sd 2.5–5 cm、范围 10–20 cm
      - [ ] 2.4.10.23.2 守线重走：站位 depth > 0.50 时后退 ~0.4 m 再走一次（每次独立抽样，P(≤0.50)≈30/60，三次 ≈87%）；需在 guard 路径放开 `_stance_converged`/`retry_closer`，10 局离线 pilot（待用户定）
      - [ ] 2.4.10.23.3 守线步长模型改交替式（大步后预测 0.08、并步后 0.23）：只让并步落在 ≥0.43 m 的局再多走一步，预计收益小；walk 场景 30 局验证不越线后再上仓库（待用户定）
      - [x] 2.4.10.23.5 0.25 守线站位（0.39–0.62）箱沿是否在可抓范围（用户 2026-09-24 问）：60 局逐 tick capture_error 最好时刻，两手都到抓线附近：跨壁 0–5 cm、沿上 0–5 cm、沿壁在 ±9 cm 允许滑动内；≤0.52 有 17/37 局真合爪。卡的不是够不够到，是最后 2–5 cm：跨壁要 ±1.8 cm（壁厚 1 cm）、要下过沿；≥0.53 沿壁向身体滑 11–19 cm 超出允许段，伸展饱和不再送达。结论：站位散布决定的是最后几厘米能不能送到，不是几何可达（2026-09-24）
      - [ ] 2.4.10.23.4 `start_advance` 相位试探 −0.25 → −0.15 / −0.35 各 10 局离线：看 depth 分布的双峰（刹车 t 4.4 vs 5.6 s）是否随起点平移（待用户定）
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
    - [~] 2.4.10.29 下一层：伸手停滞 7/20 与后退剪切 2/20。**暂定基线 = contact 版本**（用户 2026-09-24 定）：`configs/experiments/box_room_grasp_baseline_20260924.yaml`（= `box_room_grasp_contact_20260924`，6/20），之后每个实验复制它改一个旋钮
      - [x] 2.4.10.29.1 接触触发合爪 `close_trigger: contact`（`box_room_grasp_contact_20260924`，20 局，2026-09-24）：**6/20**（与 holdshort 持平）；合爪 13（此前 12）、提起 11、进后退 8、后退掉 2、脱开 5（此前 1）、停滞 7。跨壁类停滞消失了，剩下 7 局全是**手停在沿上方 1–3.7 cm 没下到沿下**（送达偏高，接触也没发生）；脱开变多是合爪精度放松的代价（合在壁顶几毫米处）。视频已带 ghost 夹爪
      - [x] 2.4.10.29.3 下落到接触为止 `contact_overdrive` 0.03 + `contact_along` 0.13（`box_room_grasp_contactdeep_20260924`，20 局，2026-09-24）：**负结果 2/20**（contact 基线 6/20）；停滞 14。轨迹：手确实下到抓线下 1–7 cm，但下落中跨壁误差从 ≤3 cm 跳到 4–8 cm（指垫沿壁面外侧滑下被顶开，沿下横向反馈又被 `lateral_hold_below_rim` 冻结，无法回正），爪口不再套壁。多下 3 cm 放大了偏转。基线回到 contact（overdrive 0）。剩余 7 局沿上停滞要换思路：小步下探 + 每步检查跨壁，或 align→descend 失败后重做一次
        - [x] 2.4.10.29.3a 首版跑 3 局停：手确实下到沿下 2–5 cm，但 `inside` 还卡在篮子时代的沿壁窗口 ±9 cm（手沿壁向身体滑 7–10 cm）和跨壁 4.5 边界；加 `contact_along` 0.13（壁长 0.32 − 指垫）。另见 round 1：指垫落在壁顶时继续下压把箱子横推 6–8 cm、抬 1.3 cm（`_stopped3`，2026-09-24）
      - [ ] 2.4.10.29.2 后退剪切：`retreat_speed` 0.35→0.25 或后退时保留全身伸手控制器夹持（待用户定）
    - [x] 2.4.10.30 ghost 夹爪规则（用户 2026-09-24 定，所有场景固定）：expert 把命令的抓取目标按 seq 记到 `trace/<label>_ghost.npz`，`render_trace_videos.py` 只在第三人称视角画半透明夹爪，ego/腕相机不画；Isaac 批渲同样转发。`docs/rendering.md` 有说明（2026-09-24）
      - [x] 2.4.10.28.1 `hold_frame pelvis` 单独（`_stopped4`，4 局）：round 2 到 hold 保持 5 s、倾斜 2°、附着 1.5 cm，但骨盆系下 2 s 窗内箱子仍漂 2.7 cm（IDLE 带载晃动）超 2.5 cm 半径，窗口反复重置，第 5 s 右指垫蠕变漏箱；4 局 3 脱开 1 unstable_hold。hold 拖得越久越掉（2026-09-24）
      - [x] 2.4.10.28.2 `hold_seconds` 1.0 + `hold_position_tolerance` 0.05 骨盆系（`box_room_grasp_holdshort_20260924`，20 局，2026-09-24）：**6/20 成功**（warehouse 线首批成功：抬起 13–21 cm、直立、后退 0.31–0.37 m、箱子在手、末态倾斜 0.9–9.6°）。漏斗：合爪提起 10 → 进后退 8 → 完成 6；后退段掉 2（倒走 0.35 起步 2 s 退 40 cm，两手前后错开 4–6 cm 剪切箱子，一侧爪口缩到 0 后 45° 翻落）；触门 2、脱开 1、空爪 1；伸手停滞 8（最大单项）。成功站位 0.39–0.46
    - [ ] 2.4.10.16 远站位（≥0.54）下落时掌先落壁顶：收站位散布或补下落段掌俯仰
        - [x] 2.4.10.26 纸箱贴图丢失（用户 2026-09-24 问）：不是渲染 bug。`warehouse` 场景（`scene_reverse_camera`）09-20 从 `outputs/scene/logistics_mujoco` 做的，早于 09-22 的贴图提交 `0bf8eed`（纸箱拆 9 块材质 + 6 张实物照片）；所以纸箱只有一个 `nor_cal_box.obj` + 纯色 rgba。修：`prepare_logistics_scene.py` 从资产包重做运行时场景 `outputs/scene/warehouse_textured`（scenes.yaml 新名 `warehouse_textured`；机器人/工作面/物体/nq 全同，ngeom 121→129），reverse_room 与 head_camera 各渲一帧已见印字。之后的 warehouse 实验改用这个名字（2026-09-24）
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
- [x] 3.12 守线 `guard_extra` 0.20 → 0.17 重标（全程 0.25，线 0.9 m，30 局）：骨盆 −34.3 ± 3.5 → −31.0 ± 4.2 cm，最前点最近 −7.4 → −5.7，0 越线 0 摔；只赚均值 3 cm，散布不变（`outputs/diagnostics/brake_guard025_extra017_20260924`，2026-09-24）
- [x] 3.13 waypoint 结尾（私有 deploy，SLOW_WALK 0.25，目标 0.9/1.2/1.5 m 各 10 局、不重发）：本体走目标的 0.64/0.64/0.71，散布 3.7/5.0/6.4 cm 随目标放大；相对 0.9 m 线骨盆 −32.4/−13.4/+16.4，1.2 m 时最前点已越线 +5.0 ± 2.7（10/10）；脚错开 10 cm，不是 09-23 的 3.9。过量下发不能替代守线（`outputs/diagnostics/waypoint_end_*_20260924`，2026-09-24）
- [x] 3.14 0.55 稳态走入 + 按相位提前刹车（用户 2026-09-24 提出）：标记 1.30–1.55 每 5 cm 各 7 局 + 09-23 的 10 局 = 52 局，0 摔。带距 21.5 ± 4.8 cm，随落地后相位周期变化（落地后 0.3 周期最短 15，摆动中段最长 27–29），相位一次谐波模型留一法 sd 3.6、最大误差 11 cm；同相位内仍有 5–13 cm 散布，脚位/速度特征不再降。只在落地瞬间刹车则 20.6 ± 1.5，但落地点按 ~27 cm 步长量子化，回到守线的问题。助跑 ≥1.0 m 才到 0.50 ± 0.02 稳态；托盘垛是 0–0.371 m 实心块，越线即碰。结论：相位刹车把停位均值从 −34 拉到可设的 −6…−10 cm、sd 3.5，代价是约 5% 局越线 ≤4 cm（`outputs/diagnostics/brake_055_mk*_20260924`，2026-09-24）
    - [ ] 3.14.1 若接受少量越线：实现相位刹车（FootWatch 已有落地事件；带距 = 21.5 + 谐波(相位)），walk 场景 30 局对线验证，再上仓库离线 10 局（待用户定）
- [x] 3.15 warehouse 10 局 Isaac 重渲（`box_room_grasp_offline_close06_20260924`，2026-09-24）：房间用 Blender 直出的 USDC，`scripts/build_isaac_warehouse_scene.py` 生成 `assets/isaac/scene/warehouse_textured/`（scene.usda 引用 /root、停用 env_light 和静态纸箱；可动纸箱包装层也引用 /root 并停用其余 237 子 prim，否则丢材质绑定渲成素灰；sidecar 偏移 (0, 0.39, −0.371)），`scripts/render_isaac_warehouse.sh` 每局 mujoco_render（新加 `--azimuth 225 --distance 2.2`，默认机位在南隔断外只见墙）→ isaac_render → 并排。10/10 成功，~1 min/局，并排预览 `previews_isaac/`（130–175 KB），全分辨率 `isaac/<局>/side_by_side.mp4`。房间贴图、纸箱印字、机器人含头在 Isaac 里完整，机位逐帧一致；Isaac 略亮
    - [x] 3.15.1 Isaac 里 G1 发白（用户 2026-09-24 问）：颜色绑定没错（MJCF 0.7/0.2/0.3/0.25 四档都取到），是曝光。房间由 5 块 24828 强度天花板 RectLight + 自发光灯板照亮，dome/平行光改强度像素不变；默认自动曝光 + ACES 把 0.2 灰抬到 108、0.7 灰推到 212（头/躯干 0.51，MuJoCo 0.23）。`isaac_render` 新增 `--rtx-setting 路径=值`；自动曝光关 + `filmIso` 30 后躯干 139（MuJoCo 136）、头/躯干 0.36，已设为 `render_isaac_warehouse.sh` 默认并重渲 10 局。Clamp 全饱和、Linear 更平，不用（2026-09-24）
- [ ] 3.11 VLA 视角：伸手阶段 latent 可学性复查（探针 B 在 box_room 数据上 reach R²≈0.16）；新专家跑通后重跑 `probe_visual_grounding_20260923.py`

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
