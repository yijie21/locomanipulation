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
- [-] 2.4.10.8 纸箱显示网格减面（46 万 → ~5k 面）或关阴影，让实时渲染/VLA 评测也跑满速；先问协作者 `assets/crates/nor_cal_box/visual/`（放弃 2026-09-24：被 4.7.1–4.7.6 的异步渲染替代；减面程度没有统一标准，治标不治本，用户定）
    - [ ] 2.4.10.9 `diag_brake` 补 reanchor（非 0 起始朝向场景一起步就转 ~90°）
    - [x] 2.4.10.12 ready 原地抬升：`visual.ready_pose lifted` + `elbow_min_deg` 旋钮（`ed5819d`，离线插值验证写成测试）；pilot `box_room_grasp_offline_lifted_20260924` 0/10，但抬手接触 0/10、箱子位移 0/10，4 局到试提（2026-09-24）
    - [ ] 2.4.10.13 “腕相机始终看到箱子”只在下落段要求；抬手/平移段放开掌心姿态；从 assisted_guidance 的逐流可见性统计各阶段哪路相机看得到（用户提出 2026-09-24，待定）
    - [x] 2.4.10.10 提交本会话代码（见 STATE 的文件清单），`make check` 已过（2026-09-23）
    - [x] 2.4.10.11 文档入口：删 `docs/STATE.md`，README/AGENTS 指向根目录 STATE/TODO/DECISIONS；`docs/handoff/warehouse-render-cost-20260923.md`；台账 crate 09-23、rendering 离线渲染、blender_scenes 已知问题（2026-09-23）
- [ ] 2.5 待验证：摔倒 6/20 为何比合成桌子多（假设：reanchor/IDLE 期间贴着托盘垛）
- [ ] 2.6 待验证：哪项部署变更把 v1.1 速度图从台阶变成连续
- [x] 2.7 `vla_rounds.py` 起点放置写死偏航 0 / 物体 π/2，房间里会放错（2026-09-27 核对属实并修复，另写死 `blue_crate`，见 4.7.6.1.2.2–3）
- [-] 2.8 `stand_lateral_gate: turn` 作为下一步 —— 它针对的横向偏置是合成场景现象，房间里没有
- [x] 2.9 warehouse 采 10 集并导出视频（用户 2026-09-25）：`box_room_grasp_10ep_20260925` 30 局保留 11；预览 `outputs/videos/box_room_grasp_10ep_20260925/` 30 个 322–493 KB + zip 11.8 MB（2026-09-25）
- [ ] 2.10 `stand_off` 0.5 → 0.44 的 20 局对照（同 seed 7000，只改这一个）：1013 局按停位深度分组 < 0.44 m 0.41、0.44–0.46 0.39、0.46–0.48 0.28、0.48–0.52 0.20、≥ 0.52 0.03；风险是躯干顶箱（box-stand-rams-torso）。待用户定（改专家参数）；分析脚本 `outputs/box_room_isaac_20260925/stand_depth_yield.py`

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
    - [ ] 4.7.11.7 待用户定：默认 `eval.client` 要不要改成 `lockstep`（校准已过；目标模型 88 vs 墙钟 81 未显著）
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
- [x] 13.4 （2026-10-02 用户：mars 全按 URDF，踝/腰 35、髋 roll 139、腕 13.4；stock 场景不变）仿真踝关节力矩上限 50 N·m 和真机固件 35 N·m 要不要对齐（改了会动所有标定）
- [ ] 13.5 待用户定：saturn 产物删不删（约 180 MB：`outputs/scene/sim_lowcom_saturn*`、`outputs/runs/bottle_{saturn,saturn_stockcam,stock}_pilot_20260930`；patch 所在的 `outputs/g1_urdf_compare_20260930/` 要留）；删前在 STATE 写明 24 小时

- [~] 13.6 mars 设为 bottle / warehouse 默认机器人（用户 2026-10-02；DECISIONS 同日一节）
  - [x] 13.6.1 （2026-10-02）第 1 步：生成器、7 个 `*_mars` 场景、`scene_meta.json` 记 `robot`、力矩上限从场景执行器读、几何测试样例换 mars；全量测试过
  - [x] 13.6.2 （2026-10-02）第 2 步：配对 10 局试跑（`outputs/mars_pilots_20261002/`）。bottle stock 9/10、mars 9/10（修网格名后重跑）；warehouse stock 6/10、mars 0/10
  - [x] 13.6.3 （2026-10-02）warehouse 退步定位：9 组单变量对照，只有腰 roll/pitch 35 → 50 N·m 救回（5/10）；35 N·m 下腰俯仰塌到 ~35°（关节限位 29.8°），stock 抓箱姿势静力矩 p95 39 N·m
    - [x] 13.6.3.1 （2026-10-02，用户问手为何抖）手抖也是腰 35 N·m：warehouse mars 手相对躯干高频抖 9.4 mm RMS、约 4 Hz、主要在肩 pitch（stock 2.8）；腰俯仰高通速度 1.26 rad/s（stock 0.13），4.7% 的时间顶在限位上反弹。腰 50 → 2.9 mm、0.12 rad/s；腕 5、质量、运动学、踝 50 都不消除；bottle mars 3.1 mm（`outputs/mars_pilots_20261002/jitter/jitter.md`）
  - [x] 13.6.4 （2026-10-02 用户：腰改 50 再试）mars 腰 roll/pitch 改 50 N·m（`make_dex1_model.MARS_EFFORT_OVERRIDES`，URDF 和 Unitree 官方都是 35，踝仍 35），7 个场景重建。warehouse 重跑 `warehouse_mars_pilot_20261002c`：6/10（stock 6/10），手抖 2.8 mm（= stock）
    - [ ] 13.6.4.1 待查：mars 的 4 局失败全是 lost_attachment，且都在箱子已抬起 4–14 cm、倾 6–13° 后一只手脱开（3 局右手）；stock 的 2 局 lost_attachment 都在抬起前（rise 0）。可能是 D405 支架、手腕 13.4 N·m 或手臂运动学；需腰 50 下的单变量对照
  - [ ] 13.6.5 第 3 步（用户看完第 2 步再做）：mars 的 Isaac 机器人资产（按机器人名分开存）、warehouse `_mars` 的 Isaac 场景与 Unreal 关卡（`scenes.yaml` 现写 unsupported 指向这里）、各实渲一局、补渲冒烟
  - [ ] 13.6.6 第 3 步：Isaac/Unreal 的 `source.json` 与两种补渲标记加 `robot`（Q12），对不上拒绝
  - [ ] 13.6.7 第 4 步：runbook（single_arm.md、warehouse.md）示例改用 `*_mars`、文档、推送（推送前问用户）
  - [x] 13.6.9 （2026-10-02，`ed26ed2`，用户：和 stock 相同的部位外观一致）逐部件比对颜色：除手腕（有意涂黑）外相同部位一致。改两处：云台俯仰支架涂黑（头部原有一根白横条；10-03 用户更正：那是头顶的 D455，真机银白，已改回）；骨盆保留的 stock 碰撞外壳改成骨盆色（MuJoCo 画碰撞体，白壳透出）。10-03 用户：骨盆（腿和上身之间的三角件）应为银色，`robot_look.SHELL_MESHES` 把 `pelvis_ver0529` 画成外壳色（URDF 和 Unitree 官方都是黑），碰撞外壳随之同色（对比图 `outputs/mars_look_compare_20261002/`）
  - [ ] 13.6.8 可选：`outputs/scene/hand_points_dex1.npy` 仍是 stock 手（mars 多了 D405 支架，最低点只在少数姿态差 >0.2 cm，最大 6 cm）；bottle 用它算手离桌高度

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
- [ ] 15.3 可选：GitHub 上 PR #1–#3 若仍显示打开，在网页上关掉（本机没有 `gh`，看不到状态）
