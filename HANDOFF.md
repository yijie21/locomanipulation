# 当前任务交接：单臂瓶子专家的走路准备、动作平滑与闭环学习性

当前续作见 [中部抓取、接触恢复与实际平滑验证](docs/handoff/bottle-contact-expert-20260915.md)。
用户要求继续实现三个优化项，并把瓶子抓在指垫中部，最终采集新的 10 集视频；不启动 VLA 训练。
当前交付 checkpoint11：中部抓取、接触恢复、有界跟踪补偿及制动/舒适姿态改动已集成；配置保留1.2秒闭爪。候选checkpoint12实测2/4，未采用，源码与证据保留。
固定四例验证为 3/4；新种子 30300 首批仅 6/10，已停止后续采集，证据保存在 `outputs/bottle_contact_20260915/final_collect`。GPU 0 / instance 0 已释放。
checkpoint11新种子30700/30800在 `outputs/bottle_contact_20260915/final_collect_v3` 完成10/15；10份合规预览已审计，ZIP4552364bytes。checkpoint12固定四例回归2/4后恢复checkpoint11，GPU0/instance0已释放。
本轮新十集视频已生成并审计；main `make check` 已通过，后续文档状态更新单独验证。尚未达到高可靠/完全自然的目标，具体剩余项见当前续作报告。

更新日期：2026-09-15 UTC。工作目录：`/workspace/code/locomanipulation`。基线提交 `fcf23d9`，
用户已要求提交并推送当前代码与文档快照，随后依次实施跟踪响应辨识与末端制动、停车与全身姿态联合规划、带物轨迹与稳定交接。稳定性验证前不缩短动作时间。上一版交接原文保留在
`docs/archive/handoff-bottle-walking-20260914.md`（`docs/archive/INDEX.md` 按约定未改）。

标记约定：**事实**＝代码、日志、manifest 或保留轨迹直接支持；**决策**＝已采用并写进代码/配置；
**假设**＝需要实验区分的解释，不是已定根因。每个实验的权威记录是
[docs/handoff/bottle-hanging-start-20260914.md](docs/handoff/bottle-hanging-start-20260914.md)
第五到第十节和 [docs/handoff/crate-formal-export-20260914.md](docs/handoff/crate-formal-export-20260914.md)
的闭环一节；日常入口是 [docs/collection.md](docs/collection.md)、[docs/evaluation.md](docs/evaluation.md)。

## 1. 目标与范围

- 总目标：单臂 bottle 与双臂 crate 两条线各自 200 集采集 → LeRobot 转换 → GR00T 10k 步微调 →
  100 局闭环评测；之后回答"为什么单臂闭环低"，并把单臂专家的走路阶段改成像双臂那样随物体连续
  调整、动作更顺、对 VLA 更可学。
- 本轮做了：双臂闭环评测器（新）、两条线的 100 局闭环、单臂失败的离线逐键替换定位、刹车深度
  0.31、走路准备三版、三种平滑变体、用户的"胶囊相切 + 像素触发 + 全身 IK"方案。
- 没做：任何新动作律下的重新采集 200 集、训练与闭环；crate 线在 71/100 之后未再动。

## 2. 已确认的约束（事实）

- 不删除 `outputs/`、数据集、权重；每段导出/引用的视频必须有 ≤500,000 字节的全时长预览并有
  `verification.json`（本轮全部满足，见第 6 节）。
- 不改验收门限；`docs/STATE.md` ≤60 行（现 59）；文档与代码同改；回复里用绝对路径。
- 两个仿真不能共用一块 GPU；单臂 pilot 固定实例 0；双臂闭环用实例 2 / 端口 5700（5591 是实例 1
  的特权端口，5590 是实例 1 的键盘端口）。
- 单臂 pilot 的启动条件：`PYTHONPATH=src`，且必须
  `export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/outputs/bottle_continuous_20260914/deploy`
  （`scripts/paths.sh` 默认的 `third_party/.../gear_sonic_deploy` 缺夹爪交接补丁，
  `require_pose_ownership` 直接拒绝）。
- GPU 上现存一个 `/workspace/miniconda3/envs/vg/bin/python` 进程（3 GB），不是本会话的，未动。

## 3. 已采用的决策及理由

| 决策 | 理由（事实） |
|---|---|
| `visible_brake_depth` 0.40 → 0.31（`configs/tasks/bottle_whole_body.json`） | 闭环失败局停在专家 park 0.46 的位置，从不做专家随后 10 cm 的全身前移；改 0.31 后 park→全身阶段的前移由 10.6 cm 降到 3.4 cm，同 20 seed 17/20 vs 15/20，髋/桌 2 vs 3 |
| 走路阶段手目标改为 v3 律（`walking_reach.py`）：腕从第一次规划起转向手指朝前，工具点固定在髋旁 x 0.08、抓点上方 `walk_clearance` 0.06，横向跟随瓶子，"至少落后瓶子 12 cm"用原始点计算，走路限速 `walk_joint_speed` 0.7，腰参与求解 | dex1 手指从掌心向下伸 19 cm，垂臂指尖 0.53 m 低于 0.62 m 桌面；0.45 rad/s 转腕要吃掉整段 3.5 s 走路（v1 2/20）；相对瓶子定义的 x 会以步速后退、手停在瓶子深度上扫倒瓶子（v2 0/20）；滤波深度在刹车时低估约 10 cm |
| 走路腰包络去掉 roll（yaw ±0.1、pitch 0–0.12） | v3 每局把 roll 打满 +0.1，骨盆侧倾，右髋在近站位贴桌（5/20）；去掉后 2/20；去 roll 后髋旁姿势 IK 误差 ≤1.7 cm |
| 三个平滑选项做成字段并默认关闭：`pose_lowpass_hz`、`reach_seconds`（`tracking_gain` 原有） | 同 20 seed 分别 7/6/6 of 20；只有增益 0.3 让身体各阶段更平但 7 局伸手超时（反馈在补 decoder 偏差） |
| 用户方案的字段：`visible_brake_edge` 0.27、`raise_trigger_px` 36、`lean_max`（JSON 设 1.0，默认 0.55） | 已实现并跑 12/20；两项在当前出生带上不起作用（第 6 节）。字段保留，值未回退，属于待用户决定 |
| 硬化手指垫为单臂默认场景（`scripts/install_scene.sh`） | 采样 20 局 0 滑脱（本轮之前已定） |

## 4. 已完成工作与修改文件（全部未提交）

新增：`src/locomani/pipeline/eval_crate_closed_loop.py`、`src/locomani/tasks/crate/vla_rounds.py`、
`src/locomani/pipeline/world_lift.py`、`configs/deployment/crate_initial_token.json`、
`configs/deployment/whole_body_initial_token.json`、`tests/test_crate_vla_eval.py`、`tests/test_world_lift.py`、
`docs/handoff/{bottle-hanging-start,crate-depth-convergence,crate-formal-export}-20260914.md`。

本轮改动的单臂文件：`src/locomani/tasks/bottle/walking_reach.py`（v3 律、raise 门、去 roll）、
`whole_body_policy.py`（`ReferenceFilter`、`reach_path`、`lean_angle`、传 `brake_depth`/`ego_px`）、
`whole_body_config.py`（`walk_clearance`、`walk_joint_speed`、`pose_lowpass_hz`、`reach_seconds`、
`raise_trigger_px`、`lean_max`）、`reach_config.py`（`visible_brake_edge`）、
`round_phases_approach.py`（`visible_stop`、`visible_speed_scale`）、`assisted_target.py`（`ego_height_px`）、
`configs/tasks/bottle_whole_body.json`、`tests/test_bottle_walking_reach.py`、`tests/test_bottle_whole_body.py`、
`tests/test_replay_obs.py`（按 schema 比较相机键）。

文档：`docs/collection.md`（复现表、采集逻辑）、`docs/evaluation.md`（两条线 2026-09-15 结果、双臂闭环一节）、
`docs/crate_collection.md`、`docs/reference/config_fields.md`（`make config-doc` 生成）、`docs/STATE.md`。
其余 `M` 文件（crate 视觉专家、导出、check_dataset、start_stack 等）属本轮之前同一未提交序列，见各 handoff。

## 5. 测试与实验结果（事实）

闭环（checkpoint-10000，seed 6000，100 局）：

| 线 | 结果 | 失败 | 目录 |
|---|---|---|---|
| 双臂 crate | 71/100，Wilson 0.61–0.79（验证批 9/10） | 20 闭合无抬起（18 局双手未接触）、5 到桌未闭合、4 reset 未稳 | `outputs/crate_eval100_20260915`（预览 `previews/`，100 段） |
| 单臂 bottle | 17/100，Wilson 0.11–0.26 | 60 到桌未闭合（最近 0.32 m）、11 spawn 视野、7 闭合无抬、3 走路、2 stand | `outputs/bottle_vla200b_20260914/eval`（预览 `previews/`，87 段） |

单臂闭环定位（`outputs/bottle_vla200b_20260914/eval_dump10`，10 局 dump，2/10）：val 闭合前后 10/10 预测
闭合；失败局观测 0/70；换入成功局两路图像 58/70，只换腕或只换 ego 0/70，换任何状态 0/70；反向
12/12→0/12。结论（事实）：闭合是两路图像的联合函数，与本体状态无关；失败局停在 0.37–0.52 m，
专家 park 0.46（0.42–0.52）、全身阶段开始 0.35。

专家同 20 seed（30000，硬垫）对照：

| 运行目录 | 变化 | 结果 | 主要失败 |
|---|---|---|---|
| `outputs/bottle_hardpads_sampled20_20260914` | 基线（刹车 0.40） | 15/20 | 髋/桌 3 |
| `outputs/bottle_brake031_sampled20_20260915` | 刹车 0.31 | 17/20 | 髋/桌 2 |
| `outputs/bottle_walkprep_sampled20_20260915` | 走路准备 v1 | 2/20 | 指/桌 8（走路末段） |
| `outputs/bottle_walkprep2_sampled20_20260915` | v2 | 0/20 | 扫倒 5、腕指/瓶 6 |
| `outputs/bottle_walkprep3_sampled20_20260915` | v3（当前律，含 roll） | 14/20 | 髋/桌 5 |
| `outputs/bottle_smooth_A_lowpass4_20260915` | + `pose_lowpass_hz` 4 | 7/20 | 髋/桌 9、无接触 4 |
| `outputs/bottle_smooth_B_gain03_20260915` | + `tracking_gain` 0.3 | 6/20 | 伸手超时 7 |
| `outputs/bottle_smooth_C_minjerk3_20260915` | + `reach_seconds` 3 | 6/20 | 扰动 5、腕/瓶 4 |
| `outputs/bottle_edge_sampled20_20260915` | 用户方案（去 roll、桌沿 0.27、触发 36 px、lean_max 1.0） | 12/20 | 无接触 3、髋/桌 2 |

每个运行的 20 段预览在各自 `previews/`，`verification.json` 全部 `all_ok`。

动作几何（事实）：v3 接管时工具点高于抓点 4.8 cm、落后 15.6 cm，右掌 0.761 m（基线 16.3 cm /
7.4 cm / 0.846 m），伸手 3.4 s（基线 5.1 s）。抖动：各阶段被跟踪的参考加速度 0.4–1.5 rad/s²、
>2 Hz 占 0.4–14%，身体 3–6 rad/s²、>2 Hz 占 24–44%；双臂走路段一样（腰 8.1 rad/s²，72%）。
日志里每次 POSE 发送只存两帧，第二帧是 preview 的 50 tick 未来终点，不是参考。

用户方案的元素效果（事实）：桌沿规则 20 局只触发 2 局（瓶子在桌盒边内 2–5 cm，深度 0.31 先到）；
36 px 触发在出生点（≈1.5 m，36–37 px）第一次规划即触发；去 roll 髋/桌 5→2；距离项 ≤0.37 rad 未
改变前倾。真实网格 guard：直立离桌沿 ≤0.23 m 右髋贴桌、0.27 m 无碰、前倾 20–40° 在 0.20 m 也无碰。
臂长：肩到工具点 0.37 m，抓点高度前伸上限约 0.60 m（70° 躯干）；双臂抓取瞬间手掌前伸 0.29–0.32 m。

检查：`make check` 通过（最后一次 2026-09-15 08:0x UTC）；相关测试
`tests/test_bottle_walking_reach.py tests/test_bottle_whole_body.py tests/test_visible_expert.py tests/test_configs.py` 55 通过。

## 6. 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/outputs/bottle_continuous_20260914/deploy
export PYTHONPATH=/workspace/code/locomanipulation/src

# 单臂专家 20 局（同 seed 对照；每次换 --out）
"$VENV_SIM_PY" -u -m locomani.pipeline.run_learn_pilot --config configs/pilot/learn_200.json \
  --task configs/tasks/bottle_whole_body.json --scene outputs/scene/sim_hardpads --out outputs/<run> \
  --set through=collect --set successes=20 --set batch_rounds=10 --set max_batches=2 --set seed0=30000 \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0
# 变体：把 reach.whole_body 块写进任务 JSON 副本（例：outputs/bottle_smooth_20260915/task_*.json）

# 双臂闭环 100 局
"$VENV_SIM_PY" -m locomani.pipeline.eval_crate_closed_loop \
  --checkpoint outputs/crate_vla200_record_20260914/model/checkpoint-10000 \
  --scene outputs/scene/crate_table30_depth04 --out outputs/<eval> --instance 2 --gpu 1 --serve-gpu 1 --rounds 100 --seed 6000

# 离线问策略"会不会闭合"（需 serve_policy 在 :5590）
third_party/Isaac-GR00T/.venv/bin/python src/locomani/pipeline/replay_obs.py --dataset outputs/bottle_vla200b_20260914/val_dataset --episode 0 --near-close
third_party/Isaac-GR00T/.venv/bin/python src/locomani/pipeline/replay_obs.py --dump <obs_dump dir> --substitute video.ego_view,video.right_wrist --donor-dump <npz>

# 预览（≤500,000 字节 + verification.json）与抖动表
nice -n 19 /usr/bin/python3 outputs/bottle_smooth_20260915/make_previews.py outputs/<run>/previews outputs/<run>/collect/videos_0*/*.mp4
python3 outputs/bottle_smooth_20260915/smooth_table.py 0   # 0 = 被跟踪参考帧；1 = preview 未来终点

PYTHONPATH=src "$VENV_SIM_PY" -m pytest -q tests/test_bottle_walking_reach.py tests/test_bottle_whole_body.py tests/test_visible_expert.py tests/test_configs.py
make config-doc && make check
```

## 7. 未解决问题与假设

- **假设**：去 roll 后抓取侧多出的 4 局失败（无接触/滑脱/指瓶）来自接管时手构型变化；闭合时深度
  1.9 cm、横向 ≤2 cm 在成功带内，20 局分不出因果与噪声。
- **假设**：身体抖动来自 deploy 跟踪与步态（依据是参考平滑而身体不平滑、双臂同样），未做控制器侧实验。
- **假设**：可见触发（像素高度、桌沿、腕相机可见）会改善策略的吸收态；任何新律都没重新采集训练。
- **事实**：刹停时髋/桌家族在相同走路代码下 20 局里 0–9 局波动（A 9、B 0、C 3、v3 5），20 局比较
  读不出 ±3 的差异；本线各版本成功率在 12–17/20 之间。
- **事实**：`visible_brake_edge` 0.27 与 `raise_trigger_px` 36 在当前出生带（瓶子桌沿内 4–8 cm、
  出生 1.1–1.3 m）上不起作用；要生效需深度刹车 ≥0.33 或去掉、触发高于出生尺寸且转腕提前。
- **事实**：`docs/archive/INDEX.md` 未加归档条目（按约定未动该文件）。

## 8. 下一步

1. 用户决定单臂默认律：回退到 `fcf23d9` 的 `walking_reach.py` + 刹车 0.31（17/20），或保留 v3 + 去 roll
   （12–14/20，动作是用户要的）。当前树是后者。
2. 若继续用户方案：`visible_brake_depth` 改 0.33–0.35 或 0 让桌沿规则成为主判据；触发像素设在出生尺寸
   之上（≥45 px）并把转腕速度或提前量相应加大；再跑一次。
3. 判据改为 100 局或"各阶段到达数 + 阶段遥测"，不再用 20 局成功率比较。
4. 律定下后再做 200 集采集 → 训练 → 100 局闭环，并用 `eval_dump10` 的替换实验复核吸收态是否消失。
5. 抖动：唯一实测有效的杠杆是反馈增益（0.3 太低），可试 0.5–0.7 或放慢反馈滤波；C 的终点改为
   抓点后 1.8 cm 再试。
