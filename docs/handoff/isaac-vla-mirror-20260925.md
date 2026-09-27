# VLA 闭环的 Isaac 画面：按需镜像，做了什么、没做什么（2026-09-25）

接手入口。任务编号见 [TODO](../../TODO.md) 4.7（4.7.9.x 是镜像本身），决策理由与被否决的方案见
[DECISIONS](../../DECISIONS.md)「2026-09-24 VLA 闭环的图像走按需 Isaac 镜像」。怎么跑见
[渲染 · VLA 闭环的 Isaac 画面](../rendering.md#vla-闭环的-isaac-画面按需镜像)。本页是状态索引和证据位置。

## 要解决的问题

VLA 要图像输入。MuJoCo 在物理循环里同步渲染，渲染慢时物理变慢，而 SONIC deploy 按墙钟推进参考，
机器人会"漂"（warehouse 旧场景 RTF 0.35，步长偏大、走得约 3 倍快）。目标：VLA 闭环评测实时看到 Isaac 画面，
物理和控制器不因渲染失步，不降画质，不减面。

## 两个结论（先读）

1. **MuJoCo 慢的根因不是面数，是单块网格超过 2²⁰ 顶点。** 单块 349,525 面 0.31 ms、349,526 面 3.91 ms；
   旧 `warehouse` 纸箱是一整块 46.3 万面（三路 36.7 ms），`warehouse_textured` 同样的面拆成 9 块（3.2 ms）。
   拆块无损；减面（TODO 2.4.10.8）放弃。证据 `outputs/diagnostics/render_speed_20260924/`，台账 crate.md 末节。
2. **图像走按需 Isaac 镜像，物理从不等渲染。** 评测时 MuJoCo 用 `backend.render: offline`（不渲相机）；
   Isaac 在另一个进程、另一张卡上，只在策略要观测时渲 sim 当下的状态。渲染耗时只让图像晚到，
   且落在现有延迟补偿的计时里（`vla_client` 先取 t0 再组观测）。

## 已做（main，已推送）

| TODO | 内容 | 位置 | 验证 |
|---|---|---|---|
| 4.7.1 | MuJoCo 三路相机渲染测速 + 2²⁰ 门槛定位 | `scripts/experiments/bench_camera_render_20260924.py`，诊断目录里的 `ab_carton.py`、`mesh_size_sweep.py`、`mesh_2pow20_check.py` | 同状态 A/B；合成网格逐面扫描 |
| 4.7.9.1 | warehouse 三路 Isaac 按需渲染耗时 | `bench_render --state-trace --streams … --gap 0.04 0.4` | GPU1：间隔 0.4 s p50 89 / p90 108 / 最大 135 ms，300 帧无空帧 |
| 4.7.9.3 | `IsaacView`、`bench_render` 的 state-trace 路径从分支移植；`--rtx-setting` 解析挪到 `stage.apply_rtx_settings` | `src/locomani/sim/isaac/{view,bench_render,stage,isaac_render}.py` | `test_isaac_view.py`；五状态三路与 MuJoCo 并排几何一致（`isaac_view_check/`） |
| 4.7.9.4 | `world_lift.RunTraces` 空 `collect/state_trace` 修复（分支移植） | `pipeline/world_lift.py` | 新测试修前失败、修后通过 |
| 4.7.9.5 | 特权状态流带完整 `qpos`，与 state trace 同 seq 行逐位相同 | `sim/privileged_pub.py` | `test_privileged_qpos.py` |
| 4.7.9.6 | 镜像服务：REQ/REP，`TF_ISAAC_MIRROR_PORT` = 5558 + 10·instance；从 eval 栈状态文件取场景和端口；场景的 Isaac stage 与曝光取 `configs/scenes.yaml` 的 `isaac` 条目；每次渲染记 `<run>/eval/isaac_mirror.jsonl` | `sim/isaac/mirror.py`、`scripts/run_isaac_mirror.sh`、`sim/ports.py`、`pipeline/run_context.isaac_scene` | `test_isaac_mirror.py`（协议、出错回复、镜像无应答后客户端恢复） |
| 4.7.9.7 | `VlaConfig.image_source: mujoco \| isaac`；`ObsBuilder` 先读本体再向镜像要图；单臂 async / blocking 与双臂 `vla_rounds --image-source` 都接上；`lockstep` + `isaac` 加载即拒 | `backends/sonic/{vla_config,vla_obs,vla_client,vla_round}.py`、`tasks/crate/vla_rounds.py`、`pipeline/eval_crate_closed_loop.py` | `test_obs_builder_image_source.py`、`test_eval_config_reaches_the_client.py`、`test_crate_vla_eval.py` |
| 4.7.9.9 | Isaac 数据集相机：训练图和评测图同源。按 main 的 MuJoCo 补渲接缝做（转换前覆盖 session 占位帧，逐行 priv_seq → trace qpos），不移植分支的"转换后副本"；只渲成功局、只渲 50→25 Hz 取到的行；marker 记 `renderer`，convert 按 `eval.image_source` 拒绝不同源 | `sim/isaac/render_dataset_cameras.py`、`scripts/render_dataset_cameras_isaac.sh`、`pipeline/render_dataset_cameras.require_rendered`、`run_overnight.sh --isaac-gpu` | `test_isaac_render_dataset_cameras.py`、`test_render_dataset_cameras.py`；`box_room_grasp_200_20260925` 批 0 两局实渲（948 / 939 次渲染，p50 76 / 84 ms，每局 86–90 s），五行×三路与 MuJoCo 同状态并排几何一致（`outputs/renders/isaac_dataset_check_20260925/`）；与采集并跑时采集 RTF 0.91–0.97 不变 |
| 4.7.9.10 | 对真实 eval 栈冒烟（无 checkpoint） | `scripts/experiments/isaac_mirror_smoke_20260924.py`，run `outputs/runs/isaac_mirror_smoke_20260924` | 50/50 应答、seq 严格递增、状态年龄 p50 11 / p90 20 ms、渲染 p50 98 / p90 120 ms（栈与镜像同在 GPU1）；同 seq MuJoCo 并排一致；SIGTERM 正常退出 |

提交：`1bc335f`（渲染根因 + 镜像全部）、`121303e`（world_lift）。`render_isaac_warehouse.sh` 也改从
`scenes.yaml` 读 stage 与曝光，离线重渲和镜像用同一份设置。

## 没做 / 未验证

| TODO | 内容 | 为什么还没做 / 卡在哪 |
|---|---|---|
| 4.7.6 | 用 Isaac 画面跑真实 VLA 闭环评测 | 2026-09-27 已跑 10 局冒烟（checkpoint-5000，9/10，台账 crate.md 末节）；100 局待跑（TODO 4.7.6.1.2.8） |
| 4.7.11 | Isaac 渲图时冻结世界（lockstep），推理延迟按帧模拟 | 墙钟下每个观测多约 95 ms 延迟；计划见 [lockstep-isaac-eval-plan-20260927.md](lockstep-isaac-eval-plan-20260927.md)，未开工 |
| 4.7.9.8 | 在线 / 离线一致性闸门：镜像记的 seq 离线重渲比像素，容差取同脚本两次的差 | RTX 有跨帧状态，按需跳着渲与逐帧渲是否同图未验证；eval 栈默认不写 state trace，要先打开 |
| 4.7.9.2 | 分支对账 | 部分完成，见下节 |
| 4.7.7 | `warehouse_textured` + `backend.render: live` 起栈实测 RTF | 只做了离线测速（3.2 ms）。影响的是 MuJoCo 实时采集和 MuJoCo 对照臂，不影响镜像 |
| 4.7.8 | 建场景闸门：单块网格 > 349,525 面时拆块或报错，建完测三路渲染耗时 | 未写 |
| — | bottle / crate 场景的 Isaac stage | `scenes.yaml` 只有 `warehouse_textured` 有 `isaac` 条目；其余要先 `build_isaac_assets.sh` 并登记 |
| — | 双臂 `vla_rounds` 与单臂 blocking 客户端的 trace 不记镜像回复头（seq / 状态年龄） | 只有单臂 async 的延迟记录里有 `image`；镜像自己的 `isaac_mirror.jsonl` 有全部渲染 |
| — | 镜像与 sim、deploy、策略服务同时满载时的耗时和 CPU | 冒烟时没有策略服务；分支 3.3.4 同一问题 |

已放弃（DECISIONS 有理由）：纸箱减面（2.4.10.8）、MuJoCo 异步渲染进程（4.7.2）、改延迟补偿起点（4.7.3），
以及它们的验收项 4.7.4（并入 4.7.7）、4.7.5。

## 与分支 `feat/isaac-sync-20260923` 的对账

分支 worktree `/workspace/code/locomanipulation-isaac-sync-20260923`，从 main `e8a0019` 拉出，本地未推送，
代码改动**未提交**（见其 STATE.md）。main 已取走 / 未取走：

| 分支内容 | main |
|---|---|
| `sim/isaac/view.py`（`IsaacView`）及测试 | 已移植，相机表改用 main 的 `sim/dataset_cameras.CAMERA_CONFIGS` |
| `bench_render.py` 的 `--state-trace`、多 gap、多路相机 | 已移植，另加 `--rtx-setting` |
| `world_lift` 空目录修复 | 已移植（测试按 main 的 `trace_rows` 接口重写） |
| 按需镜像设计（DECISIONS 2026-09-23 四节） | 已按它实现（3.4.6–3.4.8 对应这里 4.7.9.5–4.7.9.7） |
| `sim/isaac/render_dataset.py`（Isaac 数据集副本）及测试 | 不移植：4.7.9.9 改在转换前填 session（main 的接缝，不依赖转换器的裁尾/抽帧细节，能与采集并跑） |
| `CAMERA_CONFIGS` 挪到 `schema.py` | 不采用：main 已在 `sim/dataset_cameras.py`，两处不能并存 |
| 台账 `docs/ledger/bottle.md` §38（T_render）、§39（副本流水线） | **未移植**；编号与 main 新的 §38（deploy 文档版）冲突，合并时要重编号 |
| Isaac 资产在 `outputs/isaac_sync_assets/`（bottle 场景） | 未动；main 用共享 `assets/isaac/`（warehouse） |

分支后续应以 main 为准：要么只把 `render_dataset.py` 挪过来后弃用分支，要么 rebase 到 main 再继续。

## 数字（复查时用同一命令）

- MuJoCo 三路 640×480：`warehouse_textured` 3.2 / 6.9 ms，crate 1.7，bottle 1.6，旧 `warehouse` 36.7 ms。
- Isaac warehouse 三路按需（0.4 s 间隔）：89 / 108 ms；连续 72 / 107 ms。bottle 分支数据：ego 一路 52 / 70 ms，两路 63 / 94 ms。
- 推理 p50 72 ms（bottle 评测）；一轮 ≈ 160 ms，小于 0.4 s 重规划周期。
