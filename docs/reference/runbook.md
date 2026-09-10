# Runbook:当前默认的采集 / 训练 / 评测

这一页是"照着做就能复现最好结果"的唯一入口,截至 2026-09-04。任何与本页冲突的
旧说明以本页为准;每条设置后面都注明了它为什么是这个值,以及改动它会破坏什么。
背景与实验证据在 `docs/collection.md`、`docs/evaluation.md`、`docs/ledger/bottle.md`。

当前正式基线:**flow200 checkpoint-10000,干净协议,无 ensemble,50/100**。(**2026-09-10 限定**:该数字测于 2026-09-02/03,早于 `8c621fb`(09-07)引入 `ReachConfig`,因此它跑在旧的站位采样带 post_min 0.22 / azimuth 36.5 上;ready-grasp 线自 09-07 起是 0.30 / 42.0。flow200 的 2×2 内部比较不受影响,但它**不能**作为 09-07 之后任何 arm 的对标参考——那需要同带的对照。见 `docs/ledger/bottle.md` §17.7)

---

## 0. 三件最容易出错的事

1. **采集绝不能设 `NO_ENCODER=1`**。planner 模式的上肢通道需要 encoder;没有它
   deploy 会静默丢弃手臂和下蹲参考,抬臂阶段整局停滞。评测则**必须**设(VLA 走外部
   token)。两边配方长得像,抄错方向就是一整批废数据。
2. **评测前必须钉死场景**。采集每批按 `SCENE_SEED` 重建 `outputs/scene/sim`
   (桌高 0.620–0.647 m),评测栈读的是磁盘上当时的场景。`scripts/run_eval_arm.sh`
   会在起栈前用 `EVAL_SCENE_SEED=922`(桌高 0.6214 m)重建,不要绕过它。
3. **`stop_stack.sh` 会杀掉机器上所有实例**。多实例并行时用
   `scripts/stack/stop_instance.sh <INSTANCE> [policy_port]`。
4. **同一 instance 只能有一个栈**(2026-09-10 起 `start_stack.sh` 用 flock 拒绝第二个,退出码 3 并打印持有者)。被拒绝就换 INSTANCE 或先 `stop_instance.sh`,不要绕过锁;起栈前读 `docs/STATE.md` 的 RUNNING 行。

---

## 1. 采集

```bash
export SIM_VIDEO_DIR=$PWD/outputs/sim_video          # 必须在起栈前
export SONIC_DEPLOY_BIN=/workspace/deps/gear_sonic_deploy_ours_lockstep_nvdef/target/release/g1_deploy_onnx_ref

OUT=outputs/logs/collect_<name> \
SCENE_PRESET=bottle_only TARGETS=water_bottle JITTER=0.20 \
SPAWN_ARGS="--spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20 --spawn-lateral 0.20 --one-shot --mid-band 0.20 --until-ok 20" \
  bash scripts/collect.sh <批数> <每批局数上限> <首个场景种子>
```

`collect.sh` 每批重建场景、重起栈、跑到 `--until-ok` 局成功为止,并写
`manifest_<seed>.jsonl` 和 `pose_sidecar/`。产出的 exporter session 在
`$(locomani_session_root)/outputs/`,和 manifest **按顺序**配对给转换器。

**2026-09-10 起的两个默认**(台账 bottle §11):采集**不再每局重启 deploy**(`BottleTaskConfig.restart_deploy=false`,复位后 `reanchor` 代替,每局 64 → 34 s,30 局配对成功率无差);闭环评测仍**固定重启**(`eval_closed_loop` 钉 `RESTART_DEPLOY=True`,评测路径没在无重启下量过)。第三人称录像默认 `SIM_VIS=none`(`SonicBackendConfig.sim_vis`),要调试图用 `--set backend.sim_vis=full`。

代码里已经是默认值、不需要设的环境变量:

| 变量 | 默认 | 含义 |
|---|---|---|
| `GYM_ARM_CHANNEL` | `planner` | 上肢走 planner 通道,全程不切 streamed |
| `GYM_STAND_COVER` | `1` | 站位按整个可抓包络分层采样(旧行为是 2–3 个格) |
| `GYM_FACE_OBJ` | `0` | 接近段沿规划朝向走;`=1` 改成朝瓶子走,实测更差 |
| `GYM_ARM_PIN` | `0` | 不用 sim 侧手臂 pin |
| `EXPORTER_THREADS` | `4` | 容器 pid 上限 3840,不限会一个栈吃掉 2500 |

抓取阶段的关键常数(改了就不再与现有数据同分布):

| 常数 | 值 | 为什么 |
|---|---|---|
| `V_TRAVEL` / `V_DESCEND` | 0.15 / 0.10 m/s | WBC 手臂落后参考 0.2 s × 速度;0.30 时碰倒率 3/7,降速后 0/7 |
| `STAGE_UP` | 0.15 m | staging 点高于瓶顶,原来只高 5 cm 会擦到 |
| `LIFT_HEIGHT` | 0.10 m | 判据 0.06 m 之上留 4 cm |
| `MARGIN_STOP` | 0.075 | 刹车只要在带前触发,剩下的距离由 creep 闭环补 |
| creep 容差 | 方位角 10°、距离 5 cm,连续满足 0.4 s | 控制器朝向死区约 7–8°,容差落在死区里会原地蹭 |
| one-shot | 开(`--one-shot`) | 下蹲后站位出带直接丢局,数据里不留修复段 |

**采集期望值**:成功率 76–83%,单局 66–95 s。低于 70% 先查场景种子和栈日志。

**每批必看**:`grep "^EPISODE" outputs/logs/collect_<name>/gym_*.log`,以及
`stream: creep2d ...` 一行(方位角是否收敛到 10° 内)。

---

## 1b. pick-and-place 采集(2026-09-10 起)

```bash
MUJOCO_GL=osmesa third_party/Isaac-GR00T/.venv/bin/python src/locomani/sim/build_sim_scene.py \
  --preset pick_place --harden-pads outputs/scene outputs/scene/pick_place        # 一次;XML 里要有 priority="1"
bash scripts/experiments/pick_place_pilot_20260910.sh outputs/experiments/<name> <成功数> 10 <批数> <seed0>
```

`configs/tasks/pick_place.json`(target soup_can、goal basket、`reach.grip_below_center -0.03`、`place.*`)。
验收线:抓起的局里 place ≥ 8/10、释放 xy 中位 < 3 cm、掌系漂移 < 2 cm(`docs/pick_place_collection.md` §5)。
起栈前读 `docs/STATE.md` 的 RUNNING 行。

---

## 2. 数据集转换

```bash
. scripts/paths.sh
SESSIONS=$(find "$(locomani_session_root)/outputs" -maxdepth 1 -mindepth 1 -type d -newer <批次开始时的标记文件> | sort)
MANIFESTS=$(ls outputs/logs/collect_<name>/manifest_*.jsonl | sort -V)
"$VENV_DATA_PY" src/locomani/pipeline/convert_to_hf_format.py \
    --sessions $SESSIONS --manifest $MANIFESTS \
    --out datasets/g1_locomani_<name>_train \
    --holdout 20 --holdout-out datasets/g1_locomani_<name>_val
```

session 数必须等于 manifest 数,否则配对错位(转换器会自检并报错)。转换固定
25 fps;闭环评测的 `TF_DATASET_FPS` 必须与之一致。

转完跑审计:

```bash
"$GR00T_PY" -m gr00t.data.stats --dataset-path datasets/g1_locomani_<name>_train \
    --embodiment-tag UNITREE_G1_SONIC --modality-config-path configs/gr00t_sonic_2cam.py
"$VENV_DATA_PY" src/locomani/pipeline/check_dataset.py datasets/g1_locomani_<name>_train
```

**18 维动作(可选)**:同一批采集不用重采就能导出 wrist SE(3) + 夹爪 + 底盘的 18 维
动作,来源是 pose sidecar 里记录的**指令**(17 个上肢关节目标做 FK、手部指令、planner 帧):

```bash
"$VENV_SIM_PY" src/teleop_free/tools/export_action18.py outputs/logs/collect_<name> --fps 25
```

---

## 3. 训练

```bash
CUDA_VISIBLE_DEVICES=0 SINGLE_GPU=1 RUN=<name> \
  DATA=$PWD/datasets/g1_locomani_<name>_train \
  bash scripts/train_bottle.sh
```

| 设置 | 值 | 说明 |
|---|---|---|
| 步数 | 10000 | 180 局 ≈ 5.6 epoch。**不要改**,除非你打算重跑基线:lr 是 cosine 按 `MAX_STEPS` 衰减到 0,换步数后同名 checkpoint 不是同一种东西 |
| 有效 batch | 64 | 32 GB 卡上拆成 8 × 8 累积,优化器看到的仍是 64 |
| 优化器 | `adamw_bnb_8bit` | 单卡显存选择,不是优化选择 |
| checkpoint | 每 1000 步,保留 3 个 | 只用 `checkpoint-10000` |

参考 loss:flow200 0.0483,smooth200 0.0482,dart200 0.0537。约 4 小时(单卡)。

脚本会拒绝在显卡被占时启动(`SQUAT_MAX`);两卡并行训练时第二个要换
`PORT=29510`,否则 torchrun 的 29500 端口冲突。

---

## 4. 闭环评测(干净协议)

一条命令一个臂:

```bash
bash scripts/run_eval_arm.sh <臂名> 100 <INSTANCE> <GPU> <policy端口> \
     outputs/train/<run>/checkpoint-10000
```

它替你做了协议里的每一件事:重建 seed-922 场景、`CLEAN_DEPLOY=1` 起栈、
`NO_ENCODER=1`、起 policy server、用 `--client async --clean-deploy --seed 6000
--spawn-edge 1.1 1.3 --fresh 100` 跑评测、结束后收栈。

| 设置 | 值 | 说明 |
|---|---|---|
| deploy | 干净(auto-start,全程不发 planner 命令) | 退化态下行走 token 只按 1/3 幅度执行,数字不可迁移 |
| 二进制 | nvdef(NVIDIA 原始 default_angles) | SIMPLE 的移植把肩 roll 0.2 改成 0,直走 0.68 → 0.44 m |
| 客户端 | `--client async`,重规划 0.4 s | blocking 比 async 低约 30 点 |
| ensemble | **关**(`--ensemble 1`) | 在 flow200 上无收益,在退化 deploy 上有害;它的收益是数据条件化的 |
| RTC | **关** | 三个变体全部显著负收益(22% / 2% / 22% vs 50%) |
| seed / 出生 | 6000 / 1.1–1.3 m | 与所有历史臂逐轮配对的前提 |
| 局数 | 100 | 50 局分不清 10 个百分点 |
| 时长 | 每臂约 2.5 h | 两卡并行时两个臂各占一个 INSTANCE |

多实例:`INSTANCE=k` 时所有端口是 base + 10k;policy server 端口用 6590 / 6591
(5590 会和 INSTANCE 1 的 keyboard 端口撞)。

**看进度**:

```bash
watch -n 30 bash scripts/eval_progress.sh <臂名> [<臂名> ...]
```

**结果分析**:

```bash
python3 src/locomani/tasks/bottle/failure_taxonomy.py outputs/eval/<臂名> --json
```

按阶段给出 16 类失败叶子(A 导航 / B 到位未合爪 / C 合爪未抬起 / D 抬起后丢 / E 摔),
比 eval 自带的三桶细得多。当前基线的失败里 76–78% 是 B4「伸手时把瓶子撞倒」。

两个臂之间比较**必须**逐轮配对(同 seed 的同一 round id),用 McNemar,不要比总数。

---

## 5. 一次完整的实验长什么样

```
采集 300 局   ~7 h   →  转换 + 审计  ~20 min
训练 10k 步   ~4 h
评测 100 局   ~2.5 h  →  failure_taxonomy + 与 base100 配对
```

单卡串行约 14 小时,两卡并行(采集和训练不能同时占同一张卡)约 10 小时。
