# 数据集:导出与审计

> exporter 原始 session → LeRobot v2.1,数据 schema,以及转完必跑的四段审计。

## 导出 LeRobot v2.1 并对接 GR00T

exporter 的原始 session 在 `$LOCOMANI_TMP/p03/outputs/`。转成 LeRobot v2.1
(格式对齐 `hojjunekim/g1_sonic_orange_to_plate_locomani`,含 29-DoF 关节状态 +
64 维 SONIC motion latent + 三路 30fps 视频):

```bash
third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
    src/teleop_free/tools/convert_to_hf_format.py \
    --sessions "$LOCOMANI_TMP"/p03/outputs/<session1> <session2> ... \
    --manifest outputs/logs/collect/manifest_1.jsonl manifest_2.jsonl ... \
    --out datasets/g1_locomani_lerobot
```

**`--sessions` 和 `--manifest` 必须一一对应、同序**:第 k 个 session 是第 k 个
manifest 那一批产生的。`collect.sh` 的批次是按 seed 递增跑的,session 目录名是时间戳,
所以两边都按各自的自然顺序排就对上了(seed 超过 9 时记得用 `sort -V`)。

对错了不会静默 —— 转换器**硬校验**两件事,不符直接报错退出:

- manifest 里 `produced` 的行数 == session 里的 episode 数
- 每一行的 `ok` == 该 episode 在 disk 上是否被 discarded

得到的数据集可直接喂给 `third_party/Isaac-GR00T` 做 GR00T 微调。上层/底层的详细
数据流见 `docs/architecture.drawio`。

### 转完先查一遍(强烈建议)

```bash
third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
    src/teleop_free/tools/check_dataset.py datasets/g1_locomani_lerobot
```

**通过全部检查才去烧 GPU。** 有问题就退出码 1,能直接挂在 CI 或训练前的 gate 上。
四段检查,由便宜到贵:

| 段 | 回答什么 | 内容 |
|---|---|---|
| **structure** | 文件之间自洽吗 | 计数三方一致、`episode_index` 连续、parquet 行数 == 声明长度、`index`/`frame_index`/`timestamp`/`task_index` 自洽、列宽 == `info.json` 声明、无 NaN/Inf、抽检视频帧数 == parquet 行数 |
| **modality** | GR00T 认得出来吗 | 官方 `unitree_g1_sonic` 点的 8 个 state group / 3 个 action group 是否都在、切片是否越界、总维度是否 46 / 78 |
| **grasp** | 数字描述的是一次抓取吗 | 每局有闭合事件、闭合时腕已在物体 30 cm 内、闭合**之后**物体才升、末段物↔腕距离恒定 |
| **alignment** | 第 t 帧和第 t 行是同一时刻吗 | 见下 |

常用开关:`--quick`(只跑前两段,不解码视频,几秒)、`--videos N`、`--align-episodes N`。

**为什么 alignment 要单独测。** 时序错位是这类数据最贵的 bug,而且**看不出来** ——
拖着视频对曲线,差一秒以内怎么看都像对的。所以这里是**量**出来的:腕相机硬连在手臂上,
所以逐帧图像变化量是手臂速度的代理,而手臂速度在表里也有
(`|d(observation.state)/dt|`)。两者互相关,峰值所在的 lag 就是错位帧数。

```
video vs table: 16 wrist-camera measurements, lag distribution {0: 16}
action vs state: body trails the token by a median +3 frames (100 ms)
```

第二条查的是**因果方向**:token 是指令、身体是响应,身体必须**滞后**。0 或负数说明两列在
时间上被搞反了。+3 帧 ≈ 100 ms 就是 SONIC WBC 从 latent 解码到关节动起来的响应时间。

> 相关曲线在 ±3、±6 有规律小凸起,是 50→30 Hz 重采样的混叠(5:3 每 3 帧重复一次取帧
> 模式),不是真实时延 —— 脚本对小于 0.02 的领先量直接当 0,就是为了不被它骗。

**一个反直觉的坑,写在这里免得再踩:** `observation.can_pose` / `plate_pose` 是
**9 维(xyz + rot6d)且在机器人坐标系下**,不是世界坐标。所以"物体相对起始高度上升了吗"
这种判据会在**机器人下蹲**时触发(骨盆降低 → 物体相对高度跳变),在完全正常的数据上报
0/200。判据必须锚在**夹爪闭合那一帧**上。

### VLA 部署时这条链路怎么变

训练好的模型输出 78 维,**不经过 encoder**,用 protocol v4 直接喂进 :5556:

```bash
NO_ENCODER=1 bash src/teleop_free/start_stack.sh     # 关键:必须不加载 encoder
```

`NO_ENCODER=1` 不是省显存。deploy 只要加载了 encoder,`initial_encoder_mode_` 就是 0,
它的 token 路径会不停地翻回 encoder("if encoder mode is not -1, we need to turn on
the encoder",`g1_deploy_onnx_ref.cpp:3096`),外部 token 流永远压不住。不给
`--encoder-file`,mode 变成 -1,才会走 "tokens can be set externally"。

这条路已经端到端验证过(`diag_token_replay.py`):录下成功 episode 的 64 维 token
再喂回去,关节轨迹复现到 **0.2–0.9°**。开环回放抓不到罐子(浮动基座把误差积分成
0.77 m 位移),那是开环的固有性质,不是链路问题 —— VLA 部署是闭环的。

---

## 数据 schema

一条 episode 的 parquet 里列很多,但**训练用的只有下面这些**。完整推导见
[`patches/vla_action_space.md`](patches/vla_action_space.md)。

### action:78 维

| 分量 | 维度 | 说明 |
|---|---|---|
| `motion_token` | 64 | SONIC latent。**1/16 量化**(整条 episode 只有 31 个取值,约 5 bit/维)→ VLA 输出头应该做**分类**而不是回归 |
| `left_hand_joints` | 7 | 本仓库的任务全程用右手,这一列恒定 |
| `right_hand_joints` | 7 | dex1 是平行夹爪:分量 0 和 1 同步开合(−0.020..0.024),2–6 恒定 |

**这三个名字和 7 这个宽度都不能改。** Isaac-GR00T 的 `unitree_g1_sonic` 配置
(`gr00t/configs/data/embodiment_configs.py`)是**按名字**挑 action group 的,挑不到只打
一句 `Warning: Joint group 'left_hand_joints' not found` 就继续跑 —— 训练照常收敛,只是
两只手被静默丢掉。78 = 64 + 7 + 7 也正是 SONIC 推理端期望的 action space。

至于为什么只有这三个:deploy 的 pose protocol v4 是 token-only 流,必填 `token_state`,
选填 `frame_index` / `left_hand_joints` / `right_hand_joints` / `body_quat_w` ——
**planner 指令(mode / movement / facing / speed / height)没有通道**。它们仍然被写进
parquet(将来可以当辅助监督),但不是 action。

### state:46 维

官方配置只取 8 个 group:`left_leg` 6 + `right_leg` 6 + `waist` 3 + `left_arm` 7 +
`right_arm` 7 + `left_hand` 7 + `right_hand` 7 + `projected_gravity` 3。

`meta/modality.json` 是**菜单**,embodiment 配置才是**点单** —— 菜单里多列的条目
(腕位姿、root 四元数、`can_pose`/`plate_pose` 等特权信息)一个都不会进模型,所以多列
无害;但**改名会致命**,因为点单点不到。

### observation

三路 RGB(`ego_view` / `left_wrist` / `right_wrist`)+ 43 维本体状态(腿 12 + 腰 3 +
双臂 14 + 双手 14)+ 每局自己的指令(`task_index` → `meta/tasks.jsonl`)。

特权量(`observation.can_pose` / `plate_pose`,骨盆系)**留在 parquet 里但不在
modality 里**。它们是分析和写判据用的;把它们声明成 state 就等于让模型吃一条部署时
根本不存在的通道 —— deploy 只有 ego 相机和本体感受。这是这里唯一一个**会照样让 loss
下降**的错误,所以 `convert_to_hf_format.py` 的 `MODALITY` 明确把它们排除在外。

### modality.json 里的切片

转换器写出的 `meta/modality.json` / `modality_gr00t_sonic.json` 就是 GR00T 实际读的
切片表:

| | 维度 | 内容 |
|---|---|---|
| `action` | **66** | `motion_token` 64 + 左右夹爪各 1(7 宽的列是 Dex3 遗留,只取第 0 个分量) |
| `state` | **72** | 43 维关节 + 腕位姿 14 + root/gravity/offset/init_quat 15 |
| `video` | 3 | 按 session 实际有的流填,不写死 |
