# Teleop-Free Loco-Manipulation Data Engine (Unitree G1 + SONIC)

不需要遥操作的人形机器人 loco-manipulation 数据引擎:在 MuJoCo 里由**脚本化 agent**
驱动 Unitree G1(SONIC 全身控制),自动完成"走到桌前 → 抓起物体"的完整 episode,
导出 **LeRobot v2.1** 数据集,用于 GR00T + SONIC 的 VLA 微调。

**它到底产出什么。**每条成功 episode = 三路 RGB(ego + 左右腕)+ 一条 **78 维 action
序列**(SONIC 的 64 维 motion latent + 左右手关节各 7 维)。这 78 维不是设计选择,是被
deploy 的 wire protocol 定死的:v4 只运 `token_state` + 可选手关节。训练出的 VLA 输入
RGB + 指令(数据集里同时存了 29 维本体状态,要不要喂给模型是训练侧的选择),输出这
78 维,直接喂回同一个 deploy 就能驱动机器人 —— 这条链路已经端到端验证过
(见 [VLA 部署时这条链路怎么变](docs/dataset.md#vla-部署时这条链路怎么变))。

脚本化 agent 用的是仿真里的**特权信息**(物体真值位姿、可达域、桌沿距离),所以不需要
人来遥操作,也不需要人来判断成功 —— 但这些特权信息只用于**生成**动作,不进入训练输入。

---

---

## 快速开始

仓库已经装好的话,从零到第一条落盘的成功 episode 就是下面这几行(还没装看
[环境安装](docs/setup.md#环境安装),约 1 小时):

```bash
# 0. 仓库外那些东西挂在哪(依赖、conda、日志)。默认 /workspace
export LOCOMANI_WORKSPACE=/workspace

# 1. 装场景(解包 tarball + 改写路径 + 重建),约 5 s
# SCENE_PRESET=bottle_only 是复现本仓库两轮 VLA 实验的那张桌子(一个 RoboCasa
# 水瓶,别无他物)。不加的话得到的是老的多物体桌面 —— 资产都在 git 里,
# 选哪套只由这个变量决定,建场景时生效而不是起 stack 时。
SCENE_PRESET=bottle_only bash scripts/install_scene.sh

# 2. 起 stack = MuJoCo sim + C++ deploy + data exporter
#    首次约 10 min(编 TensorRT engine),之后 40 s。SIM_VIDEO_DIR 必须在这里给,
#    而且必须是绝对路径 —— sim 进程的 cwd 是 third_party/GR00T-WholeBodyControl/,
#    不是仓库根,相对路径会悄悄写到那边去
SIM_VIDEO_DIR="$PWD/outputs/sim_video" bash src/teleop_free/start_stack.sh "pick up the water bottle"

# 3. 跑 12 局,通过全部判据的落盘,失败的自动丢弃
#    后四个 spawn 参数不是可选装饰:它们定义了任务有多难,而本仓库两轮实验的
#    全部数字都是在这套窄分布上测的。照 grasp_gym 的默认跑(3 m 外、背对物体、
#    朝向偏差可达 90°)成功率会掉到 30% 出头 —— 那不是 bug,是另一个任务
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
    src/teleop_free/grasp_gym.py --rounds 12 --record \
    --targets water_bottle --edge-jitter 0.20 \
    --spawn-edge 0.7 --face-object --spawn-yaw-err 20 --spawn-lateral 0.20 \
    --manifest outputs/logs/manifest.jsonl
```

四个 spawn 参数各自干什么、为什么是这些值:

| 参数 | grasp_gym 默认 | 本仓库实验用 | 差别 |
|---|---|---|---|
| `--spawn-edge` | 无(退回 `--spawn-off`,站位后方 1–3 m) | `0.7` | 距**桌子** 0.7 m 起步,走路段基本被送掉 |
| `--face-object` | 关(朝接近线) | 开 | 醒来就正对物体,而不是正对接近线 |
| `--spawn-yaw-err` | `90` | `20` | 初始朝向偏差上限,从 ±90° 收到 ±20° |
| `--spawn-lateral` | `0.35` | `0.20` | 横向偏移上限 |
| `--edge-jitter` | `0.10` | `0.20` | **这一项我们更难**:物体沿桌沿的散布放宽一倍 |

除 `--edge-jitter` 外每一项都是把任务调容易。这是刻意的 —— 第一批数据要验证的是
「转换 → 微调 → 78 维喂回 deploy」这条链路通不通,不是模型能泛化到哪。代价是
**在 0.7 m / face-object 数据上训出来的 VLA,不应期待它能泛化到 3 m / ±90° 起步**。

跑起来长这样,一行一局:

```
EPISODE OK in 74s
EPISODE FAILED [grasp] can toppled in the up phase in 64s
[gym] 11/12 OK
```

收尾 `bash src/teleop_free/stop_stack.sh`。落盘的东西在三个地方:

| 东西 | 在哪 |
|---|---|
| exporter 原始 session(训练数据) | `$LOCOMANI_TMP/p03/outputs/<时间戳>/` |
| 第三人称录像(带 ego + 双腕画中画) | 上面第 2 步 `SIM_VIDEO_DIR` 给的那个绝对路径 |
| manifest(每局的目标、指令、成败) | 你用 `--manifest` 指定的路径 |

三条各自怎么配、有什么坑,见[跑一局 → 保存的数据在哪、怎么配路径](docs/collection.md#保存的数据在哪怎么配路径)。

**要采一整个数据集不要手跑这几步**,用两个批量脚本之一 —— 选哪个取决于你要什么:

- **复现本仓库的实验** → `scripts/collect_pilot.sh`。场景尺寸固定,只变出生点和物体
  位置,`SPAWN_ARGS` 钉住上面那套窄分布。见
  [只想先测通 pipeline](docs/collection.md#只想先测通-pipelinecollect_pilotsh)。
- **摊开随机化轴采更泛化的数据** → `scripts/collect.sh`。每批重建场景换桌高和物体
  尺寸。**它的 spawn 默认是宽的**,要窄分布得自己传 `SPAWN_ARGS`/`TARGETS`/`JITTER`
  (脚本头部有现成的一行)。见[大规模采集](docs/collection.md#大规模采集)。转成 LeRobot 见[导出](docs/dataset.md#导出-lerobot-v21-并对接-gr00t)。

---

---

## 目录结构

```
src/teleop_free/        数据引擎全部代码(模块地图见 docs/reference.md)
src/teleop_free/tools/  转换器 / 审计 / 评测 / policy server / token 工具
src/gr00t/              GR00T 训练代码(vendored,入 git,二次开发改这里;
                        出处与上游 diff 方法见 src/gr00t/UPSTREAM.md)
configs/                embodiment 配置(gr00t_sonic_2cam.py = 本项目训练用的那份)
scripts/                安装 / 启动 / 采集 / 训练 / 流水线脚本
docs/                   分主题文档(见下)+ 架构图 + 路线图
assets/                 预生成的仿真场景 tarball
patches/                对上游仓库的小改动说明(third_party 不入 git)
third_party/            clone 的上游仓库(不入 git;venv 也住在这里)
outputs/                运行时生成(场景、日志、checkpoint、评测),不入 git
datasets/               转换出的 LeRobot 数据集,不入 git
videos/                 最新一批演示视频(每次推送整体替换)
```

## 文档地图

按"你现在要干什么"索引:

| 你要 | 看 |
|---|---|
| 从零装环境 | [docs/setup.md](docs/setup.md) |
| 采数据(场景 / 单局 / 批量 / 随机化 / 相机 / 控制逻辑) | [docs/collection.md](docs/collection.md) |
| 转 LeRobot、查数据(schema / 四段审计) | [docs/dataset.md](docs/dataset.md) |
| 微调 GR00T(设置全表 / 启动 / vendored 代码怎么回事) | [docs/training.md](docs/training.md) |
| 评测(open-loop / closed-loop / 第一轮基线 / DAgger) | [docs/evaluation.md](docs/evaluation.md) |
| 搞懂 async client(实时动作分块 / 异步推理 / 时序集成) | [docs/realtime-action-streaming.md](docs/realtime-action-streaming.md) |
| 零基础图解 blocking / async 机制(时间线图) | [docs/client-mechanisms.md](docs/client-mechanisms.md)(GitHub 在线可看;[html 交互版](docs/client-mechanisms.html)) |
| 查参数、找模块、跑诊断、翻 FAQ | [docs/reference.md](docs/reference.md) |
| 下一步做什么 | [docs/ROADMAP.md](docs/ROADMAP.md) |

## 当前状态(2026-08-27)

四轮微调已闭环,**成功率 5.9% → 13.7% → 28.0% → 47.5%**:

- **Round 1**(200 局采集数据):开环贴测量噪声底(关节角差 1.05°),**闭环 6/101**,
  失败塌缩在 walk→reach 停滞 —— 训练数据没有"到桌边臂还垂着"这个状态。
- **Round 2**(+50 局靶向 DAgger 恢复数据,bottle250):同款开环质量,
  **闭环 14/102 = 13.7%(Wilson 8–22%)**,round 1 的 2.3 倍 —— 靶向 DAgger
  方向确认。诊断与两轮数字在 [docs/evaluation.md](docs/evaluation.md)。
- **Round 3**(fast200,200 局,**零 DAgger**):把脚本专家的停顿削掉(到位→伸手
  中位 4.6 s → 3.0 s,episode 35 s → 20–22 s),**闭环 28/100 = 28.0%
  (Wilson 20–37%)**,round 1 的 4.7 倍。用的还是 200 局 —— 证实病在**数据形状**
  不在数据量。失败结构整个变了:**walk / approach 零失败**,67 次失败全是
  "已到桌边(0.23–0.29 m)却不下闭合指令"。开环闭合帧误差只有 1.75 帧,说明模型
  离线知道该何时闭合、闭环下不发 —— 典型 covariate shift,下一轮 DAgger 该补
  "临近闭合"而非恢复轨迹。
- 开环指标经 **NVIDIA 官方 `gr00t/eval/open_loop_eval.py` 交叉验证**:同一批
  checkpoint 排序一致(Spearman **+0.988**),best 同为 checkpoint-9000。
- 训练按硬件自选优化器(< 40 GB 的卡走 8-bit,否则 fp32),并在启动前预检显存
  占用;冻结 DiT 消融(初步:DiT 贡献 ~12% 开环精度,闭环等价性未测)。

- **Round 4**(fast250 = 200 + 50 靶向 DAgger):**闭环 48/101 = 47.5%
  (Wilson 38–57%)**。配对 McNemar(初始条件逐局相同)把增益拆开了:
  **配置改动(FSQ 不吸附)p=0.755 不显著**,50 轮 A/B 报出的 +12 个百分点是噪声;
  **数据增益 p=0.016 显著**。教训:50 轮闭环分辨不出 10 个百分点。
  剩余瓶颈:52 个 reach 失败里 44 个是"手臂抬了但够不到物体",而"手到位不扣扳机"
  已从 12 降到 3。

wandb: locomani-bottle / bottle200 · bottle250 · bottle250_frozen_dit · fast200 · fast250。

> **这些成功率只在上面快速开始第 3 步那套窄 spawn 分布下成立**
> (0.7 m 起步、`--face-object`、朝向偏差 ±20°、横偏 ±0.20 m)。换成 `grasp_gym`
> 的默认分布(3 m 外、背对物体、±90°)是另一个更难的任务,连脚本专家自己的
> 成功率都会从 85% 掉到 30% 出头。复现这些数字必须带上那四个参数。

---

## 路线图

下一步要做什么、按什么顺序做,在 [`docs/ROADMAP.md`](docs/ROADMAP.md)。
其中**第 0 节列了已经验证过、不要重做的结论**,动手前先看那一节。
