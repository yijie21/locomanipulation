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
SIM_VIDEO_DIR="$PWD/outputs/sim_video" bash scripts/stack/start_stack.sh "pick up the water bottle"

# 3. 跑 12 局,通过全部判据的落盘,失败的自动丢弃
#    后四个 spawn 参数不是可选装饰:它们定义了任务有多难,而本仓库两轮实验的
#    全部数字都是在这套窄分布上测的。照 grasp_gym 的默认跑(3 m 外、背对物体、
#    朝向偏差可达 90°)成功率会掉到 30% 出头 —— 那不是 bug,是另一个任务
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
    src/locomani/tasks/bottle/grasp_gym.py --rounds 12 --record \
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

收尾 `bash scripts/stack/stop_stack.sh`(它会杀掉机器上**所有**实例;多实例并行用 `scripts/stack/stop_instance.sh <INSTANCE>`)。落盘的东西在三个地方:

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

## 目录结构(2026-09 重构后)

```
src/locomani/pipeline/      stack 生命周期、gym 主循环、录制、manifest、convert、审计、train、eval、配对统计
src/locomani/backends/      Backend:stack 启动、动作空间、执行器。sonic/(主线)、scalebfm/(候选替代 controller)
src/locomani/tasks/         Task 五件套(scene / expert / acceptance / schema / failures):bottle/、crate/、bottle_visual/、pick_place/
src/locomani/primitives/    真正共享的底层(arm_ik、scene_lib、primitives)
src/locomani/sim/           sim 侧:特权发布、场景随机化、Isaac 渲染
src/locomani/cli.py         统一入口:python -m locomani.cli collect|convert|train|eval|gate --task X --config Y
configs/{tasks,backends,eval,pilot}/*.json   所有旋钮(代码里没有 os.environ 旋钮);字段表见 docs/reference/config_fields.md
configs/gr00t_*.py          GR00T embodiment 配置(gr00t_sonic_2cam.py = 主线训练用的那份)
scripts/stack/              起 / 停 stack;scripts/ 其余是安装、批量采集、训练、评测、闸门(gate.py)
src/gr00t/                  GR00T 训练代码(vendored,入 git;出处见 src/gr00t/UPSTREAM.md)
docs/                       四类文档(见下)
assets/                     预生成的仿真场景 tarball、ScaleBFM 资产
third_party/                clone 的上游仓库(不入 git;venv 也住在这里)
outputs/                    运行时生成(场景、日志、checkpoint、评测、闸门结果),不入 git
```

代码与文档规范只有一份:[docs/reference/standards.md](docs/reference/standards.md)(`make check` = ruff + 尺寸/环境变量 lint + unit 测试;改主代码的 PR 附 `scripts/gate.py compare` 输出)。

## 文档地图

只有四类。**权威索引是 [docs/README.md](docs/README.md)** —— 新增文档只需要登记在那里;
下面这张表是按"你现在要干什么"挑出来的快捷入口,是它的子集,不是第二份清单。


| 你要 | 看 |
|---|---|
| **当前默认的采集 / 训练 / 评测配方(先看这个)** | **[docs/reference/runbook.md](docs/reference/runbook.md)** |
| 现在在做什么、什么在跑、卡在哪 | [docs/STATE.md](docs/STATE.md)(每个 session 从这里开始) |
| 从零装环境 | [docs/setup.md](docs/setup.md) |
| 采数据(场景 / 单局 / 批量 / 控制逻辑 / 随机化 / 相机 / 坑) | [docs/collection.md](docs/collection.md) |
| 双臂搬箱(采集到 MuJoCo/Isaac 并排视频) | [docs/crate_collection.md](docs/crate_collection.md) |
| ScaleBFM 视觉抓瓶线(SONIC 的候选替代 controller) | [docs/scalebfm_collection.md](docs/scalebfm_collection.md) |
| pick-and-place(抓起罐头放进篮子,bottle 线加一个 place 阶段) | [docs/pick_place_collection.md](docs/pick_place_collection.md) |
| 转 LeRobot、查数据(schema / 四段审计) | [docs/dataset.md](docs/dataset.md) |
| 微调 GR00T | [docs/training.md](docs/training.md) |
| 评测(open-loop / closed-loop / 配对检验) | [docs/evaluation.md](docs/evaluation.md) |
| Isaac 渲染 | [docs/rendering.md](docs/rendering.md) |
| 全部实验的总表与裁定(append-only 台账) | [ledger/bottle.md](docs/ledger/bottle.md) · [crate](docs/ledger/crate.md) · [scalebfm](docs/ledger/scalebfm.md) · [pick_place](docs/ledger/pick_place.md) · [refactor](docs/ledger/refactor.md) |
| 查模块、参数、端口、静默失败表、FAQ | [docs/reference/silent_failures.md](docs/reference/silent_failures.md) |
| 配置字段(从 dataclass 注释生成) | [docs/reference/config_fields.md](docs/reference/config_fields.md) |
| 下一步做什么 | [docs/ROADMAP.md](docs/ROADMAP.md) |
| 历史诊断、设计讨论、旧进展报告 | [docs/archive/INDEX.md](docs/archive/INDEX.md) |

## 当前状态(2026-09-10)

- 主线(单臂抓瓶,SONIC)最好的正式基线:**flow200 checkpoint-10000,干净协议,无 ensemble,50/100**
  (runbook);ready-grasp 线 07d-200 在 seed 42 上 57/100、seed 7 上 46/100 —— 同数据换 seed 差 11 个点,
  任何单 seed 的"+8"都当噪声看([docs/ledger/bottle.md](docs/ledger/bottle.md))。
- 2026-08 的四轮进展(5.9% → 47.5%)见 [docs/archive/progress_report_2026-08.md](docs/archive/progress_report_2026-08.md)。
- 2026-09-09 的 outputs/ 误删事故(所有数据集、模型、评测)与恢复选项在 [docs/STATE.md](docs/STATE.md) 顶部。
- ScaleBFM 线:controller 替换成立,VLA 最高 25/100,代码已并入 `backends/scalebfm` + `tasks/bottle_visual`
  ([docs/ledger/scalebfm.md](docs/ledger/scalebfm.md))。

> **这些成功率只在上面快速开始第 3 步那套窄 spawn 分布下成立**
> (0.7 m 起步、`--face-object`、朝向偏差 ±20°、横偏 ±0.20 m)。换成 `grasp_gym`
> 的默认分布(3 m 外、背对物体、±90°)是另一个更难的任务,连脚本专家自己的
> 成功率都会从 85% 掉到 30% 出头。复现这些数字必须带上那四个参数。

---

## 路线图

下一步要做什么、按什么顺序做,在 [`docs/ROADMAP.md`](docs/ROADMAP.md)。
其中**第 0 节列了已经验证过、不要重做的结论**,动手前先看那一节。
