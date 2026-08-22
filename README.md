# Teleop-Free Loco-Manipulation Data Engine (Unitree G1 + SONIC)

不需要遥操作的人形机器人 loco-manipulation 数据引擎:在 MuJoCo 里由脚本化 agent 驱动
Unitree G1(SONIC 全身控制),自动完成"走到桌前 → 抓起物体"的完整 episode,导出
**LeRobot v2.1** 格式数据集,用于 GR00T + SONIC 的微调训练。

核心链路:

```
MuJoCo sim (run_sim_locomani.py)
   ↑ DDS (lo)                 ──camera :5555──→  data exporter (LeRobot 原始 session)
C++ deploy (g1_deploy_onnx_ref, SONIC policy)
   ↑ :5556 planner/pose 意图流
脚本化 agent (collect_episode.py: 导航 → 两段式接近 → 流式手臂 servo 抓取)
```

## 目录结构

```
src/teleop_free/   数据引擎全部代码(sim 封装、primitives、导航、抓取、episode、诊断脚本)
src/teleop_free/tools/  LeRobot v2.1 转换器
scripts/           安装 / 启动 / 批跑脚本
assets/            预生成的仿真场景 tarball(免去 EmbodiedGen 场景生成)
docs/              架构图、reach map 可视化等
patches/           对上游仓库的小改动说明
third_party/       clone 的上游仓库(不入 git):GR00T-WholeBodyControl、Isaac-GR00T、
                   EmbodiedGen、IsaacLab、IsaacLab-Arena
outputs/           运行时生成(场景、日志、原始视频、episodes),不入 git
videos/            最新一批演示视频(每次推送会整体替换)
```

> **路径约定**:代码内的绝对路径假定仓库位于 `/workspace/code/locomanipulation`。
> 如果放在别处,先全局替换:
> `grep -rl "/workspace/code/locomanipulation" src scripts | xargs sed -i "s#/workspace/code/locomanipulation#<你的路径>#g"`,
> 场景内的路径由 `scripts/install_scene.sh` 自动改写。

## 环境安装

前置:Linux + NVIDIA GPU(本仓库在 RTX 5090 / driver CUDA 13 上验证)、
[miniconda](https://docs.conda.io)、[uv](https://docs.astral.sh/uv/)。

### 1. Clone 上游仓库到 third_party/

```bash
cd third_party
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git   # 必需:SONIC sim + deploy + exporter
git clone https://github.com/NVIDIA/Isaac-GR00T.git              # 可选:GR00T 微调
git clone https://github.com/HorizonRobotics/EmbodiedGen.git     # 可选:只有重新生成场景才需要
git clone https://github.com/isaac-sim/IsaacLab.git              # 可选:SONIC 训练(Isaac Sim)
```

### 2. 仿真与数据导出 venv(必需)

用上游自带的安装脚本,在 `third_party/GR00T-WholeBodyControl/` 下创建两个 venv:

```bash
cd third_party/GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh        # -> .venv_sim(跑 MuJoCo sim 与所有 src/teleop_free 脚本)
bash install_scripts/install_data_collection.sh   # -> .venv_data_collection(跑 data exporter)
python download_from_hf.py                        # 下载 SONIC policy 权重(policy/release)
```

### 3. 编译 C++ deploy(必需)

> **警告:不要运行 `gear_sonic_deploy/scripts/install_deps.sh`。**
> 它有 `apt install cuda` 的 fallback,会连带安装 `cuda-drivers` 覆盖宿主 NVIDIA 驱动
> (在云 GPU 容器里会直接毁掉整机 GPU)。按下面手动装依赖:

```bash
# 依赖放 /workspace/deps(路径写在 scripts/env_deploy.sh 里,可自行修改)
# 1) TensorRT 10.13.x:apt-get download 对应 deb 后 dpkg -x 解包,并把
#    include/lib 软链成官方 TAR 布局;2) ONNX Runtime 1.16.3 tarball 解包;
# 3) 小依赖:apt install libeigen3-dev libzmq3-dev cppzmq-dev libgtest-dev \
#            nlohmann-json3-dev libmsgpack-cxx-dev
source scripts/env_deploy.sh
cmake -S "$SONIC_DEPLOY_ROOT" -B "$SONIC_DEPLOY_ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SONIC_DEPLOY_ROOT/build" -j$(nproc)
```

首次运行 deploy 会编译 TensorRT engine(约 10 分钟),之后走缓存。

### 4. conda 训练环境(可选)

只在需要 SONIC 训练/eval(Isaac Sim)时安装:

```bash
bash scripts/setup_sonic.sh    # conda env "sonic":Isaac Sim 5.1 + torch cu128 + IsaacLab
bash scripts/setup_sonic2.sh   # IsaacLab 固定到 v2.3.2 并装 gear_sonic[training]
```

## 场景:直接用现成的(推荐)

场景生成(EmbodiedGen 文生 3D + 布局 + 物理注入)非常耗时,仓库直接附带了做好的场景:

```bash
bash scripts/install_scene.sh
```

解包 `assets/scene_locomani_v1.tar.gz` 到 `outputs/scene/`(双桌面 + 橙子/盘子等物体的
MJCF、抓取标注、reach map、hand points),并自动把 MJCF 里的绝对路径改写到当前 checkout。

如需重新生成场景:装好 EmbodiedGen(`src/teleop_free/install_embodiedgen.sh`),依次跑
`compose_scene.py` → `inject_physics_into_mjcf.py` → `build_sim_scene.py`,再用
`make_hand_points.py` / `reach_map.py` / `wrist_map.py` 重建缓存。

## 运行:收集数据

```bash
# 1. 起整个 stack(sim + deploy + exporter),日志在 /workspace/tmp/p03/
SIM_VIDEO_DIR=outputs/sim_video bash src/teleop_free/start_stack.sh

# 2. 采集(--record 驱动 exporter 落盘;不加就是只跑不落盘)
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
    src/teleop_free/grasp_gym.py --rounds 12 --seed 21 --record
```

要点:

- **采集入口是 `grasp_gym.py --record`,不是 `collect_episode.py`。** 后者还停在旧的
  控制路径上(走路时双臂被 pin 成平举、抓取带 sim 侧关节 pin),而且有一套独立的导航
  实现 —— 所有经过验证的逻辑都在 `grasp_gym.py` 里。`collect_episode.py` 只在需要
  完整任务(抓起 → 搬运 → 放下,`FULL_TASK=1`)时才有用,那条路径尚未迁移。
- 失败的 episode 会被 `x` 丢弃,只有通过全部 stage 谓词的才落盘。
- `INSTANCE=i` 让第 i 个实例用 `基准端口+10*i` 和 DDS domain i,多实例互不干扰;
  只有 instance 0 挂 exporter 落盘。
- `NO_ENCODER=1` 起 stack 会**不加载 encoder**,这是 token-only(VLA 部署)配置,
  见下面的路线图第 0 节。
- 一条 episode 约 67 s、约 4.7 MB(视频 + parquet)。

## 导出 LeRobot v2.1 并对接 GR00T + SONIC

exporter 的原始 session 在 `/workspace/tmp/p03/outputs/`。转换成 LeRobot v2.1
(格式对齐 `hojjunekim/g1_sonic_orange_to_plate_locomani`,含 29-DoF 关节状态 +
64 维 SONIC motion latent + 30fps 视频):

```bash
third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
    src/teleop_free/tools/convert_to_hf_format.py \
    --sessions /workspace/tmp/p03/outputs/<session...> \
    --out datasets/g1_locomani_lerobot
```

得到的数据集可直接喂给 `third_party/Isaac-GR00T` 做 GR00T 微调(SONIC 作为底层 WBC,
GR00T 输出的动作经 SONIC encoder 的 64 维 latent 通道驱动机器人)。上层/底层的详细
数据流见 `docs/architecture.drawio`。


## 路线图:从这里到大规模数据集

按顺序做。每一步都可能打破规划器的假设,所以**每加一个随机化维度都要先跑 12 局
确认成功率没崩**,再采数据 —— 否则采到的是一堆废片。

### 第 0 节:已经验证过的,不要重做

- **动作质量**:不踱步、不撞桌、站定不跳变、抓取不悬停。12/12(seed 21)、6/6(seed 33)。
- **没有 sim 侧关节 pin**。抓取全程由 policy 自己完成(`GYM_ARM_PIN` 默认 0)。
- **latent 就是 action**(`diag_token_replay.py`):录下成功 episode 的 64 维 token,
  在 `NO_ENCODER=1` 的 stack 上用 protocol v4 喂回去,关节轨迹复现到 **0.2–0.9°**。
  开环回放不会抓到罐子(浮动基座把误差积分成 0.77 m 位移),这是开环的固有性质,
  不是失败 —— VLA 部署是闭环的。
- **Action 空间 = 66 维**(token 64 + 双夹爪 1+1),由 wire protocol 定死:v4 只运
  `token_state` + 可选夹爪,planner 指令没有通道。
- **数据集能被真 LeRobot loader 加载**,不只是手工检查 parquet。
- **token 是 1/16 量化的**(整条 episode 31 个取值,约 5 bit/维)→ VLA 输出头应该
  **分类**而不是回归。

### 第 1 步:摸清场景生成的参数化能力(~15 分钟)

查 `build_sim_scene.py` 是否支持参数化**桌高**和**物体尺寸**。这一步把随机化
第 1/3/4 项的成本从"未知"变成"已知",不做的话后面无法排期。

### 第 2 步:上两个不需要重新生成场景的随机化

- **目标物体身份 + 指令**:桌上已有 plate / fork / napkin / orange,但目标永远是罐子,
  模型可以完全不读指令。把目标随机化并在指令里指名 —— **这是"抓取策略"和"VLA"的
  分界线**。注意 fork / napkin 是扁平的,dex1 平行夹爪多半抓不了,别放进目标池。
- **机器人初始位置与朝向**:现在严格在接近线后方 1.8 m、朝向 ±25°。加出生距离
  (1.0–3.0 m)、**横向偏移**、更大的初始朝向误差(到 ±90°)。转身必须发生在远处 ——
  近距离的 `face_target` 会让机器人原地转 110–160°,这是记录在案的坑。

### 第 3 步:验证(12 局)

成功率没崩再往下。崩了就先修,别继续叠维度。

### 第 4 步:外观随机化

物体/桌面材质与颜色、光照方向与强度、地面纹理、背景。**一样都没变过** —— 现在训出来的
模型很可能是在认"那个特定灰色的圆柱"。只改 material,不碰控制逻辑,零风险。

### 第 5 步:放开蹲深上限,然后上桌高与物体尺寸

前置(**这是硬阻塞**):

- `reach_map.py` 的 `PELVIS_MIN = 0.598` 是**旧下蹲机制**的上限(19.5 cm),而
  `IDEL_SQUAT` 实测能到 **0.51**(27 cm,见 `diag_squatmode.py --sweep`)。不放开的话
  矮桌会被 `plan_core_stand` 判成不可达。改完要重建 reach map。
- 每个桌高都要用 `diag_ego_geom.py` 复查罐子在决策点(约 1 m 外)是否可见:直立时
  可见区间是 0.15–1.77 m,而这个上限随物体高度变,桌子越矮远端越近。

然后才是:

- **桌高**(现在固定 0.62 m,真实桌子 0.72–0.75)
- **物体尺寸与高度**(现在固定 4.8×12 cm;夹爪开口 9.49 cm,所以 3–8 cm 直径都在能力内)。
  物体高度直接决定蹲多深 —— **现在蹲深永远是 13 cm,这个轴等于没训**。

### 第 6 步:大规模采集

一条约 67 s、4.7 MB。规模不是瓶颈(1000 条约 5 GB),但加上 5 个随机化维度后
200 条会被摊得很薄,先定目标数量。

### 以后再说

- **episode 内扰动**:接近/伸手途中推罐子(`pin_object` 现成),或给骨盆一个冲量
  (需要在 `privileged_pub` 加 `xfrc_applied`)。按之前的结论,**等第一批模型告诉我们
  缺什么再上** —— 如果失败是"轻微偏了然后越偏越远"就补扰动,如果是"完全没对准"
  就补多样性。
- **质量 / 摩擦 / 重心**:RGB 看不见却改变结果,进训练集会造成别名化。**只做评测集**。
- **完整任务**(抓起 → 搬运 → 放下):在 `collect_episode.py` 的 `FULL_TASK=1` 里,
  那条路径还没迁移到新控制逻辑。只有抓取的数据集上限比较低。

## 常用诊断脚本

`src/teleop_free/diag_*.py`(都用 `.venv_sim` 的 python,在 stack 起来后运行):
`diag_ready_walk.py` 端臂行走、`diag_reach_probe.py` reach map 实测校验、
`diag_stream.py` 流式手臂通道、`diag_nav.py` 导航等。

Phase 0 / Phase 1 期间新增的(结论都写在各自的 docstring 里):

| 脚本 | 回答什么 |
|---|---|
| `diag_channel.py` + `diag_channel_report.py` | planner 通道 vs streamed 通道的手臂投递精度(216 样本) |
| `diag_squatmode.py [--sweep]` | `IDEL_SQUAT` 的蹲深,以及 `height` → 骨盆高度的标定曲线 |
| `diag_lean.py` | 命令的腰部前倾有多少被交付(答案:2–10%,基本不交付) |
| `diag_height.py` | `height` 字段在 IDLE / SLOW_WALK 下是否有效(答案:没有) |
| `diag_ego_geom.py` | 罐子在 ego 相机里的可见距离区间,随躯干前倾的变化 |
| `diag_token_replay.py --phase capture/replay` | 录下的 token 能否复现动作(见路线图第 0 节) |
