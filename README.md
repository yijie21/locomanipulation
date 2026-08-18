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
bash src/teleop_free/start_stack.sh

# 2. 跑一条 episode(--no-record 表示只迭代不落盘)
cd src/teleop_free
../../third_party/GR00T-WholeBodyControl/.venv_sim/bin/python collect_episode.py

# 3. 批量并行收集(3 个 sim 实例并行,每实例 EPN 条)
EPN=5 bash scripts/par_batch.sh    # 结果汇总在 outputs/logs/parbatch/result.txt
```

要点:

- `INSTANCE=i` 让第 i 个实例用 `基准端口+10*i` 和 DDS domain i,多实例互不干扰;
  只有 instance 0 挂 exporter 落盘。
- `FULL_TASK=1` 启用完整任务(抓取后搬运到另一张桌子再放下);默认只做"走近 + 抓起 + 保持 2s"。
- 每条 episode 由 privileged 谓词判定成功/失败,失败的 episode 直接丢弃,不进数据集。
- 录制视频:起 stack 前设 `SIM_VIDEO_DIR=<dir>`。

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

## 常用诊断脚本

`src/teleop_free/diag_*.py`(都用 `.venv_sim` 的 python,在 stack 起来后运行):
`diag_ready_walk.py` 端臂行走、`diag_reach_probe.py` reach map 实测校验、
`diag_stream.py` 流式手臂通道、`diag_nav.py` 导航等。
