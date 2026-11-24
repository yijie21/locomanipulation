# 安装

从零装好运行[场景 RSI](rsi.md)、自身模型在公开基准上的评测（[分支 `selfwm`](selfwm_bench.md)）和已归档的三条 VLA 线（bottle、crate、warehouse）所需的环境。每一步依次写命令、做什么、怎么确认装好了。
所有命令都在仓库根目录的 bash 里执行；踩过的坑和背后的原因集中在最后一节「[安装排错与说明](#安装排错与说明)」。

## 要装哪几步

| 步 | 内容 | 谁需要 | 耗时 |
|---|---|---|---|
| 准备 | 系统软件包、克隆本仓库 | 所有人 | 几分钟 |
| 0 | 路径变量 | 所有人 | 1 分钟 |
| 1 | 克隆上游仓库 GR00T-WholeBodyControl | 所有人 | 5 分钟 |
| 2 | 两个 Python 环境、SONIC 权重、补丁 | 所有人（场景 RSI 只用其中的 `.venv_sim`） | 20 分钟 |
| 3 | 编译 SONIC 的 C++ 控制器（deploy） | VLA 线 | 主要看网速，要下载 3.9 GB |
| 4 | 场景源资产 | 所有人，每个克隆都要（`outputs/` 不在 git 里） | 5 秒 |
| 5 | GR00T 训练环境 | 要训练或评测 VLA（三条主线都要） | 按上游 README |
| 6 | conda 训练环境、EmbodiedGen | 只有训练 SONIC 本身或重新生成场景才要 | 1 小时以上 |
| 7 | Isaac 渲染环境 | warehouse 选 Isaac 出图时，以及 Isaac 重渲；selfwm 的 SIMPLE 基准 | 约 20 GB 下载 |
| 8 | Unreal 渲染环境 | warehouse 默认用它出图（实验文件写 `images.source: unreal`）；场景 RSI 只在渲染报告视频时用 | 引擎约 26 GB 下载，编译约 9 分钟 |
| 9 | 场景脚本的 Python 环境 `/venv/main` | 场景 RSI | 2 分钟 |
| 10 | ADT 数据 | 场景 RSI | 约 5 GB 下载 |
| 11 | ScaleBFM 权重 | 场景 RSI、selfwm | 1 分钟 |
| 12 | Claude CLI | 场景 RSI 的 L1 以上（内环的角色调用它） | 2 分钟 |
| 13 | SIMPLE 基准：cuRobo、SIMPLE 的补丁和资产、GPU 版 ONNX Runtime | selfwm | 编译 cuRobo 约 10 分钟 |

场景 RSI 要做准备、第 0–2 步、第 4 步和第 9–12 步，第 3、5、6、7 步不用。selfwm 的 SIMPLE 基准要做准备、第 0–2 步、第 7 步、第 11 步和第 13 步。VLA 线：做完 0–4 步就能采集并转换数据；三条主线都要训练和评测，所以还要做第 5 步；warehouse 再加第 8 步（Unreal，默认）或第 7 步（Isaac）。

## 准备

需要 Linux、NVIDIA GPU 和支持 CUDA 13 的驱动、CUDA 13 运行时开发文件（第 3 步第 0 条会检查）、[uv](https://docs.astral.sh/uv/)。

```bash
sudo apt-get install -y git git-lfs cmake build-essential curl
git lfs install          # 只需一次
git clone https://github.com/yijie21/locomanipulation.git
cd locomanipulation
git lfs pull  # 下载新场景的网格、贴图、USD 和 Blender 源文件
```

做什么：装好系统工具，克隆本仓库并下载 LFS 管理的大文件。git-lfs 不能省，原因见[排错](#git-lfs-不能省)。

## 0. 路径

仓库可以放在任何位置，仓库内的路径都是自动推导的。仓库外的东西（依赖、日志、缓存）默认放在 `/workspace` 下；你的机器不是这个布局时，设一个变量：

```bash
export LOCOMANI_WORKSPACE=/home/workspace     # 协作者最常见的情况
```

这个变量在整个安装过程中都要保持，换终端后要重新设（原因见[排错](#两条贯穿全文的约定)）。

根分区小的机器（`df -h /` 和 `df -h $LOCOMANI_WORKSPACE` 不是同一块盘，或者根分区只剩几十 GB）接着设下面几个变量，让安装时的临时文件和缓存落到大盘上，否则建环境时会报 `No space left on device`：

```bash
export TMPDIR=$LOCOMANI_WORKSPACE/tmp
export UV_CACHE_DIR=$LOCOMANI_WORKSPACE/.cache/uv
export UV_PYTHON_INSTALL_DIR=$LOCOMANI_WORKSPACE/.cache/uv/python
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
```

验证：

```console
$ python3 src/locomani/paths.py
  REPO              <repo>
  WORKSPACE         <workspace>
  SONIC             <repo>/third_party/GR00T-WholeBodyControl
  SONIC_POLICY_DIR  <repo>/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/sonic_v1_1
  SCENE_META        <repo>/outputs/scene/sim/scene_meta.json
  OUTPUTS           <repo>/outputs
  LOGS_DIR          <repo>/outputs/logs
  session_root()    <workspace>/tmp/p03
  derived from      <repo>/src/locomani/paths.py
```

`<repo>` 是你克隆的位置，`<workspace>` 是 `LOCOMANI_WORKSPACE`。要核对的是：`REPO` 是你克隆的目录；`WORKSPACE` 等于你设的值（没设就是 `/workspace`）；`SONIC`、`SCENE_META`、`OUTPUTS`、`LOGS_DIR` 都以 `REPO` 开头；`session_root()` 在 `WORKSPACE` 下；`SONIC_POLICY_DIR` 的末段是当前用的 SONIC checkpoint，默认 `policy/sonic_v1_1`。

装好以后，每个新终端先执行 `. scripts/paths.sh`。它设置所有路径变量，并把 `src/` 加进 `PYTHONPATH`。

这台机器特有、又不该进 Git 的设置，写在 `$LOCOMANI_WORKSPACE/locomani.local.sh` 里，`scripts/paths.sh` 会先读它。例如 deploy 编译在别处时写 `export SONIC_DEPLOY_ROOT=<那个目录>`，带 `pxr` 的 Python 既不是 `python3` 也不是 Blender 自带的那个时写 `export USD_PY=<那个 python>`。

其他可以单独改的位置（默认都在 `$LOCOMANI_WORKSPACE` 下）：

| 变量 | 默认 | 放什么 |
|---|---|---|
| `DEPS_DIR` | `$LOCOMANI_WORKSPACE/deps` | TensorRT 和 ONNX Runtime |
| `CONDA_BASE` | `$LOCOMANI_WORKSPACE/miniconda3` | conda（只有第 6 步用） |
| `LOCOMANI_TMP` | `$LOCOMANI_WORKSPACE/tmp` | 仿真栈的日志和 exporter 的原始录制 |
| `LOCOMANI_CACHE` | `$LOCOMANI_WORKSPACE/.cache` | pip 和 torch 的缓存，只有第 5、6 步读 |

## 1. 上游仓库

```bash
mkdir -p third_party
(cd third_party && git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git \
  && git -C GR00T-WholeBodyControl checkout --detach 1983e88888217f6c69283cf3a9d1af01e87f07af)
```

做什么：把 SONIC 所在的上游仓库克隆到 `third_party/`，并固定在本仓库验证过的提交上；第 3 步的补丁只对这个提交有效。Isaac-GR00T、EmbodiedGen、IsaacLab、SIMPLE 分别在第 5、6、7 步再克隆。

验证（检查的是文件大小，缺 git-lfs 时文件也存在，只是只有 130 字节）：

```bash
stat -c%s third_party/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/meshes/head_link.STL
```

应输出 `932784`。输出 130 左右说明 git-lfs 没装好，见[排错](#git-lfs-不能省)。以前克隆过、跟着上游 main 走的机器，见[回到钉住的提交](#以前克隆过的机器回到钉住的提交)。

## 2. Python 环境、SONIC 权重和补丁

```bash
(cd third_party/GR00T-WholeBodyControl \
 && bash install_scripts/install_mujoco_sim.sh \
 && bash install_scripts/install_data_collection.sh \
 && .venv_data_collection/bin/python download_from_hf.py \
 && .venv_data_collection/bin/python download_from_hf.py --sonic-v1-1 --no-planner)
```

做什么：用上游的安装脚本建两个环境，`.venv_sim` 跑仿真和本仓库的所有脚本，`.venv_data_collection` 跑数据录制（exporter）；再下载两份 SONIC 权重，`sonic_v1_1` 是默认用的，`release` 留作对照。

```bash
uv pip install --python third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scikit-fmm mujoco==3.11.0
# 只有要跑测试 / make check / 提交钩子才需要(它们都用 .venv_sim):上游装的环境里一个都没有
uv pip install --python third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  pytest pillow pyarrow hydra-core omegaconf
```

做什么：补上本仓库自己的依赖 `scikit-fmm`，并把 MuJoCo 固定在 3.11.0（本仓库所有结果都在这个版本上测）。第二条只有跑测试时才需要。

```bash
(
  cd third_party/GR00T-WholeBodyControl &&
  { git apply --check --reverse --include='gear_sonic/data/*' ../../patches/vla_action_space.patch 2>/dev/null ||
    git apply --include='gear_sonic/data/*' ../../patches/vla_action_space.patch; } &&
  { git apply --check --reverse ../../patches/exporter_priv_seq.patch 2>/dev/null ||
    git apply ../../patches/exporter_priv_seq.patch; } &&
  echo "补丁已打"
)
```

做什么：给上游打两个补丁，改正录制的动作空间（不打的话录下来的动作是错的），并让 exporter 记下每行对应的仿真帧号。可以重复执行，已经打过的会跳过。

验证：

```bash
(cd third_party/GR00T-WholeBodyControl \
 && .venv_sim/bin/python -c "import mujoco; print(mujoco.__version__)" \
 && .venv_sim/bin/python -c "import skfmm; print(skfmm.__version__)" \
 && ls gear_sonic_deploy/policy/sonic_v1_1/model_decoder.onnx \
 && ls gear_sonic_deploy/policy/release/model_decoder.onnx \
 && git apply --check --reverse --include='gear_sonic/data/*' ../../patches/vla_action_space.patch \
 && git apply --check --reverse ../../patches/exporter_priv_seq.patch && echo "补丁已打")
```

第一行应输出 `3.11.0`，最后打印 `补丁已打`。

## 3. 编译 deploy

deploy 是 SONIC 的 C++ 控制器。下面这一整块先检查 CUDA 13 运行时，再装三样依赖（小依赖用 apt；TensorRT 和 ONNX Runtime 下载后解包到 `$DEPS_DIR`，不装进系统），然后打三个 deploy 补丁并编译：

```bash
. scripts/paths.sh && mkdir -p "$DEPS_DIR"

# 0) CUDA 13 运行时：编译要 cuda_runtime.h，运行要 libcudart.so.13。先查：
ls "${CUDA_HOME:-/usr/local/cuda}/include/cuda_runtime.h" "${CUDA_HOME:-/usr/local/cuda}/lib64/libcudart.so.13"
#    缺了就先按 2) 加 NVIDIA 的 CUDA apt 源，再只装运行时开发包（不含驱动）：
#      sudo apt-get install -y cuda-cudart-dev-13-0
#    装在 /usr/local/cuda 以外就 export CUDA_HOME=<那个目录>

# 1) 小依赖
sudo apt-get update
sudo apt-get install -y libeigen3-dev libzmq3-dev cppzmq-dev libgtest-dev \
                        nlohmann-json3-dev libmsgpack-cxx-dev

# 2) TensorRT 10.13.3.9：只下载、解包到 $DEPS_DIR，不要 apt install。没配 NVIDIA 的 CUDA apt 源时先配（改 ubuntu2404 为你的版本）：
#      curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
#      sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
cd "$DEPS_DIR"
V=10.13.3.9-1+cuda13.0
apt-get download \
    libnvinfer10=$V libnvinfer-dev=$V \
    libnvinfer-plugin10=$V libnvinfer-plugin-dev=$V \
    libnvonnxparsers10=$V libnvonnxparsers-dev=$V \
    libnvinfer-headers-dev=$V libnvinfer-headers-plugin-dev=$V
mkdir -p TensorRT-10.13.3.9
for d in *.deb; do dpkg -x "$d" TensorRT-10.13.3.9; done
cd TensorRT-10.13.3.9
ln -sfn usr/include/x86_64-linux-gnu include
ln -sfn usr/lib/x86_64-linux-gnu lib
cd "$LOCOMANI_ROOT"

# 3) ONNX Runtime 1.16.3：解包后目录名必须改成 onnxruntime
cd "$DEPS_DIR"
curl -L -o ort.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz
tar -xzf ort.tgz && mv onnxruntime-linux-x64-gpu-1.16.3 onnxruntime && rm ort.tgz
cd "$LOCOMANI_ROOT"

# deploy 补丁：三个一起打，打在第 1 步钉住的提交上；已经打过就跳过
(cd third_party/GR00T-WholeBodyControl || exit 1
  [ "$(git rev-parse HEAD)" = 1983e88888217f6c69283cf3a9d1af01e87f07af ] ||
    { echo "不在钉住的 commit 上(现在是 $(git log -1 --oneline)):先做第 1 步末尾那段"; exit 1; }
  done=0
  for p in sonic_deploy_local_base sonic_pose_ownership sonic_hand_ownership; do
    git apply --check --reverse "../../patches/$p.patch" 2>/dev/null && done=$((done + 1))
  done
  [ "$done" -eq 3 ] && { echo "deploy 补丁已打"; exit 0; }
  if [ -n "$(git status --porcelain -- gear_sonic_deploy/src gear_sonic_deploy/deploy.sh)" ]; then
    echo "deploy 源码里已有改动(旧文档让打的 sonic_dds_domain / pose / hand 补丁,或上次只打了一部分):"
    git status --short -- gear_sonic_deploy/src gear_sonic_deploy/deploy.sh
    echo "没有自己的改动要留,就清掉后重跑这一段:"
    echo "  (cd third_party/GR00T-WholeBodyControl && git checkout -- gear_sonic_deploy && git clean -fd gear_sonic_deploy/src)"
    exit 1
  fi
  for p in sonic_deploy_local_base sonic_pose_ownership sonic_hand_ownership; do
    git apply "../../patches/$p.patch" || { echo "打不上 $p"; exit 1; }
  done && echo "deploy 补丁已打")

# 编译
source scripts/env_deploy.sh      # 把上面两个前缀喂给 cmake,并配好 LD_LIBRARY_PATH
cmake -S "$SONIC_DEPLOY_ROOT" -B "$SONIC_DEPLOY_ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SONIC_DEPLOY_ROOT/build" -j"$(nproc)" --clean-first
```

三个 deploy 补丁：`sonic_deploy_local_base` 是本仓库一直在用、没有推到上游的 6 个提交（DDS domain、planner 超时、自动启动、lockstep）；`sonic_pose_ownership` 负责双臂全身姿态的控制权交接；`sonic_hand_ownership` 让单臂和 warehouse 从全身姿态回到 planner 时夹爪不复位。采集和评测在起栈前都会检查 deploy 有没有这些补丁。

验证：

```bash
. scripts/paths.sh && ls -la "$SONIC_DEPLOY_ROOT/target/release/g1_deploy_onnx_ref"
```

可执行文件在 `target/release/` 下（不在 `build/` 下）。要确认和原作者用的是同一个控制器，见[排错](#确认-deploy-和原作者的是同一份)。

## 4. 场景源资产

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python src/locomani/sim/make_dex1_model.py
SCENE_PRESET=bottle_only bash scripts/install_scene.sh
```

做什么：第一条生成带平行夹爪（dex1）的机器人模型；第二条从仓库自带的场景包装出场景源资产，并建好 `outputs/scene/sim`。每个新克隆都要做一次（`outputs/` 不在 git 里）；场景 RSI 的 `build_scene.sh walk` 要它建出的 `outputs/scene/layout/layout.json`。三条主线各自的场景在各自 runbook 的第 1 步用 `bash scripts/build_scene.sh <场景名>` 建。

验证：`install_scene.sh` 结束时会打印它实际建出的内容，应为：

```
  preset      bottle_only
  target      water_bottle
  target_pool ['water_bottle']
  on the table ['water_bottle']
  finger pads HARDENED (priority 1, condim 4)
```

对不上见[排错](#场景的预设和指垫)。

## 5. GR00T 训练环境

```bash
(cd third_party && git clone https://github.com/NVIDIA/Isaac-GR00T.git \
  && git -C Isaac-GR00T checkout --detach 376ba890cff8c9de64d71d982772a9c36185fdd7)
# 然后按上游 README 在 third_party/Isaac-GR00T/ 下建 .venv(uv sync / pip install -e .)
bash scripts/link_gr00t_src.sh     # 回到仓库根跑;把 `import gr00t` 指到 src/gr00t
VIRTUAL_ENV=third_party/Isaac-GR00T/.venv uv pip install onnxruntime
```

做什么：克隆 Isaac-GR00T 并固定提交，按上游 README 建好训练环境；`link_gr00t_src.sh` 让训练环境里的 `gr00t` 指向本仓库的 `src/gr00t/`（训练代码在这里）；最后一条装评测指标要用的 CPU 版 onnxruntime。

验证：`link_gr00t_src.sh` 结束时会自己检查；之后 `train.sh` 每次启动也会检查，指错了会拒绝训练。

## 6. conda 训练环境（可选）

只有训练 SONIC 本身或者重新生成场景才需要，和上面几步无关：

```bash
bash scripts/setup_sonic.sh    # gear_sonic + Isaac-GR00T(IsaacLab 它自己会 clone)
bash scripts/setup_sonic2.sh   # IsaacLab 固定到 v2.3.2 并装 gear_sonic[training]
```

EmbodiedGen 要自己克隆：`(cd third_party && git clone https://github.com/HorizonRobotics/EmbodiedGen.git)`。这些脚本没有在别的机器上实测过。

## 7. Isaac 渲染环境（warehouse 选 Isaac 出图时）

本仓库只用 Isaac Sim 出图，物理始终在 MuJoCo。Isaac Sim 4.5 借用 SIMPLE 仓库的 Python 环境，不用 SIMPLE 的代码和数据。

```bash
# 1) 克隆并固定提交
(cd third_party && git clone https://github.com/physical-superintelligence-lab/SIMPLE.git)
(cd third_party/SIMPLE && git checkout 6d10628794d9c7de4596b4f2afb2c054a637c2bc)

# 2) 只拉需要的五个子模块，先把 SSH 地址改成 https
(cd third_party/SIMPLE \
  && git config url."https://github.com/".insteadOf git@github.com: \
  && git submodule update --init --recursive \
       third_party/openpi-client third_party/gear_sonic third_party/unitree_sdk2_python \
       third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 third_party/decoupled_wbc)

# 3) 系统依赖和 Python 环境（约 20 GB；根分区小的机器先设第 0 步的 TMPDIR 和 UV_CACHE_DIR）
sudo apt-get install -y curl cmake python3-dev ffmpeg gstreamer1.0-libav git-lfs
(cd third_party/SIMPLE && git lfs install && git lfs pull)
(cd third_party/SIMPLE \
  && UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match)

# 4) 打补丁并验证；--deep 会真正启动一次 Isaac，要几分钟
bash scripts/setup_render_runtime.sh
bash scripts/setup_render_runtime.sh --deep

# 5) 把机器人模型和场景转成 Isaac 用的 USD，写到 assets/isaac/
bash scripts/build_isaac_assets.sh
```

做什么：装好带 Isaac Sim 的 Python 环境（`$RENDER_PY`），打上本仓库的补丁，并生成机器人的 USD。纸箱场景的 Isaac 版本在 [warehouse runbook](warehouse.md) 第 1 步建，需要一个带 `pxr` 的 Python（`python3` 或 Blender 自带的即可，否则 `pip install usd-core` 或用 `USD_PY` 指定）。

验证：

```console
$ . scripts/paths.sh; "$RENDER_PY" -c "import importlib.metadata as m; print(m.version('isaacsim'))"
4.5.0.0
$ ls assets/isaac/robot/g1_dex1.usda
```

## 8. Unreal 渲染环境（warehouse 默认）

实验文件写 `images.source: unreal`（[rendering.md](rendering.md#图像来源imagessource)）时需要，warehouse 的默认运行就是这样；物理照样在 MuJoCo。装在 `$UNREAL_ROOT`（默认 `third_party/unreal/`）。

```bash
. scripts/paths.sh
mkdir -p "$UE_ENGINE" && unzip -q <下载目录>/Linux_Unreal_Engine_5.7.4.zip -d "$UE_ENGINE"
bash scripts/setup_unreal.sh
```

做什么：第一条把 Unreal 引擎解到 `$UE_ENGINE`。Linux 版只给登录的 Epic 账号下载，要先在 https://www.unrealengine.com/linux 手动下载 `Linux_Unreal_Engine_5.7.4.zip`（约 26 GB，解开 69 GB）。第二条先按 `scripts/unreal/host/` 建宿主工程，出图默认值写在它的配置里：阴影不走 Nanite、每次取图截 8 遍、不向 Epic 上传崩溃报告。然后克隆 URLab 插件（`efbc5e2`），打上 `patches/urlab_fresh_capture.patch` 并编译。最后建和编辑器通信的 Python 环境 `$BRIDGE_PY`（Python 3.12、mujoco 3.10.0）。它约 9 分钟，只用最后 4 个核；中途失败就重跑，做完的步骤会跳过。

验证：

```console
$ bash scripts/start_unreal.sh --gpu 1
unreal editor ready on GPU 1 (log ...)
$ "$BRIDGE_PY" -m locomani.sim.unreal.editor 'print(unreal.SystemLibrary.get_engine_version())'
5.7.4-...
```

编辑器起来以后，建一次房间关卡（每台机器一次）。先装好 Blender 5.x，做法见 [Blender 场景](blender_scenes.md#1-环境需要什么不需要什么)：

```bash
blender --background assets/blender/logistics_room.blend --python scripts/unreal/warehouse_bake.py -- outputs/unreal_room_bake
"$BRIDGE_PY" -m locomani.sim.unreal.build_level --room outputs/unreal_room_bake
```

做什么：第一条用 PATH 上的 Blender 5.x（`blender`，装法见上面链接的页面；不在 PATH 上就写它的完整路径）把物流房间的程序化材质烘成贴图，连同灯光导出成 `room.glb` 和 `room_scene.json`，约 7 分钟。第二条在运行中的编辑器里把它建成房间关卡 `/Game/Levels/Warehouse`。成功时命令正常退出，烘焙目录里的 `unreal_level_report.json` 写着 `"ok": true`；失败时命令报错退出。之后每个场景的关卡在 [warehouse runbook](warehouse.md) 第 1 步建，哪些场景有关卡看 `configs/scenes.yaml` 的 `unreal` 条目。

## 9. 场景脚本的 Python 环境（场景 RSI）

```bash
[ -x /venv/main/bin/python ] || uv venv /venv/main --python 3.12
uv pip install --python /venv/main/bin/python mujoco==3.11.0 trimesh==4.12.2 opencv-python-headless numpy scipy matplotlib pyyaml ruff==0.16.1 huggingface_hub
```

做什么：建场景的脚本（`adt_to_scene.py`、`scene_add_panels.py`、`scene_add_colliders.py` 等）和写回时的代码格式化用这个环境，不用 `.venv_sim`。Vast 的基础镜像自带 `/venv/main`，第一条会跳过。没有 `/venv/main` 写权限的机器可以建在 `~/venv/main`（两条命令里的路径照换）：`make` 和 `scripts/rsi/writeback.py` 先读环境变量 `LOCOMANI_VENV`，没有就依次找 `/venv/main`、`~/venv/main`。

验证：`/venv/main/bin/python -c "import mujoco, trimesh, cv2; print(mujoco.__version__, trimesh.__version__)"` 输出 `3.11.0 4.12.2`。`adt_to_scene.py` 经 `locomani.sim.make_dex1_model` 导入 mujoco，没有它第一步就停。

## 10. ADT 数据（场景 RSI）

```bash
python3 scripts/adt_download.py sequences Apartment_release_clean_seq131_M1292
python3 scripts/adt_download.py objects
```

做什么：先在 Project Aria 网站申请 Aria Digital Twin 的下载链接（每人一份，不进 git），把 `ADT_download_urls.json` 和 `DTC_objects_ADT_download_urls.json` 放进 `<workspace>/datasets/adt/links/`（`<workspace>` 是 `LOCOMANI_WORKSPACE`，默认 `/workspace`；三个 `adt_*.py` 脚本都从它读数据根目录）。第一条下载公寓那段序列的真值（约 20 MB），第二条下载序列里出现的所有物体模型（约 4.6 GB）。每个文件都按 sha1 校验，中断后重跑会接着下载。数据只能用于非商业研究，不能再分发。

验证：`ls <workspace>/datasets/adt/sequences/Apartment_release_clean_seq131_M1292` 里有 `instances.json`，`<workspace>/datasets/adt/objects/` 下有几百个物体目录。

## 11. ScaleBFM 权重（场景 RSI）

```bash
/venv/main/bin/python -c "from huggingface_hub import hf_hub_download as d; d('WeishuaiZeng/ScaleBFM', 'checkpoint/humanoid_transformer_m/model_22200.pt', local_dir='$LOCOMANI_WORKSPACE/tmp/scalebfm')"
mkdir -p $LOCOMANI_WORKSPACE/models/scalebfm
mv $LOCOMANI_WORKSPACE/tmp/scalebfm/checkpoint/humanoid_transformer_m/model_22200.pt $LOCOMANI_WORKSPACE/models/scalebfm/humanoid_transformer_m_22200.pt
```

做什么：下载场景 RSI 用的全身控制器 ScaleBFM（M 型号、第 22200 步，73 MB），放到代码默认读的位置 `<workspace>/models/scalebfm/`（`paths.SCALEBFM_CHECKPOINT`，`ScaleBfmBackendConfig.checkpoint` 为空串时用它；`<workspace>` 是 `LOCOMANI_WORKSPACE`，默认 `/workspace`）。它的模式表和元数据在仓库的 `assets/scalebfm/` 里，不用下载。

验证：`sha256sum $LOCOMANI_WORKSPACE/models/scalebfm/humanoid_transformer_m_22200.pt` 以 `88d5a79946c03ed2` 开头。

## 12. Claude CLI（场景 RSI 的 L1 以上）

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude
```

做什么：第一条安装 Claude Code 的命令行工具；第二条第一次运行时按提示登录，然后输入 `/exit` 退出。内环的 Reviewer、Manager、Engineer 和外环的任务发现都用 `claude -p` 调用它，模型由 `RSI_MODEL` 指定，默认 `claude-opus-5-5`。L0 冒烟和核对不需要它。

验证：`claude -p "say ok" --model claude-opus-5-5` 打印 `ok`。

## 13. SIMPLE 基准（selfwm）

selfwm 在 SIMPLE 的四个抓取任务上评测。物理在本仓库的 MuJoCo 复刻场景里跑，用 `.venv_sim`；第 7 步装的 SIMPLE 提供机器人、物体和评测局，它的环境还要编进 cuRobo，用来规划手臂路线。第 7 步说不用做的 cuRobo 和 `data/_download.sh`，这里都要做。第 11 步用的是 `/venv/main` 的 Python，没装第 9 步时换成 `.venv_data_collection` 的 Python，它也带 `huggingface_hub`。

```bash
S=$PWD/third_party/SIMPLE
# 1) 系统库：Isaac 不开窗口也要这四个 X 库，缺了会卡住、日志里不报错；unzip 解评测局时要用
sudo apt-get install -y unzip libglu1-mesa libsm6 libice6 libxt6
# 2) cuRobo：SIMPLE 的子模块，编进 SIMPLE 的环境。nvcc 的 CUDA 版本要和环境里的 torch 一致（cu128），算力按自己的卡写
(cd $S && git submodule update --init --depth 1 third_party/curobo \
  && UV_PROJECT_ENVIRONMENT=$S/.venv TORCH_CUDA_ARCH_LIST=8.9 bash scripts/install_curobo.sh)
# 3) 本仓库给 SIMPLE 的补丁，加回放和对齐两项
SIMPLE_ROOT=$S RENDER_PY=$S/.venv/bin/python bash scripts/setup_render_runtime.sh --with-replay --with-alignment
# 4) SIMPLE 的机器人和物体资产，以及 AMO 任务要的策略文件（Git LFS）
(cd $S && PYTHON=.venv/bin/python bash data/_download.sh --only robots,assets --jobs 4 \
  && git lfs pull --include "src/simple/robots/policy/*")
# 5) SONIC 的网络放在 GPU 上时，仿真环境要 GPU 版的 ONNX Runtime
VIRTUAL_ENV=third_party/GR00T-WholeBodyControl/.venv_sim uv pip install onnxruntime-gpu==1.23.2
```

做什么：让 SIMPLE 的环境能规划手臂路线，让复刻场景能读到 SIMPLE 的机器人和物体，让 SONIC 能用 GPU 跑网络。第 1 到 4 条是 2026-11-19 到 11-20 在 4090 上装 SIMPLE 时做过的，那次的脚本 `install.sh` 带国内镜像，放在装好的目录里，路径见 [4090 的交接记录](handoff/selfmodel-4090-20261123.md)。编 cuRobo 时 4090 还另外指定了 CUDA 12.8 的 nvcc 和 torch 自带的 cuBLAS 等头文件，见那份脚本的第 6 步。第 5 条的版本是 4090 上现在装的那个。没有 GPU 版 ONNX Runtime 也能跑 SONIC，设 `REACH_SONIC_NETS=allcpu`，但慢得多。

验证：

```console
$ third_party/SIMPLE/.venv/bin/python -c "import curobo.curobolib.kinematics_fused_cu; print('ok')"
ok
$ third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -c "import onnxruntime as o; print(o.get_available_providers())"
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
$ PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -m locomani.bench show --bench simple | grep signature
 "signature": "simple/perception-rgbd/pose-odometry/root-odometry/scalebfm",
```

评测局和缓存的看一眼不在这一步，见[基准 runbook](selfwm_bench.md)第 2 节。

## 安装排错与说明

### git-lfs 不能省

上游 `GR00T-WholeBodyControl` 用 LFS 存了 1275 个文件，包括机器人网格（`*.STL`、`*.obj`、`*.dae`）和共享库（`*.so`、`*.a`）。没装 git-lfs 时 `git clone` 照样成功、不报任何警告，只是把这些文件换成约 130 字节的指针文本，要到编译链接或建场景时才出错。补救：`sudo apt-get install -y git-lfs && git lfs install`，再进上游目录执行 `git lfs pull`，不用重新克隆。第 2 步检查权重的 `ls` 查不出这个问题，因为权重是另外下载的，不走 LFS。

### 两条贯穿全文的约定

1. 所有代码块都从仓库根目录出发。要进 `third_party/` 的命令都写成 `( cd ... && ... )`，执行完还在原地。少了这一层时，`. scripts/paths.sh` 找不到文件，`DEPS_DIR` 为空，而 `cd ""` 不报错、原地不动，几个 GB 的依赖就装进了上游仓库的根目录。
2. `LOCOMANI_WORKSPACE` 在整个安装过程中都要保持。丢了它会退回默认的 `/workspace`，第 3 步的依赖装到别处，编译照样成功，要到很久以后才出错。

Hugging Face 的缓存不归 `LOCOMANI_CACHE` 管。第 2 步下载 SONIC 权重时落在默认的 `~/.cache/huggingface`（根分区）；根分区小的机器先 `export HF_HOME=$LOCOMANI_WORKSPACE/huggingface_cache`。

### 以前克隆过的机器：回到钉住的提交

2026-09-25 之前照本文装的机器，上游仓库跟着 main 走。先回到钉住的提交，再重跑第 2、3 步的打补丁部分（都可以重复执行；环境和权重不用重装）：

```bash
(cd third_party/GR00T-WholeBodyControl && git fetch origin \
  && git checkout --detach 1983e88888217f6c69283cf3a9d1af01e87f07af)
```

git 会把工作树里的改动一起带过去；带不过去时报 `Your local changes ... would be overwritten` 并列出挡路的文件。没有自己的改动就 `git checkout -- <这些文件>` 后重跑。

### 下载权重

- `download_from_hf.py` 需要 `huggingface_hub`，只有 `.venv_data_collection` 里有，所以用这个环境的 Python 跑。
- 第二次下载加 `--no-planner`：两次下载的是同一个 planner 文件，脚本连只读权限一起复制，第二次覆盖会报 `PermissionError`。重跑这一段前先执行 `(cd third_party/GR00T-WholeBodyControl && chmod -R u+w gear_sonic_deploy/policy gear_sonic_deploy/planner)`。
- 默认 checkpoint 是 `policy/sonic_v1_1`（2026-09-11 起）。`release` 仍要下载，因为 2026-09-11 之前的数据集和模型都用它的 token 空间。
- deploy 第一次启动会构建 TensorRT engine，大约 10 分钟，之后缓存在 `policy/sonic_v1_1/*.trt`。

### MuJoCo 版本

上游不固定 MuJoCo 版本，uv 会装最新的（2026-09-25 是 3.14.0）。本仓库的结果都在 3.11.0 上测，物理和渲染在版本之间都会变。3.12.0 改过 `spec.to_xml()` 的属性顺序，曾让第一视角相机停在出厂朝向，成功率全部归零（已修）。采集时每局都是 `FAILED [view] object outside ego view`，先查 MuJoCo 版本，再重建场景。

### 补丁为什么要打

- `vla_action_space.patch` 把动作空间从 267 维改成 78 维，不打的话整段下蹲动作会丢失（[说明](../patches/vla_action_space.md)）。它只取 `features_sonic_vla.py` 的部分；exporter 用更新的 [`exporter_priv_seq.patch`](../patches/exporter_priv_seq.md)，它包含前者的 exporter 部分，还加了每行的仿真帧号。整份 `vla_action_space.patch` 直接打会和它冲突。
- 已经打过补丁的树上再执行 `git apply`（不带 `--check --reverse`）会报 `patch does not apply`，这不是补丁坏了。
- deploy 补丁的来由见 [`sonic_deploy_local_base`](../patches/sonic_deploy_local_base.md)、[上肢控制权](../patches/sonic_pose_ownership.md)、[夹爪交接](../patches/sonic_hand_ownership.md)。DDS domain 已经包含在 `sonic_deploy_local_base` 里，不要再单独打 `sonic_dds_domain.patch`（会冲突）。缺它时第二个栈的 deploy 会接管第一个栈的仿真（[隐性失败 · 端口](reference/silent_failures.md#端口)）。

### 编译 deploy

- 不要执行 `gear_sonic_deploy/scripts/install_deps.sh`，也不要 `apt install cuda` 或 `cuda-toolkit`：它们会装 `cuda-drivers`，覆盖宿主机注入的 NVIDIA 驱动。`scripts/env_deploy.sh` 就是替代它的。cmake 提示 `Please run: ./scripts/install_deps.sh` 时也不要照做。
- TensorRT 只由 NVIDIA 的 CUDA apt 源提供；没配源会报 `Unable to locate package libnvinfer10`，那是缺源，不是版本问题。也不要 apt install TensorRT 本身，装进 `/usr` 会撑满小的根分区。
- TensorRT 和 ONNX Runtime 的适配方法不同：TensorRT 的 deb 解出来在 `usr/...` 下，保留带版本号的目录名，加 `include`、`lib` 两个软链接，模仿官方压缩包的布局；ONNX Runtime 的压缩包布局本来就对，只是目录名必须改成 `onnxruntime`。也可以不改名，直接 `export onnxruntime_ROOT=...` 或 `export TensorRT_ROOT=...`。
- 这些输出无害：`missing components: nvparsers`（TensorRT 10 已删除该组件）；TensorRT 版本号为空；configure 阶段的 `DOWNLOAD_EXTRACT_TIMESTAMP` 警告；编译阶段 `picosha2.h` 的警告；手动执行 `build/run_tests` 时 `FK.TestFKAndGlobalVelocities` 失败（测试数据没随仓库发布）。
- 不需要装 `just`：`.justfile` 的 `build` 就是上面那两条 cmake。

### 确认 deploy 和原作者的是同一份

二进制的哈希在不同机器上本来就不同（编进了绝对路径），所以比较补丁后源码的指纹，再查二进制里的四个功能标记：

```bash
(cd "$SONIC_DEPLOY_ROOT/src" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum)
# 期望 c1360628ef7162aaf00905884d0933ee568993c37ab94a092404da17b726e0f8
for m in POSE_UPPER_OWNERSHIP_V2 PLANNER_HAND_OWNERSHIP_V1 SONIC_AUTO_START SONIC_DDS_DOMAIN; do
  grep -aq "$m" "$SONIC_DEPLOY_ROOT/target/release/g1_deploy_onnx_ref" && echo "有 $m" || echo "缺 $m"
done
```

指纹对不上，说明补丁没打全，或者上游不在钉住的提交上（`git -C third_party/GR00T-WholeBodyControl log -1`）。2026-09-25 核对过：这套补丁在 `1983e888` 上重建出的源码，和原作者 bottle、warehouse 采集用的二进制的源码逐字节相同。

### 场景的预设和指垫

- `SCENE_PRESET` 必须写在 `install_scene.sh` 这一行上。它在重建场景时读取，写在起栈的命令上不起作用。不加这个变量得到的是多物体桌面（`bottle`、`can`、`cup`、`jar`、`fork`），每局目标在几种物体之间轮换，和本仓库的实验不是同一个任务。
- 最后一行必须是 `finger pads HARDENED`。2026-09-14 起默认给夹爪指垫加硬接触，否则抬起时瓶子会从爪里慢慢滑出（同样 20 个 seed：软垫 12/20 且滑脱 5 次，硬垫 15/20 无滑脱）。`HARDEN_PADS=0 SCENE_PRESET=bottle_only bash scripts/install_scene.sh` 可以重建之前的软垫场景。

### GR00T 训练环境

训练代码是 `src/gr00t/`（从上游拷来、纳入 Git、在这里二次开发，见 [src/gr00t/UPSTREAM.md](../src/gr00t/UPSTREAM.md)）。训练环境建好时，它的 `gr00t` 指向 `third_party/` 下的上游克隆，所以必须执行 `link_gr00t_src.sh` 改过来。

### Isaac 渲染环境

- SIMPLE 的 `.gitmodules` 里有四个 SSH 地址，没配 GitHub SSH key 会卡住，所以先改成 https。只拉五个子模块，curobo、gsnet、AMO 是 SIMPLE 自己规划和抓取用的，渲染不需要。
- `uv sync` 报 `Unable to uninstall lerobot==0.3.3` 时，删掉 `third_party/SIMPLE/.venv/lib/python3.10/site-packages/lerobot-*.egg-info` 再重跑。只出图时，上游的 `scripts/install_curobo.sh` 不要跑；selfwm 的 SIMPLE 基准要它（第 13 步）。
- 只出图时不需要执行 SIMPLE 的 `data/_download.sh`（十几 GB），渲染走本仓库自己的场景文件；selfwm 只下其中的机器人和物体（第 13 步）。只有跑 SIMPLE 自己的回放工具或对照工具（`setup_render_runtime.sh --with-replay`、`--with-alignment`，见 `patches/simple.md`）才需要。
- 机器上已经有一份 SIMPLE：`ln -s <那个目录> third_party/SIMPLE`，或者 `export SIMPLE_ROOT=<那个目录>`。`RENDER_PY` 默认跟着 `SIMPLE_ROOT/.venv`，也可以指向别的装了 isaacsim 的环境（[原因](rendering.md)）。
- 桌子和台面的源网格由第 4 步装好；房间背景随 Git 提供（`assets/scene/empty_room_world.ply.gz`）。更新代码后重跑 `build_isaac_assets.sh`，新的缓存键会触发重建（[资产构建](rendering.md#资产构建)）。

### Unreal 渲染环境

- 引擎只能手动下载（要登录 Epic 账号），`setup_unreal.sh` 找不到 `$UE_ENGINE` 下的编辑器就停下。版本用 5.7.4，不要用 5.8，它的 Vulkan 渲染有回退。
- `--gpu` 是 Vulkan 的设备编号，和 `nvidia-smi` 的编号不一定相同。`start_unreal.sh` 起好后会按 `nvidia-smi` 核对，不一致就报错并关掉编辑器，这时换一个 `--gpu` 值。
- 一台机器只开一个编辑器：它的端口是固定的（URLab 用 5800–5806，命令服务用 18801），都避开了仿真栈的端口。编辑器已在另一张卡上运行时，`start_unreal.sh` 会拒绝，先 `stop_unreal.sh`。
- 编辑器和编译只用最后 4 个核（`--cpus` 或 `UE_CPUS` 可改），因为容器的 CPU 配额和仿真栈共用。看守脚本在 CPU 打满或 GPU 卡死时会杀掉编辑器，原因写在 `$UNREAL_ROOT/logs/guard.log`。CPU 打满和 GPU 卡死都按编辑器自己的占用算：别的进程把 CPU 配额或同一张卡跑满、编辑器本身空闲时，不会误杀。
- 已经装在别处的机器，在 `locomani.local.sh` 里设 `UNREAL_ROOT`（或分别设 `UE_ENGINE`、`UE_PROJECT`、`BRIDGE_PY`）。
- 关卡建在编辑器的宿主工程里，建关卡时用的场景副本记在 `$UNREAL_ASSETS/<关卡>/source.json`（默认 `$UNREAL_ROOT/levels/`，和宿主工程放在一起）。哪个检出目录建的关卡，别的检出目录都能用；场景改过之后要重建关卡，否则会被拒绝。

### 哪些步骤验证过

2026-09-25 做过一次从零复现：一个没有上下文的 agent 只照文档装完第 0–4 步，采集 bottle 7/10、crate 8/10、warehouse 3/10 成功，开局失败和摔倒都是 0。第 5、7 步没有在干净的机器上验证过。第 8 步 2026-09-29 在一个空的 `$UNREAL_ROOT` 里从 `setup_unreal.sh` 做到出图：`setup_unreal.sh` 用了 8.5 分钟，起编辑器、建房间关卡和 `warehouse_textured` 关卡、出图都通过。引擎用的是本机已解开的那份，下载和解压两步没验证。系统层面的几样东西作者机器的镜像本来就有，干净机器上要自己确认：`/usr/local/cuda` 下的 CUDA 13、MuJoCo 用的 NVIDIA EGL、转换用的 ffmpeg。
