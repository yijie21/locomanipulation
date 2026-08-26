# 环境安装

> 从零到 stack 能跑。装完回到 [README 快速开始](../README.md#快速开始)。

## 环境安装

前置:Linux + NVIDIA GPU(本仓库在 RTX 5090 / driver CUDA 13 上验证)、
[miniconda](https://docs.conda.io)、[uv](https://docs.astral.sh/uv/)。

先看要装哪些,**第 0–4 步必需,第 5 步只有微调或重新生成场景才要**:

| # | 组件 | 必需? | 耗时 |
|---|---|---|---|
| 0 | 路径变量 | 必需 | 1 min |
| 1 | clone 上游仓库 | 必需(`GR00T-WholeBodyControl`) | 5 min |
| 2 | 两个 venv + SONIC 权重 + **补丁** | 必需 | 20 min |
| 3 | 编译 C++ deploy | 必需 | 30 min(含手动放依赖) |
| 4 | 装场景 | 必需 | 5 s |
| 5 | conda 训练环境 / EmbodiedGen | 可选,只有微调或重新生成场景才要 | 1 h+ |

### 0. 路径约定

**仓库可以放在任何路径,不需要改任何代码。**仓库内的路径由 `src/teleop_free/paths.py`
和 `scripts/paths.sh` 从文件自身位置推导出来。

仓库**外**的东西(依赖、conda、日志、缓存)默认挂在 `/workspace` 下。如果你的机器
不是这个布局,export 一个变量即可:

```bash
export LOCOMANI_WORKSPACE=/home/workspace     # 协作者最常见的情况
```

需要更细的控制时,这几个可以单独覆盖(默认都在 `$LOCOMANI_WORKSPACE` 下):

| 变量 | 默认 | 是什么 |
|---|---|---|
| `LOCOMANI_WORKSPACE` | `/workspace` | 仓库外一切东西的根 |
| `DEPS_DIR` | `$LOCOMANI_WORKSPACE/deps` | TensorRT / ONNX Runtime 解包位置 |
| `CONDA_BASE` | `$LOCOMANI_WORKSPACE/miniconda3` | conda 安装位置 |
| `LOCOMANI_TMP` | `$LOCOMANI_WORKSPACE/tmp` | stack 日志与 exporter 原始 session |
| `LOCOMANI_CACHE` | `$LOCOMANI_WORKSPACE/.cache` | pip / torch / HF 缓存 |
| `LOCOMANI_ROOT` | 自动推导 | 仓库根;只有在极特殊情况下才需要手动指定 |

**验证** —— 这一步永远先做,后面每一步的路径都从它来:

```bash
$ python src/teleop_free/paths.py
  REPO         /workspace/code/locomanipulation
  WORKSPACE    /workspace
  SONIC        /workspace/code/locomanipulation/third_party/GR00T-WholeBodyControl
  SCENE_META   /workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json
  OUTPUTS      /workspace/code/locomanipulation/outputs
  LOGS_DIR     /workspace/code/locomanipulation/outputs/logs
  session_root() /workspace/tmp/p03
```

shell 一侧是 `. scripts/paths.sh; echo "$LOCOMANI_ROOT $VENV_SIM_PY"`。

### 1. Clone 上游仓库到 third_party/

```bash
cd third_party
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git   # 必需:SONIC sim + deploy + exporter
git clone https://github.com/NVIDIA/Isaac-GR00T.git              # 可选:GR00T 微调
git clone https://github.com/HorizonRobotics/EmbodiedGen.git     # 可选:只有重新生成场景才需要
git clone https://github.com/isaac-sim/IsaacLab.git              # 可选:SONIC 训练(Isaac Sim)
```

### 2. 两个 venv + 权重 + 补丁(必需)

用上游自带的安装脚本,在 `third_party/GR00T-WholeBodyControl/` 下创建两个 venv:

```bash
cd third_party/GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh        # -> .venv_sim(跑 MuJoCo sim 与所有 src/teleop_free 脚本)
bash install_scripts/install_data_collection.sh   # -> .venv_data_collection(跑 data exporter)
python download_from_hf.py                        # SONIC policy 权重 -> gear_sonic_deploy/policy/release/
```

`.venv_sim` 装完之后还缺一个包:`src/teleop_free/navigator.py` 用 `skfmm`(路径规划的
快速行进法)算距离场,但它是这个仓库自己的依赖,上游安装脚本不装,**必须**手动补上,
否则跑 `grasp_gym.py` 等脚本会在 import 阶段就报 `ModuleNotFoundError: No module named
'skfmm'`:

```bash
uv pip install --python third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scikit-fmm
```

装完之后**必须**再打补丁,否则**录下来的 action 是错的**(整段下蹲会丢失,action 维度
是 267 而不是 78):

```bash
cd third_party/GR00T-WholeBodyControl && git apply ../../patches/vla_action_space.patch
```

原因见 [`patches/vla_action_space.md`](../patches/vla_action_space.md)。

**验证**:

```bash
cd third_party/GR00T-WholeBodyControl
.venv_sim/bin/python -c "import mujoco; print(mujoco.__version__)"
.venv_sim/bin/python -c "import skfmm; print(skfmm.__version__)"
ls gear_sonic_deploy/policy/release/model_decoder.onnx   # 权重下来了
git apply --check --reverse ../../patches/vla_action_space.patch && echo "补丁已打"
```

### 3. 编译 C++ deploy(必需)

```bash
# 依赖放 $DEPS_DIR(默认 $LOCOMANI_WORKSPACE/deps)
# 1) TensorRT 10.13.x:apt-get download 对应 deb 后 dpkg -x 解包,并把
#    include/lib 软链成官方 TAR 布局;2) ONNX Runtime 1.16.3 tarball 解包;
# 3) 小依赖:apt install libeigen3-dev libzmq3-dev cppzmq-dev libgtest-dev \
#            nlohmann-json3-dev libmsgpack-cxx-dev
source scripts/env_deploy.sh
cmake -S "$SONIC_DEPLOY_ROOT" -B "$SONIC_DEPLOY_ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SONIC_DEPLOY_ROOT/build" -j"$(nproc)"
```

> **不要跑 `gear_sonic_deploy/scripts/install_deps.sh`。**它有 `apt install cuda`
> 的 fallback,而 `cuda` 元包会拉 `cuda-drivers`,覆盖掉宿主注入的 NVIDIA 驱动。
> `scripts/env_deploy.sh` 就是用来替代它的。

**验证**(二进制落在 `target/release/`,不是 `build/`):

```bash
. scripts/paths.sh && ls -la "$SONIC_DEPLOY_ROOT/target/release/g1_deploy_onnx_ref"
```

### 4. 装场景(必需)

先生成 dex1(平行夹爪)机器人模型 —— `scene_31dof.xml` / `g1_29dof_with_dex1.xml` 是
gitignored 的生成产物,新 clone 的 `third_party/GR00T-WholeBodyControl` 里没有,不生成
会导致 `MjSpec.from_file` 报 `Error opening file ... scene_31dof.xml`:

```bash
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python src/teleop_free/make_dex1_model.py
```

然后装场景。**`SCENE_PRESET` 必须给在这一行上**:

```bash
SCENE_PRESET=bottle_only bash scripts/install_scene.sh
```

`bottle_only` 是复现本仓库两轮 VLA 实验的那张桌子(一张空桌 + 一个 RoboCasa 水瓶)。
**不加这个变量得到的是默认的多物体桌面**:桌上有 `bottle / can / cup / jar / fork`,
而 `grasp_gym --targets` 默认是 `pool`,于是每局目标在四种物体之间轮换,还多一个
`fork` 当遮挡物 —— 和本仓库所有实验数字都不是同一个任务。

变量在**重建场景那一刻**被读(`scene_lib.py`),不是起 stack 时,所以补在
`start_stack.sh` 上没用,只能重跑本脚本。脚本跑完会回显它实际建出来的
preset / target / target_pool / 桌上物体,对不上就是没生效。

细节见[下一节](collection.md#场景直接用现成的推荐)。

### 5. GR00T 微调 venv(要训练/评测 VLA 就必需)

```bash
cd third_party/Isaac-GR00T
# 按上游 README 建 .venv(uv sync / pip install -e .)
cd ../..
bash scripts/link_gr00t_src.sh     # 把 `import gr00t` 指到 src/gr00t
```

最后一步不可省:**训练代码是 `src/gr00t/`**(vendored、入 git、二次开发改这里,
见 [src/gr00t/UPSTREAM.md](../src/gr00t/UPSTREAM.md)),而 venv 建好时 editable
安装指向的是 gitignore 掉的 third_party clone。`link_gr00t_src.sh` 改写 `.pth`
并自验;`train_bottle.sh` 每次启动也会验,指错会拒跑。

评测的 decoder 指标还需要 CPU 版 onnxruntime:

```bash
VIRTUAL_ENV=third_party/Isaac-GR00T/.venv uv pip install onnxruntime
```

### 6. conda 训练环境(可选,SONIC 自身的 RL 训练用,与上面无关)

```bash
bash scripts/setup_sonic.sh    # gear_sonic + Isaac-GR00T
bash scripts/setup_sonic2.sh   # IsaacLab 固定到 v2.3.2 并装 gear_sonic[training]
```

> EmbodiedGen 的 4 个安装脚本(`install_embodiedgen.sh`、
> `install_affordance_noflash.sh`、`eg_env.sh`、`setup_sonic*.sh`)同样改成了这套
> 路径变量,但**没有在别的机器上实测过** —— 它们只在重新生成场景时才用得到,
> 而重新生成场景需要几十 GB 下载。用现成场景的话完全不碰它们。
