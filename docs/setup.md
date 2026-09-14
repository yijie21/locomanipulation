# 环境安装

> 从零到 stack 能跑。装完回到 [README 快速开始](../README.md#快速开始)。

## 环境安装

前置:Linux + NVIDIA GPU(本仓库在 RTX 5090 / driver CUDA 13 上验证)、
[uv](https://docs.astral.sh/uv/),以及:

```bash
sudo apt-get install -y git git-lfs cmake build-essential curl
git lfs install          # 只需一次;不做的话下面 clone 出来的是空壳,见第 1 步
```

**`git-lfs` 不是可选的。**上游 `GR00T-WholeBodyControl` 用 LFS 存 1275 个文件 ——
不只是模型,还有机器人网格(`*.STL` `*.obj` `*.dae`)和共享库(`*.so` `*.a`)。
没装 git-lfs 时 `git clone` **返回 0、不打印任何警告**,只是把这些文件换成约 130
字节的 pointer 文本。症状要到编译链接或建场景时才炸,跟病因隔十万八千里。

第 0–4 步只要上面这些 —— venv 是 uv 建的、deploy 是 cmake 编的、场景是
`.venv_sim` 建的。**[miniconda](https://docs.conda.io) 只有第 6 步才用到**,
那一步是可选的。

先看要装哪些。**第 0–4 步是跑仿真采数据的最小集;要训练或评测 VLA 还得加第 5 步**:

| # | 组件 | 必需? | 耗时 |
|---|---|---|---|
| 0 | 路径变量 | 必需 | 1 min |
| 1 | clone 上游仓库 | 必需(`GR00T-WholeBodyControl`) | 5 min |
| 2 | 两个 venv + SONIC 权重 + **补丁** | 必需 | 20 min |
| 3 | 编译 C++ deploy | 必需 | 主要看网速(3.9 GB 下载);千兆下约 3 min,慢网可到 30 min+ |
| 4 | 装场景 | 必需 | 5 s |
| 5 | GR00T 微调 venv | **要训练或评测 VLA 就必需** | 按上游 README |
| 6 | conda 训练环境 / EmbodiedGen | 可选,只有 SONIC 自身 RL 训练或重新生成场景才要 | 1 h+ |

> 这张表的编号和下面的小节编号一一对应。想训练模型的人**不能**跳过第 5 步。

### 0. 路径约定

**仓库可以放在任何路径,不需要改任何代码。**仓库内的路径由 `src/locomani/paths.py`
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
| `LOCOMANI_CACHE` | `$LOCOMANI_WORKSPACE/.cache` | pip / torch 缓存,**只有第 5–6 步的安装脚本读它**;第 0–4 步一处都不读 |
| `LOCOMANI_ROOT` | 自动推导 | 仓库根;只有在极特殊情况下才需要手动指定 |

> **Hugging Face 缓存不归 `LOCOMANI_CACHE` 管,而且第 0–4 步根本没人设 `HF_HOME`。**
> 全仓设它的只有两处,值还不一样:EmbodiedGen 那套脚本设成
> `$LOCOMANI_WORKSPACE/.hf_home`,`scripts/train_bottle.sh` 设成
> `$LOCOMANI_WORKSPACE/huggingface_cache`。所以第 2 步下载 SONIC 权重时,它会落在
> HF 的默认位置(`~/.cache/huggingface`,即**根分区**)。根分区小的机器要自己先
> `export HF_HOME=$LOCOMANI_WORKSPACE/huggingface_cache`,和下一节那三个
> `TMPDIR`/`UV_*` 一起。

**验证** —— 这一步永远先做,后面每一步的路径都从它来:

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

这里**刻意不写任何机器上的真实路径**:`<repo>` 是你 clone 的位置,`<workspace>` 是你
`LOCOMANI_WORKSPACE` 设的地方,两者都不需要与别人一致。曾经这段贴的是作者机器的绝对路径,
于是每换一台机器就要改一次文档(2026-09-11 就发生过一次,而那次改完与另一台机器又不符)。
**要核的是不变量,不是字符串**:

* `REPO` 是你 clone 的目录,`derived from` 在它下面 —— 这两条说明仓库根是**推导**出来的,没有被某个环境变量覆盖成别处;
* `WORKSPACE` 等于你 export 的 `LOCOMANI_WORKSPACE`(没设就是 `/workspace`);
* `SONIC` / `SCENE_META` / `OUTPUTS` / `LOGS_DIR` 全部以 `REPO` 开头 —— 仓库内的东西不可配置;
* `session_root()` 在 `WORKSPACE` 下而不是 `REPO` 下 —— stack 日志与 exporter 原始 session 属于仓库外;
* `SONIC_POLICY_DIR` 末段是**当前生效的 checkpoint**,默认 `policy/sonic_v1_1`(2026-09-11 起);
  被 `SONIC_POLICY` 覆盖时这一行会跟着变,所以它也是"我现在跑的是哪个模型"的快速答案。

shell 一侧是 `. scripts/paths.sh; echo "$LOCOMANI_ROOT $VENV_SIM_PY"`。

#### 两条贯穿全文的约定

**一、所有代码块都从仓库根出发,而且用 bash。**凡是要进 `third_party/` 的都写成
`( cd ... && ... )` 子 shell,出来还在原地;整块粘进同一个 shell 就行。

**二、`LOCOMANI_WORKSPACE` 要在整个安装过程里一直 export 着。**换终端、重开
session 都要重新设。

这两条都不是洁癖,它们防的是**同一种静默故障**:东西装到了别处,而一切看起来正常。

- 漏一个 `cd ../..`:后面 `. scripts/paths.sh` 找不到文件而短路,`DEPS_DIR` 保持
  未设置,而 bash 里 `cd ""` **返回 0 且原地不动** —— 几个 GB 的依赖倒进上游 clone
  的根目录,那里是 gitignored 的,`git status` 也不提醒。
- 丢了 `LOCOMANI_WORKSPACE`:它会**静默回落到默认的 `/workspace`**,于是第 3 步那
  3.9 GB 装进 `/workspace/deps` 而不是你指定的地方 —— 而且编译照样成功、不报任何错。

两种成因,同一个后果:要到二十行后 cmake 报一个和病因毫无关系的错,或者更晚。

### 1. Clone 上游仓库到 third_party/

```bash
# third_party/ 是gitignored 的(几十 GB,各自是独立的 git 仓库),全新 clone
# 里没有这个目录 —— 必须自己建
mkdir -p third_party

# 第 0-4 步只需要这一个。5 min 的耗时估计也是指它。
# 子 shell:出来还在仓库根,下一步的相对路径才对得上
(cd third_party && git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git)

# 下面三个现在【不要】clone —— 它们分属后面的步骤,加起来几十 GB,
# 走到那一步再回来拿:
#   Isaac-GR00T   -> 第 5 步(要训练/评测 VLA 才需要)
#   EmbodiedGen   -> 第 6 步(只有重新生成场景才需要)
#   IsaacLab      -> 第 6 步(只有 SONIC 自身 RL 训练才需要)
#   SIMPLE        -> 第 7 步(只有 Isaac 离线重渲才需要;venv 约 20 GB)
```

**验证** —— 查的是**大小**不是存在性,因为缺 git-lfs 时文件照样在,只是变成了
130 字节的 pointer:

```bash
stat -c%s third_party/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/meshes/head_link.STL
# 期望 932784(约 911 KB)。如果是 ~130,就是 git-lfs 没装:
#   sudo apt-get install -y git-lfs && git lfs install
#   然后 cd 进去 git lfs pull,不用重新 clone
```

**第 2 步那条 `ls ... model_decoder.onnx` 查不出这个问题** —— `policy/` 下的权重
(`release/` 与 `sonic_v1_1/`)是 `download_from_hf.py` 下载的,不走 LFS,所以它们
存在并不说明 LFS 正常。

### 2. 两个 venv + 权重 + 补丁(必需)

> **根分区小的机器先做这一步**,否则建 venv 会在解包 torch 时炸
> `No space left on device`,而报错看起来和依赖完全无关。`uv` / `pip` 会先把 wheel
> 解到 `$TMPDIR` 再安装,几个 GB 一下就把小根分区撑满:
>
> ```bash
> export TMPDIR=$LOCOMANI_WORKSPACE/tmp
> export UV_CACHE_DIR=$LOCOMANI_WORKSPACE/.cache/uv
> export UV_PYTHON_INSTALL_DIR=$LOCOMANI_WORKSPACE/.cache/uv/python
> mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
> ```
>
> 判断要不要做:`df -h /` 与 `df -h $LOCOMANI_WORKSPACE`。两者是同一个盘、且余量
> 有几十 GB 就不用管。

用上游自带的安装脚本,在 `third_party/GR00T-WholeBodyControl/` 下创建两个 venv:

```bash
(cd third_party/GR00T-WholeBodyControl \
 && bash install_scripts/install_mujoco_sim.sh \
 && bash install_scripts/install_data_collection.sh \
 && .venv_data_collection/bin/python download_from_hf.py \
 && .venv_data_collection/bin/python download_from_hf.py --sonic-v1-1)
# install_mujoco_sim.sh      -> .venv_sim(跑 MuJoCo sim 与所有 src/locomani 脚本)
# install_data_collection.sh -> .venv_data_collection(跑 data exporter)
# download_from_hf.py        -> SONIC policy 权重落到 gear_sonic_deploy/policy/release/
#   它要 huggingface_hub,而这个包只随 install_data_collection.sh 那条依赖链进来:
#   .venv_sim 里没有,裸 python 里更没有,两者都会直接退出
# --sonic-v1-1               -> gear_sonic_deploy/policy/sonic_v1_1/,**这是本仓库
#   2026-09-11 起的默认 checkpoint**(`SONIC_POLICY`,见 scripts/paths.sh)。两份都要:
#   release 仍是对照臂,而 2026-09-11 之前采的所有数据集/模型都是 release 的 token 空间
```

> **上游不钉 MuJoCo 版本,`uv` 会解析到最新的。**本仓库的结果是在 **3.11.0** 上测
> 的;3.12.0 改了 `spec.to_xml()` 的属性发射顺序,曾经因此让 ego 相机静默停在原厂
> 朝向、成功率全局归零(已修:`build_sim_scene.py` 改用 spec API 并在编译后自检)。
> 如果采集时每局都是 `FAILED [view] object outside ego view`,先查
> `.venv_sim/bin/python -c "import mujoco; print(mujoco.__version__)"`,再重建场景。

`.venv_sim` 装完之后还缺一个包:`src/locomani/primitives/navigator.py` 用 `skfmm`(路径规划的
快速行进法)算距离场,但它是这个仓库自己的依赖,上游安装脚本不装,**必须**手动补上,
否则跑 `grasp_gym.py` 等脚本会在 import 阶段就报 `ModuleNotFoundError: No module named
'skfmm'`:

```bash
uv pip install --python third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scikit-fmm
```

装完之后**必须**再打补丁,否则**录下来的 action 是错的**(整段下蹲会丢失,action 维度
是 267 而不是 78):

```bash
(cd third_party/GR00T-WholeBodyControl && git apply ../../patches/vla_action_space.patch)
```

原因见 [`patches/vla_action_space.md`](../patches/vla_action_space.md)。

**验证**:

```bash
(cd third_party/GR00T-WholeBodyControl \
 && .venv_sim/bin/python -c "import mujoco; print(mujoco.__version__)" \
 && .venv_sim/bin/python -c "import skfmm; print(skfmm.__version__)" \
 && ls gear_sonic_deploy/policy/sonic_v1_1/model_decoder.onnx \
 && ls gear_sonic_deploy/policy/release/model_decoder.onnx \
 && git apply --check --reverse ../../patches/vla_action_space.patch && echo "补丁已打")
```

> **默认 checkpoint 是 `policy/sonic_v1_1`**(2026-09-11 用户裁定)。`sonic_v1_1/`
> 缺失时 `start_stack.sh` 会带着"这棵 deploy 树里有哪些 policy"直接失败退出,不会
> 拖到超时;首次启动会重建 TensorRT engine(~10 min,之后缓存在 `policy/sonic_v1_1/*.trt`)。
> encoder 观测维度也随之变化(v1.1 是 1751,release 是 1762),离线工具从 checkpoint
> 自己的 `observation_config.yaml` 读布局,不再写死。



### 3. 编译 C++ deploy(必需)

三样依赖。第一样正常 `apt install`;**后两样不能**(见每段的理由),它们落在
`$DEPS_DIR`(默认 `$LOCOMANI_WORKSPACE/deps`)。

```bash
. scripts/paths.sh && mkdir -p "$DEPS_DIR"

# 1) 小依赖:几 MB,正常装。久没 update 过的机器先 apt-get update,否则可能 404。
#    (这一行未在干净机器上验证过 —— 开发机上六个包本来就装着)
sudo apt-get update
sudo apt-get install -y libeigen3-dev libzmq3-dev cppzmq-dev libgtest-dev \
                        nlohmann-json3-dev libmsgpack-cxx-dev

# 2) TensorRT 10.13.3.9 —— 8 个 deb 共 3.9 GB。这些包【只】由 NVIDIA 自己的
#    CUDA apt 源提供,Ubuntu 默认源里没有。没配过就会得到
#    "Unable to locate package libnvinfer10",那不是版本问题,是缺源。
#    干净的 Ubuntu 24.04 上先加源(改 ubuntu2404 为你的版本):
#      curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
#      sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
#    配好源【只是为了 apt-get download】。千万不要顺手 `apt install cuda` 或
#    `cuda-toolkit` —— 元包会拉 cuda-drivers 覆盖宿主注入的 NVIDIA 驱动
#    (和本节末尾那条警告是同一个雷)。
#    也不要 apt install 这些 TensorRT 包本身:装进 /usr 会撑爆小根分区。
#    下载 + 解包到 $DEPS_DIR,不需要 root。
cd "$DEPS_DIR"
V=10.13.3.9-1+cuda13.0
apt-get download \
    libnvinfer10=$V libnvinfer-dev=$V \
    libnvinfer-plugin10=$V libnvinfer-plugin-dev=$V \
    libnvonnxparsers10=$V libnvonnxparsers-dev=$V \
    libnvinfer-headers-dev=$V libnvinfer-headers-plugin-dev=$V
mkdir -p TensorRT-10.13.3.9
for d in *.deb; do dpkg -x "$d" TensorRT-10.13.3.9; done
# cmake/FindTensorRT.cmake 找的是官方 TAR 的 include/ 和 lib/ 布局,
# 而 deb 解出来是 usr/... —— 这两个软链是必须的,少了就报找不到 TensorRT
cd TensorRT-10.13.3.9
ln -sfn usr/include/x86_64-linux-gnu include
ln -sfn usr/lib/x86_64-linux-gnu lib
cd "$LOCOMANI_ROOT"

# 3) ONNX Runtime 1.16.3 —— tarball 的布局本来就对(include/ lib/ 直接在根下),
#    但解出来的目录名带版本号,而 env_deploy.sh:15 认死了 $DEPS_DIR/onnxruntime,
#    所以必须改名(或者自己 export onnxruntime_ROOT=<解出来的那个目录>)
cd "$DEPS_DIR"
curl -L -o ort.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz
tar -xzf ort.tgz && mv onnxruntime-linux-x64-gpu-1.16.3 onnxruntime && rm ort.tgz
cd "$LOCOMANI_ROOT"

# 编译
source scripts/env_deploy.sh      # 把上面两个前缀喂给 cmake,并配好 LD_LIBRARY_PATH
cmake -S "$SONIC_DEPLOY_ROOT" -B "$SONIC_DEPLOY_ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SONIC_DEPLOY_ROOT/build" -j"$(nproc)"
```

**两个依赖的适配手法不一样,别拿一个去套另一个:**

| | 来源 | 目录名 | 怎么让 cmake 找到 |
|---|---|---|---|
| TensorRT | `dpkg -x` 解出来的 deb 树,东西在 `usr/...` 底下 | **保留**版本号(`TensorRT-10.13.3.9`) | 加两条软链 `include`/`lib`,把 `/usr` 树伪装成官方 TAR 布局 |
| ONNX Runtime | 官方 TAR,`include/` `lib/` 本来就在根下 | **必须改成** `onnxruntime` | 布局无需动,只是目录名要对上 |

两个 root 都是 `env_deploy.sh` 里的 `${VAR:-默认}`,所以不想改名也可以直接
`export onnxruntime_ROOT=...` / `export TensorRT_ROOT=...` 覆盖。

**这些输出无害,不要去追**(不是穷举,但常见的都在这儿):

- `missing components: nvparsers` —— TensorRT 10 里已删除,`find_package` 也没把它标为 required
- cmake 报 TensorRT 版本号为空 —— `NvInferVersion.h` 用 `TRT_*_ENTERPRISE` 宏而不是字面量,实际就是 10.13.3.9
- 单个 gtest `FK.TestFKAndGlobalVelocities` 失败 —— fixture 目录 `reference/bones_072925_test/` 没随仓库发布,不是编译问题。**按上面的命令你不会看到它**:`run_tests` 只编不跑,`.justfile` 的 `test` target 是个空壳(只 echo 一行)。只有手动执行 `build/run_tests` 才会撞上

- configure 阶段的 `CMake Warning (dev) ... DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is not set`(来自 `FetchContent_Declare`),以及编译阶段 `picosha2.h:210:23: warning: writing 1 byte into a region of size 0` —— 都是上游代码的告警,与本仓库无关

`just` 不需要装:`.justfile` 的 `build` 就是上面这两条 cmake。

双臂全身 POSE 交接还需要 [上肢控制权补丁](../patches/sonic_pose_ownership.md)：
应用 `patches/sonic_pose_ownership.patch` 后重编。采集入口会检查实际选择的二进制，
旧版本会明确报错。自定义 `SONIC_DEPLOY_BIN` 也必须包含该补丁。

> **其余部署补丁按用途安装。**
> `vla_action_space.patch` 是真补丁,第 2 步已经打过了(不打 action 就是错的)。
> `sonic_dds_domain.patch` + `.md`:补丁本体加说明。**照本文装出来的 deploy 没打它,因此无法参与任何多实例场景** ——
> 而 [collection.md](collection.md) 的并行采集就是多实例。这不是"高级用法才需要",
> 是本文装出来的东西不满足那份文档的前提。
>
> 只跑单条 stack 就不用管它(单条完全正常)。要跑多条就得先打上再重编:
>
> ```bash
> (cd third_party/GR00T-WholeBodyControl && git apply ../../patches/sonic_dds_domain.patch)
> cmake --build "$SONIC_DEPLOY_ROOT/build" -j"$(nproc)"
> ```
>
> 之所以必须打,
> 因为库存二进制把 DDS domain 硬编码成 0、根本不读 `SONIC_DDS_DOMAIN`,而
> `INSTANCE` 只管得住 Python 侧和端口。失败模式很隐蔽:第二个 stack 的 sim 在
> domain 1、它自己的 deploy 却在 domain 0,于是**那个 deploy 跑去接管第一个 stack
> 的 sim**,自己的 sim 一条指令都收不到。三方状态表见
> [reference.md 端口](reference/silent_failures.md#端口)。

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
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python src/locomani/sim/make_dex1_model.py
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
`start_stack.sh` 上没用,只能重跑本脚本。

**验证** —— 这一步自己会报告结果,不用另外敲命令。脚本跑完回显它实际建出来的
preset / target / target_pool / 桌上物体:

```
  preset      bottle_only
  target      water_bottle
  target_pool ['water_bottle']
  on the table ['water_bottle']
```

对不上就是 `SCENE_PRESET` 没生效(最常见的是给在了 `start_stack.sh` 上)。

细节见[下一节](collection.md#2-环境与场景)。

### 5. GR00T 微调 venv(要训练/评测 VLA 就必需)

```bash
(cd third_party && git clone https://github.com/NVIDIA/Isaac-GR00T.git)
# 然后按上游 README 在 third_party/Isaac-GR00T/ 下建 .venv(uv sync / pip install -e .)
bash scripts/link_gr00t_src.sh     # 回到仓库根跑;把 `import gr00t` 指到 src/gr00t
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
bash scripts/setup_sonic.sh    # gear_sonic + Isaac-GR00T(IsaacLab 它自己会 clone)
bash scripts/setup_sonic2.sh   # IsaacLab 固定到 v2.3.2 并装 gear_sonic[training]
```

EmbodiedGen 要自己拿:
`(cd third_party && git clone https://github.com/HorizonRobotics/EmbodiedGen.git)`。

> EmbodiedGen 的 4 个安装脚本(`install_embodiedgen.sh`、
> `install_affordance_noflash.sh`、`eg_env.sh`、`setup_sonic*.sh`)同样改成了这套
> 路径变量,但**没有在别的机器上实测过** —— 它们只在重新生成场景时才用得到,
> 而重新生成场景需要几十 GB 下载。用现成场景的话完全不碰它们。

### 7. Isaac 渲染 runtime:SIMPLE(可选,只有离线重渲 / Isaac 像素才需要)

渲染线(`docs/rendering.md`)从 SIMPLE 只拿**一样东西:一个装了 `isaacsim` 4.5 的 venv**。
它不 import `simple` 包,不读 SIMPLE 的 `data/`。所以 SIMPLE 和第 1 步的仓库一样 clone 到
`third_party/`,固定到 `scripts/paths.sh` 里的 `SIMPLE_PIN`,**不下载它的资产**。

```bash
# 1) clone 并固定 commit(源码本身不到 1 GB)
(cd third_party && git clone https://github.com/physical-superintelligence-lab/SIMPLE.git)
(cd third_party/SIMPLE && git checkout 6d10628794d9c7de4596b4f2afb2c054a637c2bc)

# 2) submodule。SIMPLE 的 .gitmodules 里有四个写的是 git@github.com: 的 SSH 地址,
#    没配 GitHub SSH key 会在这里卡住 —— 先把 SSH 改写成 https(只影响这一个 clone;
#    六个 path 依赖都在 GitHub 上公开,已用 ls-remote 核过)。
#    只拉 uv sync 的 pyproject path 依赖要的五个;curobo / gsnet / AMO 是 SIMPLE 自己的
#    规划与抓取用的,渲染线不需要,不要 --init 全部(几 GB)。
(cd third_party/SIMPLE \
  && git config url."https://github.com/".insteadOf git@github.com: \
  && git submodule update --init --recursive \
       third_party/openpi-client third_party/gear_sonic third_party/unitree_sdk2_python \
       third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 third_party/decoupled_wbc)

# 3) 系统依赖 + uv sync。和第 2 步同一套 TMPDIR / UV_CACHE_DIR 约定 —— venv 约 20 GB,
#    isaacsim 的 wheel 从 pypi.nvidia.com 拉,根分区小的机器不设这两个变量必爆。
sudo apt-get install -y curl cmake python3-dev ffmpeg gstreamer1.0-libav git-lfs
(cd third_party/SIMPLE && git lfs install && git lfs pull)
(cd third_party/SIMPLE \
  && UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match)
#    上游 README 记录的坑:sync 若报 "Unable to uninstall lerobot==0.3.3",
#    rm -f third_party/SIMPLE/.venv/lib/python3.10/site-packages/lerobot-*.egg-info 后重跑。
#    上游的 scripts/install_curobo.sh【不要】跑,渲染线用不到。

# 4) 打补丁 + 验证。脚本检查 pin、应用 patches/simple/0001(必需),解析 isaacsim 但不 import。
bash scripts/setup_render_runtime.sh
#    第一次真正 import 会启动 Kit、拉 extension cache,数分钟;脚本已带 OMNI_KIT_ACCEPT_EULA=YES:
bash scripts/setup_render_runtime.sh --deep

# 5) 把机器人 MJCF 和场景转成 USD(-> assets/isaac/,gitignored 的生成产物)
bash scripts/build_isaac_assets.sh
```

> 桌子和台面的源网格/贴图由第 3 步 `install_scene.sh` 从仓库场景包安装；
> 房间背景随 Git 放在 `assets/scene/empty_room_world.ply.gz`（约 41 MB）。
> 不需要复制本地 EmbodiedGen 输出。旧版本可能缓存了缺少桌子/房间的场景；
> 更新代码后重新运行 `build_isaac_assets.sh`，新缓存键会触发重建。
> 必需源资产缺失或转换失败会明确报错，不再把灰底场景当作构建成功。
> 详见 [资产构建](rendering.md#资产构建)。

**验证**:

```console
$ . scripts/paths.sh; "$RENDER_PY" -c "import importlib.metadata as m; print(m.version('isaacsim'))"
4.5.0.0
$ ls assets/isaac/robot/g1_dex1.usda
```

> **不需要下载的东西**:SIMPLE 的 `data/_download.sh` 会从 HF `USC-PSI-Lab/SIMPLE` 拉
> robots / assets / objaverse / hssd scenes / vMaterials,十几 GB。主线渲染走我们自己的
> `scene_locomani.xml` 经 Isaac 的 MJCF importer,一个字节都不读它们。只有跑 SIMPLE 自己
> 的 replay CLI 或 MuJoCo/Isaac 对照 harness(`setup_render_runtime.sh --with-replay` /
> `--with-alignment`,见 `patches/simple.md`)才需要,届时先 `bash data/_download.sh --list`
> 看大小再按 `--only` 挑组。

> 机器上已有一份 SIMPLE 检出在别处?不用搬:
> `ln -s <那个目录> third_party/SIMPLE`,或者 `export SIMPLE_ROOT=<那个目录>`。
> `RENDER_PY` 默认跟着 `SIMPLE_ROOT/.venv` 走,指向别的装了 isaacsim 的 venv 也行
> (`docs/rendering.md` 解释了为什么这是两个变量)。

> 第 2、3 步未在干净机器上验证过 —— 作者机器上的 venv 是按上游 README 装的;第 4、5 步
> 的脚本是可重复执行的,在作者机器上跑过。
