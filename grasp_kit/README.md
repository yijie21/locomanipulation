# G1 视觉抓取包（g1_grasp_kit）

**输入：**
- G1 头部相机拍的一帧画面（彩色图 + 深度图）；
- 一句指令，例如 "pick up the red cup"。

**输出：**右手（Dex3 三指手）的抓取方案——
- 手掌该放到哪、以什么姿态去抓；
- 抓之前停在哪（预抓取位姿）；
- 手指张开、合拢的角度。

```
一帧画面：彩色图 + 深度图 + 相机内参 + 相机在机器人身上的位姿
 ① 深度补全   LingBot-Depth 把深度图里缺的、错的地方补好（可以关掉）
 ② 找目标     SAM 3 按指令里的物体名，在彩色图上把目标分割出来
 ③ 目标点云   目标像素的深度 → 机器人坐标系下的点；同时量出物体所在台面的高度
 ④ 生成抓取   GraspGen-X 在点云上生成几百个手掌位姿并打分
 ⑤ 挑选       取得分最高、而且右手够得着的那个；沿掌心法向后退 8 cm 得到预抓取位姿
输出：grasp.json（位姿）+ check.png（检查图）
```

**不包括：**
- 手臂怎么移过去（逆解、轨迹、避障）；
- 手怎么合拢。

这两件事用你们自己的控制器做，第 5.4 节有我们在仿真里的做法。

> **验证情况：**
> - 这套流程来自我们在 SIMPLE 仿真基准上的抓取实验：G1 + Dex3，头部相机画面由 Isaac Sim 渲染。
> - **只在仿真画面上验证过，从没接过真实相机。**
> - `examples/` 里的 6 帧就是仿真画面，`run_examples.sh` 会和仿真里的结果逐项比对。
>
> 真机上最可能出问题的地方：
> - 相机外参（第 5.3 节，一定要先检查）；
> - 真实深度的噪声；
> - 物体名称的说法。

## 1. 需要什么

- **系统：**Linux x86_64。
- **GPU：**NVIDIA，显存建议 12 GB 以上。我们只在 RTX 5090 上测过，跑示例时显存增加 5–10 GB；同一张卡上还有别的进程，所以量得不精确。
- **驱动：**≥ 570。只有 560 的驱动，安装前先 `export TORCH_CUDA=cu126`（RTX 50 系必须用默认的 cu128）。
- **硬盘：**约 25 GB：两个环境 14.5 GB，三个源码仓库 3.8 GB，模型 6.5 GB。
- **工具：**`git`，以及 [uv](https://docs.astral.sh/uv/)：`curl -LsSf https://astral.sh/uv/install.sh | sh`。
- **账号：**一个 Hugging Face 账号（SAM 3 要先在网页上同意许可）。
- **部署方式：**模型跑在一台带 GPU 的电脑上，不在 G1 自带的计算板上。机器人只负责拍照、执行动作，两边通过文件或网络传数据。

## 2. 安装（我们实测 2–17 分钟，取决于网速）

```bash
cd g1_grasp_kit
bash setup/install.sh                 # 两个 Python 环境：envs/perception（SAM 3、LingBot-Depth）和 envs/grasp（GraspGen-X）
envs/perception/bin/hf auth login     # 先在 https://huggingface.co/facebook/sam3 页面同意许可，再登录
bash setup/download_weights.sh        # 模型权重约 6.5 GB，放到 weights/
```

**为什么是两个环境：**GraspGen-X 锁定了较旧的 numpy、timm、huggingface-hub，和 SAM 3 冲突。两个环境的 PyTorch 版本也不同：感知环境用 2.11.0，抓取环境用 2.7.1。`run.sh` 会自动用对应的环境，不用手动切换。

**网络和磁盘：**
- 国内网络：
  - Hugging Face 用镜像：`export HF_ENDPOINT=https://hf-mirror.com`；
  - PyPI 用镜像：`export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple`（PyTorch 仍从 download.pytorch.org 下载）。
- GitHub 访问不了：把三个仓库（见 `THIRD_PARTY.md`）手动放到 `third_party/sam3`、`third_party/lingbot-depth`、`third_party/GraspGenX`，再运行 `install.sh`，它会检出测试过的版本。
- `/tmp` 空间小于 15 GB：先 `export TMPDIR=<大盘上的目录> UV_CACHE_DIR=<大盘上的目录>`，否则安装 PyTorch 时会报 "No space left on device"。
- 想把环境或模型放到别的盘：安装前设 `ENVS`、`WEIGHTS`（见 `setup/env.sh`）。

## 3. 先跑示例，确认装好了

```bash
bash run_examples.sh
```

**大约 1 分钟，最后一行应是 `all ok`。**它把 6 帧示例各跑两遍：
- 一遍直接用示例里的深度图；
- 一遍从示例的双目图像重新算深度。

然后和仿真里的结果比对：

| 列 | 意思 | 要求 |
|---|---|---|
| IoU | 目标分割和仿真里的重合度 | 1.000 |
| centre cm / table cm | 目标中心、台面高度和仿真里差多少 | 0.00 |
| depth cm | 重新算的深度和示例深度差多少 | 0.00 |
| d cm / d deg | 选中的抓取和仿真里选中的差多少 | 只做参考：每次采样不同，几厘米、几度以内都正常 |

`TabletopGraspMP_L1_e7` 是故意放进去的失败例子：指令没说物体名，SAM 3 找到的"最大物体"不是桌上那个。所以这一行显示 known failure 是对的。

## 4. 怎么跑一次抓取

```bash
bash run.sh out/ look.npz                           # 指令写在 look 文件里
bash run.sh out/ look.npz -- --object "red cup"     # 或者直接给物体名（推荐）
bash run.sh out/ look.npz -- --depth raw            # 不做深度补全，直接用相机的深度
```

| 输出文件 | 内容 |
|---|---|
| `out/look.grasp.json` | 选中的抓取，第 5.4 节逐项解释 |
| `out/look.check.png` | 检查图。左：画面，目标涂绿，红线是抓取时的手掌，蓝线是预抓取。中、右：目标点云的俯视图和侧视图，虚线是台面 |
| `out/look.target.npz` | 目标掩码、点云、台面高度 |
| `out/look.grasps.npz` | GraspGen-X 给的全部候选和分数 |

**速度：**加载模型时，感知环境约 8 s，抓取环境约 6 s；之后每帧感知约 0.2–0.6 s、生成抓取约 0.4 s。要连续调用时，可以在 Python 里让 `g1grasp.target.Sam3`、`g1grasp.depth.LingBot`、`g1grasp.grasp.GraspGenX` 这三个对象常驻，不用每次重新加载。用法照 `perceive.py`、`plan_grasp.py` 里的写法。

## 5. 接到真机

### 5.1 坐标系

| 坐标系 | 定义 |
|---|---|
| base（所有输入输出都在这个坐标系） | 原点在骨盆（pelvis），x 指向机器人正前方（水平），z 竖直向上，单位米 |
| 相机 | OpenCV 光学坐标系：x 右，y 下，z 朝前 |

base 用"水平的骨盆系"，是因为台面高度、"手指朝前"这些判断都要用到重力方向。如果你们的手臂控制器用别的坐标系，自己乘一次变换。

### 5.2 一帧画面（look 文件）

每帧存成一个 `.npz`，字段见 `g1grasp/look.py`：

| 字段 | 内容 |
|---|---|
| `rgb` | H×W×3 uint8 彩色图 |
| `depth` | H×W float32，单位米，**对齐到彩色图**，未知处填 0 |
| `K` | 3×3 彩色相机内参 |
| `T_base_cam` | 4×4，相机在 base 系下的位姿 |
| `instruction` 或 `object` | 指令，或物体名 |

三种生成方式：
- `tools/capture_realsense.py`：直接从头部 D435(i) 拍一帧。先装 pyrealsense2：`uv pip install -p envs/perception/bin/python pyrealsense2`。
- `tools/make_look.py`：已经有彩色图和对齐好的深度图（例如 ROS 录的）时用。
- 自己写：调用 `g1grasp.look.save(...)`。

### 5.3 相机外参（最关键的一步）

`g1grasp/frames.py` 的 `head_camera(waist, imu_wxyz, offset)` 按 Unitree 的 URDF 算头部相机在 base 系下的位姿，要用到三样东西：
- 骨盆 IMU 的俯仰和横滚；
- 腰部三个关节的读数（锁腰的机器人就是 0 0 0）；
- URDF 里 D435 的安装位置：在躯干上，向下倾斜 47.6°。

URDF 给的是相机外壳的位置，不是彩色传感器的光心，两者差 1–2 cm，再加上装配误差，**用之前一定要检查**：

1. 机器人站好，让相机能同时看到地面和一张桌子，拍一帧。
2. 运行：
   ```bash
   envs/grasp/bin/python tools/check_camera.py look.npz --ply cloud.ply
   ```
   它会列出画面里找到的水平面，以及每个面的高度和倾斜度。
3. 对照两个标准：
   - **地面：**高度应约等于"骨盆离地高度"的负值（站立时约 −0.75 m，以你们的状态估计为准），倾斜小于 1°；
   - **桌面：**桌面高度减地面高度，应等于卷尺量的桌高，倾斜也小于 1°。
4. 不对就修正：
   - 用 `--offset`（彩色传感器在 d435_link 中的位置）调整；
   - 或者直接用你们自己手眼标定的结果作为 `T_base_cam`。

误差有多大影响：相机俯仰差 1°，0.6 m 外的点高度就差约 1 cm。手指只有 4 cm 长，几厘米的误差就会抓空或撞到物体。

### 5.4 输出怎么用

`grasp.json` 里的字段：

| 字段 | 内容 |
|---|---|
| `grasp.palm` | `right_hand_palm_link` 的目标位姿（4×4，base 系） |
| `grasp.wrist` | 同一个抓取下 `right_wrist_yaw_link` 的位姿（多数逆解以手腕为末端，用这个） |
| `grasp.pregrasp_palm` / `pregrasp_wrist` | 预抓取位姿：从抓取位姿沿掌心法向后退 8 cm |
| `grasp.score` | GraspGen-X 给的分数（0–1） |
| `alternatives` | 后面几个可选抓取。第一个手臂够不着时，依次试这些 |
| `fingers_open` / `fingers_close` | 张开、合拢时的手指角度，按 Dex3 的关节名给。`raw` 是 GraspGen-X 手模型给的原值，有几个超出 URDF 限位；`clipped` 是截到限位后的值 |
| `table_top` / `centre` | 物体所在台面的高度；目标点云的中心 |

Dex3 SDK 里电机的顺序不一定和这里一样，请按关节名一一对应。

**我们在仿真里的执行顺序：**
1. 手指张开到 `fingers_open.clipped`。
2. **先把手抬到预抓取的高度，再移到预抓取位姿。**我们在仿真里发现，手从桌沿下方直接往前伸，会卡在桌沿上。
3. 从预抓取位姿沿直线前进 8 cm，到抓取位姿。
4. 合手：朝 `fingers_close` 的方向慢慢合，手指停住（碰到物体）就停，不要硬把手指驱动到 `fingers_close`。
5. 抬起。

## 6. 可调参数（`g1grasp/config.py`）

| 参数 | 默认 | 含义，什么时候改 |
|---|---|---|
| `sam_floor` | 0.3 | SAM 3 分数低于它，就认为没看到目标。目标找不到时可以适当调低 |
| `own` / `reach` | 0.2 / 1.0 m | 指令不说物体名时，只在骨盆前方这个距离范围里找"最大的物体" |
| `erode` / `radius` | 1 px / 0.3 m | 去掉掩码边缘 1 个像素（那里的深度混着前景和背景）；去掉离目标中心超过 0.3 m 的离群点 |
| `ahead` | 0.7 | 手指方向和机器人正前方夹角的余弦下限，0.7 约等于 45° 以内 |
| `roll` | −30°…110° | 掌心朝向：0° 朝左（手背朝上），90° 朝下 |
| `clear` | 0.06 m | 手掌至少比物体底部高这么多，中指在手掌下方 4 cm |
| `pre` | 0.08 m | 预抓取后退的距离 |
| `num_grasps` | 500 | GraspGen-X 采样的数量 |

`ahead`、`roll`、`clear` 是按 G1 右臂在仿真里够得着的范围定的。你们的机器人站位离桌子更远或更近时，可能需要调整。

## 7. 已知限制

- **只在仿真里验证过。**真实相机、真实光照、真实深度噪声都没试过。
- **只支持右手 Dex3。**GraspGen-X 也有 Inspire 手的模型；换手要改 `g1grasp/grasp.py` 里的手名、手掌坐标变换和关节名，我们没验证过。
- **指令不说物体名时容易选错。**例如 "grasp the object on the table"，会选 SAM 3 认为是"物体"、离骨盆 0.2–1.0 m、面积最大的那个，示例 `TabletopGraspMP_L1_e7` 就选错了。真机上建议总是用 `--object` 给出物体名。
- **反光或玻璃台面的深度不准。**仿真里桌沿处的深度能差 8–19 cm。
- **只检查手的朝向和高度。**不检查手臂够不够得着、会不会撞到别的东西。选出的抓取够不着时，试 `alternatives`，或者让机器人换个站位。
- **小目标或被挡住大半的目标找不到。**有深度的目标像素少于 100 个时，就视为没看到。
- **手模型有小偏差。**GraspGen-X 的 Dex3 手模型和 Unitree URDF 里的手略有差别（手指根部差 7.3 mm），已在 `g1grasp/grasp.py` 的 `T_GRASP_PALM` 里补偿。

## 8. 目录

```
g1_grasp_kit/
  README.md  THIRD_PARTY.md
  run.sh                  整条流程：perceive.py（感知环境）→ plan_grasp.py（抓取环境）
  run_examples.sh         安装检查
  perceive.py             ①–③：深度、目标分割、点云、台面高度
  plan_grasp.py           ④–⑤：生成抓取、挑选、检查图
  g1grasp/                config.py（参数）  frames.py（坐标系、G1 相机和手掌）  look.py（一帧的文件格式）
                          depth.py  target.py  grasp.py  viz.py
  tools/                  capture_realsense.py  make_look.py  check_camera.py  check_examples.py
  setup/                  env.sh（路径）  install.sh  download_weights.sh
  examples/               looks/（6 帧仿真画面）  expected/（仿真里的结果）
```
