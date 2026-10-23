# 不读仿真真值的场景 RSI：用相机和视觉基础模型代替特权信息（设计，2026-10-23）

状态：设计稿，没有实现，没有跑过。分支 `design/vision-rsi-20261023`（worktree
`/workspace/code/locomanipulation-vision-rsi`），不改 `main` 上的文件、不碰主线的沙盒和 GPU。第 10 节的 D1–D8 待用户定。
上游：[场景 RSI 方案](scene-rsi-plan-20261016.md)、[RSI 设计](../rsi_design.md)、
[grasp_atomic 线的记录与复现](grasp-atomic-line-20261023.md)（现任：封存集 38 / 50）、
[仿真数据零样本上真机](real-robot-sim2real-plan-20261006.md)（真机三相机）、
[离线抓取缓存](grasp-cache-20261010.md)（10-10 用户定的“仿真里用物体全部信息”，本设计在这条线上改掉它）。

## 0. 目标

用户 2026-10-23：“现在所有的 RSI 过程其实都是基于仿真器的 priviledge information。但是现在我们为了弥补 sim2real
gap 的话，还是要基于机器人的本体感知才行……我们如果现在依然要在仿真器中实现 RSI 过程，但是完全不依赖
priviledge information 了，而是依赖于视觉和视觉 foundation model 来 behave，仿真是在 mujoco 里面做，渲染的画面用
unreal，这样的话，需要哪些模型或者信息才能实现。”

- **不变的**：MuJoCo 物理、Unreal 渲染（`adt_apartment_mars` 关卡）、机器人 mars（G1 + Dex1）、控制器 ScaleBFM、
  RSI 的角色（Reviewer / Manager / Engineer / Composer）和验收方式。
- **要改的**：机器人自己的代码（技能、种子、挑选器、准备动作层）在运行时能读什么。今后它们只能读传感器和由传感器
  算出来的估计，读不到 `sim.d`、场景文件里的位姿、碰撞网格和接触对。
- **要量的**：同一批起点上，用感知时的成功率比用真值时（38 / 50）低多少，差距分别来自哪一层（定位、地图、物体
  位姿、形状、抓取、接触判断）。

## 1. 现状：主线在哪里读了真值

10-23 的盘点结论：场景 RSI 主线没有一处用感知。`src/locomani/tasks/scene/` 和 `scripts/rsi/loop.py` 里没有任何
相机、深度或分割的 import。

| 环节 | 现在读的真值 | 位置 | 感知要给出的东西 |
|---|---|---|---|
| 物体位姿 | 自由物体的 `sim.d.xpos / xquat`；静态物体取场景文件的 `init_pos` | `world.py:210` `object_pose`，所有技能起步时调用 | 每个物体的 6D 位姿，带时间戳和不确定度 |
| 抓取伺服 | 每个控制周期读一次物体 `xpos`（`grip = obj_p + offset`） | `skills/grasp_program.py:132-182` | 控制频率下可用的物体位姿，带延迟 |
| 物体形状 | 碰撞网格顶点、geom 尺寸 | `graspability.py:97-260`，`pick.py:486` `object_vertices` | 物体点云或网格 |
| 物体高度、放置面 | 网格顶点的世界 z | `world.py:169-208` `underside`、`seat` | 物体底面和支撑面的高度 |
| 抓住、碰到 | 指垫和物体的接触对 | `backends/scalebfm/motor.py:250-275` | 夹爪开度和电机力，加视觉确认 |
| 自身位姿 | 骨盆 `qpos[:7]` | `pick / place / near / into / navigate` 多处 | 里程计或 SLAM |
| 跌倒 | `qpos[2] < 0.40` | `backends/scalebfm/runtime.py:249` | IMU 加腿式里程计的高度 |
| 走路地图 | 场景文件的家具表面建出的占据图 | `src/locomani/primitives/navigator.py:372` `build(meta)` | 深度建图得到的占据图 |
| 站位 | `approach_edges`（`scripts/rsi/stance_edges.py` 离线在仿真里试站出来）、矮家具位姿 | `choose.py:124-191` | 由感知地图算出的可站边、矮家具 |
| 走廊和邻居 | 其他自由物体的包围盒 | `choose.py:232`，`scripts/rsi/seeds/grasp_c11/clutter.py:117` | 感知到的邻近物体 |
| 准备动作 | 物体网格在假想场景里挪动后的位姿 | `prep.py:148-216` | 感知网格和位姿 |
| LLM 诊断 | 每局 npz 里的 `obj_<body>` 真值轨迹、`scene_meta.json`、在真值几何上跑的 `graspability / body / choose / prep` 工具 | `scripts/rsi/prompts/reviewer.md:6-16` | 见第 5 节 |

在 `src/locomani/tasks/scene/` 里直接读 `sim.d`、`world.sim` 或 `meta[...]` 的地方（`grep -c`，不含测试）：
`pick.py` 23 处、`place.py` 22、`into.py` 16、`near.py` 10、`world.py` 10、`graspability.py` 6、`prep.py` 6、
`cover.py` 5、`choose.py` 4。改接口的工作量主要在这五个技能文件上。

已经有、不读真值、但没接进 RSI 的部件：

| 部件 | 位置 | 做什么 |
|---|---|---|
| 腿式里程计 | `src/locomani/backends/scalebfm/odometry.py` `LegOdometry` | 29 个关节编码器加 IMU，估骨盆位置；只给一次起始位姿 |
| 相机观测 | `src/locomani/backends/scalebfm/camera.py` `RobotCameraObserver` | MuJoCo 渲染头相机和右腕相机，给 VLA 用 |
| 相机延迟 | `src/locomani/pipeline/camera_delay.py` | 给图像加延迟 |
| Unreal 镜像 | `src/locomani/sim/unreal/mirror.py`、`view.py` | 输入一个 MuJoCo 状态，输出几路相机图像；仓库场景 3 路相机、每次 8 帧收敛时约 125–140 ms 一次 |
| 图像伺服 | `src/locomani/tasks/bottle/image_servo.py` | bottle 线的视觉抓取误差伺服 |

## 2. 核心设计：机器人代码只读“估计的世界”

在仿真和技能之间划一条线，分四层：

| 层 | 读什么 | 产出 | 谁用 |
|---|---|---|---|
| 传感器层 | 仿真状态 | 真机同样拿得到的数据：Unreal 渲出的图像、深度、IMU、关节编码器、夹爪电机位置和力，每条带时间戳和延迟 | 感知层 |
| 感知层 | 只读传感器层 | 估计的世界（下称 `PerceivedView`）：物体位姿和不确定度、物体形状、地图、自身位姿、抓住与否 | 技能、挑选器、准备动作层 |
| 技能层 | 只读 `PerceivedView` 和本体感觉 | 给控制器的指令 | 控制器 |
| 评测与诊断 | 真值 | 成功判定、失败分类、给 LLM 的对照轨迹 | 验收门、Reviewer、Engineer |

接口做法：把今天 `SceneWorld` 里“查询世界”的那部分方法（`object_pose`、`state`、`underside`、`seat`，以及挑选器读的
地图和邻居）抽成一个协议 `WorldView`，两个实现：

- `TruthView`：把现在的调用原样包一层，行为不变，作为对照（L0）。
- `PerceivedView`：同样的方法，返回估计值，并多给几个字段：协方差、最后一次看见的时间、来源（头相机、腕相机、
  记忆）。另加一个 `look(targets)`，请求一次观测（第 7 节）。

`SceneWorld` 留在仿真一侧，负责摆物体、推进物理和评测。技能拿到的 `ctx.world` 换成一个 `WorldView`。

**怎么保证技能真的读不到真值：**
1. `PerceivedView` 不持有 `MjModel / MjData`。感知层放在单独的进程里，只收传感器消息。
2. 加一条静态检查（测试加验收门）：`src/locomani/tasks/scene/skills/`、`scripts/rsi/seeds/` 和候选补丁不许出现
   `.sim`、`.d.`、`meta["objects"][...]["init_pos"]`、`mj_` 调用；违反的候选在验收门前直接拒掉。
   `rules.md` 加一条对应的规则。
3. `motor.py` 的接口拆成两半：本体感觉（关节角、夹爪开度、执行器力，`d.actuator_force` 相当于真机的电机电流）
   给技能；接触对（`between_fingers / gripping / touching`）只给评测。

**任务怎么指定目标物体：** 任务文本可以给物体的类别名（场景文件里的物体名，例如“木勺”），不给 body id、位姿或
网格。“哪一个木勺”由感知层去认（第 4.3 节）。

## 3. 传感器：仿真里要模拟什么

实验室真机的相机配置已经在 mars 模型里（`configs/scenes.yaml` 第 154 行起的注释，
`adt_apartment_mars/scene_locomani.xml` 里的三个相机）：

| 传感器 | 真机 | 仿真现状 | 要补的 |
|---|---|---|---|
| 头部 D455 | 装在头上的云台支架，俯仰 52°，30 fps | `head_camera`，fovy 65；仿真里没有头部关节，角度固定 | Unreal 渲双目图（D455 基线 95 mm），跑双目深度；或深度缓冲加噪声模型（D3） |
| 双腕 D405 | 每个夹爪一台，30 fps | `left_wrist / right_wrist`，fovy 58 | 同上（D405 基线 18 mm，量程约 7–50 cm），最后一段接近主要靠它 |
| LiDAR | 原厂 G1 头部带 Livox MID-360；实验室这台有没有 **待确认** | 没有 | 如有：用 `mj_multiRay` 按它的扫描图案打射线，加噪声（D4） |
| IMU、29 个关节编码器 | 有 | 有，`LegOdometry` 已在用 | 加噪声和偏置 |
| Dex1 夹爪 | 电机位置和电流，没有触觉 | 有开度；执行器力可读 | 给技能一个“抓住了没有”的本体信号（第 4.7 节） |

真机 D455 / D405 的内参没有提供，仿真用的是规格书视场（[真机设计](real-robot-sim2real-plan-20261006.md) 末节）。

**渲染的代价：** Unreal 镜像每次请求约 125–140 ms（仓库场景、3 路相机、每次 8 帧让 Lumen 和 TSR 收敛）。
感知在锁步下最多 7 Hz 左右，除非把每次的帧数降到 1（stream 模式）。今天的抓取伺服每个控制周期读一次物体位姿，
感知只有 5–10 Hz 时，伺服要能吃带延迟、有时过期的估计。

**不要直接用 Unreal 的深度缓冲当 L3 的深度。** 那是一份完美深度，等于把真值从另一扇门放回来。L2 可以用它
（第 6 节），用来把“局部视角和遮挡的影响”和“模型误差的影响”分开量。

## 4. 感知模块和候选模型

“本机”一列指 `/workspace/huggingface_cache/hub` 或 conda 环境里已经有的权重（10-23 列出，属于本机其他项目，不能删）。

| 模块 | 首选 | 备选 | 输出 | 本机 |
|---|---|---|---|---|
| 4.1 自身定位 | `LegOdometry` 加视觉 SLAM（VGGT-SLAM 2.0 或 MASt3R-SLAM，头相机） | 有 LiDAR 时用 FAST-LIO2 | 骨盆位姿，地图坐标系 | VGGT-1B、VGGT-Omega、map-anything、Pi3X |
| 4.2 深度 | Fast-FoundationStereo（CVPR 2026，实时双目） | Depth Anything 3 metric（单目）；FoundationStereo | 每路相机的稠密深度 | DA3-GIANT-1.1、DA3METRIC-LARGE、DA3NESTED、MoGe-2；FFS 要下 |
| 4.3 检测和指代 | SAM 3（文字提示分割，带视频跟踪） | Grounding DINO 加 SAM 2.1；VLM 消歧：Qwen3-VL-8B（本地）或 Gemini Robotics-ER 1.5（API） | 每帧的物体掩码和跟踪 id | `facebook/sam3`、`sam2.1-hiera-large`、`grounding-dino-base`、Qwen3-VL-4B / 8B |
| 4.4 地图和场景图 | 深度和位姿融合成 TSDF 占据图；RANSAC 拟合水平面得到台面和边；SAM 3 的掩码跨视角并成 3D 物体（ConceptGraphs 的做法） | nvblox | 占据图、台面多边形和高度、可站边、物体节点 | — |
| 4.5 物体形状 | 接近过程中多视角融合的点云，最后用腕相机补近景 | SAM 3D Objects（单张图出网格，再用深度 ICP 定尺度）；Hunyuan3D-2.1、TRELLIS | 物体网格或点云、包围盒 | conda 环境 `sam3d`（`sam3d_objects`、`sam3`、`pytorch3d`），Hunyuan3D-2.1、TRELLIS-image-large |
| 4.6 位姿跟踪 | FoundationPose，用 4.5 重建的网格当模板 | 只需要位置时：掩码质心加深度 | 6D 位姿、最后看见的时间 | FoundationPose 要下 |
| 4.7 抓取候选 | 在 4.5 的网格上用现有的反向法线采样加物理验证（[离线抓取缓存](grasp-cache-20261010.md) 的方法，输入从真网格换成感知网格） | GraspGen / GraspGen-X（跨夹爪，Dex1 支不支持未核实） | 带分数的抓取，再交给挑选器按 `body.reach()` 包络过滤 | `adithyamurali/GraspGenModels` |
| 4.8 抓住、碰倒 | 夹爪开度停在大于 0、执行器力超过阈值，判为抓住；物体轨迹在没抓住时离开支撑面，判为碰倒 | 视觉确认：掩码跟手一起动、抬起后深度里物体离开台面 | `held`、`knocked` 两个估计 | — |
| 4.9 跌倒 | IMU 姿态加 `LegOdometry` 的骨盆高度 | — | `fallen` | — |

几点说明：

- **抓取候选不让基础模型自己做主。** 10-08 在 65 个物体上比过，失败几乎都在执行（指垫没贴上、抬起打滑、进场碰倒），
  不在规划；控制器也给不出倾斜和俯抓的手部姿态。所以抓取模型只出候选，挑选器（`choose.py`）继续用身体包络过滤。
  4.7 的首选沿用 10-10 的缓存方法：在仿真里只有手的模型上合爪、抬、晃，这一步验证的是机器人自己的估计，真机上也能跑，
  不算读真值。
- **最后 10 cm 的遮挡。** 手会挡住头相机看物体。做法：保留最后一次估计并标成过期，切到腕相机，再用夹爪信号确认。
  今天抓取的误差余量只有 10–14 mm，这一段最容易出事。
- **视觉网格和碰撞网格要对得上。** 感知看到的是 Unreal 渲染的视觉网格，物理用的是 MuJoCo 的碰撞网格。两者差几毫米，
  “感知对了、抓取还是失败”就会被误判成感知问题。L2 之前先逐个物体量一遍差异。

## 5. 评测、诊断和 LLM 角色

- **成功判定继续用真值。** 它是测量，不是行为，不构成 sim2real 问题。
- **Reviewer、Engineer、Composer 两份轨迹都看。** 每局记录真值轨迹和 `PerceivedView` 的估计轨迹，并排给出，
  另加几列感知误差：位姿误差、掩码 IoU、地图误差，以及失败局的几帧 Unreal 图像（叠掩码和估计位姿）。这样它们能分清
  一次失败是感知错了还是控制错了。`scripts/rsi/failure_class.py` 加几类：`perception_miss`（没看见）、
  `perception_stale`（用了过期估计）、`wrong_instance`（认错了物体）、`shape_error`（形状估错导致抓法错）。
- **风险：Engineer 用真值学到感知的系统偏差，再写一个常数去补。** 这个偏差是仿真渲染加这个模型的性质，真机上会不同。
  对策：每局随机感知噪声的参数（偏置、尺度、延迟、丢帧），让针对某一种偏差调出来的常数在验收集上不占便宜。

## 6. 分级推进和归因

一次只换一层，失败才能归到具体的层上。所有级别都跑同一批封存集起点（run C 的 50 个），按起点配对比较。

| 级别 | 感知来源 | 回答的问题 |
|---|---|---|
| L0 | `TruthView`（现状） | 基线 38 / 50；接口拆开后必须原样复现 |
| L1 | 真值加噪声、延迟、遮挡丢帧、过期 | 技能对感知误差有多敏感；需要的精度是多少（不用渲染，很便宜） |
| L2 | Unreal 深度缓冲加真值分割掩码（stencil），得到局部点云 | 只看得见一面、被手挡住这些本身伤多少，暂时不受模型能力影响 |
| L3 | 第 4 节的模型，逐个换上 | 基础模型带来的差距 |
| L4 | 再加双目伪影、IMU 偏置、运动模糊 | 接近真机 |

再做一个逐模块换回真值的矩阵：定位、地图、物体位姿、形状、抓取、抓住判断六个模块，每次只把一个换回真值，
看成功率回来多少。回来最多的那个模块就是差距的主要来源。

## 7. 主动感知：RSI 可以长出来的新技能

今天的技能从不“看”：物体位姿总是现成的。换成感知之后，这些动作会变得有用，而且正好是 RSI 循环可以自己发现、
自己写的：

- 走到能看清的位置再停；
- 抓之前换个角度再看一次；
- 改用腕相机接近；
- 不确定度高于门槛时先观测，再决定抓法。

它们和 33.5 的准备动作层（转、挪、清）是一类东西：准备动作的目标是“挪完之后出现一个可站的站位”，观测动作的目标是
“看完之后不确定度降到门槛以下”。挑选器的打分要带上位姿和形状的不确定度。

## 8. 资源

- **GPU：** 本机两张 RTX 5090（各 32 GB）。Unreal 编辑器占一张，感知模型（SAM 3、深度、FoundationPose、抓取）放
  另一张，MuJoCo 在 CPU 上。主线的 RSI 也要用 GPU（`RSI_GPUS`），这条线只在主线不占卡的时候跑，并且走现有的
  实例锁，不和主线的记录栈共用一张卡。
- **时间：** 每个感知周期多一次 Unreal 渲染和一轮模型推理。每局多出多少，等 L2 冒烟测出来再定预算，不提前估。
- **磁盘：** 每局要存图像。3 路相机 × 5 Hz × 60 s 是 900 帧，JPEG 按 100 KB 算约 90 MB 一局，150 局约 13 GB 一次迭代。
  默认只存失败局和抽样成功局的图像，其余只存估计轨迹。新下的权重（Fast-FoundationStereo、FoundationPose）是几 GB。

## 9. 分步计划

| 步 | 做什么 | 完成的标准 |
|---|---|---|
| 0 | 拆出 `WorldView`、`TruthView`；`motor.py` 拆本体感觉和接触；加静态检查 | 封存集 50 局在 `TruthView` 下和 run C 现任一致：38 / 50，成败逐局相同 |
| 1 | L1：`TruthView` 加噪声、延迟、过期 | 成功率随位置误差 0.5 / 1 / 2 / 4 cm、延迟 0 / 100 / 200 ms 的曲线 |
| 2 | L2：Unreal 镜像接进循环；深度缓冲加 stencil 掩码出点云；巡一圈建地图 | 封存集 50 局的 L2 成功率，和 L0 按起点配对 |
| 3 | L3：SAM 3、Fast-FoundationStereo、FoundationPose、感知网格上的抓取缓存、SLAM，逐个换上 | 每换一个出一行配对结果，加逐模块换回真值的矩阵 |
| 4 | 在 `PerceivedView` 上跑一次 RSI，从 run C 现任出发 | 循环能不能把 L3 的差距追回来，追回的改动是不是第 7 节那类观测动作 |
| 5 | L4 噪声；必要时从空种子跑 | 和 L3 配对 |

## 10. 待定决定

| 编号 | 问题 | 选项 | 推荐 |
|---|---|---|---|
| D1 | 真值边界 | 技能只读估计，评测读真值，LLM 角色两份都看 / LLM 角色也只看估计 | 前者：诊断需要知道是谁错了 |
| D2 | 允许多少先验地图 | 1. 回合前机器人巡一圈，用感知建图（不是真值）；2. 只给房间级粗图；3. 什么都不给 | 先做 1；真机部署前扫一遍房间本来就合理 |
| D3 | L3 的深度从哪来 | Unreal 双目图加 Fast-FoundationStereo / 深度缓冲加噪声模型 / 单目 DA3 | 双目加 FFS，和真机 D455、D405 的用法一致 |
| D4 | 要不要模拟 LiDAR | 实验室真机有就模拟 / 不模拟 | 先确认真机有没有 |
| D5 | 感知和物理的时序 | 锁步：感知算完再推进物理 / 实时：感知有真实延迟 | 先锁步，把精度和延迟分开量；L4 再上实时 |
| D6 | 任务怎么点名目标 | 给类别名（场景文件里的物体名） / 给一张参考图 / 只给自然语言描述 | 类别名 |
| D7 | 从哪套技能出发 | run C 现任改成读估计 / 空种子 | 现任：能直接量出感知带来的差距 |
| D8 | “抓住了”怎么判 | 只用夹爪开度和执行器力 / 再加视觉确认 | 两者都用，接触对只给评测 |

## 11. 还没核实的

- GraspGen-X 支不支持 Dex1 这种两指夹爪的自定义几何、权重和许可证；VGGT-SLAM 2.0 的代码是否已公开；FoundationPose
  有没有 2026 年的公认后继（10-23 的检索没找到）。知识截止在 2026-06，之后半年的模型需要专门检索一次。
- 实验室真机有没有 LiDAR；D455 / D405 的真实内参。
- `adt_apartment_mars` 里自由物体的 Unreal 视觉网格和 MuJoCo 碰撞网格差多少。
- Unreal 编辑器在公寓关卡上占多少显存；感知模型和它能不能分在两张卡上同时跑。
- 在公寓关卡里，每次只渲 1 帧（stream 模式）时图像质量够不够模型用。

## 参考

- SAM 3 与 SAM 3D：<https://ai.meta.com/blog/sam-3d/>
- Depth Anything 3（ICLR 2026）：<https://arxiv.org/abs/2511.10647>
- Fast-FoundationStereo（CVPR 2026）：<https://arxiv.org/abs/2512.11130>；FoundationStereo：<https://arxiv.org/abs/2501.09898>
- FoundationPose（CVPR 2024）：<https://arxiv.org/abs/2312.08344>
- GraspGen：<https://arxiv.org/abs/2507.13097>，<https://github.com/NVlabs/GraspGen>；GraspGen-X 与 Grasp-MPC：
  <https://blogs.nvidia.com/blog/cvpr-research-grasping-driving-agent-training/>
- VGGT-SLAM 2.0（RSS 2026）：<https://arxiv.org/abs/2601.19887>；VGGT-SLAM：<https://arxiv.org/abs/2505.12549>
- Gemini Robotics-ER 1.5：<https://developers.googleblog.com/en/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-1-5/>
- ConceptGraphs：<https://arxiv.org/abs/2309.16650>
