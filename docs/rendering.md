# 渲染:MuJoCo 物理 + Isaac 像素

## Video export

导出任何任务的视频、提供播放链接或打包 ZIP 前，必须满足
[AGENTS 的视频大小规则](../AGENTS.md#shared-resources-and-artifacts)。
这是面向查看和下载的交付步骤；原始采集、训练和审计视频保留在原位置。

1. 在新的预览目录中压缩；保持完整时长、正常播放速度和画面比例。
   可以降低分辨率、帧率和码率。声音有信息时保留，不能通过裁掉任务阶段或加速来达标。
2. 从以下命令开始，再按实测大小调整 CRF、分辨率和帧率。固定 CRF 不保证文件大小。

   ```bash
   ffmpeg -v error -nostdin -n -i input.mp4 \
     -map 0:v:0 -map '0:a?' -vf 'scale=640:-2,fps=10' \
     -c:v libx264 -preset medium -crf 32 -pix_fmt yuv420p \
     -c:a aac -b:a 24k -movflags +faststart preview.mp4
   wc -c preview.mp4
   ffmpeg -v error -xerror -i preview.mp4 -f null -
   ```

3. 每个文件都要检查实际字节数、完整解码、起止画面，以及与原片的时长差
   （只允许重采样到输出帧率时的一帧舍入差）。超限就继续压缩。
   长视频若无法兼顾上限和可辨识性，拆成编号连续片段，覆盖原片完整时间范围并明确标注。
4. 播放页面和回复链接指向合格预览。ZIP 仅包含这些预览和必要的小型说明文件；
   打包后再次核对成员和总体积，并报告单片范围及 ZIP 大小。历史原片和旧大包保留作审计。

## 渲染与物理

物理永远是 MuJoCo,wall-clock,SONIC 的 ZMQ 时序一行不改。Isaac 只负责像素,
按已录轨迹**事后重渲**,不进采集回路。这与 SIMPLE 自己的 SONIC 路线一致
(`docs/source/sonic-wbc/replay_render.md`:把 Isaac 放进实时回路会饿死
ZMQ/DDS 通信并触发控制器心跳超时)。

票表在 `.scratch/isaac-render/issues/`,先读 `00-context.md`。

## 运行时

```bash
bash scripts/setup_render_runtime.sh          # 检查 pin、打补丁、验证 isaacsim
bash scripts/setup_render_runtime.sh --deep   # 外加一次真正的 import(会启动 Kit,数分钟)
```

可重复执行:第二次每个补丁都报 `already applied` 且不改动任何文件。三个状态由
`git apply` 自己的检查判定,不靠 grep 标记串:反向检查通过就是已打,正向检查通过
就打上,两者都不通过说明工作树漂移了,脚本拒绝继续而不是静默重打。

## 两个变量,以及它们为什么是两个

`scripts/paths.sh` 里:

| 变量 | 默认 | 含义 |
|---|---|---|
| `SIMPLE_ROOT` | `$LOCOMANI_ROOT/third_party/SIMPLE` | SIMPLE 检出位置(安装见 `docs/setup.md` 第 7 步)|
| `SIMPLE_PIN` | `6d10628…` | 期望的 commit |
| `RENDER_PY` | `$SIMPLE_ROOT/.venv/bin/python` | 装了 `isaacsim` 的 python |

`RENDER_PY` 单独成一个变量是刻意的。SIMPLE 在自己的 `pyproject.toml` 里硬 pin
`isaacsim[all,extscache]==4.5.0`,而 3DGS/NuRec 需要 Isaac Sim 5.0 以上。渲染
运行时必须能在不等 SIMPLE 松开这个 pin 的前提下升级,把 `RENDER_PY` 指向另一个
venv 即可,其它什么都不用改。

## SIMPLE 是被 pin 的检出,不是 vendor 进来的依赖

和其它上游仓库一样放在 gitignored 的 `third_party/` 下,固定到一个 commit;clone、submodule、
venv 的步骤在 `docs/setup.md` 第 7 步,补丁与 pin 见 `patches/simple.md`。它的 `data/`
(HF 上十几 GB 的资产)主线渲染一个字节都不读,不用下载。

更要紧的是**离线重渲不 import `simple` 包**。它把我们自己的
`scene_locomani.xml` 走 Isaac 的 MJCF importer,关节名 1:1,状态同步就是一次
qpos 拷贝;相机由 MuJoCo FK 驱动,因为 MJCF importer 把 link 原点放在 armature
joint 上,直接挂相机会低约 24 cm。所以对 SIMPLE 的真实依赖只有两条:一个装了
`isaacsim` 的 venv,以及它转换好的场景与 robocasa 物体 USD。

`simple_ext`(dex1 机器人与 hug-container 任务注册)和烘焙了 d435_link 的
flat USD 属于 MuJoCo/Isaac 像素对齐的对照 harness,不在主线上,用
`--with-alignment` 才装。

## 资产构建

```bash
bash scripts/build_isaac_assets.sh                                  # 机器人 + 当前工作场景
bash scripts/build_isaac_assets.sh --snapshot <批次的 scene_<seed> 目录>
bash scripts/build_isaac_assets.sh --robot-only | --force
```

首次安装先执行 `bash scripts/install_scene.sh`。桌子、台面及贴图来自仓库中的
`assets/scene_locomani_v1.tar.gz`；空房间来自已跟踪的
[`assets/scene/empty_room_world.ply.gz`](../assets/scene/README.md)，约 41 MB。
不需要从另一台机器拷贝 EmbodiedGen 输出，也不需要 Git LFS。

旧版本只读取被忽略的 `outputs/scene/{layout,surface_b}/asset3d` 和背景目录，
源文件缺失时仍写出缺少桌子/房间的 USD。现在构建前检查必需源资产；缺失就报错，
转换失败也不会写入成功缓存。升级后重新执行上面的构建命令即可；旧缓存会被保留
为 `.previous.<时间>`，新的源文件、贴图、场景 XML、构建器、检出位置和 USD 输出目录参与缓存校验。
`--snapshot` 的默认 label 是目录完整 basename，与 `render_wall_pilot.sh` 一致；
只有显式 `--label` 才改变它。旧的 `scene_<seed>` 去前缀命名不再作为默认。

产物在 `assets/isaac/`,已 gitignore,约 650 MB 起,随时可重建。分两层,因为两者
变化频率不同:

| 层 | 位置 | 缓存键 | 耗时 |
|---|---|---|---|
| 机器人 | `robot/g1_dex1.usda` | 机器人 MJCF 的哈希 | 10 s,92 MB |
| 场景 | `scene/<label>/` | 元数据、XML、网格/贴图、背景、构建器及检出路径的哈希 | 20 s,280 MB |

机器人与场景无关,桌子多高都是同一具关节体,所以只建一次。场景按 batch 变,
`collect.sh` 每批用 `SCENE_SEED` 重建,桌面高度随之改变,所以按 seed 分别建、
分别缓存。缓存命中的开销是 10 毫秒,放进每集的后处理循环里也无所谓。

## 两个步骤都用 RENDER_PY

```bash
"$RENDER_PY" -m locomani.sim.isaac.compute_cam_traj --session <dir> --episode 0 \
    --sidecar '<glob>' --out cam_traj.npz [--scene-xml <该批快照>]
"$RENDER_PY" -m locomani.sim.isaac.isaac_render  --session <dir> --episode 0 \
    --sidecar '<glob>' --cam-traj cam_traj.npz --out <dir> [--scene-xml <该批快照>]
```

`compute_cam_traj` 不启动 Kit,但它同时需要 mujoco 和 pandas,而只有 render venv
两个都有,sim venv 有 mujoco 没有 pandas。迁移前的 docstring 说它跑在 sim venv,
是错的(2026-09-05 实测)。

`--scene-xml` 默认指当前工作场景,只有在没有别的 batch 重建过它的前提下才正确;
重渲旧数据要指向该批的快照 `$OUT/scene_<seed>/scene_locomani.xml`。

## Isaac 的渲染不是逐位可复现的

同一份脚本、同一份输入连跑两次,解码后没有一帧逐位相同,最大通道差 123,平均绝对
差 0.85。DLSS 是时域累积的。所以任何"和参考图对比"的验收都必须给容差,不能要求
bit-identical;判定重构或迁移是否等价,要用"同脚本跑两次"做对照。

## 采集侧的 flag

`scripts/collect.sh` 认 `RENDER_BACKEND`:

| 值 | 行为 |
|---|---|
| 不设 / `mujoco` | 默认,老路径逐字节不变 |
| `isaac` | 每集之后追加离线重渲阶段 |
| 其它 | 报错退出 |

它切换的是**视觉后端,不是物理**。

## T_render:闭环评测要不要 hybrid clock(2026-09-05 裁定)

**结论:不需要。纯 async 就够,票 10 的 hybrid clock 不做。**

复现:`"$RENDER_PY" -m locomani.sim.isaac.bench_render`,原始结果在
`outputs/renders_isaac/bench/t_render_2026-09-05.json`。

分母先说清楚,搞错就会把结论放大一倍:动作流跑在 `TF_DATASET_FPS`,
`run_eval_arm.sh` 钉死为 **25 fps**,不是 50 Hz;40 步的 chunk 是 1.6 s,
0.4 s 的 replan 间隔就是这么来的。

**预算**,来自两条 100 局干净协议臂的真实 trace(`tools/latency_dist.py`):

| 臂 | 稳态 obs→obs 间隔(帧 @25fps) | 首次推理 |
|---|---|---|
| clean100 | p10/p50/p90 = 10 / 10 / 10,max 18 | p50 3,max 13 |
| base100 | p10/p50/p90 = 10 / 10 / 10,max 20 | p50 3,max 16 |

也就是每个 replan 周期身体照旧执行约 10 帧(0.4 s)的旧计划。

**开销**,640×480(与数据集一致,没有为了压低数字降质量),两块 GPU 空闲:

| 节奏 | p50 | p90 | p99 | 折合帧 p50 / p90 |
|---|---|---|---|---|
| 连续渲染(离线重渲) | 16.0 ms | 19.7 ms | 23.9 ms | 0.40 / 0.49 |
| 间隔渲染(live mirror,0.2 s 间隔) | 27.8 ms | 36.2 ms | 42.6 ms | 0.70 / 0.91 |

间隔渲染比连续渲染慢约七成,因为 RTX 在连续帧之间复用时域状态,停顿后这部分作废。
live mirror 是间隔那一行,所以判定按它。

**判定**:p90 加不到 1 帧,预算是 10 帧,增幅约 9%。这远在延迟敏感度的平坦区内,
不值得为它引入冻结机制、也不值得承担冻结机制自身的验证成本(empty-freeze 对照)。

**四条限制,写下来免得以后误用这个数字:**

1. 这是进程内的渲染耗时。live mirror 还要加 IPC 往返,所以这是它的**下界**,
   票 08 必须自己量一次往返。
2. 机器空闲时测的。策略 server 和渲染抢同一块卡会更差,评测本来就要求 wall-clock
   臂串行跑。
3. Isaac Sim 4.5。升到 6.0(票 12)之后这个数字作废,基准工具留在仓库里就是为了重跑。
4. 不含 Kit 启动和冷缓存的首帧,预热 40 帧之后才开始计时。评测自己的首次推理本来
   就单独处理。

## 相机位姿需要两次步进(2026-09-06)

绝对世界相机位姿设进 `SensorCamera` 之后,要**两次** `world.step(render=True)` 才进入
渲染管线。只步进一次,每一帧画的都是上一帧的相机。

实测,状态跳 50 帧之后与 MuJoCo 同帧对比:

| 步进次数 | 桌面 IoU | 质心偏差 |
|---|---|---|
| 1 | 0.428 | (−11.6, +67.2) px |
| 2 | 0.756 | (−1.1, −1.3) px |
| 3 / 5 | 0.758 / 0.759 | 约 1 px |

相邻帧之间这个滞后很小、看不出来,但它是系统性的,会原样进数据集,所以比明显的错误更糟。
SIMPLE 不会遇到:它把相机挂在机器人下面,位姿和关节体在同一次 fabric 更新里走。我们用的是
逐帧绝对世界位姿,代价就在这里。`stage.SETTLE_STEPS = 2`。

**对 T_render 的影响**:每帧渲两次,所以票 04 量的耗时要翻倍,p90 从 0.91 帧变成约 1.8 帧
(25 fps 下)。稳态预算是 10 帧,增幅约 18%,"纯 async 够用、不需要 hybrid clock"的裁定不变。

## 关节体的根是 torso_link,不是 pelvis(2026-09-06)

`set_world_pose` 移动的是 PhysX 认定的关节体**根链接**,而 PhysX 把它排在
`body_names` 第一位。我们这具 G1 的 MJCF 浮动基座是 `pelvis`,根却是 `torso_link`。
按骨盆位姿下发,等于把躯干钉在骨盆的位置上,腰以下反着动。

一个原因造成了三种看起来无关的现象:零位姿下 34 个链接偏同一个 54.15 mm 向量;
真实位姿下残差随运动链增长到 112 mm;误差只由三个腰关节贡献,手臂、腿、夹爪、
基座位置与朝向全部精确。决定性的测量是驱动单个腰关节时躯干的**角度**误差,它精确
等于指令角度(0.25 rad → 14.324°,0.50 rad → 28.648°),即躯干根本没转。

修法是 `stage.root_body(robot)` 取 `body_names[0]`,replay 下发该链接的位姿。修后
零位姿和两个真实位姿下每个链接都是 0.00 mm(最大 0.03),桌面 IoU 从 0.886/0.786
升到 0.938/0.888。

**两个探针陷阱**,写下来免得重犯:只量被驱动关节**自己的子链接**,再坏的关节也读
0.00 mm,因为子链接就在关节原点上,要看下游;`get_link_transforms` 返回的四元数是
**xyzw**,按 wxyz 读会让位置精确的链接报出 180° 的假误差。

## 已解决:多可视网格的 body 丢网格,和手腕"断开"的两个原因

用户看到的症状是一个:Isaac 里前臂和钳口之间空一段。原因是两个,互相掩盖。

**1. importer 的合成丢网格。** 每个 mesh 资产各自导成 `/meshes/<asset>` 原型(一个带
geom pos/quat 的 Xform,里面一个叫 `mesh` 的几何);带多个可视 geom 的 body 被写成
**一个** `/visuals/<body>` prim 上的 reference **列表**。USD 合成时同名子节点合并,每个原型的
几何都叫 `mesh`,所以合成结果里只剩一个 `mesh`,其余原型的几何**不是被遮住、不是放错,
是不存在**。受影响的四个 body:`pelvis`(丢轮廓壳)、`torso_link`(四个网格丢三个:logo、
head、waist_support)、左右 `wrist_yaw_link`(丢自己的圆柱,只剩 Dex1 托板)。

修法是 `locomani/sim/isaac/split_visual_refs.py`:在导出的 `.usda` **文本**上把 reference
列表改成每个 reference 一个子 prim(`<asset>_visual`),原型、变换、其余 prim 一字不动;
`scripts/build_isaac_assets.sh` 在 convert 之后自动跑,缓存 key 同时含 MJCF 和这个脚本的
hash。验证(2026-09-07,ep05 第 1050/1100 帧):合成后手腕 1→2、躯干 1→4、骨盆 1→2 个
Mesh prim;34 个链接的 PhysX 位姿与未拆分构建**逐位相同**;像素差只在新出现的网格上。

之前记的"拆 reference 会让机器人整体位移(IoU 0.49)"是当时 articulation root 还取错
prim(见上一节)时的现象,不是拆分本身的问题;那时三种试法都是在**活 stage** 上改
(`ClearReferences` / `BatchNamespaceEdit`),后者根本没生效。在 layer 文本上改绕开了
这两条。`mjcf_split.py`(上游改 MJCF)仍留着没接,Isaac 4.5 的 importer 对它 SIGABRT。

**2. 头部网格从内部被看见。** 拆分之后 `head_link` 终于合成出来,结果画面右下角出现一个
大灰色圆顶。`head_camera` 在 torso 系的位置是 (0.06, 0, 0.45),而 head_link 的包围盒是
x∈[-0.064, 0.075]、z∈[0.269, 0.476]:**相机在头的网格内部**,离前表面 1.5 cm。MuJoCo 看不到
它只是因为 MuJoCo 剔除背面(把 `mjRND_CULL_FACE` 关掉,MuJoCo 里同一个圆顶盖住大半画面);
RTX 不剔除背面(`/rtx/hydra/faceCulling/enabled` 在 stage 加载前后设都无效),而从封闭网格
内部看到的每个面都是背面,所以**隐藏这个网格就等于 MuJoCo 的剔除结果**。
`StageCfg.hidden_visuals` 默认隐藏 `torso_link/head_link_visual`,prim 不存在时直接报错
(意味着机器人 USD 是拆分前的旧构建)。

最终对照在 `outputs/renders_isaac/ep05/wrist_fix_vs_mujoco.png`:MuJoCo | 修前 | 修后。
