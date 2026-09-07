# 渲染:MuJoCo 物理 + Isaac 像素

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
| `SIMPLE_ROOT` | `$LOCOMANI_WORKSPACE/code/SIMPLE` | SIMPLE 检出位置 |
| `SIMPLE_PIN` | `6d10628…` | 期望的 commit |
| `RENDER_PY` | `$SIMPLE_ROOT/.venv/bin/python` | 装了 `isaacsim` 的 python |

`RENDER_PY` 单独成一个变量是刻意的。SIMPLE 在自己的 `pyproject.toml` 里硬 pin
`isaacsim[all,extscache]==4.5.0`,而 3DGS/NuRec 需要 Isaac Sim 5.0 以上。渲染
运行时必须能在不等 SIMPLE 松开这个 pin 的前提下升级,把 `RENDER_PY` 指向另一个
venv 即可,其它什么都不用改。

## SIMPLE 是被 pin 的检出,不是 vendor 进来的依赖

工作副本约 35 GB,几乎全是 `data/`,所以不进本仓库。补丁与 pin 见
`patches/simple.md`。

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

产物在 `assets/isaac/`,已 gitignore,约 650 MB 起,随时可重建。分两层,因为两者
变化频率不同:

| 层 | 位置 | 缓存键 | 耗时 |
|---|---|---|---|
| 机器人 | `robot/g1_dex1.usda` | 机器人 MJCF 的哈希 | 10 s,92 MB |
| 场景 | `scene/<label>/` | 该 seed `scene_meta.json` 的哈希 | 20 s,280 MB |

机器人与场景无关,桌子多高都是同一具关节体,所以只建一次。场景按 batch 变,
`collect.sh` 每批用 `SCENE_SEED` 重建,桌面高度随之改变,所以按 seed 分别建、
分别缓存。缓存命中的开销是 10 毫秒,放进每集的后处理循环里也无所谓。

## 两个步骤都用 RENDER_PY

```bash
"$RENDER_PY" -m teleop_free.isaac.compute_cam_traj --session <dir> --episode 0 \
    --sidecar '<glob>' --out cam_traj.npz [--scene-xml <该批快照>]
"$RENDER_PY" -m teleop_free.isaac.isaac_render  --session <dir> --episode 0 \
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

复现:`"$RENDER_PY" -m teleop_free.isaac.bench_render`,原始结果在
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

## 未解决:多可视网格的 body 会丢掉一个网格

带多个可视网格的 body 被 importer 合成到**一个 prim、多个 reference** 上,而每个原型
内部的几何都叫 `mesh`,同名子节点相撞,只有最强的那个 reference 存活。受影响的四个
body 是 `pelvis`、`torso_link`、左右 `wrist_yaw_link`。手腕保留了 `Dex1_base_link`
的托板、丢掉了自己的圆柱,画面上表现为前臂和钳口之间空一段。

这不影响位姿正确性,只影响外观完整性。**四种改法都试过,都不成立**,不要重复:

| 改法 | 结果 |
|---|---|
| 每个 reference 移到各自子节点 | 机器人整体位移,剪影 IoU 0.49-0.62,质心偏 23-31 px |
| 保留 reference[0],只移多余的 | 同样位移,说明是 `ClearReferences` 本身改变合成 |
| `Sdf.BatchNamespaceEdit` 给原型内的 `mesh` 改名 | `Apply()` 返回 False,完全没生效 |
| 上游改 MJCF,多余 geom 拆进无质量子 body | 变体本身正确(41 body、同样 34 关节 77 geom、运动学 0.0000 mm),但 Isaac 4.5 的 MJCF importer 直接 SIGABRT,带不带 inertial 都崩 |

预处理器留在 `teleop_free/isaac/mjcf_split.py`,没有接进构建,换新版 Isaac 后值得重试。
