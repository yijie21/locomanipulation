<!-- Archived from root HANDOFF.md on 2026-09-20; historical claims are retained. -->
# HANDOFF — Blender 物流间场景导入(2026-09-19/20)

当前进展：[G1 条件可达域设计、验证与视频](g1-reachability-atlas-20260920.md)。

本文件用于把当前工作交接给下一个 session。

最新进展：已新跑原双臂 expert 一条对照（行走抓取成功，抬升20.8cm），并修正 ghost 任务录像起点：先完成准备与站稳，诊断另存，真正开启新任务录像后再 reach。本次单组成功，主视频3.92秒、0.02秒后进入reach，没有初始化重置。见 [对照视频与验证](g1-original-expert-task-boundary-20260920.md)。

仓库既有的交接件放在 `docs/handoff/`,本文件按用户要求放在仓库根。

---

## 1. 目标

把 `assets/blender/logistics_room.blend`(一个物流/仓储房间)变成:

1. **Isaac 渲染场景**(USD)
2. **MuJoCo 物理场景**(MJCF)

并在房间中央的 HP 纸箱托盘平台上放一个 NOR-CAL 纸箱,让 G1 站在前面,
**复现"低桌双臂抓取"那条线的布置**,用于后续采集数据与训练。

最终产物要求:两边布置一致,能渲出一张左 MuJoCo / 右 Isaac 的同视角配对图。

---

## 2. 约束(session 级,必须延续)

- 回复中**所有文件路径写绝对路径**。
- 视频预览 ≤ 500,000 字节,**全时长、正常速度**,原片保留。
- 单臂 expert **冻结在 `stage4_27_wrap`**,不得改动。
- fresh20 样本库(seed 30900)不得改动。
- **禁止 `git add -A`**;按文件名 stage;push/merge 前检查是否混入 `outputs/` 路径或符号链接。
- 只停自己这一份 stack 实例,不得波及其他 session。
- commit trailer:`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- 文档行数上限由 `scripts/lint_repository.py` 强制(README ≤100 行)。
- 产物一律放 `outputs/`,不放 `/tmp`。

---

## 3. 已确认的决策及理由

### 3.1 Isaac 管视觉,MuJoCo 管碰撞(架构基线)

**决策**:房间网格只进 Isaac 做视觉;MuJoCo 只放少量碰撞图元。

**理由(事实)**:
- `docs/rendering.md` 明写"房间网格是视觉背景,**不会自动增加 MuJoCo 的墙体碰撞**"。
- 实测现有每个 MuJoCo 场景**只有 1 个 `type="plane"`、零堵墙**;5 box + 4 cylinder 是桌子道具,
  115 个 mesh 是机器人自己的连杆。
- 489k 三角面做碰撞体要对每个货架做凸分解,而仿真跑 50 Hz 全身控制器。

### 3.2 不做光照烘焙(原方案 c)

**决策**:放弃"把 Cycles 的 GI 烘进 UV 贴图"。

**理由(事实)**:路径追踪已经把 GI/接触阴影找回来(结构差异 4.2% → 2.9%);
而全房 **只有 12% 的物体有 UV**(434/3604),为剩下 88% 做展开代价远大于收益。

### 3.3 地板:plane 管物理,房间网格管画面

**决策**:保留 `type="plane"` 作碰撞面,但**不渲染**(`rgba` alpha=0、`group=3`),
抬到 `z=0.001` 与房间地板网格顶面对齐。

**理由(事实)**:
- 原来的蓝色是基础场景自带 `groundplane` 棋盘贴图(`rgb1="0.2 0.3 0.4"`,本身蓝灰)与房间地板网格打架。
- plane 是平面碰撞的正确图元:精确、无限大、开销为零、接触求解最稳;用网格当地面更贵且易接触抖动。
- 效果:MuJoCo↔Isaac 结构差异 **17.2% → 13.8%**。

### 3.4 灯光增益不写进 USD

**决策**:`blender_to_usd.py` 的 `--light-gain` 默认 **1.0**,增益由渲染方在运行时施加。

**理由(事实)**:见 §7.1 的未解问题;烘进 USD 的版本渲染结果不正确。
仓库既有数据集重渲走 `locomani.sim.isaac.isaac_render`,**不受此影响**。

### 3.5 不发布 Isaac 单帧渲染脚本

**决策**:写好的 `scripts/render_usd_frame.py` **已删除,不随提交发布**。

**理由(事实)**:它产出的图**结构性不符**(19.4%,同内容在组合场景脚本里是 4.2%),
且提高增益不解决(增益 4800 时亮度上去但结构差异仍 19.4%)。
在一个故障目录写满"配置声明了但没传下去"的仓库里,发布会产出错误图像的工具是最不该做的事。

### 3.6 纸箱朝向:长边(0.45 m)朝向平台边缘

**决策**:用户明确指定。已照做。

**风险(事实)**:双手间距因此是 **0.450 m**。
`assets/crates/blue_crate/model.xml` 记着硬约束:原生 605×405 让把手相距 **0.58 m,手臂交付不了**,
故缩到 403×270(**0.403 m**)。0.450 落在已知可行与已知失败之间,**未验证**。

---

## 4. 已完成工作

### 4.1 本 session 的提交(均已推送到 `origin/main`)

| commit | 内容 |
|---|---|
| `7710591` | **本次主体**:`.blend` → MuJoCo/Isaac,5 个脚本 + runbook + 台账 + README 行 |
| `5a861b4` | 单臂出生点 A/B 台账 + 同批双臂 20 局 |
| `4cb0d82` | `collect_bottle.py` 交付缺陷修复(`--video-include-init` 曾被硬编码) |
| `5c506b4` | 单臂文档导出器说法修正 + 两份 20 局采集配置 |
| `97c1edd` | 第三视角视频按集号进数据集 |

`f5002ad` 及更早为本 session 之前的双臂 clean-deploy 工作。

### 4.2 交付的脚本(全部实跑验证)

```
scripts/blender_to_usd.py        blend → USD(含四处单位修正)
scripts/blender_to_mujoco.py     blend → 按材质拆分的 OBJ
scripts/build_mujoco_scene.py    OBJ → scene_locomani.xml
scripts/render_mujoco_frame.py   渲一帧
scripts/compare_frames.py        对比两帧,报 raw 与 structural
```

### 4.3 文档

- `docs/blender_scenes.md`(105 行)—— runbook,README「功能与文档」表格单开一行
- `docs/ledger/blender_scenes.md`(85 行)—— 台账,记数字与发现过程

---

## 5. 关键命令

```bash
# 前置:需要 Blender 5.x(自带 USD API)
which blender

# Isaac 场景
blender --background assets/blender/<room>.blend \
  --python scripts/blender_to_usd.py -- outputs/scene/<name>

# MuJoCo 场景(两步)
blender --background assets/blender/<room>.blend \
  --python scripts/blender_to_mujoco.py -- outputs/scene/<name>_mj
python scripts/build_mujoco_scene.py outputs/scene/<name>_mj

# 渲一帧 MuJoCo
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 "$VENV_SIM_PY" \
  scripts/render_mujoco_frame.py \
  outputs/scene/<name>_mj/scene_locomani.xml /tmp/mj.png --camera <名字>

# 对比
python scripts/compare_frames.py <isaac.png> /tmp/mj.png --out /tmp/pair.png
```

Isaac 侧需 `OMNI_KIT_ACCEPT_EULA=YES`,否则无头导入被 EULA 提示阻塞。

---

## 6. 产物路径(均在 `outputs/`,不进 git)

```
outputs/logistics_room/                   119M  Isaac USD(初版)+ 贴图
outputs/logistics_room/renders/            59M  33 张渲染图(含全部中间态)
outputs/scene/logistics_room_box/          94M  房间+纸箱 的 USD
outputs/scene/logistics_mujoco/            88M  MuJoCo 场景:scene_locomani.xml + mesh/ + textures/
```

**关键图**:

| 文件 | 内容 |
|---|---|
| `renders/pair_mujoco_isaac.png` | **左 MuJoCo / 右 Isaac 配对图**(用户要的最终产物) |
| `renders/01_blender_reference.png` | Blender Cycles 参考 |
| `renders/06_isaac_dome_fixed_pathtraced.png` | Isaac 最佳质量(结构差异 2.9%) |
| `renders/g1d_reverse.png` | G1 + 纸箱侧视,配对图的 Isaac 半边 |
| `renders/02..04` | 三个单位 bug 的现场(全黑 / 2 米飘带 / 黑角) |

---

## 7. 实验结果(事实)

### 7.1 Blender USD 导出器的四处单位错误

| 项 | 导出值 | 应为 | 症状 |
|---|---|---|---|
| 相机 `focalLength`/`horizontalAperture` | `0.185` / `0.36` | `18.5` / `36` | 视场角两边算得一样(88.4°),但 RTX 不认亚毫米焦距,**取景完全跑偏** |
| 灯光 `intensity` | `20.69`(Blender 65 W) | ×1200 | 画面几乎全黑 |
| `BasisCurves.widths` | `2.0`(舞台单位是米) | `0.002` | **一根地面走线渲成 2 米宽灰飘带**,106 条 |
| `DomeLight` 环境贴图 | 465 字节纯色 EXR | 丢贴图改 `inputs:color` | RTX 报 `MirroredBall environment format not supported yet` 后**整个环境光失效** |

四条都是**渲出来对比才发现的**,没有一条是读文档读到的。

### 7.2 渲染差异

| 对比 | 结果 |
|---|---|
| Blender vs Isaac(原始) | 20.3% |
| Blender vs Isaac(曝光归一化) | 5.8% |
| + DomeLight 修复(RTX 实时) | **4.2%** |
| + 路径追踪 | **2.9%** |
| MuJoCo vs Isaac(修地板前) | 17.2% |
| MuJoCo vs Isaac(修地板后) | **13.8%** |

黑角落实测:Blender 左上角亮度 109.0,Isaac **63.9** → 修 DomeLight 后 135.1 / 137.1。

**渲染器分工建议**:数据集用 RTX Real-Time(4.2%,便宜);出图给人看用路径追踪(2.9%,慢得多)。

### 7.3 几何测量

| 项 | 数值 |
|---|---|
| 房间 | 8.1 × 6.9 × 3.24 m,489,284 三角面,3,604 网格,36 材质,METRIC scale 1.0 |
| 托盘平台顶面 | **0.371 m**,footprint 1.2 × 0.9,近边 `y = -0.555` |
| NOR-CAL 纸箱 | 0.45 × 0.32 × **0.33** m,原点在底面中心,462,752 三角面,**开口有沿** |
| 纸箱位置 | `(0, -0.39, 0.371)`,近端面 `y=-0.550`,内缩 5 mm |
| G1 | 高 **1.323 m**,肩宽 0.464 m,**+x 为正前**(`head_camera pos="0.06 0 0.45"` 确认) |
| G1 站位 | `y = -1.205` 朝 +y,站距 **0.265 m**(双臂台账标定值) |

### 7.4 与现有双臂线的差距

| | 现有 crate 线 | 本场景 |
|---|---|---|
| 工作面高度 | 0.30 m | **0.371 m** |
| 抓取高度 | 0.47 m | **0.701 m** |
| 双手间距 | 0.403 m | **0.450 m** |

### 7.5 三个踩过的坑(避免重复)

1. **Blender 的 USD 导入器解析不了 `assets/isaac/robot/g1_dex1.usda` 的 140 个 reference**
   —— 导入 178 个物体但世界变换全是 ±1.0 占位值。机器人合成必须在 Isaac 里做。
2. **`g1_dex1.usda` 没有 `defaultPrim`** —— `AddReference(路径)` 不指定 prim 会引用到空,
   必须写 `AddReference(ROBOT, "/World/scene")`。
3. **G1 的 MJCF 没有 `<material>` 也没有 `<texture>`** —— 外观全靠 77 个逐 geom `rgba`
   (`0.7` 外壳 52 个、`0.2` 关节 11 个)。Isaac 的 MJCF 导入器把它们全丢了(USD 里 Material 数 = **0**)。
   还原需从 MJCF 读回再绑;而链接几何是**实例代理**,USD 禁止在其上写属性,
   必须先 `SetInstanceable(False)` 解实例化(68 个 prim)。

---

## 8. 未解决问题

### 8.1 【事实】Isaac 单帧渲染的结构性差异 —— 未解

同一个 USD、同一台相机:
- 在组合场景脚本(含机器人引用 + 解实例化 + 绑材质)里渲 → luma ~118,结构差异 **4.2%**
- 在独立单帧脚本里渲 → luma **53.6**,结构差异 **19.4%**

症状:**带贴图的材质渲成黑色**,无贴图材质正常。提高增益不解决。

**已排除九条,下一次请从第十条开始**:

1. 贴图文件(44 个字节级完全相同)
2. 贴图路径解析(43 个引用,0 缺失)
3. 色彩空间(sRGB 15 / raw 28,两边一致)
4. 材质着色网络(节点、类型、输入一致)
5. 相机(世界变换与焦距一致)
6. 累积采样(subframes 32 → 96 无变化)
7. 资产流式加载(等待 stage loading 完成无变化)
8. 灯光写入时机(运行时重写同值、调整与渲染器配置的先后顺序,均无变化)
9. Hydra 结构重同步(增删 dummy prim 无变化)

**影响范围**:仅限这个一次性对比工具。仓库既有的数据集重渲走 `isaac_render.py`,不受影响。

### 8.2 【事实】场景还不能采集 —— 缺 `scene_meta.json`

专家代码读 `surfaces.A`(`top` / `half_extent` / `xy`)与 `objects.<name>`
(`init_pos` / `height` / `mesh` / `surface`)来规划。**没有它,场景搭好了也跑不了采集。**

本场景应为:`surfaces.A.top = 0.371`,`objects.nor_cal_box`。

### 8.3 【假设】抓取可行性 —— 全部未验证

以下四条都是**推断,不是结论**:

- 0.450 m 手间距可达(位于已知可行 0.403 与已知失败 0.58 之间)
- 0.701 m 抓取高度可行(**在现有标定之外**;0.70 m 是站立抓取,
  与该线"弯腰够 30 cm"、躯干 55–58° 是相反姿态)
- 纸箱约 5 mm 的壁沿可被 dex1 夹持(crate 抓的是 30×24 mm 边梁,闭合范围 84.8 mm)
- 现有双臂 expert 可以"重新标定"而非"重建"

**注意**:§7.4 的三项差距意味着**整套站位/躯干前倾/屈膝标定作废,不是微调**。
纸板还更软,`MuJoCo grip creep` 台账记过默认软接触下物体 0.5 cm/s 滑出夹爪,需单独硬化夹爪垫片 geom。

### 8.4 【事实】遗留清理

- `assets/blender/` 下有 **71 MB** 首次手工导出时 `unpack_all` 留下的解包残留(贴图+字体)。
  新脚本不再产生,但这份需要删。该目录**未进 git**。
- 本 session 之前遗留的未跟踪实验配置:`configs/experiments/agent_*.yaml`、
  `crate_doc_walkthrough.yaml`、`crate_twitch_probe.yaml`。非本次产物,未处理。

### 8.5 【事实】本 session 之外的未决项

- **F10** 导出器丢帧根因,入口 `docs/handoff/exporter-frame-loss-20260917.md`
- 用 `outputs/runs/bottle_simple_200_20260916/model/checkpoint-10000` 跑一次**真实 100 轮单臂评测**
  (其历史闭环数字在 `9ead4a0` 之前全部失效)
- 重建双臂基线:**所有历史 crate 数字都是在降级 deploy 上测的**,与 `f5002ad` 之后的默认不可比

---

## 9. 下一步(建议顺序)

1. **写 `scene_meta.json`** —— 唯一挡住"能不能采集"的东西。`surfaces.A.top = 0.371`。
2. **先渲/仿一次够取检查**,再动专家 —— 用 §7.4 的三个数判断 0.701 m / 0.450 m 是否可达。
   比直接改专家便宜得多。
3. 若可达 → 重新标定站位/躯干/屈膝(**不是微调**);若不可达 → 考虑把纸箱换成有沿容器,
   或调整平台高度。
4. 删 `assets/blender/` 的 71 MB 残留。
5. §8.1 的渲染差异:想查就从第十条假设开始,**不要重复已排除的九条**。

---

## 10. 环境状态(交接时)

- `main` 与 `origin/main` 同步,HEAD = `7710591`
- 两张 GPU 空闲,无残留 sim / exporter / deploy
- `/workspace` 约 69 GB 可用(本 session 清理了 45 GB 的两个 100 步冒烟 checkpoint)
- headroom / fast-jev-compaction 两个压缩插件**已卸载并移除 marketplace**
  (headroom 在本 session 进程内仍驻留,下个 session 消失)

---

## 11. 建议下个 session 调用的 skill

- `/grill-me` —— 若要决定"重新标定 vs 换容器 vs 改平台高度",这是有分支的设计决策
- `/diagnosing-bugs` —— 若要接手 §8.1 的渲染差异
