> **台账(append-only)**:本文件只在末尾追加新的日期化章节,不改写、不删除已有结论。
> 要更正一条结论,在新章节里写清楚并链接回旧的那一条,不要就地覆盖。

# Blender 场景导入台账

## 1. 物流间首次贯通:四个单位错误,以及一个没解开的渲染差异(2026-09-19/20, 本 session)

**素材**:`assets/blender/logistics_room.blend`(102 MB)与 `NOR_CAL_Box_Refined_V2.blend`(50 MB)。

**房间体检**:公制、scale 1.0,8.1 × 6.9 × 3.24 m,**489,284 三角面**,3,604 个网格物体,
36 材质,49 张打包贴图,6 相机,6 盏面光 + 世界环境光,Cycles 3840×2880 / 1024 spp。
比仓库自带的 `empty_room_world.ply.gz`(419 万面)**轻 8.5 倍**,渲染侧完全吃得下。

### Blender USD 导出器的四处单位错误

逐条都是渲出来对比才发现的,不是读文档读到的:

| 项 | 导出值 | 应为 | 症状 |
|---|---|---|---|
| 相机 `focalLength`/`horizontalAperture` | `0.185` / `0.36` | `18.5` / `36` | 视场角 88.4° 两边算得一样,但 RTX 不认亚毫米焦距,取景完全跑偏 |
| 灯光 `intensity` | `20.69`(来自 65 W) | 需 ×1200 | 首帧几乎全黑 |
| `BasisCurves.widths` | `2.0`,舞台单位是米 | `0.002` | 106 条装饰线缆渲成 2 米宽灰飘带,糊住半个房间 |
| `DomeLight` 环境贴图 | 一张 465 字节纯色 EXR | 丢掉,改 `inputs:color` | RTX 报 `MirroredBall environment format not supported yet`,**环境光整体失效**,背光角落全黑(左上角亮度 63.9 对 Blender 的 109.0) |

### Blender 与 Isaac 的实测差距

同一台相机(`01 Figure1 overview`,USD 导出时相机和灯光一并带过来,所以是同机位同灯组):

| 版本 | structural 差异 | 左上角亮度(Blender 109.0) |
|---|---|---|
| 修相机 + 曲线,DomeLight 仍坏 | 5.6% | **63.9** |
| 再修 DomeLight | 4.2% | 135.1 |
| 再换路径追踪 | **2.9%** | 137.1 |

曝光对齐前 raw 20.3%,对齐后 5.8% —— **四分之三的差异只是曝光**。
剩下的是 Cycles 路径追踪 GI 与 RTX Real-Time 的固有差别:接触阴影、环境光遮蔽、噪点。

**裁定**:数据集重渲用 RTX Real-Time(4.2%,便宜);出图给人看用路径追踪(2.9%,慢得多)。
**烘焙方案(把光照烘进 UV 贴图)不做** —— 路径追踪已经把 GI 找回来了,而全房 88% 的物体没有 UV。

### 未解决:单独渲染 `blender_to_usd.py` 的产物偏暗

同一份 USD、同一台相机,在组合场景脚本里渲是 luma 118 / structural 4.2%,
用独立的单帧脚本渲是 luma 53.6 / structural 19.4%,提高增益只会更亮但结构差异不降(4800 倍时 19.4%)。

**排除过九条**:贴图文件(字节级相同)、贴图路径(43 个引用 0 缺失)、色彩空间(sRGB 15 / raw 28 两边一致)、
材质着色网络(节点与输入一致)、相机(世界变换与焦距一致)、累积采样(32→96 无变化)、
资产流式加载(等 `is_stage_loading()` 排空后无变化)、灯光强度写入时机(运行时重写无效)、
Hydra 结构性重同步(建删临时 prim 无效)。

因此:**灯光增益不烘进 USD**,`blender_to_usd.py` 的默认增益改为 1.0,由渲染方在运行时施加。
独立单帧渲染脚本**未随本次提交发布**,因为它产出的图结构性不符。
仓库的数据集重渲走 `locomani.sim.isaac.isaac_render`,不受此影响。

### MuJoCo 侧

36 个按材质拆分的视觉网格 + 15 张贴图(**MuJoCo 不读 JPG**,自动转 PNG);
碰撞只有地面 plane、托盘平台 box(1.20 × 0.90,顶面 0.371)、纸箱 5 块板。
做法是在 `crate_table30_depth04/scene_locomani.xml` 上做 XML 手术,继承整棵 G1 树、默认值与执行器。

地面 plane 保留碰撞但关闭渲染(`rgba` alpha 0、`group=3`、抬到 z=0.001):
base 场景的 `groundplane` 棋盘贴图是蓝灰的(`rgb1="0.2 0.3 0.4"`),和房间自己的地板网格打架。
**改完 MuJoCo↔Isaac 的 structural 从 17.2% 降到 13.8%。**

### 场景几何(供后续标定)

| 量 | 值 |
|---|---|
| 托盘平台顶面 | **0.371 m**(对比现有双臂线的 0.30 m 桌面) |
| 平台尺寸 | 1.2 × 0.9 m,近边 y = -0.555 |
| NOR-CAL 纸箱 | 0.45 × 0.32 × **0.33 m**,原点在底面中心 |
| 箱沿高度 | **0.701 m**(现有双臂线抓取高度 0.47 m,**高 23 cm**) |
| 双手间距(长边朝外) | **0.450 m**(标定值 0.403,已知失败值 0.58) |
| G1 | 高 1.323 m,肩宽 0.464 m |

**尚未验证**:这个抓取高度和手间距都在现有双臂标定之外。0.70 m 是站立抓取的高度,
与该线"弯腰够 30 cm"(躯干 55–58°)是相反姿态,**整套站位/躯干/屈膝标定作废,不是微调**。

### 两个踩过的坑

- **Blender 的 USD 导入器解析不了 `g1_dex1.usda` 的 140 个 reference**,导入 178 个物体但世界变换全是 ±1.0 的占位值。机器人合成改在 Isaac 里做。
- **`g1_dex1.usda` 没有 `defaultPrim`**,`AddReference(路径)` 不指定 prim 会引用到空;要写成 `AddReference(ROBOT, "/World/scene")`。
- **G1 的 MJCF 没有 `<material>` 也没有 `<texture>`**,外观全靠 77 个逐 geom `rgba`(0.7 外壳 52 个、0.2 关节 11 个),
  Isaac 的 MJCF 导入器把它们全丢了(USD 里 Material 数为 0)。要还原得从 MJCF 读回来再绑;
  而链接几何是**实例代理**,USD 禁止在其上写属性,必须先 `SetInstanceable(False)` 解实例化(68 个 prim)。

## 2. Isaac 画质 A/B:高度图被当成法线、分辨率、路径追踪(2026-09-28)

**问题**:Isaac 重渲比 Blender 静帧"糙"。先对比两边的配置:
- Blender:Cycles 1024 spp + OIDN,12 次反弹,AgX,3840×2880。
- Isaac:RTX Real-Time(旧版)、FXAA,640×480,ACES + ISO 30。

**A/B**:`box_room_grasp_200_cap120_20260925` 的 `crate_vla_004`。同一条 trace、同一条 `cam_traj`、同一套曝光;
取前 1500 个状态、stride 6,共 250 帧,只渲主画面。产物在 `outputs/experiments/isaac_render_quality_ab_20260928/`。
这一局有机器人,没有 Blender 参照,结构差异都是对现状(Real-Time 640)算的;有 Blender 参照的数字仍是第 1 节的 4.2% / 2.9%。

| 组 | ms/帧(含加载与预热) | 对现状的结构差异 |
|---|---|---|
| Real-Time 640×480(现状) | 76 | — |
| Real-Time 1280×960 | 109 | 1.5–1.8% |
| PathTracing 64 spp + OptiX 降噪,640×480 | 433 | 2.6–2.8% |
| PathTracing 64 spp + OptiX 降噪,1280×960 | 1501 | 2.7–2.9% |
| Real-Time 640×480,屏蔽 14 条高度图法线 | 74 | 2.3% |

**结论**:
- **每个照片纸箱满面的白色噪点来自导出器**:它把 Bump 节点的灰度高度图接进了 `UsdPreviewSurface.normal`(scale 2,bias −1)。
  - 高度值 0.50 ± 0.03,折成法线约为零向量,逐 texel 翻号。
  - 屏蔽这 14 条连接,Real-Time 下噪点消失,耗时不变。
  - 路径追踪的降噪能盖住大部分,但纸箱仍斑驳。
- **材质丢失**:房间 48 个 Principled 材质里,36 个的法线、15 个的底色来自程序化节点(Bump、ColorRamp、Noise),`UsdPreviewSurface` 全部丢失。这一项没修,要修得先烘焙,而 88% 的物体没有 UV。
- **分辨率**:1280×960 下纸箱印字和机器人边缘不再锯齿,耗时 1.4 倍。
- **路径追踪**:补上间接光和软接触阴影,但耗时是 Real-Time 的 5.7 倍(640×480)到 14 倍(1280×960);平均亮度约高 10/255,ISO 是在 Real-Time 下调的。
- **版本**:Isaac 5.0/5.1 没有新的写实渲染器;RTX Real-Time 2.0(实时路径追踪 + DLSS)要到 Isaac 6.0 才是默认,它要 Kit 110、Python 3.12。

**改动**:
- `blender_to_usd.py` 新增第五项修复:断开来自 Bump 高度图的 `normal` 连接。重新导出实测 `disconnected 14`。
- 已打包的 v1 USDC 不重导(当前 .blend 导出 43 张贴图,v1 是 49 张)。改由 `build_isaac_warehouse_scene.py` 在两个引用层里屏蔽单通道贴图的 `normal`,重建后组合结果为 0 条连接。
- `render_isaac_videos.sh` 默认 `--size 1280x960`,新增 `--path-trace`;`mujoco_render` 按需放大离屏缓冲(场景是 960×720)。
- 实测:一局 66 s 的评测带 `--insets`,1280 Real-Time 用 249 s;`--path-trace --size 640x480 --stride 16` 用 105 s。

**影响**:`warehouse_textured` 的 Isaac stage 从此不带噪点,数据集补渲和评测镜像的画面随之改变。
checkpoint-5000 的 Isaac 评测数字(81/83/88)是在旧 stage 上测的。

## 3. 四边线的纸箱尺寸:MuJoCo 场景与 Isaac stage 对齐(2026-09-28)

**做了什么**:`warehouse_textured` 派生出 4 个场景 `warehouse_box_45x{32,20}x{33,18}`(`locomani.tasks.crate.warehouse_box_scene`),
纸箱 9 个照片网格按尺寸非均匀缩放(照片跟着拉伸),碰撞壁与惯量同 `box_platform`;Isaac stage 用
`build_isaac_warehouse_scene.py --scene <场景>`,缩放从该 MJCF 读,`offset_pos` 随缩放同乘。`45x32x33` 的 stage 与 `warehouse_textured` 逐字节相同
(只多一个 `source.json`)。每条边的空地用水平射线对房间网格测(房间无碰撞体):ym 1.285、yp 1.629、xm 1.112、xp 1.048 m
(挡住的分别是黑货架、北侧墙边、蓝色滚筒台、钢货架)。

**核对**:同一组状态走 MuJoCo 与 Isaac(`IsaacView`,各场景 `isaac` 条目的曝光):机器人站在 −y 边前 0.42 m 的头相机,
Isaac 的纸箱掩码 = 有箱 − 箱沉到地下两帧之差(> 40/255)。产物 `outputs/experiments/warehouse_edges_20260928/`(`isaac_check.py`、`compare_*.png`)。

| 场景 | MuJoCo 箱像素 | Isaac 箱像素 | IoU | Isaac 在 MuJoCo 外 > 3 px | MuJoCo 在 Isaac 外 > 3 px |
|---|---|---|---|---|---|
| 45x32x33 | 30258 | 29655 | 0.974 | 13 px(11 个 ≤ 3 px 的点) | 0 |
| 45x20x33 | 24154 | 23772 | 0.976 | 11 px(9 个) | 0 |
| 45x32x18 | 21112 | 20639 | 0.971 | 11 px(9 个) | 0 |
| 45x20x18 | 16021 | 15879 | 0.983 | 14 px(12 个) | 0 |

轮廓在 3 px 内一致;散点是渲染噪声。四个 `edge_<边>` 机位两边都拍到机器人站在该边空地上、纸箱在该边台上。
