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

## 2. warehouse 的 Isaac 图比 Blender 暗：主要是曝光按 MuJoCo 标的，不是按 Blender（2026-09-29，分支 loco_unreal）

在 Blender 参考相机 `reverse_room` 上对照，参考图是 `outputs/unreal_warehouse_20260928/reference/blender_reverse_room.png`（Cycles，AgX Medium High Contrast，曝光 −0.25）。
Isaac 用生产用的 stage（`stage.assemble_stage` + warehouse 的 RTX 设置）。脚本和图在 `outputs/isaac_blender_match_20260929/`。

| 区域（灰度中位数） | Blender | Isaac 现在（ISO 30） | Isaac 路径追踪 | Isaac ISO 60 | Isaac ISO 80 | Unreal |
|---|---|---|---|---|---|---|
| 天花板 | 181 | 53 | 97 | 104 | 130 | 112 |
| 后墙 | 214 | 171 | 186 | 211 | 223 | 212 |
| 地面（前 / 左） | 80 / 76 | 39 / 36 | 43 / 40 | 70 / 65 | 87 / 81 | 86 / 63 |
| 货架纸箱 | 197 | 138 | 153 | 187 | 203 | 199 |
| 与 Blender 的平均绝对差 | 0 | 58.9 | — | 23.5 | 17.1 | — |

- **暗的主因是曝光。** `filmIso 30` 是 2026-09-24 按 MuJoCo 的机器人躯干亮度标的（`docs/rendering.md` warehouse 一节），从来没有对 Blender 标过。ISO 60–80 时墙、地面、纸箱都回到 Blender 的水平。
- **全局光不是主因。** RTX 实时模式的间接漫反射本来就开着（`/rtx/indirectDiffuse/enabled = True`，2 次反弹）。换路径追踪只把平均值从 101 提到 114，地面仍是 43。
- **剩下的差距在天花板**（只靠反射光照亮）：ISO 80 时 130 对 181。来源是 Cycles 12 次反弹对 RTX 2 次，加上色调映射不同（ACES，`tonemap/op = 6`，对 Blender 的 AgX）。
- **`env_light` 被关掉的理由已经过时。** `build_isaac_warehouse_scene.py` 以"RTX 不认导出的 EXR"为由关掉它，但仓库里的 USDC 早已修成纯色 dome（强度 144）。实测：打开它、关掉通用 dome + sun，平均值只变 0.1，封闭房间里 dome 几乎照不进来。
- **还有一处和 Blender 不同，这次没量：材质。** Blender 的 USD 导出器丢掉所有程序化节点（40 个材质里 28 个是 Noise 纹理）。Unreal 那条路把它们烘成了贴图，Isaac 这条路没有烘。

## 3. Isaac 的锯齿和小字：生产渲染只有 FXAA（2026-09-29，分支 loco_unreal）

生产渲染的抗锯齿是 `/rtx/post/aa/op = 2`（FXAA），1 spp，没有别的后处理；纹理流式加载的预算是显存的 60%，不是贴图分辨率的问题。
同一 stage（`warehouse_textured_env`，ISO 80，隐藏机器人和纸箱）在三台 Blender 参考相机上对照，指标是对 Cycles 参考图的结构相似度（SSIM 的结构项），1 = 一致。
脚本和图在 `outputs/isaac_blender_match_20260929/`（`probe_aa.py`、`aa/`、`aa_crops.jpg`）。

| 区域 | FXAA（现在） | 不抗锯齿 | TAA | 2× 超采样 | 2× 超采样 + FXAA |
|---|---|---|---|---|---|
| reverse_room 左货架纸箱 | 0.909 | 0.860 | 0.933 | 0.922 | 0.947 |
| reverse_room 右货架纸箱 | 0.889 | 0.832 | 0.924 | 0.897 | 0.926 |
| 头部相机 2400 托盘侧面标签 | 0.943 | 0.906 | 0.962 | 0.944 | 0.957 |
| 头部相机 2400 货架印字 | 0.928 | 0.902 | 0.950 | 0.944 | 0.956 |

- 2× 超采样 = 按 2 倍分辨率渲、再用面积插值缩回来。
- TAA 的分数是在静止相机上攒了 24 帧得到的，是它的最好情况。闭环镜像每张图是姿态跳变后只走 2 步，数据集补渲却是连续帧，TAA 会让两边的图不一样。
- DLSS（op 3）和 DLAA（op 4）在同一会话里这样切换出来是噪点，没收敛。这是用法问题，不是画质结论。

**采用 2× 超采样（用户 2026-09-29 定）。** 保留 FXAA，每个相机按 2 倍分辨率渲、按 2×2 块平均缩回（`isaac.supersample: 2`，`stage.downsample`）。
- 逐帧确定，镜像的按需渲染和数据集的连续渲染是同一张图；TAA 做不到这一点。
- 实测代价（`outputs/isaac_ssaa_20260929/`，GPU0，`bench_render` 走 `IsaacView`，三路相机，100 次）：连续渲 p50 94 → 209 ms，间隔 0.4 s 按需渲 95 → 218 ms，约 2.2 倍，不是先前估的 1.4 倍。
- 同一状态三路相机的平均亮度差 < 0.3/255；地砖缝从虚线变成连续线（`ss1_vs_ss2_crop.png`）。

## 4. 天花板偏暗和程序化纹理：前者用间接光 ×2 + Hejl–Hable 色调映射修好，后者在 640×480 看不出来（2026-09-29，分支 loco_unreal）

对照方法同第 2 节，但换成三台 Blender 参考相机（`reverse_room` 加两个头部相机），隐藏机器人和纸箱。指标是灰度平均绝对差。脚本和图在 `outputs/isaac_blender_match_20260929/`（`sweep_gi*.py`、`gi*/`、`gi_compare.jpg`）。

**更正第 2 节的天花板数字。** 实时模式的间接光有时域累积，单次渲染的天花板在 107–182 之间跳，取决于前面渲过什么。第 2 节的 130 就是这种单次值。
改成候选之间交错、每个重复 4 次、每次先把相机挪开再回来之后，天花板的标准差只有 1–3。下表是这样量的（稳态 = 回来后渲 30 步；镜像式 = 回来后只渲 2 步，和闭环镜像一致）：

| 设置（ISO） | 平均绝对差 稳态 / 镜像式 | 天花板（Blender 183） | 左墙 / 后墙（231 / 214） | 地面（81） | 货架纸箱（198） |
|---|---|---|---|---|---|
| 原来（80） | 11.1 / 11.5 | 151 | 242 / 225 | 88 | 206 |
| 间接光 ×2（60） | 9.9 / 10.4 | 174 | 238 / 220 | 74 | 197 |
| Hejl–Hable 色调映射（80） | 8.7 / 9.2 | 140 | 228 / 208 | 86 | 188 |
| **间接光 ×2 + Hejl–Hable（80）** | **6.9 / 7.6** | **176** | **231 / 213** | 89 | 193 |
| 同上再关环境光遮蔽（80） | 6.5 / 7.5 | 176 | 231 / 213 | 89 | 196 |

- 采用第四行：`/rtx/indirectDiffuse/scalingFactor=2.0`、`/rtx/post/tonemap/op=4`，写进 `configs/scenes.yaml` 的 warehouse 条目。关环境光遮蔽只再好 0.1–0.4，不值得多改一项。
- 三台相机每步耗时都在 33–44 ms，各设置之间没有差别。
- 增加反弹次数（`/rtx/indirectDiffuse/maxBounces` 2 → 4、8）效果远不如直接放大间接光：天花板只到 135–148。
- 其他色调映射（Reinhard、Modified Reinhard、HableUc2、Iray）都比 ACES 差。
- 颜色（Lab 的 a、b 通道）误差略降，0.88 → 0.76；饱和度略升（房间 14.8，Blender 12.8）。

**程序化纹理的缺失在 640×480 看不出来。** 第 2 节说墙面、地面丢了 Blender 的噪声纹理，量下来并不成立：
- 地面是照片材质（`Reference photograph | Rubber`），Isaac 本来就有；只是照片的高度图凹凸被屏蔽了，Unreal 那边也一样。
- 墙面在 Blender 里几乎是平的，高通能量 0.28，Isaac 是 0.53。
- 货架、托盘、纸箱这些有烘焙材质的区域，Isaac 的细节都不比 Blender 少（比如托盘侧面 21.4 对 16.9）。
- Isaac 多出来的是锯齿颗粒，属于第 3 节的抗锯齿问题。
- 所以没有把 Unreal 的烘焙贴图搬进 Isaac。给人看的高分辨率视频如果以后需要，做法是：保留 USDC 里的灯、照片和纯色材质，只把 14 个烘焙材质的网格换成 `room.glb` 里的对应网格。USDC 每个网格只绑一种材质，可以按网格替换。

## 5. Unreal 对 Blender：灯板不发光是 glTF 母材质的问题；曝光本来就对，天花板差在反弹光（2026-09-29，分支 loco_unreal）

对照只用 `reverse_room`：它是固定的世界相机，位姿和 Blender 参考图完全一样。参考图里头部相机的位姿来自另一条 trace，和现有 trace 差 4 cm，没法用。
机器人和纸箱（Blender 图里没有）所在区域不计入。脚本和图在 `outputs/unreal_blender_match_20260929/`（`match.py`，`before_after.jpg`）。
每个候选都是：停 PIE → 改关卡的后处理体积 → 启动 PIE（URLab 在 BeginPlay 把体积设置复制进相机）→ `fresh` 取图（8 遍）。
同一候选重复 2 次，各区域的标准差都 ≤ 0.8。

**灯板不发光。**
- glTF 经 Interchange 导入后，材质用的是 `/InterchangeAssets/gltf/M_Default`，它在这些截图里不渲自发光。
- 证据：强度设为 0、2049、20490 nit 时灯板都是 167；把底色改红，灯板就变红；把自发光改绿，完全不出现。
- 修法：`unreal_warehouse_level_20260928.py` 的 `emissive_materials`，把 Blender 的自发光材质（`room_scene.json` 记录的两种）换成自建的 `/Game/Loco/M_LocoEmissive`，强度 s × 683 nit。
- 结果：灯板 167 → 230（Blender 238）。

**曝光不用改。** 按物理推算的 −9.40（`place_exposure`）下，墙 231 / 218、地面 79，Blender 是 229 / 217、80。+0.5 档会把地面推到 109。

**天花板差在反弹光。** 天花板只靠反弹光照亮：关掉 GI 时是 40，Lumen 90，Blender 174。

| 后处理设置（关卡体积） | 平均绝对差 | 天花板（174） | 墙（229 / 217） | 地面（80） | 货架纸箱（153） | 托盘顶（201） |
|---|---|---|---|---|---|---|
| 原来 | 23.1 | 90 | 231 / 218 | 79 | 141 | 204 |
| `lumen_diffuse_color_boost` 3 | 18.2 | 109 | 232 / 220 | 83 | 153 | 209 |
| **`lumen_diffuse_color_boost` 4** | **17.4** | **114** | **233 / 221** | **84** | **154** | **209** |
| `lumen_diffuse_color_boost` 6 | 16.6 | 120 | 233 / 222 | 85 | 157 | 211 |
| 增益 4 + 曝光 −0.15 | 19.1 | 104 | 230 / 217 | 75 | 148 | 205 |

- 采用增益 4，写进 `place_exposure`（`LUMEN_DIFFUSE_COLOR_BOOST`）。再往上收益很小。
- 下面几项没有作用：`indirect_lighting_intensity` 1.5–4、Lumen 的 final gather 和 scene lighting 质量 4、表面缓存分辨率 1.0，天花板变化都在 1 以内。
- 剩下的差距：天花板仍比 Blender 暗 60。

**在 PIE 里直接改相机的后处理不起作用。** 改 `SceneCaptureComponent2D.post_process_settings` 后读回值是对的，但截图不变（曝光 +1 档也不变）。之后插件崩了两次。

## 6. box_platform 的 Isaac 盒子颜色同样没转线性（2026-09-29，分支 loco_unreal）

- 原因：`build_isaac_primitive_scene.py` 把 MJCF 的 rgba 直接写进 `diffuseColor`，和机器人（`stage.G1_LOOKS`）是同一个错误。纸箱 0.86 / 0.78 / 0.62 渲得近白。
- 修法：先从 sRGB 转成线性。纸箱 0.71 / 0.57 / 0.34，台子 0.17 / 0.19 / 0.21；地面 `GROUND_RGB` 本来就是按线性值选的，不变。
- 新 stage 是 `box_platform_srgb`。旧的 `box_platform` 留在共享资产缓存里，给用它渲过的运行。
- 对照：`outputs/isaac_ssaa_20260929/box_platform_video/before_after.jpg`，纸箱从近白变成和 MuJoCo 一致的纸板色。
