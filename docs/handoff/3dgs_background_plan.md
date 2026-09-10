# 3DGS 背景集成:方案台账与实施指导

> **状态:未实施,零代码改动。**本文是 2026-09-10 的调研 + grilling 记录。
> 两条已裁定,渲染路线**故意不裁定** —— 所有可行方案并列记录,以后探索时再选。
> 写它的目的:下一个 session 不用重做调研就能直接开工。
> 归属:`docs/handoff/`(跨 session 计划),与 `refactor-plan.md`、`sonic_lockstep_collection_plan.md` 同类。

---

## 0. 一页速览

**想做的事**:物理仍由 MuJoCo 负责,桌子 / 箱子 / 瓶子 / 机器人继续按逐帧状态同步,
把**背景房间**换成 3D Gaussian Splatting 场景,离线重渲成训练图像。

**已裁定(2026-09-10,用户)**

| # | 裁定 | 影响 |
|---|---|---|
| D1 | 目的是 **appearance domain randomization**,不是单一真实房间的保真还原 | 需要 N 个场景而不是 1 个;资产首选批量数据集;重渲进训练集而不只是演示视频 |
| D2 | **不单独验收**这条线的效果,当作普通数据增强,只看总闭环成功率 | 不做 held-out 外观的闭环对照;风险见 §9 第 1 条 |

**未裁定(下次开工前要回答,清单见 §9)**:前景用哪个渲染器、资产用哪套、
渲几种外观、渲哪几路相机、物理一致性怎么保、阴影怎么办。

**三个最重要的事实**

1. **同一间房的 3DGS 版本已经在磁盘上**,和现在渲染用的 mesh 同坐标系,毫米级吻合。
   见 §3。这意味着打通链路的第一步不需要任何对齐工作。
2. **Isaac 4.5 完全不支持渲 splat**。原生支持从 5.0 才有且是残的,6.0 才真正可用。
   所以"把 3DGS 放进 Isaac 渲染"在当前环境下**不存在直接做法**,只能合成或升级。见 §5。
3. **今天采的 200 集全部带 `state_trace`**,34 个目录逐帧 qpos 齐全,
   这批数据不用重采就能重渲。ROADMAP 当年标记的"现在采的数据重渲染不了"这个地基缺口**已经补上**。

---

## 1. 目标与它在路线图里的位置

### 1.1 目的:多样性,不是保真

`docs/ROADMAP.md` 的「以后再说」一节在 2026-08-25 就 framed 过这件事,当时的判据是:

> **UE 还是 3DGS,判据是一个问题:真机会部署在一个特定的已知环境里吗?**
> 是 → 3DGS 更直接,扫进来,sim 的像素就是真机的像素;要跨环境泛化 → UE。

**这条判据的前提在 2026 年变了。**当时把"多样性"划给 UE,是因为 3DGS 被认为只能做单场景还原。
但 InteriorGS(1000 个室内场景、单位米、Z-up、带语义 OBB)这类批量资产出现之后,
3DGS 也能做多样性,而且资产准备成本远低于 UE 建场景。D1 就是在这个新前提下做的。

同一节还给出了回报的量级,这条依然成立:

> 同一段轨迹渲成 N 个外观 —— 域随机化从"重采 10 倍数据"变成"重渲 10 倍图像"。
> 这个乘数比并行采集那 3 倍大一个数量级,而且两者不冲突。

### 1.2 优先级

ROADMAP 的原话:**"优先级排在 DAgger 修复与闭环成功率之后:当前瓶颈是 covariate shift,不是渲染。"**
本文不改这个排序。2026-09-10 的失败结构仍然支持它:
`visible200` 的失败里 stand infeasible 占 72/101,不是外观问题。

### 1.3 既有的镜像 vs 内嵌裁定(仍然有效)

ROADMAP 已裁定**选镜像不选内嵌**:渲染器作为独立进程订阅状态、渲染相机、把图发回 `:5555`,
物理 / 控制 / deploy / 数据格式一行不动。UnrealRoboticsLab 那种把 MuJoCo 塞进渲染器进程的做法被明确否掉。
本文所有方案都在这个约束下。

---

## 2. 现在的渲染链路,以及接缝在哪

### 2.1 训练图像的来路

sim 进程渲染三路相机 → ZMQ `:5555` → exporter → LeRobot 数据集。
VLA 实际吃两路:`ego_view` 和 `right_wrist`(`configs/gr00t_sonic_2cam.py`)。
**闭环评测走同一条路**,也就是评测看到的是 MuJoCo 渲染器的像素。这一点决定了 §6 的路线取舍。

### 2.2 已有的两条离线重渲链路

| 入口 | 做什么 |
|---|---|
| `src/locomani/sim/isaac/mujoco_render.py --state-trace` | MuJoCo 重渲跟踪机位,并导出逐帧世界系相机轨迹 `cam_traj.npz`(`cam_pos [T,3]`、`cam_quat [T,4]` wxyz、`fovy`) |
| `src/locomani/sim/isaac/isaac_render.py --state-trace` | Isaac 重渲同一段,`replay.MujocoReplay` 每帧一次 `mj_forward`,关节 / 物体 / 相机全部从同一个状态读 |

两者共用 `replay.py`,所以"两个渲染器渲的是同一个瞬间"是构造保证的,不是巧合。
对齐实测:表面 mask IoU 0.852(1209 帧),瓶子质心 3.9 px 均值 / 5.0 px P90,无系统性偏移。

### 2.3 Isaac stage 的组装,背景是**单个 prim**

`src/locomani/sim/isaac/stage.py:assemble_stage()` 打开 `assets/isaac/scene/<label>/scene.usda`,
然后:机器人 USD 作为 **sublayer**(不是 reference,原因写在代码注释里),
可动物体按 `/World/<name>/geom` 引用并逐帧设位姿,
灯光是 dome 300 + 两盏 ceiling sphere light(强度 40000,z = 2.70 m)。

背景房间由 `src/locomani/sim/isaac_scene.py` 建成**一个** prim:`/World/Background`,
一张顶点色网格,来自 `mesh_model.ply`。摆放常数就三个:

```python
BG_ROTATE = (90.0, 0.0, 90.0)  # 资产是 y-up,场景是 z-up
BG_SCALE = 1.15
BG_SHIFT_XY = (-0.05, 0.25)  # 把房间的 footprint 对准 layout,而不是原点
```

外加一个把重建地板的"碗形"压平的 warp(`flatten_floor`),让视觉地板和 MuJoCo 的平面地板一致。

**接缝就在这里。**换背景 = 换 `/World/Background` 这一个 prim,前景那套完全不动。

### 2.4 前置条件:逐帧世界位姿(已满足)

ROADMAP 当年标了一条"【前置,且必须赶在下一轮采集之前】把机器人世界位姿记进逐帧数据",
理由是原始 exporter 的 32 列里没有任何一列是机器人在世界系的位置,重渲染摆不出场景。

**这条已经解决**:`privileged_pub.py` 的 `StateTrace` 在 sim 进程内写
`<label>_state_trace.npz`(`seq`、`qpos`、`qvel`),由 `EpisodeControlServer` 在与录像同一个边界启停,
`SIM_STATE_TRACE_DIR` 开启。`visible200_20260910` 的 34 个批次全部有,例如
`collect/state_trace_00/gym_1789023819_state_trace.npz` 是 `qpos (425, 47)`。

---

## 3. 本地已有的资产与工具

**这一节是本文最省时间的部分。**当前 Isaac 渲染里那间房不是 Isaac 自带的,
是 EmbodiedGen 生成的,而它当时同时产出了同一间房的 mesh 和 3DGS 两种表示。

目录:`third_party/EmbodiedGen/outputs/bg_scenes/scene_0000/`

| 文件 | 大小 | 内容 |
|---|---|---|
| `mesh_model.ply` | 92 MB | 2,212,151 顶点,顶点色网格。**当前渲染用的就是它** |
| `gs_model.ply` | 124 MB | 2,207,613 个 splat,gsplat 训练产物 |
| `gsplat_cfg.yml` | 1.3 KB | 训练配置,`max_steps: 4000`、`normalize_world_space: true` |
| `pano_image.png` | 616 KB | 全景图,两种表示共同的来源 |
| `prompt.txt` | — | `a tidy home kitchen with a wooden dining table in the middle, open floor space around the table` |

**两者同坐标系**,实测包围盒:

| | min | max |
|---|---|---|
| `gs_model.ply` | `[-3.071, -0.016, -3.090]` | `[2.244, 2.727, 2.474]` |
| `mesh_model.ply` | `[-3.065, -0.014, -3.089]` | `[2.242, 2.720, 2.471]` |

差异毫米级。**所以 §2.3 那三个摆放常数和 floor warp 可以直接搬到 splat 上**,
第一次打通链路不需要任何对齐工作。y 是高度轴,和 `BG_ROTATE = (90, 0, 90)` 一致。

### 3.1 这个 splat 的局限,必须先知道

`gs_model.ply` 每个 splat 只有 14 个属性:`x y z / f_dc_0..2 / opacity / scale_0..2 / rot_0..3`。
**没有 `f_rest_*`,即 SH degree 0,颜色与视角无关。**
而且 splat 数(2.207 M)≈ mesh 顶点数(2.212 M),两者是同一次全景重建的两种表示。

**结论:它不会比现在的 mesh 真实多少。**它的价值是**零成本验证链路**,
不是画质。真正的画质提升要换资产,见 §4。

### 3.2 本地工具

- **gsplat 1.5.3** 已装在 `/workspace/miniconda3/envs/embodiedgen`。
- `third_party/EmbodiedGen/embodied_gen/models/gs_model.py` 的
  `GaussianOperator.render(c2w, Ks, image_width, image_height)` 返回
  **rgb + depth + alpha** 三样,正好是深度合成需要的全部 buffer。
- `third_party/EmbodiedGen/embodied_gen/scripts/render_gs.py` 是它的 CLI,
  目前是环绕机位,但底层接受任意 `c2w`,把 `cam_traj.npz` 喂进去是小改动。
- `scene3d-cli --prompts "..." --output_dir outputs/bg_scenes/ --seed 0`
  可以再生成同规格的房间(mesh + gs 成对),这是一条**自产多样性**的路。

### 3.3 Isaac 环境现状

`third_party/SIMPLE/.venv` 里是 `isaacsim 4.5.0.0`,Python 3.10,
`isaacsim/exts` 下**没有任何 gaussian / nurec / splat 扩展**。
渲染配置在 `stage.configure_renderer()`:RTX real-time、关 sampled lighting、
关 async、`hydraEngine/waitIdle`、**故意不开 DLSS**(它从更低内部分辨率上采样)。
`SETTLE_STEPS = 2`,因为绝对世界相机位姿要两次 `world.step(render=True)` 才到渲染器。

---

## 4. 外部可用资产

| 资产 | 规模 | 格式 / 尺度 | License | 备注 |
|---|---|---|---|---|
| **InteriorGS** (`spatialverse/InteriorGS`) | 1000 个室内场景 | 压缩 `.ply`,**单位米,Z-up** | 自定义 InteriorGS License | **多样性路线的首选**。带 755 类语义 3D OBB、1024² 俯视占据图、户型图。重力对齐已经做好 |
| **`nvidia/PhysicalAI-Robotics-NuRec`** | 9 个室内环境 | USDZ,**metric**,77 GB | **CC-BY-4.0** | 唯一 license 干净且开箱即用的。自带 collision mesh、occupancy map、`.tum` 轨迹。需 Isaac **5.1+** |
| **SceneSplat-7K / 49K** | 7,916 / ~46,000 个**真实**室内 | 预训练 splat | 逐源继承,**全部 non-commercial** | 来自 ScanNet++/Replica/Hypersim/ARKitScenes/Matterport3D/DL3DV。PSNR 29.64 dB,depth-L1 0.035 m |
| INRIA mip-360 / T&T / DB | 室内只有 4 个 | 预训练模型 | non-commercial | **COLMAP 任意尺度,无重力对齐**,要自己拟合相似变换。不推荐 |
| SplatSim / GSWorld | 桌面实验室场景 | 训练好的 splat + COLMAP | SplatSim README 未声明 license | 机器人特定,规模小 |
| 自己拍(LEGS 路线) | 想要几个拍几个 | iPhone 1–2 分钟 → COLMAP → splatfacto 30k iter,单卡 10–15 分钟 | 自有 | 唯一能让 sim 的房间等于真机那间房的做法 |
| 自己生成(EmbodiedGen) | 想要几个生成几个 | `scene3d-cli`,mesh + gs 成对 | 本地已装 | 与现有链路天然兼容,但真实感受限于生成质量 |

---

## 5. Isaac 原生渲 3DGS 的版本真相

功能有三个名字指同一件事:**NuRec**(产品名)、**Neural Volume Rendering**(Isaac 文档页名)、
**Particle Fields**(Kit 110 的 OpenUSD `ParticleField` 几何类型)。

| Isaac Sim | GA | Kit | Python | 3DGS 状态 |
|---|---|---|---|---|
| **4.5.0** | 2025-02 | 106.5 | 3.10 | **完全没有**(本仓库装的就是它) |
| 5.0.0 | 2025-08-08 | 107.3.1 | 3.11 | 首次有,**compositing 实现** |
| 5.1.0 | 2025-10-21 | 107.3.3 | 3.11 | 同一套引擎 |
| **6.0.0** | 2026-06-04 | 110 | **3.12** | **原生 RTX ParticleField**,full path-tracing |
| 6.0.1 | 2026-06-22 | 110 | 3.12 | 加 `isaacsim.replicator.nurec_utils` |
| 6.1.0 | 2026-09-10 | 110.x | 3.12 | 最新 GA |

**5.x 的三个残缺**(升到 5.1 是买 bug 不是买功能):

- 深度遮挡有公开 bug:`isaac-sim/IsaacSim#108`,"远的挡住近的",标记 `pending-repo-update`,无公开修复确认。
- **只支持 pinhole 相机**。
- splat **不接收阴影**,只能用 ≤4 个 proxy mesh + Matte Object 兜底;
  官方自承 proxy 与 splat 几何不贴合会产生**白边**,而且反射质量完全由 proxy 决定。

**6.0+ 的原生**:RTX 直接 path-trace particle field,官方原话是
"supersedes the previous compositing-based approach";支持全部相机类型(含 defocus、motion blur);
**与 mesh 互相投影和收影**(cast 默认关,要勾 Casts Shadows);出现在反射折射里;
深度遮挡在光追内部解决,不是 2D 合成。

**升级代价**

- **Python 死绑**:4.5=3.10、5.x=3.11、6.x=**3.12**(`isaacsim==6.1.0.0` 的元数据是硬 `Requires-Python ==3.12.*`)。
  必须新建 venv,render venv 里每个 wheel 都要有 py3.12 版本。
- 6.0.0 **删掉了 `omni.isaac.*` 兼容层**。
- 6.0.0 把 **`isaacsim.sensors.camera`** 标 deprecated —— **本仓库正在用它**(`isaac_render.py`)。
- Linux 需 GLIBC 2.35+。GA 后 wheel 会滞后发布(6.0 GA 2026-06-04,wheel 到 06-19 才齐)。
- **资产格式**:`.ply` 不能直接读,必须用 **3DGRUT**(Apache-2.0)转 USDZ / ParticleField。
- **未验证**:找不到任何一手证据演示"pip wheel + headless 渲 NuRec"。当作很可能可行但未证实。

---

## 6. 可选路线(并列记录,不裁定)

四条都可行,差别主要在**前景用哪个渲染器**,以及**是否升级 Isaac**。

### 路线 A:MuJoCo 前景 + gsplat 背景合成

MuJoCo 渲前景(隐藏地板 / skybox,读深度得到 matte),gsplat 用同一组 `c2w` / `Ks` 渲背景,按深度合成。

- **优点**:训练集里唯一变的就是背景,**前景像素与闭环评测同分布**,没有第二个混淆变量;
  `mujoco_render.py` 已能从 state_trace 重渲;不需要每个 scene seed 建一次 USD 资产;
  不需要 Isaac venv;重渲快;不受 Isaac RTX "非位级可重现"的困扰。
- **缺点**:前景是 OpenGL 着色,没有 PBR / 光追,画面上限低于 Isaac;
  Isaac 那套已验证的对齐工作用不上。
- **适合**:D1(多样性)+ D2(不单独验收)。因为不单独验收,减少混淆变量的价值反而更高。

### 路线 B:Isaac 4.5 前景 + gsplat 背景合成(LEGS 路线)

Isaac 渲前景并输出深度(`distance_to_image_plane` annotator),隐藏 `/World/Background`,
gsplat 渲背景,逐像素按深度合成,再套 LEGS 的两段色彩标定。

- **优点**:前景最真;`isaac_render --state-trace` 已验证与 MuJoCo 对齐到 2 px;
  与 LEGS 已发表的做法同构,有论文背书。
- **缺点**:训练前景与评测前景来自两个渲染器;每个 batch 的 scene seed 要建一次 USD
  (`build_isaac_assets.sh` 按 scene_meta hash 缓存,采集时桌高随 SCENE_SEED 变);
  Isaac RTX 不是位级可重现(同脚本两次跑,13 帧无一相同,最大通道差 123);
  `SETTLE_STEPS = 2` 让 T_render 翻倍。
- **参考成本**:LEGS 报的是一种外观条件重渲 50 集 ≈ 0.1 GPU-h(4090)。

### 路线 C:升级 Isaac 6.1,原生 NuRec

- **优点**:唯一能让 splat 与 mesh **互相投影收影**的方案,图像最自洽;
  深度遮挡在光追里解决,没有合成接缝;支持全部相机类型;
  可以直接吃 `PhysicalAI-Robotics-NuRec` 的 metric USDZ(自带 collision mesh)。
- **缺点**:py3.12 新 venv、`isaacsim.sensors.camera` 迁移、`.ply` 要 3DGRUT 转 USDZ、
  headless pip wheel 无一手证据。属于大改动。
- **建议形态**:**独立 venv 里的实验**,永远不要就地升级现有 render venv。

### 路线 D:全 splat(SplatSim 路线)

连机器人和物体也用 splat 表示,按 sim 的关节变换刚性搬运 splat。

- **优点**:没有前景 / 背景合成问题,不存在深度接缝。
- **缺点**:与本仓库的 mesh 资产体系不兼容,要为 G1 和每个物体重建 splat;
  和"MuJoCo 物体与渲染同步"的既定架构冲突。**记录以备完整,不推荐**。

### 对比表

| | A: MuJoCo+gsplat | B: Isaac4.5+gsplat | C: Isaac 6.1 原生 | D: 全 splat |
|---|---|---|---|---|
| 要不要升 Isaac | 否 | 否 | **是,py3.12** | 否 |
| 前景与评测同分布 | **是** | 否 | 否 | 否 |
| 前景画质 | 低 | 高 | 高 | 中 |
| 深度接缝问题 | 有 | 有 | **无** | 无 |
| splat 收阴影 | 否 | 否 | **是** | 不适用 |
| 每 scene seed 建 USD | 否 | 是 | 是 | 否 |
| 可直接吃 `.ply` | **是** | **是** | 否,要转 USDZ | 是 |
| 新代码量 | 中 | 小(链路已有) | 大 | 大 |

---

## 7. 四个硬问题(每条路线都躲不掉)

### 7.1 物理一致性 —— 本仓库已经踩过一次

`isaac_scene.py` 的注释记着为什么当初换掉了自带的厨房背景:

> 那间房是个完整厨房,它的橱柜台面只存在于渲染里、不存在于 MuJoCo,
> 机器人会穿过自己看得见的家具。空壳才能让视觉与物理一致,每个物体在两边都存在。

**任何真实拍摄或批量下载的房间都有同样问题。**三种应对:

1. 只用**空房间 / 空区域**,把机器人活动范围限制在没有家具的地方。
2. 用带**语义 OBB**(InteriorGS)或 **collision mesh**(NVIDIA NuRec)的资产,把看得见的家具同步进 MuJoCo。
3. 只让背景出现在机器人到不了的远处,近处全是 mesh。

这是选资产时最该看的一栏,不是画质。

### 7.2 阴影

splat 的光照是**烘死的**。3DGRT 的原话:"Gaussian particles are not affected by light sources"。
Isaac ≤5.1 splat 不收阴影。合成路线里机器人在地板上的**接触阴影会整个消失**,
而这是"机器人站在地上"最强的视觉线索之一。

- 社区的三渲一乘方案(forum 365450,2026-04-02,Isaac 5.1.0,**NVIDIA 至今未回**):
  关阴影渲一次 RGB + 开阴影渲一次纯白 → OpenCV multiply。
  **只在默认 dome light 下成立**;换成 Distant / Sphere / Rect 灯,proxy 会渲成黑色漏进合成。
  本仓库的场景恰好用两盏 **ceiling sphere light**,所以这条路对我们**不通**。
- 可用的替代:matte shadow-catcher 地面做一趟 multiply pass(地面是平的且已知);
  或者直接画软 blob 阴影;或者走路线 C 让原生光追解决。

### 7.3 尺度与重力对齐

COLMAP 出来的场景是任意尺度、无重力对齐。业界三种解法:
ArUco 标定尺度(GSWorld)、LiDAR 手机扫描(VLK 用 Polycam)、
或者直接用已经是 metric 的资产(InteriorGS 单位米 Z-up;NVIDIA NuRec metric)。
NVIDIA 自己的 Marble 博客也承认 PLY→USDZ 仍需手动对齐和定尺度。

**本地那个 `gs_model.ply` 不受此困**,它和 mesh 同框,复用现成常数即可。

### 7.4 深度接缝与色彩

- **splat 没有唯一深度**:一条光线上多个 gaussian 贡献,任何 depth map 都是代理。
  gsplat 的 `ED` 是不透明度加权均值 z,在轮廓处有偏 → mesh 边缘与 splat 交界出现**光晕 / 毛边**。
- **sRGB vs linear**:RTX 把 ParticleField 当 display-referred sRGB,与 mesh 渲染的线性空间混用是典型静默 bug。
- **LEGS 的两段色彩标定可直接抄**:
  1. 每个 mesh 在 linear RGB 做对角缩放,使其渲染均值匹配扫描图在 SAM2 mask 内的均值,clip 到 `[0.5, 3.0]`;
  2. 用 24 色卡在两台相机各拍一次,最小二乘拟合 3×3 CCM。
  前提是两边都关 AE / AWB。**注意:这一段是为真机对齐设计的,D1 的多样性目的下第 2 步用不上。**
- **离采集轨迹越远画质越差**。NVIDIA 的 Fixer 博客里相机偏 3 m,PSNR 从 16.5809 到 16.6147,
  却被描述为"显著更真实" —— 说明 **PSNR 在这件事上没有判别力,只能看图**。

---

## 8. 可抄的现成实现

| 项目 | License | 可抄什么 |
|---|---|---|
| **gsplat** (`nerfstudio-project/gsplat`) | Apache-2.0 | `rasterization(..., render_mode="RGB+ED")` 一次给出 RGB + expected depth + alpha,正好是合成需要的三个 buffer。相机模型支持 pinhole / fisheye / FTheta |
| **j3soon/omni-3dgs-extension** | Apache-2.0 | 合成路线的参考实现:Omniverse 渲深度 → 3DGS 做深度测试 + alpha blend。老 Kit,但思路直接可用 |
| **NVIDIA 3DGRUT** (`nv-tlabs/3dgrut`) | Apache-2.0 | **PLY ⇄ USD ⇄ NuRec 转换器**,走路线 C 的必备工具。另有 Playground 演示 PBR mesh 插进 splat 场景 |
| **LEGS** (arXiv 2606.01458) | 论文 CC BY 4.0 | 与本仓库几乎同构的架构 + 两段色彩标定配方。**代码到 2026-09-10 仍是 Coming Soon** |
| **GSWorld** (arXiv 2510.20813) | 见仓库 | Gaussian-on-Mesh:把 gaussian 绑到 mesh 三角面,**遮挡变成 mesh 定义的**,结构上最干净的深度解 |
| **SplatSim** (arXiv 2409.10161) | README 未声明 | 路线 D 的参考,含训练好的 splat 与 COLMAP 数据 |

---

## 9. 待裁定清单(下次开工前回答)

1. **验收(D2 已裁定为"不单独验收",此处记录风险)**。
   闭环评测跑在 MuJoCo 且外观固定,所以这个评测**只能证明"没掉",证明不了"涨"**。
   现有的 `SCENE_RANDOM` 只改地板棋盘格色相、灯光方向和纯色材质,改不了贴图,
   和"换一个房间"不是同一个随机化轴。
   风险:涨了不知道是否因为它,没涨也不知道是否实现错了 —— 仓库已经吃过一次这种亏
   (单 seed 的 +8 后来被证明是噪声,`docs/ledger/bottle.md`)。
   **即使不验收效果,仍建议保留"实现正确性"闸门**:沿用已有的表面 mask IoU 与物体质心 px 误差,
   确认重渲的几何没错。这与验收效果是两回事。
2. **前景渲染器**:路线 A / B / C(§6)。
3. **资产**:InteriorGS(批量、metric、带语义 OBB)/ NVIDIA NuRec(license 干净、需 Isaac 5.1+)/
   自己拍 / EmbodiedGen 自产 / 先用本地那个跑通。
4. **渲几种外观、渲哪几路相机**:`ego_view` 和 `right_wrist` 都渲还是只渲 ego?
   腕相机贴近桌面,背景占比小,可能不值得。N 是多少?
5. **物理一致性策略**:§7.1 的三种应对选哪个。
6. **阴影**:§7.2 的三种应对选哪个,或先接受没有阴影。
7. **训练集怎么混**:MuJoCo 原图与重渲图的比例;是否保留原图以免自造域差。
8. **数据落地形式**:重渲图像是写进新的 LeRobot 数据集,还是作为同一数据集的额外 video key。

---

## 10. 不论选哪条路线都要做的骨架

### 10.1 最小可行第一步(不需要任何新资产,不需要对齐)

用本地的 `gs_model.ply` 把合成链路打通,只验证三件事:
相机约定对不对(`cam_traj.npz` 的 wxyz / USD 约定 → gsplat 的 `c2w`)、
深度合成对不对、边缘接缝什么样。
因为它与现在的 mesh 同框,**这一步的产出可以和现有渲染逐帧对比**,是最干净的对照。

### 10.2 代码落位(遵循 `docs/reference/standards.md`)

- 渲染代码属于 `src/locomani/sim/`。合成器建议 `src/locomani/sim/gs/`,
  与 `sim/isaac/` 平级,内部同样保持"所有重依赖 import 放在函数里"的写法(`stage.py` 的先例)。
- **所有旋钮进 config dataclass**,禁 `os.environ.get`(白名单在 `scripts/lint_env.py`)。
  新增一个 `GsBackgroundConfig` 之类,字段带注释写明含义与出处,
  `docs/reference/config_fields.md` 由 `scripts/gen_config_reference.py` 生成,不手写。
- 文件 ≤ 600 行,函数 ≤ 80 行。
- 一次性 runner 放 `outputs/experiments/<name>/launch.sh` 跑,源码提交到 `scripts/experiments/<name>.sh`。
- 改主代码的 PR 附 `scripts/gate.py compare` 输出。
- 新资产不入 git:`assets/isaac/` 已经是 gitignored 的先例,splat 资产同理。

### 10.3 机器资源注意

- 3DGS 渲染要 GPU。本机两张 5090,经常被采集 / 训练 / 另一个 session 占着,
  开工前看 `docs/STATE.md` 的 RUNNING 行和 `nvidia-smi`。
- CPU 配额只有 15.36 核(`/sys/fs/cgroup/cpu.max`,`nproc` 的 32 是宿主机),
  用 `scripts/cpu_headroom.sh` 判断余量,起 stack 需要 ≥ 6 核空闲。
- 本地 `gs_model.ply` 2.2 M splat,124 MB;InteriorGS 单场景规模相近,
  批量下载前先 `df -h /workspace`。

---

## 11. 参考

**调研原始记录**(含逐条链接与日期):本次调研写在 session scratchpad
`3dgs_isaac_research.md`,297 行,内容已全部吸收进本文,原文不入库。

主要一手来源:
[Isaac Sim 6.0 NuRec 文档](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_nurec.html) ·
[Omniverse Particle Fields](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/particle-fields.html) ·
[RTX 110.1 release notes](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer-release-notes/110_1.html) ·
[IsaacSim#108 遮挡 bug](https://github.com/isaac-sim/IsaacSim/issues/108) ·
[isaacsim on PyPI](https://pypi.org/project/isaacsim/) ·
[PhysicalAI-Robotics-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec) ·
[InteriorGS](https://github.com/manycore-research/InteriorGS) ·
[SceneSplat-7K](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k) ·
[LEGS](https://arxiv.org/abs/2606.01458) ·
[3DGRUT](https://github.com/nv-tlabs/3dgrut) ·
[gsplat](https://github.com/nerfstudio-project/gsplat) ·
[omni-3dgs-extension](https://github.com/j3soon/omni-3dgs-extension) ·
[GSWorld](https://github.com/luccachiang/GSWorld) ·
[shadow compositing 论坛帖](https://forums.developer.nvidia.com/t/casting-shadows-on-gaussian-splats-in-isaac-sim-via-shadow-compositing-need-feedback-help/365450) ·
[Fixer 博客](https://developer.nvidia.com/blog/how-to-enhance-3d-gaussian-reconstruction-quality-for-simulation/)

仓库内相关:
`docs/ROADMAP.md` 的「以后再说」一节(UE vs 3DGS 的旧判据、镜像不内嵌的裁定、回报估算)·
memory `isaac-render-mujoco-physics-plan`(离线重渲的完整设计与全部踩坑记录)·
memory `isaac-crate-render-pipeline` · `src/locomani/sim/isaac/` · `src/locomani/sim/isaac_scene.py`
