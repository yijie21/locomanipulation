# 真实感渲染全景调研（2026-08）

> 目标：替换/增强 MuJoCo 原生画面，做到"够真够炫酷"，同时服务 sim2real。
> 前提架构：渲染器在我们的管线里天然可插拔 —— exporter 和 VLA 从 :5555 拿图，
> 物理/控制/deploy 不动（ROADMAP 镜像方案）；根位姿通道已在实现中，落地后
> 支持**离线重渲染**（不受 50Hz 墙钟约束，一条轨迹渲 N 种外观）。
> 本机：2×RTX 5090（32GB）、Vulkan/GL/OptiX 可用、**无 docker-in-docker**。

## 一、表征轴：四个家族

### 1. 网格 + 光栅化/路径追踪（经典管线）
制作方式与游戏/影视相同：资产 + 材质 + 灯光 + 渲染器。
- **上限**：路径追踪可达影视级；**代价**：资产与灯光的美术工作量决定一切，
  资产差则再好的渲染器也假。
- 平台：Isaac Sim（RTX/OptiX 路径追踪 + vMaterials）、UE5（Lumen 实时 GI /
  Nanite / Path Tracer）、Blender Cycles（离线）、Genesis Nyx（见下）。

### 2. 真实场景重建：3DGS（NeRF 已被取代）
手机视频 1–2 分钟 → COLMAP → 3DGS，渲染误差直接对真实照片最小化，
**照片级是构造保证的**。
- **LEGS（arXiv:2606.01458）在我们同款栈（G1+SONIC+GR00T）上验证过**：
  mesh 前景（机器人+物体）深度合成到 3DGS 背景 + 两段式颜色标定，
  纯仿真数据零样本上真机，且胜过 mesh-only 渲染 —— 他们的核心结论就是
  "photorealistic rendering is a key enabler"。
- 生态（2024–2026 爆发）：SplatSim（换皮 mesh→splat，86% 零样本 sim2real）、
  RoboSplat（编辑重建场景做六类增广，单演示 87.8%）、GSWorld（闭环照片级
  manipulation 套件）、RoboGSim（real2sim2real）、RL-GSBridge。
- **限制**：背景静态；反光/透明差；换场景 = 重新扫一段视频（便宜但非零）。

### 3. 生成式重渲染：扩散/视频世界模型
- **Cosmos Transfer（NVIDIA，2503.14492；2026 已到 Transfer 2.x + 蒸馏版）**：
  输入 sim 的结构化控制信号（**depth / segmentation / edge / 视频本身**），
  输出保结构的照片级视频。**不需要重建场景、不需要接任何引擎** ——
  对已录轨迹的渲染帧直接做风格提升；NVIDIA 官方教程就是
  "Isaac 渲染 → Cosmos 增广" 这条 sim2real 路线，Agility 已采用。
  限制：逐帧扩散推理成本；时序一致性要验收；细小物体（瓶口、指尖）
  的保真需要抽查 —— 对 manipulation 是风险点，必须先做小样本验证。
- **DreamGen / GR00T-Dreams（2505.12705）**：视频世界模型直接"梦"出新轨迹
  （neural trajectories，300k 条/827 小时规模），用 IDM 反标 action。
  注意这是**另一个轴**：它生成新行为/新环境的数据，不是重渲染我们已有轨迹。
  与我们 scripted 引擎是互补而非替代 —— 留作 scaling 阶段的选项。

### 4. 混合合成（LEGS 配方）
mesh 前景（物理照常）+ 3DGS 背景（照片级）+ 颜色标定。
兼得动态物理与照片级背景，是当前 humanoid sim2real 的已验证配方。

## 二、平台轴

| 平台 | 真实感上限 | 与我们架构的接口成本 | 关键事实 |
|---|---|---|---|
| **Isaac Sim** | 高（RTX 路径追踪 + vMaterials） | **低-中**：SIMPLE（2606.08278）已验证 "MuJoCo 物理 + Isaac **离线重放**渲染" 正是我们的镜像架构；pip 装、headless 可跑，不需 docker | Newton 集成进行中；Replicator 域随机化现成 |
| **UE5** | 很高（Lumen/Nanite，游戏级资产生态 Fab/Megascans） | **高**：C++ 工程，MJCF→UE 资产转换，headless Linux 渲染在容器里可行（Vulkan ✓）但折腾 | UnrealRoboticsLab（ICRA 2026）证明 UE 内嵌 MuJoCo 可行；**炫酷上限最高** |
| **Genesis / Genesis World** | 中-高（Nyx 路径追踪，目标 4ms@1080p） | **不适配**：它是物理+渲染一体平台，render-only 用法不是其设计；换它 = 换物理 = 丢 SONIC/MuJoCo 栈 | 开源版有 OptiX 光追；Genesis World 1.0 走商业部署路线（2026 底目标客户交付）。**观望，不采用** |
| **Blender Cycles** | 高（离线影视级） | 中：python API 全脚本化，离线重渲染天然契合；MJCF→Blender 导入要自己写；**本机已在用 Blender** | 免费；每帧秒级 —— 只适合离线，不能进实时闭环 |
| **Cosmos Transfer** | 高（生成式照片级） | **最低**：不接引擎，吃我们现有渲染帧+depth/seg 即可，纯离线后处理 | 蒸馏版速度已可用；需 VRAM（5090 32GB 应可跑推理） |
| Habitat 3 / OmniGibson | 中 | 中 | 场景数据集价值大于渲染器本身 |

## 三、AI 场景构建工作流

1. **EmbodiedGen —— 已在我们 third_party/ 里**：image/text→3D 资产→场景，
   当初桌面场景就是它生成的。重新生成场景的管线是现成的，只是下载量大。
2. **3DGS 扫描工作流（最便宜的"真"）**：手机绕拍 1–2 分钟 → COLMAP → 3DGS。
   LEGS 全程就这么做的。想要"真实实验室/厨房/客厅"，扫就行了。
3. **程序化照片级**：Infinigen / Infinigen Indoors（CVPR'24，Blender 基座，
   无限生成照片级室内，免费）—— 域随机化的量产机。
4. **LLM agentic 布局**：Holodeck（CVPR'24）→ SAGE（CVPR'26，agentic 自验证，
   能出 "Cyberpunk game den" 这种炫酷主题场景）、SceneCraft（LLM 写 Blender
   代码）、PhyScensis（物理增强布局）。文本→场景，适合快速铺任务多样性。
5. **单图→场景/物体**：SceneGen（单图一次前传出 3D 场景）、SAM3D / TRELLIS
   （单图→网格，LEGS 用 SAM3D 做前景物体）。

## 四、可用数据集 / 游戏资产

- **物体**：Objaverse-XL（千万级）、RoboCasa 资产（在用）、SAM3D 从照片抠
- **场景**：HSSD-200（Habitat 合成家居，高质量）、ScanNet++ / HM3D /
  ARKitScenes（真实扫描）、Infinigen 无限生成
- **游戏级**：UE Fab / Megascans（扫描材质）、City Sample（整城）、
  Sketchfab —— 只有走 UE 才能吃到这个生态
- **材质/光照**：PolyHaven HDRI（免费）、NVIDIA vMaterials（Isaac）

## 五、结论：三轨推荐（按投入产出排序）

**轨 A —— Cosmos Transfer 离线增广（先做，两周内可出效果）**
零引擎集成：拿我们已有的 843 条第三人称视频/三路 ego 帧 + MuJoCo 可导出的
depth/seg，离线过 Cosmos Transfer，直接得到照片级版本。同时回答两个问题：
画面炫不炫（demo 即刻升级）、VLA 在照片级数据上训练是否更好（可做 A/B）。
风险：小物体保真、时序一致性 —— 先抽 10 条验收再上量。

**轨 B —— LEGS 配方：3DGS 背景 + mesh 前景合成（主线，配合根位姿通道）**
同款栈已被 LEGS 验证到真机。步骤：手机扫目标场景 → 3DGS → 镜像渲染进程
订阅根位姿+物体位姿 → 前景光栅化+深度合成 → 发回 :5555 或离线重渲染。
这是"特定场景照片级还原"的正解，也是将来真机部署环境对齐的正解。
依赖：根位姿通道（99 处进行中）、相机内外参对齐、颜色标定。

**轨 C —— Isaac Sim 离线重放（量产域随机化，SIMPLE 配方）**
录 MuJoCo 轨迹 → Isaac 离线重放 + Replicator 随机化（材质/光照/视角/干扰物）
→ 批量渲染。SIMPLE 用这条路渲了 6000 episodes。适合 200 条轨迹 × N 外观的
数据放大。UE5 是它的高上限替代（资产生态最炫），但工程成本高一档，
建议 Isaac 先行、UE 留给 demo 大片。

**不采用**：Genesis（换物理，与 SONIC 栈冲突，商业化路线观望）、
NeRF（被 3DGS 取代）、实时闭环内换渲染器（没必要 —— 闭环评测用 MuJoCo
原生画面即可，照片级只需要出现在训练数据和 demo 里；除非将来要证明
"照片级闭环评测"本身）。

**与三条 claim 的关系**：轨 A/B 直接补 imitability 故事的视觉面
（teacher 的观测分布 vs 学生部署时的观测分布）；LEGS 已证明渲染轴的
增益，我们不必重证 —— 引用即可，把火力留给 cadence/imitability 轴。

## 主要来源

- Genesis World 1.0 blog: https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward
- Genesis repo: https://github.com/Genesis-Embodied-AI/genesis-world
- Cosmos Transfer1 (arXiv:2503.14492): https://arxiv.org/pdf/2503.14492
- NVIDIA sim-to-real Cosmos 教程: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html
- Isaac Sim Replicator+Cosmos: https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_cosmos.html
- DreamGen (arXiv:2505.12705): https://arxiv.org/abs/2505.12705 · https://github.com/nvidia/gr00t-dreams
- LEGS (arXiv:2606.01458)、SIMPLE (arXiv:2606.08278)：papers/ 下有全文
- SplatSim/RoboSplat 线: https://arxiv.org/abs/2504.13175 · GSWorld: https://arxiv.org/pdf/2510.20813 · RoboGSim: https://arxiv.org/pdf/2411.11839
- SAGE (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/papers/Xia_SAGE_Scalable_Agentic_3D_Scene_Generation_for_Embodied_AI_CVPR_2026_paper.pdf
- Holodeck (CVPR 2024): https://arxiv.org/pdf/2312.09067
- 3DGS 合成数据行业综述: https://blog.pebblous.ai/report/isaac-sim-3dgs-vla-synthetic-data-2026-04/en/
