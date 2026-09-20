# 双臂改动与新物流间场景同步（2026-09-20）

分支：`main`；本记录覆盖 `7710591` 之后此前未提交的双臂迁移工作，以及本次场景资产整理。
GitHub 发布以包含本记录的提交为准；本地实验产物不是仓库文件。

## 纳入范围

| 改动 | 当前说明 / 入口 |
|---|---|
| 箱沿几何、实际站位下的静态全身 IK、候选碰撞检查 | [箱沿规划](../box_rim_planning.md)，`box_rim_*` 模块与 `configs/planning/box_rim_example.json` |
| 沿用原双臂 expert 接近、闭合、试提的独立仿真入口 | [箱沿执行](../box_rim_execution.md)，`scripts/probe_box_grasp.py` |
| 空地固定 ghost、测量判据、条件可达域与箱沿区间查询 | [可达实验](../bimanual_reachability.md)，`probe_ghost_reach.py` / `map_ghost_reachability.py` |
| 原 expert 共享站稳准备、录像边界和状态诊断 | `expert.prepare_stationary_ready`、`round_runner.prime`、ghost CLI；[对照记录](g1-original-expert-task-boundary-20260920.md) |
| `CoordinatedTarget`、reach-only 和调用方 IK 关节限制 | `coordinated_reach.py` / `primitives/whole_body_ik.py`；不改变默认蓝篮目标几何 |
| Reset 输入先校验，拒绝错误的 29 关节数据及起始索引 | `sim/privileged_pub.py`，`test_episode_reset.py`；非法输入不应先改变场景 |
| 室内固定第三视角 | [录像机位](../rendering.md#video-export)，`scene_meta.video_camera` 与相机回归测试 |
| 新物流间 MuJoCo / Isaac 场景及源文件 | [场景包](../../assets/scene/logistics_room_box_v1/README.md)、[场景说明与尺寸](../blender_scenes.md) |
| 四份已有实验配置 | `agent_single_arm.yaml`、`agent_bimanual.yaml`、`crate_doc_walkthrough.yaml`、`crate_twitch_probe.yaml`；归档现有配置，不新增成功率主张 |

相关测试、生成字段表、代码导航和此前的实验 handoff 一并纳入。
根 README 提供当前功能与 LFS 下载入口；原根 HANDOFF 全文移至
[场景开发历史](logistics-room-development-history-20260920.md)，当前 HANDOFF 只保留入口、已验证边界及下一步。

## 场景与尺寸

新纸箱外尺寸 **0.450 × 0.320 × 0.330 m**，平台顶面 z=0.371 m、箱沿 z=0.701 m。
旧蓝篮当前使用约 **0.403 × 0.270 × 0.170 m**；原始约 0.605 × 0.405 × 0.170 m 的资产
已在水平两轴缩小，不能用原始尺寸解释当前双臂 case。
实际视觉/碰撞边界和纸箱壁厚的区别在场景说明及资产 manifest 中记录。

大型网格、贴图、USD、Blender 源文件使用 Git LFS；XML/JSON/说明直接版本管理。
`outputs/` 中源资产保留，通过复制和依赖整理形成版本化场景包，禁止跟踪运行目录或 symlink。
场景加载/完整性检查证明依赖可解析，不等于新的物理抓取成功。

## 保留在本地的内容

- `outputs/`、`third_party/`、`datasets/`、视频和运行缓存：按 AGENTS 的运行路径规则不入 Git。
- `.serena/`、`node_modules/`、仅含 headroom-ai 的 `package.json` / `package-lock.json`：本机辅助工具。
- `uv.lock`：只有 Python 3.14 头信息的本机生成文件，不是项目仿真环境锁文件。
- `rite_config_snapshot(p`：历史命令残留的重复源码文件，不属于代码入口。
- `assets/blender/textures/` 中不属于场景重建依赖的重复解包文件；必要资源由场景包提供。

这些文件保留原位，不为清理工作区而删除。需要的原始 `.blend` 与实际资源依赖单独按路径纳入。

## 验证、结果与限制

此前可达域实现的 16 项相关测试和完整 CPU 检查通过；14 个代表点实测仅 2 个通过，
5 个超时、7 个被原 expert 支撑检查拒绝。没有因上传场景而重新标记这些结果。
本次最终提交执行安装的 commit hook（仓库检查与完整默认 CPU 测试）；
四份纳入的实验配置均通过 `scripts/pipeline.py --dry-run`（没有启动仿真或采集）。
场景专项检查及上传核实见本次交付结果。

场景专项检查：

- 105 个场景负载文件的哈希与清单一致；107 个 MJCF 文件引用为可解析的相对路径。
  房间/纸箱资源随包分发，G1/Dex1 网格仍来自标准 SONIC 安装。
- MuJoCo CPU 编译通过：36 bodies、121 geoms、nq=47、nv=45、nu=33。
- USD 通过 Blender 的 pxr API 打开，49 个贴图引用可解析；纸箱世界包围盒与记录一致。
- 运行目录准备工具生成 37 bodies、121 geoms、nq=47 的场景，新增的是平台的 identity body，
  没有移动平台碰撞体；表面高度 0.371 m、箱体 qpos 地址 40 及 metadata 均从编译结果核实。
- Blender 源准备/贴图重映射和 MJCF 相对引用检查通过，未重新执行完整 Blender 场景导出。
  未启动 GPU/Isaac RTX 渲染或新增抓取试验。完整 MuJoCo 重建仍需要清单指明的校准 base 输入。


下一步是实际停车姿态下的候选复核、闭环跟踪误差与箱沿接触/负载验证；
当前不声明完整五维动态可达域，也不声明新纸箱已稳定搬运成功。
