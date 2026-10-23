# 历史记录与开发者索引

主线（场景 RSI）的文档和已归档的 VLA 文档清单都在[仓库 README](../README.md)。这一页只收 README 不列的东西：给开发者的入口，以及按时间留下的交接记录和归档。

## 开发者入口

- [AGENTS.md](../AGENTS.md)：改代码的共同约定，从这里开始。
- [代码导航](reference/change-map.md)：按要改的东西找代码、配置和测试。
- [开发规范](reference/standards.md)：模块边界、接口和配置约定、证据规则。
- [会话流程](agents/session-workflow.md)：验证范围、完成标准和交接。
- [文档维护规则](agents/doc-ownership.md)：每类内容写在哪一页，以及面向读者的页面怎么写。
- [STATE](../STATE.md)、[TODO](../TODO.md)、[DECISIONS](../DECISIONS.md)：当前进度与卡点、任务账本、每次决策的理由与被否决的方案。
- 流水线的结构（一次运行一份实验文件、每个阶段一个脚本、并行就是多开终端）怎么定下来的：[2026-09-21 重构](handoff/refactor-flat-bash-20260921.md)；更早的 [2026-09-16](handoff/infra-plan-20260916.md)、[2026-09-17](handoff/refactor-flat-scripts-20260917.md) 两次只作历史。

## 台账

场景 RSI 的实测数字和结论在[进展报告](handoff/scene-rsi-report-20261019.md)和[方案](handoff/scene-rsi-plan-20261016.md)里。已归档的 VLA 线，每一轮实验的实测数字和结论：[bottle](ledger/bottle.md)、[crate 与 warehouse](ledger/crate.md)、[pick-and-place](ledger/pick_place.md)、[ScaleBFM](ledger/scalebfm.md)、[Blender 场景](ledger/blender_scenes.md)、[上一次重构](ledger/refactor.md)。

## 排查线索

- [隐性失败](reference/silent_failures.md)：历史上的排查线索，路径和默认值要对照代码逐项核实。
- [F10：exporter 静默丢帧](handoff/exporter-frame-loss-20260917.md)：仍未关闭，根因没找到；也是 2026-09-17 冷启动走查 11 条发现（F1–F11）的索引。
- [双臂评测开局的突然动作](handoff/crate-eval-prelude-twitch-20260917.md)：为什么 deploy 的第一局会先蹲下再站起，以及试过但无效的五种修法。

## 双臂可达性的历次记录

- [G1 能力包络](handoff/g1-capability-envelope-20260921.md)：[双臂前伸包络](reach_envelope.md)那张图的测量与推断。
- [五维联合实测](handoff/g1-joint-reach-search-20260920.md)：1,277 次尝试、原生复核、倾角对照与独立复测。
- [早期条件切片](handoff/g1-reachability-atlas-20260920.md)：最初 atlas 的覆盖范围与局限。
- [平地 ghost 手预览](handoff/g1-reachability-preview-20260920.md)：可达实验的环境和采样范围，只展示目标分布。
- [14 组动态伸手](handoff/g1-ghost-reach-execution-20260920.md)：6 组成功，8 组在伸手阶段超时；[初始化诊断](handoff/g1-startup-jump-diagnosis-20260920.md)确认了初始化缺陷。
- [修正重置后的同 14 组重跑](handoff/g1-ghost-reach-fixed-reset-repeat-20260920.md)：4 组成功、9 组超时、1 组初始化倒地，启动同步问题当时没有解决。
- [原专家实跑与录像起点修正](handoff/g1-original-expert-task-boundary-20260920.md)：原版行走抓取成功；前一轮见[同 14 组重录](handoff/g1-shared-expert-ready-repeat-20260920.md)。

## 交接与归档

- `handoff/`：各阶段的交接记录和计划，文件名带日期。
- [技能的接口设计](handoff/primitive-contracts-design-20260928.md)：每个技能的前置条件、参数和返回、计划执行器（2026-09-28）。
- [scene_discovery 决策](handoff/scene-discovery-plan-20261002.md)、[场景文件格式](handoff/scene-file-format-20261003.md)、[put_down_single 技能](handoff/put-down-single-20261003.md)：从一个场景自动发现任务、补技能、采集数据的设计和第一个新技能（2026-10-02 起）。
- [仿真数据零样本上真机的设计](handoff/real-robot-sim2real-plan-20261006.md)：数据集 50 fps、官方推理客户端结构、Dex1 单位、`src/real_robot/` 的结构与后续任务（2026-10-06）。
- [warehouse 补渲时随机外观的计划](handoff/appearance-randomization-plan-20261008.md)：随机哪些属性、训练与留出范围、实现步骤、之后怎么测；光照开关已实现（2026-10-08，TODO 21.3.1）。
- [GraspNet 物体上 IK 规划与 SIMPLE 的 CuRobo 规划对比](handoff/graspnet-bottle-planner-comparison-20261008.md)：graspnet-baseline 抓取、bottle 专家执行、65 个物体的左右对比视频（2026-10-08，TODO 21.3.4）；10-09 补了 ghost 和固定抓取对比。
- [离线抓取缓存替代 graspnet 感知](handoff/grasp-cache-20261010.md)：物体网格上采反向法线抓取、只有手的 MuJoCo 模型合爪抬晃验证、在线按测得位姿变换；65 个物体 × 3 朝向的验收对照 21.3.4（2026-10-10，TODO 23.2）。
- [SONIC 对 ScaleBFM 控制器对比](handoff/wbc-compare-20261012.md)：只换全身控制器重跑 65 物体抓取线；五个范围决定、手指律统一、27 格开环探针与偏置常数、手程序移植到 ScaleBFM；结果：受控臂 129 局配对 SONIC 48 对 ScaleBFM 77（p < 0.001），SONIC 的偏置常数和腰限修复都不改成功率；ScaleBFM 的姿态交付包络（156 姿态探针，31 可交付）预测它的成败，按包络选抓取的放开臂 104/129（§8）；空地终点线刹车（§9）：越线惯性 ScaleBFM 0.5–4.5 cm 对 SONIC 4–25 cm，不越线程序脚尖停线前 4–8 对 13–22 cm
- [场景驱动的 RSI 方案](handoff/scene-rsi-plan-20261016.md)：只给场景和 WBC（mars + ScaleBFM），外环发现任务、分桶、选缺口，内环用 warehouse 的闭环写技能并统计验收；D1–D12 用户 2026-10-16 全部按推荐定
- [会换做法、知道自己身体的抓取进化](handoff/scene-rsi-strategy-plan-20261021.md)：身体模型、准备动作层、定向验收、“换个做法”提议方向、外环按出路频次排序（TODO 33，用户 2026-10-21：五点都做）
- [场景 RSI 进化记录](handoff/scene-rsi-report-20261019.md)：给导师和合作者看的进度报告，通俗写法：系统怎么找任务、写技能，15 次被采纳的改动各自根据什么、和上一版差在哪（网页版带前后对比视频），做成的 62 个任务和偶尔成的 12 个，没成的、人插手的、判错的（2026-10-19）
- [从原子技能开始的抓取进化：记录与复现](handoff/grasp-atomic-line-20261023.md)：`grasp_atomic` 四次运行（空种子 → 封存集 38 / 50）、每版改了什么、现任还败在哪、从空种子跑和续跑的命令、已做好没跑的部件（2026-10-23）
- [NVIDIA ASPIRE v1.0 审读](handoff/aspire-v1-review-20261019.md)：拉下来的代码和流程哪些能补我们的场景 RSI（经验库、打分与验证、错误码），哪些不能照搬（无统计、BEHAVIOR 多次尝试计分、靠自觉的隔离）；核实过的缺陷与建议（2026-10-19）
- [ScaleBFM warehouse 专家自主重标定方案](handoff/scalebfm-warehouse-rsi-20261014.md)：TODO 25.9.1，借 RoboRSI 的骨架，补控制器包络、干预归因和配对统计验收三环；三个起点池（诊断 / 验收 / 封存）；待用户定 D1–D7
- [ScaleBFM bottle 专家自主重标定](handoff/scalebfm-bottle-rsi-20261015.md)：TODO 26.3，warehouse 那套 RSI 闭环换到 bottle 任务的第二个实例；适配层、起点池、停止线（比 SONIC 多成 ≥ 3）和视频导出
- [SONIC → ScaleBFM 替换评测计划](handoff/controller-compare-20261013.md)：用户逐轮定的 35 个决定与开跑前冻结的判据；范围是仿真采集、仿真评测和真机；L1 原语层（bottle B1–B8、warehouse W0–W8、复杂动作 C1–C7、mode 扫描与切换、真机扰动 a–h、估计位姿）是关口，过了才移植专家跑 L2 配对（每场景 ≥ 40 局）；最后是部署程序和真机冒烟
- [单臂主线接 CuRobo 出路线的计划](handoff/single-arm-curobo-route-plan-20261009.md)：去掉原地抬手、CuRobo 避障直达预抓，每一条怎么验证没 bug、有效、不变坏（2026-10-09）；10-10 实现为 `reach.simple.arm_route: curobo`，默认关，关节空间目标两批 71/80 对基线 69/80。
- [归档索引](archive/INDEX.md)：冻结的原始报告和被取代的入口文档，包括 2026-09-14 的[维护路线图](archive/roadmap_maintenance_20260914.md)。
- [第一次迁移](handoff/agent-maintenance-plan.md)、[维护交付](handoff/maintenance-completion.md)、[规范审计与探针](handoff/agent-guideline-completion-20260914.md)。

较新的说法或撤回不会自动比旧记录更可信。保留原始记录，链接到更正，并写清新核实的观察适用的范围。
