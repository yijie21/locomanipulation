# G1 五维联合采样与实际 reach（2026-09-20）

分支 `main` 工作树，基线 `fcf1b9d3186ab592b7f65fdebccb2fc69ddaa709`；本轮未提交、未推送。
结果根目录：`outputs/g1_joint_reach_search_20260920/`。
本轮接续[条件切片 atlas](g1-reachability-atlas-20260920.md)，完成五个变量联合变化的真实闭环试验，
没有修改原 expert 的控制逻辑或验收阈值。环境只有平地与自由站立的 G1，没有箱体、平台或负载。

## 范围、采样与标准

坐标沿用[可达实验](../bimanual_reachability.md)：以初始化站位为参考，x 向前、y 向左，h 为离地高度。
x 是双抓点中点的前向距离，不是脚尖到箱体前壁的距离。

| 变量 | 联合采样范围 |
|---|---|
| x | 0.15–0.75 m |
| y | −0.24–0.24 m |
| h | 0.50–0.80 m |
| 双点间距 d | 0.25–0.55 m |
| 连线偏角 yaw | −30–30° |

初始 908 条包括：512 个 scrambled Sobol 五维联合样本、完整 3⁵=243 网格、3 个历史成功参考点、
150 个纸箱专门组合。纸箱组合固定 h=0.701 m，d=0.44/0.45 m，
x=0.25/0.35/0.45/0.55/0.65 m，y=−0.12/0/0.12 m，yaw=−30/−15/0/15/30°。
初始倾角为向身体倾斜 30°。新纸箱外形 45×32×33 cm、壁厚 1 cm，
两侧壁中心间距约 44 cm，所以同时保留 44 和 45 cm 的目标。

随后完成 256 个成功／失败交界处的独立加密点；生成输入只使用真实进入 reach 且在线与原生判定一致的记录。
冻结的 `refinement/source_observations.json` 和阶段计划保存源身份与哈希；加密点不与初始点重复。
24 个位置再分别测 15°、45°、0°，共 72 次；这是离散倾角对照，不是连续朝向优化。
16 个目标各做两次独立复位复测，共 32 次，覆盖全部 7 个当时的争议目标、6 个已确认成功目标
（含至少两个纸箱目标）和两个随机对照。最后对 9 个仅有初始化／基础设施异常、尚无有效结论的身份补跑一次。
已有普通 reach 失败不因此重试。

每次独立准备初始站姿，释放初始化固定约束，通过 quiet admission 后才开始任务录像；初始化另存。
目标中心在单次执行中固定，不发送走路命令。每个目标最多 12 s 仿真时间，
双手位置误差各 ≤2 cm、姿态误差各 ≤15°，脚部水平漂移／抬升各 ≤3 cm，连续保持 ≥0.5 s。
原 expert 的控制器仍可在 observe 阶段拒绝目标；这不算真正执行 reach。

## 最终结果

| 阶段 | 尝试 | 实际 reach | 在线成功 | 在线＋原生确认成功 | 实际 reach 失败（在线） | 伸手前拒绝 | 初始化／基础设施异常 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 初始覆盖 | 908 | 562 | 101 | 100 | 461 | 340 | 6 |
| 边界加密 | 256 | 253 | 106 | 105 | 147 | 0 | 3 |
| 倾角对照 | 72 | 48 | 10 | 9 | 38 | 24 | 0 |
| 独立复测 | 32 | 29 | 18 | 16 | 11 | 2 | 1 |
| 异常补跑 | 9 | 6 | 2 | 2 | 4 | 3 | 0 |
| 总计 | **1,277** | **898** | **237** | **232** | **661** | **369** | **10** |

以双重证据分类，898 次实际 reach 中：232 次共同确认成功、656 次共同不成功、10 次判定分歧。
分歧包括 5 次在线成功但原生保持不足、5 次在线超时但原生达到保持标准；原始标签不改写。
原生审计使用 50 Hz MuJoCo qpos、真实工具坐标与脚位，独立验证两手同时满足标准的连续窗口；
碰撞、倒地及控制器状态仍采用执行时证据。

共 1,164 个不同五维坐标、1,236 个不同“坐标＋倾角”身份。
补跑后每个身份都有至少一次正常准入拒绝或实际执行结论，未决的纯基础设施异常身份为 0；
这不代表所有身份都已伸手，也不代表所有结论都没有临界分歧。
所有 1,277 条记录的走路命令总数为 0。

### 倾角与复测

72 次倾角对照中，9 次相对原来失败的目标获得在线＋原生共同确认成功；
只看原生运动学则有 11 次，额外两次仍是在线超时争议，不能混为共同确认。
例如 `initial:765` 在 30° 时超时，同坐标 `alternate_tilts:1/2` 在 15°/45° 时成功，
原生保持分别 0.54/0.66 s。各次均独立复位，不能仅凭这一组对照推断倾角变化的确定性收益。

16 个复测身份中，按原生二元结果：5 个三次均成功、2 个三次均不成功、7 个混合、2 个存在无二元结论的尝试。
同时要求在线也成功时，只有 `initial:541`、`initial:546`、`initial:831` 三个身份三次均获共同确认。
这是刻意包含边界与争议点的复测集，不是全域可靠性比例。
纸箱目标 `initial:831` 为 x=0.45 m、y=0、h=0.701 m、d=0.44 m、yaw=0、倾角30°，
其 `repeats:15/16` 均再次成功；另一已确认纸箱目标 `initial:772` 则出现成功／失败混合。

初始纸箱网格 150 条有 17 条共同确认成功（在线18条）、90条在线 reach 失败、42条准入拒绝。
成功分散于不同 x/y/yaw 组合，不能把成功点的外包盒或二维投影涂成连续可达区。

### 初始化与数据完整性

898 次实际 reach 的 executor 起点全部通过 quiet 复核，任务中没有混入准备 reset。
录像首帧与执行起点分开检查：`refinement:83` 的录像首帧速度 0.063988 m/s，略超0.06阈值；
20 ms 后执行起点为0.053577 m/s并通过，最终为碰撞失败。
另有未进入 reach 的 `initial:380/419` 首帧不通过；不据此重标有效执行结果。
7 条异常记录没有任务原生轨迹，保持为无运动学结论；其存在不影响其他正常试验。

`artifact_audit.json` 核对所有阶段 bank、逐条结果、汇总、checkpoint、progress 身份与数量，
共1,277条完全一致；5,134个引用产物均可读取／解析，无缺失、损坏或未引用文件。
NPZ 完整解压读取，原始 MP4 用 ffprobe 检查；交付预览额外做完整解码检查。
初始 shard2 的 launcher 在完整结果写入后退出143，原因未确定；当时产物完整性核验通过，
遗留的本次实例已通过 Stack launcher 清理。其余19个 launcher 均退出0。

## 报告、图和视频

根目录下 `analysis/report.html` 可筛选五轴、倾角、阶段、结果和原生审计状态，点击样本查看误差与身份。
顶部同时显示在线成功237与共同确认232；紫色表示判定分歧。
`analysis/samples.json`、`report.json`、`final_stats.json` 保留逐条及配对统计。
`joint_5d_projections.png` 为投影，`box_focus_measured.png` 为纸箱确切参数网格；均无可达域插值认证。
独立16,384个 Sobol 探针的归一化五维最近距离仅衡量几何覆盖，不能用于推算可达体积或成功率。

交付预览都保留完整任务时长、正常速度、首尾帧并通过完整解码；初始化与源录像保留。

| `videos/` 下的预览 | 时长 | 字节 |
|---|---:|---:|
| `box_three_attempts_preview.mp4`：831及两次独立复测 | 16.32 s | 438,964 |
| `box_forward45_initial836_preview.mp4`：45 cm间距、正对成功 | 5.76 s | 405,123 |
| `box_yaw15_initial837_preview.mp4`：同距离、偏角15°超时 | 12.48 s | 428,440 |
| `tilt30_initial765_preview.mp4`：倾角30°超时 | 12.48 s | 422,987 |
| `tilt15_alternate001_preview.mp4`：同抓点、倾角15°成功 | 3.76 s | 385,873 |
| `success_initial007_preview.mp4`：一般目标成功 | 5.60 s | 416,730 |
| `rejected_initial001_preview.mp4`：伸手前拒绝 | 1.28 s | 258,485 |

`g1_joint_reach_previews.zip` 为上述7个合规预览及说明、清单，共2,750,427 bytes；没有原始录像。
`delivery_manifest.json`、各视频的 `.verification.json` 和 `videos/concat_audit.json` 保存大小、哈希、时长与验证。
三次对照拼接保留三个源片段的全部204帧、16.32 s，不加速、不裁任务阶段。

## 实现、复现与验证

使用[联合采样入口](../../scripts/generate_joint_reach_bank.py)与
[实测入口](../../scripts/probe_ghost_reach.py)，实现位于
[采样模块](../../src/locomani/tasks/crate/joint_reach_sampling.py)和
[执行CLI](../../src/locomani/tasks/crate/ghost_reach_execution_cli.py)。
新增任意1..N目标集、索引／区间／分片选择、逐条原子保存、checkpoint、严格身份检查的resume；
`execution_started`仅表示调用executor，真实伸手必须检查`controller_phase="reach"`。

结果根目录保存 `initial_sampling.json`、`refinement_sampling.json`、各阶段bank、request与`phase_plan.json`。
各阶段分析快照保留为 `initial_analysis/`、`refinement_analysis/`、`alternate_analysis/`、`repeats_analysis/`。
`frozen_runtime/manifest.json`保存实验期间9个源码哈希及原文件，`runtime_completion.json`确认全部试验结束时仍一致。
仅在全部运行结束后格式化两个入口文件；`runtime_format_ast_audit.json`确认9个文件AST与冻结版本完全等价。
`deployment_assets.json`记录运行期间观察到的部署二进制、ONNX与配置的磁盘哈希及进程参数；
这是磁盘观测，不是对进程内存镜像的追溯证明。

离线重建（在仓库根目录，使用已保存的运行产物）：

```bash
REACH_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
REACH_OUT=outputs/g1_joint_reach_search_20260920
PYTHONPATH=src "$REACH_PY" "$REACH_OUT/audit_native.py" --phase initial --phase refinement --phase alternate_tilts --phase repeats --phase recovery
PYTHONPATH=src "$REACH_PY" "$REACH_OUT/analyze_search.py" --phase initial --phase refinement --phase alternate_tilts --phase repeats --phase recovery
PYTHONPATH=src "$REACH_PY" "$REACH_OUT/final_stats.py"
PYTHONPATH=src "$REACH_PY" "$REACH_OUT/plot_search.py"
python "$REACH_OUT/check_browser.py"
```

执行器12项、采样5项针对性测试此前通过；最终全阶段原生审计、产物审计、配对来源完整性检查、
真实Chrome交互检查、预览完整解码与拼接帧数检查通过。
最终命令 `make check` 通过：对108个变更路径执行完整验证，Ruff格式与lint、仓库／环境／文档／Mermaid／
配置引用检查全部通过，完整CPU测试集1,123项通过。
四个仿真实例锁及stack.json均已释放，无本次sim/deploy遗留；渲染使用独立管理的离线job。

这次得到的是有限范围内、当前原expert、固定站位和空载条件下的实测证据，
不是G1机械极限或任意箱子的连续可达保证；尚未测试平台碰撞、合爪、负载和搬运。
后续规划应把未测点保留为未知，并利用复测和执行反馈验证候选，不能从成功点外包范围直接接受抓取。
