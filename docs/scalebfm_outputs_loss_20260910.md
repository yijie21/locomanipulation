# ScaleBFM outputs删除后的影响核查

核查日期2026-09-10。现场事实：outputs目录为空；未找到ABCD运行进程；3项test_bfm_pelvis_ablation测试通过。未启动GPU、未重新采集或训练。

## 保留

src/teleop_free/scalebfm内核心专家/VLA/ABCD干预代码、tests、configs和docs/HANDOFF仍在。许多文件仍为Git untracked；存在不等于已提交或有版本备份。ABCD关键参数可由代码与交接记录恢复。

## 缺失

ABCD实验目录的run.py、PLAN.md、plan.json、progress.json、逐场状态、轨迹、日志、源码快照和结果；yplus200微调checkpoint-3000、200条数据、100场原始评估记录；小视频和ZIP；outputs/scene构建场景及旧实验的runtime_assets.json/base_model_snapshots.json。

因此不是完全没有代码可用，但目前不能直接运行或断点续跑。试验编排脚本放在outputs里，也一起丢失；这部分可据对话重建，不能把重建的计划当作原始实验结果。

## 结果证据边界

旧VLA25/100及分层数值仍有docs/HANDOFF文字记录；原始逐集证据已缺失。
ABCD最后一次在对话中核实是2/120，不能据此断言删除时只完成2场，也不能认定已完成；删除前最终进度/结果现在无法从本地验证。不能编造统计。

## 恢复顺序（未执行）

先查外部备份及用户已下载文件；若无微调checkpoint备份，需重建场景/依赖清单和训练数据、重新训练后才可重做对照。基础模型缓存不等于已训练的这个checkpoint；其它SONIC exporter会话也不能当作这200条ScaleBFM数据。
将编排脚本/计划移到src/scripts/docs等源码位置，输出目录只保存实验产物；之后保存独立备份。重建结果使用新实验ID，与遗失的实验明确区分。

删除原因以docs/STATE.md INCIDENT的另一会话记录为依据：合并误跟踪的outputs符号链接；本次独立核查确认目录与文件缺失，未尝试块设备级恢复。

## 路径核对

- 存在：`src/teleop_free/scalebfm/visual_pelvis_ablation.py`
- 存在：`tests/test_bfm_pelvis_ablation.py`
- 存在：`configs/gr00t_bfm_visual_2cam.py`
- 缺失：`outputs/experiments/scalebfm_pelvis_abcd_20260909/run.py`
- 缺失：`outputs/experiments/scalebfm_pelvis_abcd_20260909/progress.json`
- 缺失：`outputs/experiments/scalebfm_yplus200_s17_20260909/model/checkpoint-3000`
- 缺失：`outputs/scene/sim/scene_locomani.xml`
- 缺失：`outputs/experiments/scalebfm_visual_goal_20260908/runtime_assets.json`
- 缺失：`outputs/experiments/scalebfm_visual_goal_20260908/base_model_snapshots.json`
