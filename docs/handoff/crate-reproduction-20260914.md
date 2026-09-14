# 双臂 main 与 6/10 实验复现核对

用户反馈协作者 50 次尝试全部失败。协作者实际命令、版本和失败日志尚未提供，
因此这份核对不能确定其 0/50 的原因。

## 已核实

- `git fetch origin` 后，远端 main 为 `0f9e0f2`，包含连续动作提交 `797371c`。
- `outputs/crate_continuous_motion_20260914/final10/config_snapshot.json`
  记录的 crate 与 SONIC backend 源码哈希均与该 main 相同。
- `configs/tasks/crate_visual_low_reach.json` 的解析结果与实验的 `task_config.json` 相同。
- 本机默认二进制与实验指定二进制 SHA-256 都是
  `4add30143c97a0214c1937802d25f65350a6ca9c7a15c5ac4a9053e392e4713f`，
  都包含 `POSE_UPPER_OWNERSHIP_V2`。这不证明协作者已使用相同构建。
- 用户指南已指向 `--expert visual`，但裸 `run_crate_pilot` 仍默认 `wall`。
  旧 shell 诊断入口与新 visual 专家并非同一策略。

## 发现并修复的场景差异

旧指南从默认 `crate_bimanual` 场景直接降桌高，遗漏实验中的 **4 cm 桌内 inset**。
编译后比较 22 组机器人/物体/接触参数，差异集中在篮子初始 x：

| 场景 | 篮子初始 xyz（m，XML 精度） |
|---|---|
| 6/10 实验 | `[-0.328008, 0, 0.387]` |
| 旧指南构建结果 | `[-0.368008, 0, 0.387]` |

桌面高度均为 0.30 m，解析后的控制配置相同。4 cm 改变接近、支撑与抓取几何，
属于需要消除的复现差异；没有对照实验就不能断言它独自导致 0/50。
修正后重新编译，以上 22 组物理字段全部一致；报告在
`outputs/bottle_continuous_20260914/crate_corrected_table30/audit.json`。
原始比较保留在 `outputs/bottle_continuous_20260914/crate_recipe_audit/audit.json`。

`bimanual_scene --crate-setback 0.04` 现在显式设置 XML 和 reset 元数据，
[当前指南](../crate_collection.md#2-环境与场景)使用独立的 depth04 场景目录。
runner 拒绝 wall 搭配 whole-body 配置，并在启动时打印实际专家、控制模式和桌高。

## 协作者核对顺序

1. 更新 main；按安装指南确认 v1.1、POSE ownership 补丁与重新编译的二进制。
2. 使用指南的 depth04 构建步骤和完整 `--expert visual --config ...` 命令。
3. 先跑固定 seed 41 的 10 次尝试，保留全部失败、manifest 和运行快照。
4. 对照 `commands.jsonl`、`task_config.json`、场景副本和 `deploy_receipts.jsonl`。
5. 输入身份一致后，再按初始化、接近、接触、倾斜、抬升与站直失败分类比较。

6/10 是一个固定小批次的观测，不是 60% 成功率承诺。正式 visual `--record`
仍未验证，不能通过改用旧 wall 录制入口来复现这批动作。
