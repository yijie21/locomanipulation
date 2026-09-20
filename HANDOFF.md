# HANDOFF — 双臂搬箱与可达域

当前分支：`main`。本次整理及场景发布的范围、验证和排除项见
[同步记录](docs/handoff/scene-and-bimanual-sync-20260920.md)。

## 当前入口

- [仓库 README](README.md)：功能与安装导航。
- [Blender 场景](docs/blender_scenes.md)：新物流间的 MuJoCo / Isaac 资产与加载方式。
- [箱沿规划](docs/box_rim_planning.md)和[仿真执行](docs/box_rim_execution.md)：
  从箱体尺寸、位姿和机器人实测站位生成双抓点，再复用原双臂 expert。
- [双臂可达实验](docs/bimanual_reachability.md)：固定脚站姿的 ghost reach、条件 IK 候选域与实测验证。
- [STATE](docs/STATE.md)：当前工作和共享资源状态。

## 已验证与限制

- 原 expert 的站稳准备由同一函数维护；初始化和任务录像分开。
  [原版对照](docs/handoff/g1-original-expert-task-boundary-20260920.md)保留原始证据。
- 第一版可达域计算 7 个条件切片、1,120 个目标，448 个 IK 候选；
  14 个代表点实测 2 成功、5 超时、7 被支撑条件拒绝。
  [可达域记录](docs/handoff/g1-reachability-atlas-20260920.md)包含视频、误差与输入身份。
- IK 候选不等于动态可靠区域；新纸箱的端到端抓取/负载搬运尚未证明。
  下一步在实际停车状态复核候选，并完善闭环跟踪、箱沿接触和负载验证。

## 历史

原根目录 HANDOFF 的场景导入过程、失败尝试和未决项完整保存在
[场景开发历史](docs/handoff/logistics-room-development-history-20260920.md)。
历史主张按当时的输入和代码理解，不覆盖上述当前验证范围。
