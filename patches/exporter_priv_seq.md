# exporter_priv_seq.patch

`gear_sonic/scripts/run_data_exporter.py` 相对上游 HEAD 的**累积**工作区 diff 快照
(2026-08-28)。包含两代改动:

1. 早先的 pose 列支持(`_add_object_poses`、`TF_POSE_OBJECTS`、:5581 订阅)——
   与 `vla_action_space.patch` 中的 exporter hunks 有重叠,以本文件为准(更新)。
2. 本次新增:**per-row priv_seq 旁路**。exporter 对 :5581 特权流做 CONFLATE 采样
   且从不记录用了哪条消息——这是 pose↔parquet 对齐结构性不可行的根因
   (docs/pose_parquet_alignment.md,七种离线对齐全败)。现在每行记下所用消息的
   `seq`,在所有 save/discard 路径 flush 成
   `<dataset>/priv_seq/episode_XXXXXX.json`(LeRobot schema 不动,旁路写失败
   绝不影响 episode 本体)。

配套(在主仓库内,非 patch):`privileged_pub.py` 的 publisher 给每条消息加单调
`seq`,PoseSidecar 行内记录同一个 `seq`——两侧从此有共享键,parquet 行 ↔ sim 侧
位姿记录可精确对齐,且能实测 exporter 的真实行速率(相邻行 seq 差)。

恢复方法:`git -C third_party/GR00T-WholeBodyControl checkout gear_sonic/scripts/run_data_exporter.py && git apply --directory=third_party/GR00T-WholeBodyControl patches/exporter_priv_seq.patch`
