# 文档地图

只有四类,四类之外不新建;带日期的诊断写进台账,不建独立文件。

## 教程(每个 pipeline 阶段一篇)

`setup.md`(环境安装)、`collection.md`(数据采集)、`dataset.md`(导出与审计)、
`training.md`(训练)、`evaluation.md`(评测)、`rendering.md`(渲染)、
`crate_collection.md`(双臂搬箱)。代码路径随主线保持更新,判据和数字不动。

## 台账(`ledger/`,每条产品线一个,append-only)

`ledger/bottle.md`(单臂抓取实验总表)、`ledger/crate.md`(双臂搬箱验收/结果)。
一条结论只记一处;要更正,在新的日期化章节里写并链接回旧的,不要就地覆盖。

## 参考(`reference/`)

`reference/standards.md`(代码与文档规范)、`reference/silent_failures.md`(模块地图 /
参数 / 诊断 / 静默失败表 / FAQ)、`reference/runbook.md`(当前默认配方的唯一入口)。

## 状态

`STATE.md`(现在在做什么)、`ROADMAP.md`(排期)。只这两个。

## 其余:`archive/`

历史诊断、设计讨论、进度报告——内容冻结,只加 `archive/INDEX.md` 索引,不改内容。
`handoff/`(进行中的跨 session 计划)与 `agents/`(agent 规则)原地保留,不算"其余"。

---

**四类之外不新建;带日期的诊断写进台账。**
