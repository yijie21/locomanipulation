> **台账(append-only)**:本文件只在末尾追加新的日期化章节,不改写、不删除已有结论。
> 要更正一条结论,在新章节里写清楚并链接回旧的那一条,不要就地覆盖。
> 分类与格式规则见 `docs/reference/standards.md` §3。

# 台账:crate(双臂搬箱)线

## 2026-09-07 wall pilot 验收判据(来自 `docs/crate_collection.md` §5,表格原样保留在教程里)

`run_wall_pilot.sh` 10 集 pilot 的验收表,`WALL_FEATURE=rail`:

| 指标 | 合格 | 为什么 |
|---|---|---|
| 成功 | ≥ 8/10 | 历史:ready-stand pilot 16 20/20;走入 pilot 21 20/20,22 9/10,23 8/10 |
| `move_s` | = lift 开始时刻(约集长 − 4 s) | lift 前篮子不能被碰。pilot 18 的 20 集都在走路时被指尖扫转 7–14°,当时以为的"站位 yaw 门"失败其实是这个 |
| `trig` | 绝大多数 `contact` | `floor` 是兜底,`seated` 是毫米判据,都不该是主流 |
| `static` 中位 | ≤ 0.10 | 吸收态预算(单手线 cover400 的 creep 就是这么来的) |
| 站位 `so` | 0.13–0.32 内 | reach 预算 0.38 减去深度 [0.06, 0.25] |
| 站位 `\|yaw\|` 中位 | ≤ 3° | 篮子没被碰时步态自身的散布 |

## 2026-09-07 设计文档里嵌入的实测结果(来自 `docs/archive/bimanual_grasp_design.md`,原文未加日期,按文中"今天"与相邻 pilot 编号推断为 09-07 前后)

这些是设计文档正文里夹带的实测数字,不是设计目标;裁定摘录于此,详细上下文见归档原文。

- **R1(单手线)91/95 局失败是相位错位**(脚本走路时抬手、数据里手已在桌边抬好);改成 coupled(蹲 + 伸同时)后 **8/8**,去停顿使闭环翻倍(`docs/archive/bimanual_grasp_design.md` §B)。
- **heading 死区实测**:recoverable 策略 89 轮里仍有 3 轮在 spawn 处因朝向死区(7–8°)超时("could not turn"),`diag_turn.py` 量出 15–120° 各档都停在 6.5–8.1° 残差(`docs/archive/bimanual_grasp_design.md` §A)。
- **夹爪硬化前的蠕滑**:0.30 kg 瓶子默认接触下 0.5 cm/s 蠕滑掉落;codex 07 独立集 24 集里 **3 集在 hold 阶段滑出约 1 m**,是 pad 硬化(`PAD_HARDEN`)成为前置条件的直接证据(`docs/archive/bimanual_grasp_design.md` §D)。

## 2026-09-10 随机化 10 集 pilot + MuJoCo|Isaac 并排视频(branch session,GPU0 / instance 2)

用户要求:再录 10 集 SONIC 双臂搬箱,场景在合理范围内随机化,成功失败都要,第三视角 + ego/左腕/右腕画中画,左 MuJoCo 右 Isaac。
配置:`configs/tasks/crate_wall.json` 默认(`rail`,`WALL_WALK=1`),seed 910;随机轴 = 现有三条(箱子横向 ±8 cm、出生点
1.0–1.3 m / ±0.15 m / ±10°、站位由步态散布),箱子 yaw / 桌高 / 外观仍未打开(教程 §8)。GPU1 被外部任务占 96%,
所以 pilot 和渲染都在 GPU0(教程默认 GPU1)。

| 指标 | 结果 | 合格线 |
|---|---|---|
| 成功 | **8/10** | ≥ 8/10 |
| 失败 | r2 `[hold] lost contact: ['right']`;r5 `[lift] rose 0.0 cm`(站位横向 +5.9 cm、箱子 dy −6.7 cm,双手闭在箱沿旁边) | — |
| `trig` | 10/10 `contact` | contact 为主 |
| `so` | 0.18–0.26 | 0.13–0.32 |
| `\|yaw\|` 中位 | 2.3° | ≤ 3° |
| `static` 中位 | **0.15**(max 0.30) | ≤ 0.10 —— **不合格**,吸收态预算超了 |
| 每集 | 25–47 s(r0–3 约 26 s,r4–9 约 44 s;`deploy_restarts` 全 0) | 40–60 s |
| move_s | 5.6–8.0 s,均在 lift 段(len 10–12 s) | = lift 起点 |

`static` 超线是本次唯一的红项,r1/r2 达 0.26/0.30;没查原因(本次是采集性能测试,不是调参)。
产物:pilot `outputs/videos/crate_random10_20260910/`(`wall_s910_NN.mp4`、`trace/`、`manifest.jsonl`、`pilot_s910.log`);
并排视频 `outputs/renders_isaac/crate_random10_20260910/sbs_roundNN.mp4`(10 个,1280×480,12.5 fps,10–12 s,共 70 MB;
左 MuJoCo 跟踪相机 + 三路画中画,右 Isaac 同一相机轨迹)。渲染约 3 min/集。
`summarize_wall_pilot.py` 之前把 trace 名写死为 `wall_s0`,非 0 seed 的 move_s/static 全是 nan,已改为按 manifest 的 video 名取 label(a0cd7d4)。

## 2026-09-10 更正:上一节的 10 集是 ready-stand 变体,走入版重跑 4/10(主机 load 130–176)

上一节(同日,seed 910)按教程 §4 的 `WALL_WALK=1 WALL_FEATURE=rail` 起的 pilot **没有走路段**:重构 step 3 之后这两个
环境变量不再被读,`wall_walk` 默认 false,10 集全是出生在站位后 0.45 m 瞬移的 ready-stand 变体(每集 25–47 s、视频
10–12 s、日志无 `[walk]` 行)。那 10 个并排视频作为"抓取段"仍有效,但不是完整 episode;教程、runner、静默失败表已改
(b07caef)。同 seed 用 `--config configs/tasks/crate_wall.json` 重跑:

| 指标 | 走入版(本节) | ready-stand(上一节) | 合格线 |
|---|---|---|---|
| 成功 | **4/10** | 8/10 | ≥ 8/10 |
| 失败 | r1 `[stand]` 横向 −9.6 cm(3 次走位后);r2 停太近 0.11 m;r3 `[descend]` never seated;r5 停太远 0.38 m;r6 站位处摔倒;r7 停太远 0.45 m | r2 hold 失接触;r5 lift 0 cm | — |
| 站位 `so`(成功集) | 0.14–0.23 | 0.18–0.26 | 0.13–0.32 |
| `static` 中位 | 0.01 | 0.15 | ≤ 0.10 |
| 每集 | 29–87 s;视频 9–33 s | 25–47 s | 40–60 s |
| `deploy_restarts` | 0 | 0 | — |

**裁定:4/10 不能当作 expert 的成功率**。5/6 失败在走入停站,走入 pilot 历史是 20/20、9/10、8/10(09-07);本次主机
load average 131 → 176(32 核,`/workspace/code/video_grasp` 的 teacher-benchmark,5 个 python 进程,其中 3 个在 pilot 中途
启动),GPU1 另有 97% 的外部任务。走路阶段对 20 ms 周期最敏感,ready-stand 变体绕过了它所以看不出。要得到可信的走入
成功率,机器安静时重跑同 seed;若仍 ≤ 6/10 再查 `STAND_OFF_WALK` / 走位逻辑。`static` 0.01 说明上一节的 0.15 来自
ready-stand 的抬臂 + 瞬移前的等待,不是走入流程的问题。
产物:pilot `outputs/videos/crate_walk10_20260910/`;并排视频 `outputs/renders_isaac/crate_walk10_20260910/sbs_roundNN.mp4`
(10 个,含走路段,9–33 s,共 106 MB;左 MuJoCo + 三路画中画,右 Isaac)。
