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
