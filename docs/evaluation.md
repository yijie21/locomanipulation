# 评测:open-loop 与 closed-loop

> 两套评测回答两个不同的问题,谁也替代不了谁。训练在
> [training.md](training.md)。

## 为什么必须两套

本仓库已有的硬证据:把**真值** token 开环回放,关节轨迹复现到 0.2–0.9°,机器人
却抓不到瓶子(浮动基座 1 分钟积分出 0.77 m 漂移)。所以:

- **open-loop** 回答"模型学没学到东西"——快、可逐 checkpoint 跑、用来选 checkpoint;
- **closed-loop** 回答"这个策略行不行"——唯一的验收标准。

bottle200 第一轮把这条原则演成了实例:开环关节角差 1.05°(贴测量噪声底),
闭环 6%。两个数字都对,测的不是一件事。

## Open-loop:`src/teleop_free/tools/eval_open_loop.py`

```bash
third_party/Isaac-GR00T/.venv/bin/python src/teleop_free/tools/eval_open_loop.py \
    --model-path outputs/train/<run>/checkpoint-N \
    --dataset datasets/..._val \
    --out outputs/eval/<run>/open_loop_checkpoint-N.json \
    --wandb-run <run>          # val 曲线写回训练 run,同一根 x 轴
```

四个指标,每个都配 persistence 基线(块首真值动作保持整块 —— 数据 30.6% 的帧
token 与前帧相同,不摆这条线任何 MSE 都读不懂):

| 指标 | 为什么是它 |
|---|---|
| nMSE,token 与 gripper **分开** | nMSE = MSE/var = 1−R²,尺度无关;不分开的话决定成败的 1 维 gripper 被 64 维淹没 |
| 吸附后命中率(逐维 + 整帧) | FSQ 空间每维实际只用 2–5 个格点,离散空间里命中率可读、MSE 不可读 |
| **关节指令差(度)** | 头条数字。经 SONIC decoder 换算,有物理单位;脚本自测噪声底(数据 30 Hz vs decoder 50 Hz 历史,~1.1°),低于它的差异不作数 |
| 闭合时刻误差(帧) | 抓取成败系于闭爪时机,任何 MSE 都看不见它 |

跨 `execution_horizon ∈ {1,8,16,40}` 各报一遍 —— 误差随 horizon 的增长曲线
就是"一个 chunk 敢信多少步"的直接证据(选 checkpoint 只跑 h16 即可,全 horizon
留给选出的最优者,省 3 小时)。

### bottle200 基线(checkpoint-10000,20 局 val)

| horizon | nMSE token(persistence) | 关节角差(persist.) | 闭合误差(persist.) |
|---|---|---|---|
| 1 | 0.120 | 1.05°(—) | 1.4 帧 |
| 16 | 0.142(0.323) | 0.93° | ~3 帧 |
| 40 | 0.176(0.895) | 1.30°(4.21°) | 4.9 帧(23.4) |

误差随 horizon 增长很缓,整块 1.33 s 可信;关节角差已到噪声底(1.16°)。

## Closed-loop:`src/teleop_free/tools/eval_closed_loop.py`

三个进程,两条硬前提:

```bash
# 1. stack 必须 token-only(带 encoder 时外部 token 流压不住,cpp:3096)
NO_ENCODER=1 SCENE_RANDOM=0 bash src/teleop_free/start_stack.sh "pick up the water bottle"

# 2. policy server,第二张卡
CUDA_VISIBLE_DEVICES=1 third_party/Isaac-GR00T/.venv/bin/python \
    src/teleop_free/tools/serve_policy.py --model-path <ckpt> --port 5590

# 3. 评测本体(.venv_sim —— 它要 gear_sonic 的相机客户端和 pinocchio FK)
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
    src/teleop_free/tools/eval_closed_loop.py --fresh 100 \
    --out outputs/eval/<run>/closed_loop.json
```

设计要点,每条都有伤疤背书:

- **复用 grasp_gym 自己的采样器**(`VLA_POLICY` 钩子,fork 点在 settle 之后)——
  参照线是脚本专家在同分布上的 **200/210 = 95.2%**,自己另写采样器 = 悄悄换分布,
  对照作废。
- **观测用 exporter 同款代码路径构建**(同一个 FK、同一个 gravity、图像原样透传),
  与训练数据构造上一致。
- VLA 局**出生即朝向 approach heading、跳过脚本转向**:token-only 模式下没收到
  token 时 `token_state` 置零(cpp:3066),站立撑得住、主动转向是坏的;而训练
  episode 本来就从转向完成后开始录。
- eval 会 `export NO_ENCODER=1`:gym 每局重启 deploy(`restart_deploy.sh`),
  不传下去的话第 2 局就被换回带 encoder 的 deploy。
- 结果 = 成功率 + Wilson 95% 区间 + **分阶段失败分布**(walk / reach / close /
  fell / timeout)。分阶段才可行动:失败在 close = 忠实继承了数据的缺陷;失败在
  walk = 没学会,触发加历史的消融。
- 每局 trace 落盘(含停滞位姿),它就是下一轮靶向补数据的靶单。

### bottle200 基线(101 局 fresh)

```
6/101 = 5.9%  (Wilson 3–12%)      脚本专家同分布 95.2%
失败:reach 91 · close 2 · walk 1 · pre-vla 1
```

91/95 的失败是同一个固定点:干净走到抓取站位 → 臂垂爪开站满 90 秒。训练数据里
专家**边走边抬臂**,"到桌边臂还垂着"零出现,模型对该状态的回答是"站着",而站着
还是该状态。6 局成功是随机逃逸(与出生距离无关)。教科书 covariate shift。

### 处方:靶向 DAgger(`scripts/collect_dagger.sh`)

让**专家出生在学生卡住的位置**(离桌沿 0.35–0.55 m;臂垂就是自然站姿,无需在线
接管 —— 也绕开了"专家标签要 encoder、学生驱动要 NO_ENCODER"的硬冲突),录它
抬臂→伸手→抓的恢复轨迹。`PROBE=1` 先 10 局探站立净空(实测 9/10 通过)。采完与
原数据合并转换(`--strip-pose-columns` 统一混代际 schema)重训。
