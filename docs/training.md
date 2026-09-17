# 训练:GR00T + SONIC 微调

从 LeRobot 数据集(`convert` 阶段的产物)到 checkpoint。数据集制作见
[dataset.md](dataset.md),训练完的 checkpoint 怎么评见 [evaluation.md](evaluation.md)。

## 1. 任务与能力

| 项目 | 当前支持 |
|---|---|
| 输入 | `convert` 阶段写的 `train_dataset/`(+`val_dataset/` 做 checkpoint 选择) |
| 训练代码 | vendored `src/gr00t/`,`launch_finetune` 微调 GR00T-N1.7-3B 的 projector + diffusion action head |
| embodiment | `g1_single_arm`(单臂,bottle/pick_place 系任务)、`g1_bimanual`(双臂,crate) |
| 硬件 | 单卡(8-bit AdamW,量化省显存)或多卡(DeepSpeed ZeRO-2,`gpus>1`) |
| 输出 | `outputs/runs/<name>/model/checkpoint-<steps>`、`train/manifest.json` |
| 验收 | checkpoint 事后选(开环 nMSE),真正的验收标准是闭环成功率(见 evaluation.md) |

## 2. 环境与场景

训练代码是 **`src/gr00t/`**——NVIDIA Isaac-GR00T 的 `gr00t` python 包的 vendored
副本,**入 git、可改、可提交**;`third_party/Isaac-GR00T` 整个目录被 gitignore,
二次开发不能放在那边,只留作上游参照和 venv 的宿主(出处/diff/改动清单见
[`src/gr00t/UPSTREAM.md`](../src/gr00t/UPSTREAM.md))。venv 仍在
`third_party/Isaac-GR00T/.venv`,editable 安装由 `scripts/link_gr00t_src.sh` 指向
`src/`,新机器建好 venv 后跑一次:

```bash
bash scripts/link_gr00t_src.sh
# ok: gr00t -> .../src/gr00t/__init__.py
```

`scripts/train_bottle.sh`(train 阶段内部调用它)每次启动前都会验证这个解析,
指到 third_party 会拒跑——那种失败原本是无声的:同名同版本的包,过期的代码,
训练照样跑完。GPU 显存需求见下文单卡/多卡两节。

## 3. 跑一局(+读日志)

前提是同一个运行目录下已经有 `convert/manifest.json`(即先跑过
`collect` 和 `convert`,见 [数据采集](collection.md)、[数据集转换](dataset.md))。
在此基础上单独跑训练:

```bash
python scripts/train.py outputs/runs/<run_name> --gpu 0
python scripts/train.py outputs/runs/<run_name> --gpu 0 --dry-run   # 先看会传哪些环境变量
```

`train.py` 从运行目录的 `summary.json` 拿到 `convert` 写下的 `train_dataset` 路径,
把运行自己的 `train`/`embodiment` 两个块变成环境变量后调用 `scripts/train_bottle.sh`,
再把 checkpoint 路径、步数和所用数据集写回 `summary.json`;若 `<run>/model`
已存在会直接拒绝重跑,避免覆盖一半的 checkpoint。

日志在 `<out>/train.log`(`locomani.pipeline.stages.Stage` 统一写的位置),
启动命令记在 `<out>/commands.jsonl`,耗时记在 `<out>/timings.jsonl`。盯
训练进度就 `tail -f <out>/train.log`,找 loss 曲线和 `Saving checkpoint` 字样;
`summary.json` 的 `train` 条目出现、且 `model/checkpoint-<steps>/` 真的在那里,
才算这局跑完,不要只看进程退出码。

## 4. 批量(+验收)

端到端跑完整条流水线(采集 → 转换 → 训练 → 闭环评测)就是一条命令:

```bash
python scripts/pipeline.py configs/experiments/bottle_simple_200.yaml --gpu 1
```

它按 `task.plugin` 选对应的脚本、按顺序跑,产物已存在的阶段自动跳过(要重跑就删掉
那个产物:数据集目录、`model/`、`eval/closed_loop.json`)。`--from`/`--to` 限定跑哪几段。
并行就是**再开一个终端**换一张卡,没有作业文件也没有调度器:

```bash
python scripts/pipeline.py configs/experiments/A.yaml --gpu 0    # 终端 1
python scripts/pipeline.py configs/experiments/B.yaml --gpu 1    # 终端 2
cat outputs/runs/*/summary.json                                  # 比较结果
```

或者分阶段逐个手动运行,方便在训练前后插手工检查:

```bash
python scripts/collect_bottle.py configs/experiments/bottle_simple_200.yaml --gpu 0  # 采集
python scripts/convert.py outputs/runs/bottle_simple_200                             # 转换
python scripts/train.py   outputs/runs/bottle_simple_200 --gpu 0                     # 训练
python scripts/eval_bottle.py outputs/runs/bottle_simple_200 --gpu 1                 # 闭环评测
```

## 5. 产物与验收

以下路径相对于本次运行目录(即 `outputs/runs/<name>/`):

| 路径 | 内容 |
|---|---|
| `model/checkpoint-<steps>/` | 训练产出的 checkpoint(权重 + 归一化 processor) |
| `train/train.log` | `launch_finetune` 的标准输出(loss、保存记录) |
| `train/manifest.json` | `checkpoint`/`train_dataset` 路径、`steps`/`gpus`/`global_batch` 指标、config 与上游 hash |
| `run_config.yaml`/`run_config.json` | 这次运行完整解析后的配置,训练用到的所有字段都能在这里查到 |

### bottle200 实测配置(模型与输入输出)

| 项 | 值 |
|---|---|
| Base model | `nvidia/GR00T-N1.7-3B`,embodiment `unitree_g1_sonic`(projector **slot 11**,公开权重里该槽为空,从随机初始化训) |
| 视觉输入 | 2 路 RGB:`ego_view` + `right_wrist`,crop 230 → 256×256。**不用** left_wrist(左臂全程不动,纯静态贴图);这份历史配方用的是 `configs/gr00t_sonic_2cam.py`,不是当前 `g1_single_arm` embodiment 的默认 `gr00t_sonic_3cam.py`(见 §6 modality 表) |
| State | **53 维** = 官方 46 + right_wrist_pos 3 + right_wrist_abs_quat 4(FK 腕位姿,pelvis 系,真机可得非特权)。绝对 yaw 类(root_orientation 等)**刻意不进** —— 随机出生 yaw 下它是查表捷径 |
| Action | **65 维** = motion_token 64 + right_gripper 1。左手 7 维 std 恒为 0、右手 7 维 rank-1,官方 78 里其余 13 维是常数,留着只会稀释指标 |
| Action horizon | 40 步 chunk @30 Hz(1.33 s);推理去噪 4 步 |
| 语言 | 单指令;SONIC 的语言键是 `annotation.human.task_description` |
| 时间历史 | 无(单帧观测)。列为第一消融项 |

### 可训练参数

| 模块 | 参数量 | 训练? |
|---|---|---|
| DiT(flow-matching action head) | 1,091.7 M | ✓ |
| vl_self_attention + vlln | 201.4 M | ✓ |
| Projector bank(32 槽) | 325.8 M(**有效梯度只有 slot 11 的 10.2 M**,其余 31 槽白占 ~2.5 GB Adam 状态) | ✓ |
| 视觉塔 / 语言塔(Cosmos-Reason2-2B) | — | ✗ 冻结。200 局单场景解冻视觉塔 = 把通用特征换成对这张橙桌的记忆 |
| **合计** | **1.62 B** | |

### 优化

| 项 | 值 |
|---|---|
| lr / schedule | 1e-4,cosine,warmup 5% |
| optimizer | AdamW fused,weight decay 1e-5 |
| batch | 64(2×5090 单次装下,实测 ~1.3 it/s,10k 步 2h07m) |
| checkpoint | 每 1000 步、**10 个全留**(`save_only_model`,各 6.5 GB)。默认 `save_total_limit=5` 会把前半程删光,而过拟合恰恰在前半程见分晓 |
| 归一化 | q01/q99 percentile,**统计量来自训练集并冻进 checkpoint 的 processor** —— 评测/部署自动用训练统计,无泄漏 |

### 为什么 checkpoint 事后选、不训练中选

GR00T 没有验证集机制(`DatasetFactory.build` 断言 `eval_strategy == "no"`)。
流程是:数据集层面切 180/20(`convert` 阶段的 `--holdout`,按采集 chunk 分层,
字段是 `train.holdout`)→ 训完对若干 checkpoint 逐个跑开环评测 → 按 held-out
token nMSE 选,平票取**更早**的那个。bottle200 实测 nMSE 0.314 → 0.1415
**单调降**,最优 = 10000,无过拟合。

### Token 的三个不能不知道

1. **motion_token 是 FSQ 离散码**,不是连续 latent:每维 32 格、步长 1/16(从
   encoder ONNX 的 `/quantizer` 节点直接读出)。回归输出必须吸附回格点
   (`src/locomani/backends/sonic/sonic_token.py`),训练/评测/部署三处一致。
2. **token 空间的 MSE 没有物理意义**,要读误差得过一遍 SONIC 自己的 decoder 换算
   成关节角度(`src/locomani/backends/sonic/sonic_decoder.py`,自带噪声底 ~1.1°)。
3. **GT token 开环回放也抓不到瓶子**(浮动基座 1 分钟漂 0.77 m),开环指标只能
   当体温计,验收必须闭环。

## 6. 调整任务(控制逻辑+判据表)

`train` 阶段的 `TrainConfig`(`configs/train/*.yaml`,`src/locomani/pipeline/train_config.py`)
只暴露 `steps`/`save_steps`/`holdout`/`gpus`/`global_batch`/`squat_max`/`base_model`
七个字段,通过 `train.<field>=<value>` 覆盖,例如
在实验文件里把 `train.steps` 改成 20000、`train.gpus` 改成 2,再跑 `scripts/train.py`。
它们只是 `launch_finetune` 完整超参数的一个子集;下面是完整表,直接调用
`bash scripts/train_bottle.sh`(而不经过 `scripts/train.py`)时才用得上全部选项。

### `launch_finetune` 完整超参数表

完整超参数来自 `tyro.cli(FinetuneConfig, ...)`
([`src/gr00t/configs/finetune_config.py`](../src/gr00t/configs/finetune_config.py))
——这个 dataclass 的每个字段就是一个 CLI flag(tyro 把 `snake_case` 转成
`--kebab-case`;`bool` 字段同时生成 `--flag`/`--no-flag` 两个开关,
`train_bottle.sh` 里的 `--tune-projector`/`--no-tune-visual` 就是这么来的)。
**脚本末尾的 `"$@"` 会把额外参数原样透传**,覆盖任意一项不用改脚本,直接追加
即可,例如 `bash scripts/train_bottle.sh --learning-rate 5e-5`。

#### 数据 / 模型路径(必填,无默认值)

| flag | 说明 |
|---|---|
| `--base-model-path` | 预训练权重路径(HF hub 或本地目录) |
| `--dataset-path` | 数据集根目录,多个数据集用 `os.pathsep` 分隔 |
| `--embodiment-tag` | embodiment 标签(大小写不敏感) |
| `--modality-config-path` | modality config `.py` 路径;不给就用 `embodiment_configs.py` 里预注册的(见下节) |

#### 模型微调开关

| flag | 默认 | 说明 |
|---|---|---|
| `--tune-llm` | False | 是否微调 LLM 骨干 |
| `--tune-visual` | False | 是否微调视觉编码器 |
| `--tune-projector` | **True** | 是否微调 vision/language → 共享空间的 projector |
| `--tune-diffusion-model` | **True** | 是否微调 diffusion action decoder |
| `--state-dropout-prob` | 0.2 | state 输入的 dropout 概率 |

上表只是 `FinetuneConfig` 暴露给 CLI 的部分。`Gr00tN1d7Config`
([`src/gr00t/configs/model/gr00t_n1d7.py`](../src/gr00t/configs/model/gr00t_n1d7.py))
里实际的 `tune_*` 训练开关共 6 个,CLI 只暴露上表 4 个,另 2 个改不了:
`tune_top_llm_layers`(LLM 最后 N 层单独解冻,默认 0 不启用)和 `tune_vlln`
(`vlln` LayerNorm + `vl_self_attention`,**永远为 True、永远参与训练**,
对应上文「可训练参数」表里的 "vl_self_attention + vlln 201.4 M")——要改这两个
只能直接编辑该文件的默认值。同一份文件里还有几个不在 CLI 里的架构选择开关:
`select_layer`(12,LLM 只保前 12 层,是砍大小不是冻结)、`use_vlln`、
`use_alternate_vl_dit`(`AlternateVLDiT` 还是普通 `DiT`)、`add_pos_embed`。

#### 数据增强

| flag | 默认 | 说明 |
|---|---|---|
| `--random-rotation-angle` | None | 图像随机旋转最大角度(度) |
| `--color-jitter-params` | None | dict:`brightness`/`contrast`/`saturation`/`hue`;不给用预训练模型自带的默认增强 |
| `--use-percentiles` | True | state/action 归一化用 q01/q99 百分位,而非 min/max |

#### 训练超参数

| flag | 默认 | 说明 |
|---|---|---|
| `--global-batch-size` | 64 | 梯度累积**之前**的总 batch |
| `--gradient-accumulation-steps` | 1 | 累积步数;有效 batch = 两者乘积(`__post_init__` 在 >1 时会打印提醒,防止把累积前的值当成真实 batch) |
| `--learning-rate` | 1e-4 | 初始学习率 |
| `--warmup-ratio` | 0.05 | warmup 占总步数比例 |
| `--weight-decay` | 1e-5 | AdamW weight decay |
| `--optim` | `adamw_torch` | 可选 `adamw_bnb_8bit`(需 bitsandbytes,量化 Adam m/v,单卡塞得下的关键,见下面「单卡训练」) |
| `--max-steps` | 10000 | 总训练步数(即 `train.steps`) |
| `--num-gpus` | 1 | 参与训练的 GPU 数(即 `train.gpus`) |

#### 输出 / checkpoint / 日志

| flag | 默认 | 说明 |
|---|---|---|
| `--output-dir` | `./outputs` | checkpoint/日志输出目录 |
| `--experiment-name` | None | wandb run 名;不给用 `output_dir` 的 basename |
| `--save-steps` | 1000 | 每多少步存一次 checkpoint(即 `train.save_steps`) |
| `--save-total-limit` | 5 | 最多保留几个 checkpoint,超了删旧的(`train.py` 固定传 `SAVE_LIMIT=1`) |
| `--save-only-model` | False | 只存模型权重,不存 optimizer/scheduler/RNG 状态(省盘,但不能 resume) |
| `--resume-from-checkpoint` | False | 从 `output_dir` 里最新的 `checkpoint-*` 续训;和 `save-only-model=True` 不兼容(`experiment.run` 里强制检查) |
| `--use-wandb` | False | 是否记录到 wandb(`train.py` 固定关闭) |

数据集分片/采样(`--shard-size`/`--episode-sampling-rate`/`--num-shards-per-epoch`/
`--ds-weights-alpha`)控制预加载与多数据集混合权重,默认值对单数据集训练够用。

### `--modality-config-path`:训练脚本怎么读 LeRobot 数据集

`--modality-config-path` 由 `train` 阶段自动传成
`run_cfg.embodiment.gr00t_modality_config`(`configs/embodiment/*.yaml` 的字段);
`g1_single_arm` 当前默认 [`configs/gr00t_sonic_3cam.py`](../configs/gr00t_sonic_3cam.py)
(官方 46/78 布局,三路相机全录),`g1_bimanual` 默认
[`configs/gr00t_sonic_3cam_bimanual.py`](../configs/gr00t_sonic_3cam_bimanual.py)。
上一节 bottle200 的 53/65 维配方用的是另一个仍然存在的文件
[`configs/gr00t_sonic_2cam.py`](../configs/gr00t_sonic_2cam.py)(改
`configs/embodiment/g1_single_arm.yaml` 或覆盖 `train.base_model` 能切回它)。
不管选哪个,它 `import` 时都会原地覆写 `MODALITY_CONFIGS["unitree_g1_sonic"]`
(NVIDIA 官方 slot 11 配置)。以 2cam 配方为例,它声明的字段是:

| modality | modality_keys | 维度 |
|---|---|---|
| `video` | `ego_view`、`right_wrist`(丢弃 `left_wrist`) | 2 路 |
| `state` | `left_leg`、`right_leg`、`waist`、`left_arm`、`right_arm`、`left_hand`、`right_hand`、`projected_gravity`、`right_wrist_pos`、`right_wrist_abs_quat` | 53 |
| `action` | `motion_token`、`right_gripper`,`delta_indices=range(40)`(预测 40 步的 action chunk) | 65 |

这些 `modality_keys` 名字必须能在数据集的 `meta/modality.json` 里按名字找到
(`lerobot_episode_loader.py:328` 的 `_extract_joint_groups`);找不到**不会
报错**,只打一句 `Warning: Joint group '...' not found` 然后静默丢掉那个字段
(`lerobot_episode_loader.py:339-342`)。**换一个新数据集训练前,先拿这张表的
`modality_keys` 逐项核对它的 `meta/modality.json`**,少一个,训练照样收敛,
只是悄悄瘸了一路输入或监督,不核对根本看不出来。

`meta/stats.json`(不要跟数据集自带的逐集统计 `meta/episodes_stats.jsonl` 搞混,
前者是 GR00T 要的**跨全部 episode 合并**的归一化统计,天生没有)是另一个硬
依赖,但会自愈:`LeRobotEpisodeLoader._load_metadata()` 对它是硬 `assert`
(`lerobot_episode_loader.py:180`),但 `DatasetFactory.build()` 会先在 rank-0
上调 `generate_stats()`/`generate_rel_stats()`(`factory.py:56-59`)——**缺失时
自动生成,不是人工步骤**。`train_bottle.sh` 的启动前检查不会替你等生成,缺失
时直接报错退出并打印生成命令(排错见 §7)。

### 单卡训练:`SINGLE_GPU=1`

`train.gpus=1` 时 `train.py` 会自动设 `SINGLE_GPU=1`;直接调用脚本
时也可以手动指定:

```bash
SINGLE_GPU=1 RUN=<name> DATA=<train_ds> bash scripts/train_bottle.sh
```

双卡需求从来不是算力,是 Adam:1.62B 可训参数的 fp32 m/v 就要 13 GB,加上主权重、
梯度、冻结 VLM 和激活约 33 GB,超出单卡 31.4 GB。ZeRO-2 靠**分片**解决;这个开关
靠**量化**解决(`adamw_bnb_8bit`,13 → 3.2 GB)。三个数字全是实测:per-device 32
OOM(峰值 32.1 GB)、16 能跑但只剩 950 MiB 余量、**8(accum 8)峰值 27.0 GB** ——
有效 batch 恒为 64,lr 与实验可比性不变。代价 ~1.4 s/it,10k 步约 3.9 h
(双卡 2.1 h),换回第二张卡做并行实验。

## 7. 排错与停止

| 现象 | 先查哪里 |
|---|---|
| 训练拒跑,报 gr00t 解析到 third_party | 重新执行 `bash scripts/link_gr00t_src.sh`(§2) |
| 报 `meta/stats.json` 缺失就直接退出 | 按提示手动跑一次 `gr00t.data.stats`,或改走 `DatasetFactory.build()` 自动生成(见 §6);`scripts/convert.py` 产出的数据集通常已经算过 |
| `WANDB` 相关报错 | `train.py` 已固定 `WANDB=0`;手动调脚本时没登录会在启动阶段**立即报错**,不是等到 torchrun 下才挂死 |
| 训练/评测同盘 ENOSPC | 两个训练同时存 checkpoint 前先算账 —— checkpoint 大小 × 保留个数 × 任务数。2026-08-25 的事故:bottle250(6.5 GB × 10)+ frozen_dit(12 GB × 8)并行把 165 GB 吃穿,闭环评测和训练双双死于 ENOSPC。旧 checkpoint 评完即删,或调小保留个数 |
| `<out>/model` 已存在,拒绝重跑 | 这是防止覆盖一半的 checkpoint;换一个新的 `name=` 或先确认旧 checkpoint 已经不需要 |

前台运行用 Ctrl-C 停止训练进程本身;`pipeline.py` 起的阶段同样可以
Ctrl-C 中止,已完成的阶段(manifest `ok=true` 且哈希未变)下次重跑会被跳过,
不必从头再来。

## 8. 进一步阅读

- [运行基础设施](infra.md)：配置组怎么组合、阶段产物与断点、多卡并行的作业文件与资源准入。
- [数据集转换](dataset.md):train_dataset/val_dataset 的字段、图像/动作对齐、`check_dataset` 审计。
- [评测](evaluation.md):开环选 checkpoint、闭环验收、部署参数如何影响成功率。
- [`src/gr00t/UPSTREAM.md`](../src/gr00t/UPSTREAM.md):vendored GR00T 包对上游的改动清单。
- [数据采集](collection.md):单臂采集入口与任务配置。
