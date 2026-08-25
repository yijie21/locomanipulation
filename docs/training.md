# 训练:GR00T + SONIC 微调

> 从 LeRobot 数据集到 checkpoint。评测在 [evaluation.md](evaluation.md),
> 数据集制作在 [dataset.md](dataset.md)。

## 代码在哪(以及为什么在那)

训练代码是 **`src/gr00t/`** —— NVIDIA Isaac-GR00T 的 `gr00t` python 包的
vendored 副本,**入 git、可改、可提交**。`third_party/Isaac-GR00T` 整个目录被
gitignore,在里面改的每一行对本仓库都不可见,所以二次开发的代码不能放在那边;
那份 clone 只留作上游参照和 venv 的宿主。出处、diff 上游的方法、本地改动清单
都记在 [`src/gr00t/UPSTREAM.md`](../src/gr00t/UPSTREAM.md)。

venv 仍在 `third_party/Isaac-GR00T/.venv`(环境不是代码)。它的 editable 安装
由 `scripts/link_gr00t_src.sh` 指向 `src/`,新机器上建好 venv 后跑一次:

```bash
bash scripts/link_gr00t_src.sh
# ok: gr00t -> .../src/gr00t/__init__.py
```

`train_bottle.sh` 每次启动前都会验证这个解析,指到 third_party 会拒跑 ——
因为那种失败是无声的:同名同版本的包,过期的代码,训练照样跑完。

## 启动

```bash
bash scripts/train_bottle.sh                  # 全部默认
DRY=1 bash scripts/train_bottle.sh            # 只打印命令,不跑
RUN=bottle250 DATA=datasets/..._train bash scripts/train_bottle.sh
```

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `DATA` | `datasets/g1_locomani_bottle200_train` | 训练集(须已有 `meta/stats.json`) |
| `RUN` | `bottle200` | **同时**是 wandb run 名和 checkpoint 目录叶子 |
| `GPUS` | 2 | 2 卡走 DeepSpeed ZeRO-2;1.62B 可训参数单卡装不下 |
| `GLOBAL_BATCH` / `GRAD_ACCUM` | 64 / 1 | 前者是**优化** batch(lr 按它选的,别动);后者是**显存**旋钮,OOM 时升 accum 降 batch,乘积保持 64 |
| `MAX_STEPS` | 10000 | ≈5.6 epoch(180 局 × ~630 帧 @30 Hz) |
| `WANDB` | 1 | 凭据预检:没登录会**立即报错**而不是在 torchrun 下挂死 |
| `DRY` | 0 | 1 = 打印最终命令行 |

端到端流水线(等转换 → stats → 审计 → 训练 → 逐 checkpoint 开环评测):

```bash
nohup bash scripts/run_pipeline.sh > outputs/logs/pipeline.log 2>&1 &
```

## 训练设置(bottle200 实测配置)

### 模型与输入输出

| 项 | 值 |
|---|---|
| Base model | `nvidia/GR00T-N1.7-3B`,embodiment `unitree_g1_sonic`(projector **slot 11**,公开权重里该槽为空,从随机初始化训) |
| 视觉输入 | 2 路 RGB:`ego_view` + `right_wrist`,crop 230 → 256×256。**不用** left_wrist(左臂全程不动,纯静态贴图) |
| State | **53 维** = 官方 46 + right_wrist_pos 3 + right_wrist_abs_quat 4(FK 腕位姿,pelvis 系,真机可得非特权)。绝对 yaw 类(root_orientation 等)**刻意不进** —— 随机出生 yaw 下它是查表捷径 |
| Action | **65 维** = motion_token 64 + right_gripper 1。左手 7 维 std 恒为 0、右手 7 维 rank-1,官方 78 里其余 13 维是常数,留着只会稀释指标 |
| Action horizon | 40 步 chunk @30 Hz(1.33 s);推理去噪 4 步 |
| 语言 | 单指令;SONIC 的语言键是 `annotation.human.task_description` |
| 时间历史 | 无(单帧观测)。列为第一消融项 |

这套选择的实现是 [`configs/gr00t_sonic_2cam.py`](../configs/gr00t_sonic_2cam.py) ——
modality.json 是菜单,embodiment config 是点单;数据集始终保有全部列,改回官方
46/78 只是换个 config,无需重转数据。

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
所以流程是:数据集层面切 180/20(转换器 `--holdout`,按采集 chunk 分层)→ 训完
对 10 个 checkpoint 逐个跑开环评测 → 按 held-out token nMSE 选,平票取**更早**的
那个。bottle200 实测:nMSE 0.314 → 0.1415 **单调降**,最优 = 10000,无过拟合。

## Token 的三个不能不知道

1. **motion_token 是 FSQ 离散码**,不是连续 latent:每维 32 格、步长 1/16
   (从 encoder ONNX 的 `/quantizer` 节点直接读出)。回归输出必须吸附回格点
   (`tools/sonic_token.py`),训练/评测/部署三处一致。
2. **token 空间的 MSE 没有物理意义**。要读误差就过一遍 SONIC 自己的 decoder
   换算成关节角度(`tools/sonic_decoder.py`),它自带测量噪声底(~1.1°)。
3. **GT token 开环回放也抓不到瓶子**(浮动基座 1 分钟漂 0.77 m)——
   开环指标只能当体温计,验收必须闭环。
