# SIMPLE 抓取对比：vastgpu 上 11-19 到 11-20 的记录，交给 4090

写给 4090 上接手的 session，也是这两天的留底。11-19 中午 4090 那边把“渲染对比视频、做成报告”交到 vastgpu，交接说明是 `/workspace/tmp/simple_film_20261119/HANDOFF.md`。这份文档记从那时起到 11-20 02:00 UTC 做了什么、用户定了什么、哪些数变了哪些没变、接下来要 4090 做什么。命令在 [runbook 第 7 节](../selfwm.md#7-simple-评测集)。

## 现在的状态

- 分支 `selfwm`。这次提交加了：用 SIMPLE 的 Isaac 画面做开局那一眼的代码、桌面抓取找目标的新规则、画感知过程和对比视频的脚本、几个诊断脚本。提交前 GitHub 上的 `selfwm` 停在 `4014ec8e`。
- 4090 上的评测代码（提交 `a4816f1`，`simple_bench.py`）不在 GitHub 上，也不在 vastgpu 上。vastgpu 连不上 4090。所以报告里的 30 局结果表 vastgpu 复现不了，也没有重跑。
- **报告结果表里的每一格仍是 11-19 在 4090 上跑的数。** vastgpu 只重做了“看一眼”这一步。
- 下一步在 4090：拉这次提交，把自己的提交接在后面并 push；用新的抓取重跑感知栏。清单见[要 4090 做的](#要-4090-做的)。

## 用户的决定

按时间，引号里是原话。

| 日期 | 决定 |
|---|---|
| 11-19 | “查表修正，闭环反馈都不应该有啊……一直调整是犯规的，这样的话不论多差的模型都能抓到了。” lookup 和 servo 从报告里去掉。servo 以后在任何对比里都不出现，当参考线也不行 |
| 11-19 | 关于 inverse：“我不记得我有训练过这个模型。” 随后“先把逆模型隐藏掉吧”。它是谁在什么时候加的、怎么训练的，用户还在等 4090 那边回答 |
| 11-19 | “把 1 拿掉”：model_pose 去掉。它和 model 是同一个网络，8 格里 0 格更好、4 格更差 |
| 11-19 | 得知感知栏的相机画面是 MuJoCo 复刻场景画的以后：“做，并且我想你把感知得到抓取点的整个过程都可视化到视频里。” 这一眼改用 SIMPLE 的 Isaac 画面 |
| 11-20 | 桌面抓取找目标的规则“用新的” |
| 11-20 | 评测代码“暂时先不拷过来”，30 局评测仍在 4090 上跑 |
| 11-20 | 代码和文字记录走 git，数据全部同步 |

更早定下、仍然有效的：和 HITCH 比的任何一格都不能用仿真真值；SIMPLE 上用官方的 G1 加 Dex3。

报告现在只有两种方法：direct，把目标手掌位置直接当命令发；model，自身模型求出命令。

## 报告里的数

11-19 在 4090 上跑的，代码 `a4816f1`，每格 30 局，种子 0，每格只跑一次。大数是 45 秒内抓起来的局数，括号里是手进入抓取点时的偏差中位数。这些数 vastgpu 没有重跑过，证据是 4090 上的逐局结果。

| 任务 | 感知栏 direct | 感知栏 model | 真值栏 direct | 真值栏 model |
|---|---|---|---|---|
| XMovePick | 28（2.79 cm） | 30（0.81 cm） | 30（1.38 cm） | 22（0.47 cm） |
| XMoveBendPick | 27（3.44 cm） | 30（0.72 cm） | 0（3.53 cm） | 22（0.44 cm） |
| BendPick | 10（6.20 cm） | 30（2.63 cm） | 0（3.84 cm） | 0（2.34 cm） |
| TabletopGrasp | 18（4.76 cm） | 14（16.23 cm） | 17（1.59 cm） | 24（1.58 cm） |

- BendPick 官方只给 8 秒，两栏两种方法在 8 秒内都是 0 局。
- 感知栏当时用的抓取是 `cloud_blind/`，相机画面由 MuJoCo 复刻场景渲染，白底、没有房间、深度是理想值。核对方法：两局选中的抓取分数和 vastgpu 上那份缓存逐位相同，BendPick 第 1 局 0.965591，Tabletop 第 0 局 0.929242。
- 隐藏掉的 inverse：真值栏 15、19、0、25，感知栏 30、30、30、16。

报告是一个私有页面，https://claude.ai/artifact/GrMKuETgedkcm1Vw8FVmot ，现在是第 8 版。页面和每一版的留底在数据包的 `film_20261119/report*/`。

## 做了什么

### 对比视频

`isaac_film_vast.py` 把 4090 录下的状态逐帧写进 SIMPLE 的 Isaac 场景再渲染，`grid.py` 把两种方法拼成左右两格。六段：真值栏 XMoveBendPick 第 0、6 局，真值栏 XMovePick 第 4 局，真值栏 BendPick 第 0 局，感知栏 BendPick 第 1 局，感知栏 Tabletop 第 0 局。

`isaac_film_vast.py` 是从 4090 给的 `isaac_film.py` 改的，改了两处：XMoveBendPick 这个任务 SIMPLE 在 `eef474f` 里删掉了，要先注册它删除前的任务文件，参数是 `--old-task`；加了 `--no-head`。另外按仓库的格式化规则重排过。4090 合并时把这两处并进自己的 `isaac_film.py`，再删掉这个文件。

### 用 Isaac 画面重做“看一眼”

机器人在复刻场景里站 1 秒。`eyes_server.py` 把这一刻的状态按关节名写进 SIMPLE 的 Isaac 场景，渲染 8 次让画面稳定，读头部左右两张图。`depth_server.py` 用双目匹配加 LingBot-Depth 补全得到深度。之后和原来一样：SAM 3 找目标，GraspGen-X 出 716 个抓取，取第一个右手做得到的。物理仍然是复刻场景，只换了眼睛。入口是 `blind.py look --eyes`，四个任务一起跑用 `look_all.sh`。

四个任务各 30 局，和 MuJoCo 画面比：

| 任务 | 找对目标 | 目标上的深度误差中位数 | 选中的抓取移动超过 5 cm | 选中从上往下抓的局数，MuJoCo 画面和 Isaac 画面 |
|---|---|---|---|---|
| XMovePick | 30 | 0.34 cm | 7 | 0 和 7 |
| XMoveBendPick | 30 | 0.29 cm | 7 | 0 和 5 |
| BendPick | 30 | 0.38 cm | 23 | 8 和 28 |
| TabletopGrasp | 27 | 0.73 cm | 找对的 27 局里 5 局 | 30 和 29，高度约 0.90 m |

找没找对是事后拿仿真里目标的真实轮廓判的，重合过半算对，机器人不用这个信息。数据在 `look_stats_eyes_vs_blind.json`、`look_stats_tabletop_rules.json`、`grasp_types_by_look.json`。第一份和第三份由 `look_stats.py`、`grasp_types.py` 算出；第二份是当时在命令行里按同样的算法算的，没有留脚本。

BendPick 的抓法变了：MuJoCo 画面下多是从侧面低位抓，手掌比目标点云的最低处高 8.0 cm（中位数）；Isaac 画面下 28 局是从上往下抓，高 19.3 cm。这对抓取结果有什么影响要重跑才知道。

### 桌面抓取找目标的规则

这个任务的指令是“grasp the object on the tabletop.”，不说物体名。

- **原规则**，`see.largest_thing`：只看深度，取比桌面高出 1.2 cm 的最大一块。在 Isaac 画面里 30 局只找对 12 局，找错的 18 局里有 8 局没有右手做得到的抓取。
- **原因**：这张桌子长 2.06 m，被 SIMPLE 随机成纯色或反光的材质。桌面的双目深度偏近，在 0.3 到 0.6 m 处偏 1.0 cm，0.9 到 1.2 m 处 3.8 cm，1.5 到 2.0 m 处 9.7 cm。匹配视差差 0.26 到 0.52 像素，另外三个任务的桌子上是 0.02 到 0.14 像素。空桌面上的鼓包于是高过了 1.2 cm 的门槛。双目为什么在这种桌面上偏，没有查清。脚本是 `diag_depth.py` 和 `diag_tabletop.py`。
- **新规则**，`see.largest_seen`，开关是 `blind.py look --unnamed sam`，带 `--eyes` 时是默认：SAM 3 的提示词用 “object”，取骨盆前 0.2 到 1.0 m 内面积最大的一个。

同样 30 局的 Isaac 画面，离线比较几种规则，再整条流程重跑新规则：

| 规则 | 找对 |
|---|---|
| 原规则 | 12 |
| 桌面改成拟合的平面 | 20 |
| 只看前方 1.0 m 内 | 22 |
| 上面两条都加 | 24 |
| 新规则，离线 | 27 |
| 新规则，整条流程重跑 | 27，这 27 局都有右手做得到的抓取 |

前四行来自 `try_rules.py`，第五行来自 `try_sam.py`。新规则是看到失败以后才定的，等于在测试局上调过，报告里写明了这一点。

没找对的 3 局：`L1_e2` 什么都没找到，`L1_e7` 圈出的位置和目标只重合三分之一，`L2_e4` 圈到了别的东西。

### 感知过程的视频

`percept_film.py` 把一局的这一眼按六步画成 33 秒的视频：看一眼、找目标、量距离、变成三维点、出抓取、定抓取点。画的都是机器人当时用的数据，来自 `LOOK_DUMP` 存下的中间结果。报告里放了五段。

### 用户问的两格

11-20 用户问：感知栏 TabletopGrasp 的 14 和真值栏 BendPick 的 0 更新了没有。都没有更新。vastgpu 上只有录像那几局的逐局结果，所以下面只能说到第 0 局。

- **感知栏 TabletopGrasp 第 0 局**：direct 和 model 的手都停在离抓取点 34 cm 处，到预抓取点时就差 30 cm，IK 残差 0.01 cm。这一局目标是找对的，看到的位置差 3.63 cm。所以这一局的失败和方法无关，也不是感知的问题。记录里 `grasp_h` 是 `[0.327, -0.104, 0.894]`，`pre_dir` 是 `[0, 0, 8]`，`walk_err_cm` 是 `[-1.07, -9.37]`。model 的偏差中位数为什么比 direct 大这么多，要 30 局的逐局结果才能说。
- **真值栏 BendPick 第 0 局**：model 到预抓取点差 0.215 cm，进入后差 2.233 cm，物体没有被抬起来。同一种抓取在 XMoveBendPick 第 0 局是 0.171 和 0.494 cm，抓起来了。两局进入时都是拇指先碰到物体。

## 要 4090 做的

1. **代码。** `git fetch`，把 `a4816f1` 和之后的提交接在这次提交后面，push。`blind.py`、`see.py` 两边都可能改过，冲突在这一步解决。
2. **数据。** 数据包见[东西在哪](#东西在哪)。在仓库根目录用 `tar xkf` 解开，`k` 保证不覆盖已有的文件。
3. **重跑感知栏。** 抓取指到 `outputs/simple_eval_20261112/cloud_eyes_final/`，direct 和 model，四个任务各 30 局，别的都不变。
   - 这个目录里前三个任务来自 `cloud_eyes/`，TabletopGrasp 来自 `cloud_eyes_sam/`，格式和 `cloud_blind/` 相同。
   - 缺 `TabletopGraspMP_L1_e2`，这一局直接记失败，不要退回 `cloud_blind/`，更不要用真值。
   - 没找对的 `L1_e7`、`L2_e4` 照常跑，按实际结果记。
4. **拷回逐局结果。** 这次重跑的全部 240 局；11-19 那一版里用户问的两格各 30 局，就是感知栏 TabletopGrasp 和真值栏 BendPick 的 direct 和 model；录像状态六局，和上面对比视频那六局相同。
5. **回答用户。** `inverse_wrist.pt` 是谁在什么时候决定加的，网络结构、输入输出、训练数据怎么划分。

重跑的结果回来以后，报告的结果表、对比视频和开头的结论都要跟着改。

## 没查清的

- 桌面抓取没找对的 3 局，目标看上去都是同一个薯片罐，都在随机化 1、2 档。Isaac 把它画成半透明的壳加一个木纹的实心体，木纹体比复刻场景里的轮廓低十几个像素。可能是 SIMPLE 的材质随机化本来就这样，也可能是 `eyes_server.py` 写状态时这个物体没对上。后一种情况下这 3 局不算感知的错。
- 双目深度在纯色或反光桌面上偏近的原因。相机没有做随机化，桌子是一个普通的长方体。
- 真值栏 BendPick 为什么进入时偏到 2 cm 以上。
- 感知栏 TabletopGrasp 里 model 为什么比 direct 差。
- 11-12 用更早的流程测过一次：XMovePick 在 SIMPLE 官方环境里 19 / 30，在复刻场景里 28 / 30。复刻场景的数可能偏乐观，这一点还没有用现在的流程核对。
- 根目录 README 里 11-15 写的投稿故事仍把“靠反馈纠偏”当基线。用户 11-19 已经否掉 servo，README 这一段没有改。

## 东西在哪

代码都在 `scripts/experiments/simple_eval_20261112/`：

| 文件 | 做什么 |
|---|---|
| `eyes_server.py` | 在 SIMPLE 的 venv 里跑，把复刻场景的状态画进 SIMPLE 的 Isaac 场景，返回头部双目图。一个进程只能开一个任务 |
| `blind.py` | 加了 `--eyes`、`--unnamed`、`--cloud`，`LOOK_DUMP` 可以是目录 |
| `see.py` | 加了 `largest_seen` |
| `look_all.sh` | 四个任务的看一眼加出抓取 |
| `look_stats.py`、`grasp_types.py` | 上面两张表的数 |
| `diag_depth.py`、`diag_tabletop.py`、`try_rules.py`、`try_sam.py` | 桌面抓取的诊断和规则比较 |
| `percept_film.py` | 感知过程的视频 |
| `isaac_film_vast.py`、`render_all.sh`、`grid.py` | 对比视频 |

数据不进 git。数据包在 vastgpu 的 `/workspace/tmp/simple_film_20261119/vastgpu_records_20261120.tar`，1.5 GB，1001 个文件加一份清单，要经 Mac 中转。

- 整包的 sha256：`a024b8ad9b7a2736b890ae9779e086d9374bfca7ef85be48dc0f922e0ffd85a6`，同目录有 `.sha256` 文件。
- 包里的路径都以 `outputs/simple_eval_20261112/` 开头。逐文件的 md5 在 `outputs/simple_eval_20261112/MANIFEST_vastgpu_20261120.md5`，解开后在仓库根目录跑 `md5sum -c` 核对。

| 包里的目录或文件 | 内容 |
|---|---|
| `cloud_eyes_final/` | 重跑要用的抓取 |
| `cloud_eyes/`、`cloud_eyes_sam/` | 原规则四个任务，新规则的桌面抓取 |
| `look_dump_eyes/`、`look_dump_eyes_sam/` | 每局这一眼的中间结果：左右图、两种深度、目标的圈、点云、相机位姿 |
| `look_logs/` | 每次运行的日志，`look_stats.py` 从里面读深度误差 |
| `film_20261119/` | 渲染出的片段、4090 录的状态、报告各版、当时用的脚本、SIMPLE 删掉的那个任务文件 |
| 顶层的 `*.json` | 三份统计，以及 11-12 到 11-18 旧流程的逐局结果 |

没有放进包的：`cloud_blind/`，4090 上已经有同一份；11-12 到 11-18 旧流程的其他输出，约 870 MB。

## 验证

- 搬进仓库的脚本重新格式化过。`look_stats.py` 和 `grasp_types.py` 重跑，输出和保存的 JSON 逐字节相同；`try_rules.py` 重跑得到同样的 12、20、22、24；`percept_film.py --check` 选中的抓取和之前相同。
- `look_all.sh TabletopGraspMP:0` 从仓库里跑了一局，不加 `--unnamed`：点数 3140 和之前相同，目标位置差 0.006 cm，圈出的区域重合 0.998。
- `try_sam.py`、两个 `diag_*.py`、`render_all.sh`、`isaac_film_vast.py`、`grid.py` 搬过来以后没有重跑，只过了语法和静态检查。它们的结果是搬之前在临时目录里跑出来的。
- 没有做的：30 局评测；整套流程在 4090 上能不能跑。`look_all.sh` 里 main 的检出目录、深度模型的 Python、GPU 编号都是 vastgpu 的默认值，换机器用环境变量改，见脚本开头。

## vastgpu 上跑这些要知道的

- 0 号卡留给别的任务，Isaac、深度模型和 GraspGen-X 都只用 1 号卡。
- 同机的 esc session 会同时起三个 Isaac 服务，负载到过 100 以上，而容器只有 15 个核。那时 Isaac 启动要 5 分钟，一次 30 局的看一眼会超时。可以设 `NICE=10 LOOK_TIMEOUT=5400`。
- 它的脚本叫 `depth_server_esc.py`、`official_server_esc.py`。查找或结束自己的进程要按完整路径匹配，不要按 `depth_server` 这样的短名。
- SAM 3 服务在 127.0.0.1:18832，是别的 session 起的，只调用不要动。
