# 旧数据转成当前格式（2026-10-07 对齐官方之后）

> **已归档（2026-10-20）**：VLA 数据采集与训练这一系列暂不使用，这一页不再随代码更新，命令可能已经跑不通。现在的主线是[场景 RSI](rsi.md)。

2026-10-06 到 10-07，数据集、训练和评测按 NVIDIA 官方 SONIC + GR00T 教程做了对齐（逐项见[与官方教程的对照](official_comparison.md)）。这一页说明在这之前采的数据怎样转成现在的格式。

结论：不用重新采集。这次对齐没有改采集链路：专家、仿真、SONIC 控制、exporter、`start_stack.sh`、`run_agent_server.sh` 都没变；exporter 一直按 50 Hz 每 20 ms 写一行，现在要用的行原始录制里都有。要重做的只有补渲（离线出图的运行）和转换。

| 改了什么 | 对旧数据意味着什么 |
|---|---|
| 数据集帧率 `dataset.fps` 改成 50（warehouse mars 线的实验文件；bottle、crate 的默认仍是 25，要对齐官方也按这一页转） | 以前的数据集是 25 fps，转换时隔一行取一行。要按 50 重新转换；离线出图的运行还要先把每一行的图补渲出来 |
| 训练图加相机延迟 `images.camera_delay` | 只用于离线补渲：图按一台 30 fps、延迟 20 到 30 ms 的相机出。实时录制的运行不能加 |
| state 改回 46 维 | 不用做任何事。选哪些字段由训练配置决定，数据集里都有 |
| 评测节拍、真机客户端 | 和数据无关 |

## 1. 先看你的数据是哪一种

```bash
. scripts/paths.sh
OLD=outputs/runs/my_old_run
awk '/^backend:/{b=1} b&&/^  render:/{print $2; exit}' "$OLD/config.yaml"
```

做什么：打印旧运行的 `backend.render`。命令都在仓库根目录执行，`my_old_run` 换成你的运行名。
得到什么：打印 `offline` 是离线出图（warehouse 线，采集时仿真不出图，图由 Unreal、Isaac 或 MuJoCo 事后补渲），按第 3 节做。什么都不打印或打印 `live` 是实时录制（bottle、crate 线，图是仿真采集时自己出的，约 33 Hz），按第 2 节做。

两种都要先做下面这一步：从旧运行派生出一个 50 fps 的新运行。运行的 `config.yaml` 开跑后就不能改，所以要另建运行目录；旧运行和它已经转好的数据集保持不动。

```bash
NEW=outputs/runs/my_old_run_50hz
mkdir -p "$NEW"
rsync -a --exclude train_dataset --exclude val_dataset --exclude model --exclude eval \
  --exclude summary.json --exclude config.yaml --exclude convert.log \
  --exclude 'audit_*.log' --exclude 'stats_*.log' --exclude train.log "$OLD/" "$NEW/"
python3 - "$OLD" "$NEW" <<'EOF'
import json, os, sys
old, new = sys.argv[1:]
collect = json.load(open(os.path.join(old, "summary.json")))["collect"]
collect["manifests"] = [os.path.join(os.path.abspath(new), os.path.basename(m)) for m in collect["manifests"]]
json.dump({"collect": collect}, open(os.path.join(new, "summary.json"), "w"), indent=1)
EOF
sed "s/^name: .*/name: $(basename "$NEW")/" "$OLD/config.yaml" > "$NEW/config.yaml"
grep -n '^dataset:' "$NEW/config.yaml" || printf 'dataset:\n  fps: 50\n' >> "$NEW/config.yaml"
```

做什么：复制采集留下的东西（`collect/`、`trace/`、`manifest_*.jsonl`、场景目录等），不复制数据集、模型和评测；`summary.json` 只留 `collect` 条目，manifest 指向新目录里的副本；`config.yaml` 换运行名，末尾加 `dataset: {fps: 50}`。
得到什么：`$NEW/` 下一份可以转换的运行。最后一条如果打印了一行 `dataset:`，说明旧文件里已经有这个块，要手工把它的 `fps` 改成 50。
原始录制的 session 两个运行共用（路径记在 `summary.json` 和 `collect/inputs.json` 里），实时录制的运行只读它们；离线出图的运行补渲时会写它们，见第 3 节。

## 2. 实时录制的数据：重新转换

```bash
bash scripts/convert_data.sh "$NEW"
```

做什么：从原始录制里每一行都取（50 fps），转成 LeRobot 数据集，留出验证集，并审计。不用 GPU。
得到什么：`$NEW/train_dataset/` 和 `$NEW/val_dataset/`，帧数是原来 25 fps 数据集的两倍。2026-10-07 用 `crate_smoke` 试过：训练集 407 帧变成 814 帧；逐帧比对，39% 的帧和上一帧相同，不同的画面每秒约 30 张，和官方真机数据（30 fps 相机铺在 50 Hz 网格上）是一回事。
要注意：
- 那次试转审计有一项没过，`body trails the token by a median +0 frames`（25 fps 时是 +3 帧，通过），所以命令以非零状态结束，但数据集已经写出来了。那次训练集只有 1 条，可能是样本太少；完整数据转完要看 `audit_train.log` 里这一项。
- 不要加 `images.camera_delay`：录下的图是仿真自己出的，没法重画，加了转换会拒绝。

## 3. 离线出图的数据：重新补渲，再转换

先给新运行打开相机延迟（warehouse mars 线现在的做法），确认 `images` 块里有这一行：

```bash
grep -n -A3 '^images:' "$NEW/config.yaml"
sed -i '/^images:/a\  camera_delay: {fps: 30, latency_ms: [20, 30]}' "$NEW/config.yaml"   # 没有 camera_delay 时才执行
```

做什么：第一条列出 `images` 块；已经有 `camera_delay` 就跳过第二条。第二条在 `images:` 下面加一行，补渲时每一行的图取自比状态早 20 到 63 ms 的时刻。

然后决定原始 session 要不要保留原样。补渲按 `collect/inputs.json` 里的路径直接改写 session 里的视频：

- 共用（省磁盘，也是 `docs/handoff/real-robot-sim2real-plan-20261006.md` 第 4.3 节的做法）：旧运行已经转好的数据集不受影响（转换时视频是重新编码的副本），但旧运行以后不能再按原来的设置补渲或转换。
- 保留原样：先把 session 复制一份，再把新运行的 `summary.json` 和 `collect/inputs.json` 里的 `sessions` 都改成副本的路径。两处都要改：只改 `summary.json` 时，补渲仍然写进原始 session。

```bash
for s in $(python3 -c "import json; print(' '.join(json.load(open('$NEW/collect/inputs.json'))['sessions']))"); do
  rm -f "$s/meta/offline_render.json"
done
```

做什么：删掉每个 session 的补渲标记，之后每一行都重渲。标记在时补渲会拒绝：Unreal 和 Isaac 在 25 fps 时只渲了一半的行（报 `drawn for a 25 fps dataset and dataset.fps is 50; remove`）；MuJoCo 虽然每行都渲过，但当时没有相机延迟（报 `not mixing`）。

用旧运行的渲染器补渲（`images.source` 是什么就用哪条）：

```bash
bash scripts/render_dataset_cameras_unreal.sh "$NEW"            # Unreal：先 bash scripts/start_unreal.sh --gpu 1
bash scripts/render_dataset_cameras_isaac.sh "$NEW" --gpu 1     # Isaac
bash scripts/render_dataset_cameras.sh "$NEW" --gpu 0           # MuJoCo
```

做什么：按状态轨迹给每个成功局的每一行渲三路相机图，写回 session，打印 `pictures 20-63 ms older than the state`。
得到什么：每局一行统计。2026-10-07 实测：Unreal 每局约 95 秒（550 局约 15 小时），MuJoCo 每局约 5 秒。要用 2026-10-07 之后的代码：之前的版本在还没有补渲标记的 session 上第一局就报 `KeyError: 'dataset_fps'` 退出。

最后转换：

```bash
bash scripts/convert_data.sh "$NEW"
```

得到什么：同第 2 节，50 fps。审计按相机延迟查图和表的错位：2026-10-07 用一个 6 条的旧 warehouse 运行试过，训练集期望 +2.6 帧、实测中位 +2 帧，通过；验证集只有 1 条、2 次测量，中位 +1 帧，没过。条数少时这一项不可靠，同日 50 条的 `warehouse_mars_50` 两边都过了。

## 4. 之后

- 训练照常：`bash scripts/train.sh "$NEW" --gpu 0`，state 自动是 46 维。
- 闭环评测：crate 和 warehouse 的 lockstep 评测接受 50 fps；bottle 线的评测器（`eval_closed_loop`）目前只接受 25 fps，50 fps 的 bottle checkpoint 只能走[真机部署客户端](../src/real_robot/README.md)的 sim2sim。
- 还会再转换一次：夹爪改用 Dex1 原始单位（TODO 16.3）做完后要重新转换，同样不用重新采集。
