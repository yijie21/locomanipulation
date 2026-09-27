# STATE — 2026-09-27（会话：warehouse checkpoint-5000 开环/闭环 → 评测器修复 → `images.source` → 6.9 提交 → lockstep 计划）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，handoff 提交与本次文档复查都已推送。
lockstep 计划与其链接（4.7.10.9）已随文档复查（TODO 4.6.5，新增 `docs/reference/glossary.md`）一起提交推送。
没有运行中的栈、镜像、训练或渲染；GPU1 上约 3.7 GB 是别的项目的。

## 工作树

- 未跟踪、与本项目无关、不要提交：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`。
- 运行 `box_room_grasp_200_cap120_20260925` 的 `config.yaml` 被手改（`eval.rounds: 10`、warehouse token），`summary.json` 手补 train 条目；原件 `*.bak_20260927`。

## 当前进度（已确认，数字在台账 crate.md 2026-09-27 末节）

- 已确认：checkpoint-5000 开环 MSE 0.00123 / MAE 0.0217（预测均值 0.0350，块首保持 0.00179）。
- 已确认：评测器修 4 处（摆放、物体名、初始 token、夹爪下限）后，Isaac 镜像闭环 10 局 **9/10** 抬起；9/9 成功后再持 10 s 不掉。录像 `outputs/videos/warehouse_ckpt5000_closed_loop10_isaac_20260927.zip`。
- 已确认：训练图（81 个有成功局的批次全部 `renderer: isaac`）与闭环图（713/713 个 chunk）都是 Isaac。
- 已确认：图像来源是整次运行的 `images.source`，冲突组合加载即拒；`box_platform` 有 Isaac stage，冒烟走通采集→补渲→convert→录像→镜像。
- 已确认（实测）：Isaac 三路按需渲 p50 95.5 ms、连续补渲中位 69.8 ms；双臂 blocking 周期 810 ms，取图+推理 166 ms 期间不发指令、参考冻结、之后跳 4 帧。

## 卡点 / 待验证

- 待验证：10 局 9/10 能否在 100 局里站住（这条线 10 局方差大）。
- 待验证：lockstep 能否复现墙钟成绩（2026-09-02 单臂校准未过，46 vs 32/100，原因未查明）——Isaac 闭环对齐现实的方案以它为前提。
- 假设：间隙里参考冻结 + 跳帧会压低闭环表现；未做对照。
- 待验证：按需渲与连续渲是否同图（4.7.9.8）。

## 下一步（编号见 `TODO.md`，顺序等用户定）

- **4.7.6.1.2.6 / .8**："后退"一列怎么改，然后 100 局闭环（先把 run 的 `eval/` 挪到 `eval_smoke10_20260927/`，`eval.rounds` 改回 100）。
- **4.7.11**（先 W0）：lockstep + Isaac 评测，计划 `docs/handoff/lockstep-isaac-eval-plan-20260927.md`；过渡是 4.7.6.1.2.11 补帧。
- **6.9**：③ 附着反馈是否进默认配置；同几何房间 26% vs 台上 43% 的差异。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export PYTHONPATH="$LOCOMANI_ROOT/src" TMPDIR=/workspace/tmp
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
bash scripts/cpu_headroom.sh; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑；停栈后等 ≥20 s 再起
R=outputs/runs/box_room_grasp_200_cap120_20260925
# 闭环（栈+策略 GPU0，Isaac 镜像 GPU1）；评测前逐条核对见 docs/warehouse.md §6.1
bash scripts/run_overnight.sh configs/experiments/box_room_grasp_200_cap120_20260925.yaml --gpu 0 --isaac-gpu 1 --line bimanual --from eval
python3 $R/eval/analysis/rounds_table.py                     # 逐局表（抬起/合爪时刻、倾角、后退）
bash scripts/render_isaac_videos.sh $R --traces $R/eval/state_trace --out $R/eval --insets --gpu 1   # Isaac 录像，一局 2–3.5 min
bash scripts/open_loop_eval.sh --model-path $R/model/checkpoint-5000 --dataset-path $R/val_dataset --out-dir $R/eval/open_loop_ckpt5000
# 提交：make check（STATE ≤ 60 行）→ 按文件名 add（混着别人改动的文件用 hash-object + update-index 只暂存本次部分）
#       → 在只含暂存内容的干净 worktree 里跑全套测试 → git commit --no-verify → fetch 确认无远端新提交再 push
```
