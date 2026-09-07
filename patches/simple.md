# SIMPLE 本地补丁

渲染线用 SIMPLE 只为两件事:一个装了 `isaacsim` 的 venv,以及它转换好的场景与
robocasa 物体 USD。**离线重渲不 import `simple` 包**,场景走我们自己的
`scene_locomani.xml` 经 Isaac 的 MJCF importer,相机由 MuJoCo FK 驱动。

SIMPLE 工作副本约 35 GB(绝大部分是 `data/`),所以不把它 vendor 进本仓库,
改为**固定 commit 加补丁文件**。

- remote: `https://github.com/physical-superintelligence-lab/SIMPLE.git`
- pinned commit: `6d10628794d9c7de4596b4f2afb2c054a637c2bc`
- 默认位置: `$LOCOMANI_WORKSPACE/code/SIMPLE`,用 `SIMPLE_ROOT` 覆盖

用 `bash scripts/setup_render_runtime.sh` 应用,可重复执行。

| 补丁 | 必需 | 内容 |
|---|---|---|
| `0001-mujoco-renderer-get.patch` | **是** | `engines/mujoco.py` 把 `self.renderers[name]` 改成 `.get(name)`。MJCF 里写的相机若没有配套 sensor renderer,原代码抛 KeyError,我们的场景必然命中 |
| `0002-replay-video-recorder.patch` | 否 | `cli/replay*.py` 去掉重复的 `VideoRecorder` 包装,包两次会对同一个 mp4 开两个 cv2 writer 把文件写坏。只在跑 SIMPLE 自己的 replay 时需要 |
| `0003-dex1-registration.patch` | 否 | 注册 dex1 机器人与 hug-container 任务。只给对照 harness(票 11)用 |
| `0004-dex1-new-files.patch` | 否 | 上一条需要的两个新文件本体。与 0003 成对使用 |

0002 到 0004 与主线无关,`--with-replay` / `--with-alignment` 才会应用。
