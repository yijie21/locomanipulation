# Public benchmarks and the run switches (TODO 49)

A run on a public benchmark is one set of switches: the benchmark, where the robot's knowledge of objects and grasps
comes from (simulator truth or perception), perception's input (stereo pair or one RGB-D camera, with or without depth
noise), where the robot believes it is (truth or leg odometry), the pelvis ScaleBFM is told, and the controller.
`python -m locomani.bench table` prints what each benchmark supports and what runs here yet.
Back to [domain navigation](change-map.md). Test names are relative to `tests/`.

| Change | Entry / owner | Configuration or example | Tests |
|---|---|---|---|
| Run switches: choices, resolution (code defaults, profile, `--bench-set`), consistency, signature, record | [options.py](../../src/locomani/bench/options.py) | [configs/bench/](../../configs/bench/simple.yaml) | `test_bench_options.py` |
| A benchmark's capability table (ready / planned per value, with reasons) | [spec.py](../../src/locomani/bench/spec.py), `bench/<name>/__init__.py` | `python -m locomani.bench table` | `test_bench_options.py` |
| Camera streams and privileged state a benchmark session hands out | [channels.py](../../src/locomani/bench/channels.py) | | `test_bench_options.py` |
| Views (truth / camera), sensors (stereo / RGB-D), self pose | [perception/](../../src/locomani/perception/__init__.py) | | `test_bench_options.py` |
| SIMPLE adapter: config (former env switches), env server, client, one episode, session (slots, servers, look and GraspGen-X pass, batch), judge, practice pools | [bench/simple/](../../src/locomani/bench/simple/__main__.py) (`python -m locomani.bench.simple run`) | [config.py](../../src/locomani/bench/simple/config.py) | `test_simple_config.py`, `test_simple_session.py`, `test_simple_judge.py`, `test_simple_episodes.py` |
| SIMPLE seed skill package (loop-editable) and its guard | [seed/](../../src/locomani/bench/simple/seed/__init__.py), [guard.py](../../src/locomani/bench/simple/guard.py) | | `test_simple_package.py`, `test_simple_guard.py` |
| Camera tools, head geometry, SAM 3 and stereo depth services, leg odometry, truth view, grasp planning | [perception/](../../src/locomani/perception/head.py) | | `test_simple_package.py` |
