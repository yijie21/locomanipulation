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
