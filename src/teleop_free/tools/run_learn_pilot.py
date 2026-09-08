"""Learnability pilot: random-spawn ready-grasp collection -> convert -> train -> 100-round closed loop.

Differs from run_ready_pilot.py on purpose:
  * random spawns drawn from the CLOSED-LOOP EVAL distribution (edge 1.1-1.3 m,
    yaw 20 deg, lateral 0.20 m, jitter 0.20) instead of a frozen 0.9-1.0 m case
    bank -- the question is whether the policy is learnable on the benchmark
    distribution, not whether the expert passes a gate;
  * full stack restart every batch of rounds (exporter thread growth,
    container pid limit 3840);
  * evaluation is the standard fresh-round benchmark (seed 6000, same spawn
    args as base100), not a case bank, so the number sits next to 50/100.
Stage receipts make a rerun resume at the first incomplete stage.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / 'third_party/GR00T-WholeBodyControl/.venv_sim/bin/python'
DATA = ROOT / 'third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python'
TRAIN = ROOT / 'third_party/Isaac-GR00T/.venv/bin/python'
SRC = ROOT / 'src/teleop_free'
SESSIONS = Path('/workspace/tmp/p03/outputs')
SPAWN = ['--spawn-edge', '1.1', '1.3', '--face-object', '--spawn-yaw-err', '20',
         '--spawn-lateral', '0.20', '--edge-jitter', '0.20']


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reach-config', type=Path, required=True)
    ap.add_argument('--scene', type=Path, required=True, help='frozen scene dir (LOCOMANI_SIM_DIR)')
    ap.add_argument('--initial-token-file', type=Path,
                    default=ROOT / 'configs/deployment/ready_grasp_initial_token.json')
    ap.add_argument('--successes', type=int, default=60)
    ap.add_argument('--batch-rounds', type=int, default=10)
    ap.add_argument('--max-batches', type=int, default=12)
    ap.add_argument('--holdout', type=int, default=6)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--save-steps', type=int, default=1500)
    ap.add_argument('--train-gpu', default='1')
    ap.add_argument('--eval-rounds', type=int, default=100)
    ap.add_argument('--eval-seed', type=int, default=6000)
    ap.add_argument('--seed0', type=int, default=27000)
    ap.add_argument('--through', choices=('collect', 'convert', 'train', 'eval'), default='eval')
    ap.add_argument('--gym-env', action='append', default=[], metavar='KEY=VAL',
                    help='extra environment for the collection gym only (e.g. GYM_SHUFFLE_BAND=0)')
    args = ap.parse_args()

    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    cfg = out / 'reach_config.json'
    if not cfg.exists():
        shutil.copy2(args.reach_config, cfg)
    scene = out / 'scene'
    if not scene.exists():
        shutil.copytree(args.scene, scene)
    token = args.initial_token_file.resolve()
    src_files = sorted(p for b in ('src/teleop_free', 'configs', 'scripts')
                       for p in (ROOT / b).rglob('*') if p.suffix in ('.py', '.json', '.sh'))
    prov = out / 'provenance.json'
    if not prov.exists():
        prov.write_text(json.dumps(dict(
            git=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            reach_config=digest(cfg), initial_token=digest(token),
            scene={p.name: digest(p) for p in scene.iterdir() if p.is_file()},
            spawn=SPAWN, gym_env=args.gym_env, started=time.time(),
            sources={str(p.relative_to(ROOT)): digest(p) for p in src_files}), indent=1))

    base_env = dict(os.environ, LOCOMANI_SIM_DIR=str(scene), SIM_VIS='none', SCENE_RANDOM='0',
                    SIM_VIDEO_INCLUDE_INIT='1', HF_HOME='/workspace/huggingface_cache',
                    PYTHONUNBUFFERED='1', WANDB='0')
    for k in ('NO_ENCODER', 'CLEAN_DEPLOY', 'SIM_VIDEO_DIR', 'TF_INIT_TOKEN_FILE', 'CUDA_VISIBLE_DEVICES'):
        base_env.pop(k, None)

    def run(name, cmd, extra=None):
        t0 = time.time(); log = out / f'{name}.log'
        print(f'{name}: start  log={log}', flush=True)
        with log.open('a') as f:
            f.write('\nCOMMAND ' + json.dumps([str(x) for x in cmd]) + '\n'); f.flush()
            rc = subprocess.call([str(x) for x in cmd], cwd=ROOT,
                                 env=base_env if extra is None else dict(base_env, **extra),
                                 stdout=f, stderr=subprocess.STDOUT)
        with (out / 'timings.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=name, start=t0, seconds=time.time() - t0, rc=rc)) + '\n')
        print(f'{name}: rc={rc} in {time.time() - t0:.0f}s', flush=True)
        if rc:
            raise RuntimeError(f'{name} exited {rc}: {log}')

    def done(stage): return (out / f'{stage}.done').exists()
    def mark(stage): (out / f'{stage}.done').write_text(str(time.time()))

    stack_up = False
    def stop():
        nonlocal stack_up
        if stack_up:
            # Scoped to instance 0: stop_stack.sh's pkill is unscoped and took
            # down another session's instance-2 stack twice on 2026-09-07.
            run('stop_stack', ['bash', SRC / 'stop_instance.sh', '0', '5590']); stack_up = False
            subprocess.call(['bash', '-c', 'for p in $(pgrep -f run_instance_exporter); do grep -qz "TF_CAMERA_PORT=5555" /proc/$p/environ 2>/dev/null && kill -9 $p; done'])

    server = None
    try:
        # ---------------- collect ----------------
        collect = out / 'collect'; collect.mkdir(exist_ok=True)
        inputs = collect / 'inputs.json'
        state = json.loads(inputs.read_text()) if inputs.exists() else dict(sessions=[], manifests=[], kept=0, batches=0)
        while not done('collect') and state['kept'] < args.successes and state['batches'] < args.max_batches:
            b = state['batches']; tag = f'{b:02d}'
            manifest = collect / f'manifest_{tag}.jsonl'
            if manifest.exists():
                raise RuntimeError(f'{manifest} exists but is not in inputs.json; inspect before continuing')
            env = dict(SIM_VIDEO_DIR=str(collect / f'videos_{tag}'),
                       SIM_STATE_TRACE_DIR=str(collect / f'state_trace_{tag}'),
                       TF_FRAME_AUDIT_PATH=str(collect / f'frame_audit_{tag}.jsonl'),
                       TF_EXPORTER_SCRIPT=str(SRC / 'tools/run_instance_exporter.py'))
            env.update(dict(kv.split('=', 1) for kv in args.gym_env))
            stack_up = True
            before = set(SESSIONS.iterdir())
            run(f'start_stack_{tag}', ['bash', SRC / 'start_stack.sh', 'pick up the water bottle'], env)
            # Wait for THIS stack's exporter dir. max(mtime) alone raced: a slow
            # teardown could leave the previous batch's dir newest, giving an
            # off-by-one session<->manifest join that only surfaces at convert.
            session = None
            for _ in range(60):
                fresh = [d for d in SESSIONS.iterdir() if d not in before]
                if fresh:
                    session = max(fresh, key=lambda p: p.stat().st_mtime); break
                time.sleep(1)
            if session is None:
                raise RuntimeError('no new exporter session dir appeared after start_stack')
            state['sessions'].append(str(session)); state['manifests'].append(str(manifest))
            inputs.write_text(json.dumps(state, indent=1))
            try:
                run(f'collect_{tag}', [SIM, SRC / 'grasp_gym.py', '--reach-config', cfg, '--targets', 'water_bottle',
                                       '--rounds', str(args.batch_rounds), '--until-ok', str(args.successes - state['kept']),
                                       '--seed', str(args.seed0 + 100 * b), '--stand-retries', '0', '--record',
                                       *SPAWN, '--manifest', manifest], env)
            finally:
                stop()
            rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()] if manifest.exists() else []
            got = sum(bool(r.get('ok') and r.get('produced')) for r in rows)
            state['kept'] += got; state['batches'] += 1
            inputs.write_text(json.dumps(state, indent=1))
            print(f'collect: batch {tag} {got}/{len(rows)} ok, total kept {state["kept"]}/{args.successes}', flush=True)
        if not done('collect'):
            if state['kept'] < args.successes:
                raise RuntimeError(f'collection stopped at {state["kept"]}/{args.successes} after {state["batches"]} batches')
            mark('collect')
        if args.through == 'collect': return

        # ---------------- convert ----------------
        train_ds, val_ds = out / 'train_dataset', out / 'val_dataset'
        if not done('convert'):
            for d in (train_ds, val_ds):
                if d.exists(): shutil.rmtree(d)
            run('convert', [DATA, SRC / 'tools/convert_to_hf_format.py', '--sessions', *state['sessions'],
                            '--manifest', *state['manifests'], '--trim-verification-tail', '--out', train_ds,
                            '--holdout', str(args.holdout), '--holdout-out', val_ds, '--jobs', '4'])
            for label, ds in (('train', train_ds), ('val', val_ds)):
                run(f'stats_{label}', [TRAIN, '-m', 'gr00t.data.stats', '--dataset-path', ds,
                                       '--embodiment-tag', 'UNITREE_G1_SONIC',
                                       '--modality-config-path', ROOT / 'configs/gr00t_sonic_2cam.py'])
                run(f'audit_{label}', [DATA, SRC / 'tools/check_dataset.py', ds])
            mark('convert')
        if args.through == 'convert': return

        # ---------------- train ----------------
        ckpt = out / 'model' / f'checkpoint-{args.steps}'
        if not done('train'):
            if (out / 'model').exists(): shutil.rmtree(out / 'model')
            run('train', ['bash', ROOT / 'scripts/train_bottle.sh'],
                dict(DATA=str(train_ds), RUN='model', OUT_ROOT=str(out), MAX_STEPS=str(args.steps),
                     SAVE_STEPS=str(args.steps), SAVE_LIMIT='1', WANDB='0', WORKERS='2',  # one checkpoint (disk)
                     GPUS='1', SINGLE_GPU='1', CUDA_VISIBLE_DEVICES=args.train_gpu,
                     # Another instance's MuJoCo EGL context sits on device 0 (~1.6-2.1 GB)
                     # whatever CUDA_VISIBLE_DEVICES says; the default 2048 MiB guard trips on it.
                     SQUAT_MAX=os.environ.get('SQUAT_MAX', '4096')))
            if not ckpt.exists(): raise RuntimeError('no checkpoint after training')
            mark('train')
        if args.through == 'train': return

        # ---------------- eval ----------------
        if not done('eval'):
            ev = out / 'eval'; ev.mkdir(exist_ok=True)
            stack_up = True
            run('start_eval_stack', ['bash', SRC / 'start_stack.sh', 'pick up the water bottle'],
                dict(NO_ENCODER='1', SIM_VIDEO_DIR=str(ev / 'videos')))
            slog = (ev / 'serve.log').open('a')
            server = subprocess.Popen([str(TRAIN), str(SRC / 'tools/serve_policy.py'), '--model-path', str(ckpt),
                                       '--port', '5590'], cwd=ROOT,
                                      env=dict(base_env, CUDA_VISIBLE_DEVICES=args.train_gpu),
                                      stdout=slog, stderr=subprocess.STDOUT)
            t_end = time.monotonic() + 600
            while 'listening' not in (ev / 'serve.log').read_text():
                if server.poll() is not None or time.monotonic() > t_end:
                    raise RuntimeError('policy server did not come up')
                time.sleep(3)
            run('eval', [SIM, SRC / 'tools/eval_closed_loop.py', '--fresh', str(args.eval_rounds),
                         '--stand-retries', '0', '--reach-config', cfg, '--initial-token-file', token,
                         '--client', 'async', '--budget', '90', '--replan-interval', '0.4',
                         '--spawn-edge', '1.1', '1.3', '--seed', str(args.eval_seed),
                         '--out', ev / 'closed_loop.json', '--trace-dir', ev / 'traces'],
                dict(NO_ENCODER='1', TF_DATASET_FPS='25', SIM_VIDEO_DIR=str(ev / 'videos')))
            mark('eval')
    finally:
        if server is not None:
            server.terminate()
            try: server.wait(timeout=10)
            except subprocess.TimeoutExpired: server.kill(); server.wait()
        stop()
    print('pilot complete', flush=True)


if __name__ == '__main__':
    main()
