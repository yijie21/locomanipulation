"""Sequential GPU0 experiment driver; durable stages survive context loss.

Never starts a stack, kills another process, repeats a completed stage, or tunes
on held-out results. Stops on any failed stage for diagnosis. Run only after the
GPU reservation and preregistered plan in docs have been checked.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from probe_visual_expert import atomic_json

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / 'third_party/Isaac-GR00T/.venv/bin/python'
HERE = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


class Driver:
    def __init__(self, root):
        self.root = root
        self.path = root / 'pipeline_v2_state.json'
        self.cohorts = {arm:root / f'cohort_{arm}_v2' for arm in 'ABC'}
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(created_utc=utc(), stages={})
        self.state['pid'] = os.getpid()
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', NO_ALBUMENTATIONS_UPDATE='1',
                        PYTHONPATH=str(ROOT / 'src'))
        atomic_json(self.path, self.state)

    def run(self, name, command, gpu=False):
        stage = self.state['stages'].get(name)
        if stage and stage['status'] == 'complete':
            return
        if stage and stage['status'] == 'failed':
            raise RuntimeError(f'{name} previously failed; diagnose and record recovery before restarting')
        if stage and stage['status'] == 'running':
            raise RuntimeError(f'{name} has a running/interrupted stage record; inspect PID{stage.get("pid")} before adopting or resuming')
        log = self.root / f'{name}.driver.log'
        env = self.env.copy()
        if not gpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        stage = dict(status='running', started_utc=utc(), command=list(map(str,command)), log=str(log))
        self.state['stages'][name] = stage
        self.state['current_stage'] = name
        with log.open('a') as output:
            child = subprocess.Popen(list(map(str,command)), cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            stage['pid'] = child.pid; atomic_json(self.path, self.state)
            print(json.dumps(dict(stage=name, pid=child.pid, started_utc=stage['started_utc'])), flush=True)
            code = child.wait()
        stage.update(status='complete' if code == 0 else 'failed', returncode=code, ended_utc=utc())
        atomic_json(self.path, self.state)
        if code:
            raise RuntimeError(f'{name} failed with exit{code}; inspect {log}')

    def script(self, name, script, *args, gpu=False):
        self.run(name, [PYTHON, HERE / script, *args], gpu)

    def collect(self, arm):
        path = self.cohorts[arm]
        status_file = path / 'run.json'
        if status_file.exists():
            status = json.loads(status_file.read_text())
            if status['status'] == 'running':
                # Adopt an already-authorized collection, e.g. B started during
                # interface validation. Do not launch a duplicate GPU process.
                pid = status['pid']
                while status['status'] == 'running':
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        status = json.loads(status_file.read_text())
                        if status['status'] == 'complete':
                            break
                        raise RuntimeError(f'Collection{arm} exited without a completion marker; inspect it')
                    self.state['current_stage'] = f'waiting_existing_cohort_{arm}'
                    atomic_json(self.path, self.state)
                    time.sleep(5)
                    status = json.loads(status_file.read_text())
            if status['status'] == 'complete':
                return
        extra = ['--normal-cohort', self.cohorts['B']] if arm == 'C' else []
        if arm == 'B':
            retired = json.loads((self.root / 'cohort_B/retirement.json').read_text())
            extra += ['--prior-attempts', str(retired['physical_attempts_started'])]
        # This driver implements the old registered experiment, not new default collections.
        self.script(f'collect_{arm}', 'collect_visual_cohort.py', '--arm', arm, '--out', path,
                    '--scene-profile', 'legacy_edges_v1', *extra, gpu=True)
        if json.loads(status_file.read_text())['status'] != 'complete':
            raise RuntimeError(f'Collection{arm} exhausted its registered attempt budget')

    def execute(self):
        acceptance = json.loads((self.root / 'c3_final_acceptance.json').read_text())
        if not acceptance.get('ready_for_formal_comparison'):
            raise RuntimeError('Final expert/runtime qualification is incomplete')
        for arm in 'BAC':
            self.collect(arm)
        for arm in 'ABC':
            for split in ('train', 'val'):
                dataset = self.root / f'dataset_{arm}_{split}'
                extra = ['--baseline-control'] if arm == 'A' else []
                self.script(f'export_{arm}_{split}', 'export_visual_dataset.py', '--inputs',
                    self.cohorts[arm] / f'{split}_inputs.json', '--out', dataset, *extra)
                self.run(f'stats_{arm}_{split}', [PYTHON, '-m', 'gr00t.data.stats', '--dataset-path', dataset,
                    '--embodiment-tag', 'UNITREE_G1_SONIC', '--modality-config-path', ROOT / 'configs/gr00t_bfm_visual_2cam.py'])
        # Freeze all six resolved configurations before the first comparative
        # optimizer update. The shared algorithmic settings are identical.
        for seed in (17,29):
            for arm in 'ABC':
                self.script(f'freeze_{arm}_s{seed}', 'train_visual_vla.py', '--dataset', self.root / f'dataset_{arm}_train',
                    '--out', self.root / f'model_{arm}_s{seed}', '--seed', str(seed), '--dry-run')
        for seed in (17,29):
            for arm in 'ABC':
                model = self.root / f'model_{arm}_s{seed}'
                self.script(f'train_{arm}_s{seed}', 'train_visual_vla.py', '--dataset', self.root / f'dataset_{arm}_train',
                    '--out', model, '--seed', str(seed), gpu=True)
                checkpoint = model / 'checkpoint-3000'
                state = json.loads((checkpoint / 'trainer_state.json').read_text())
                if state['global_step'] != 3000:
                    raise ValueError('Training did not complete the frozen3000-step budget')
                self.script(f'selection_{arm}_s{seed}', 'eval_visual_vla.py', '--model', checkpoint, '--out',
                    self.root / f'selection_{arm}_s{seed}', '--validation-dataset', self.root / f'dataset_{arm}_val', gpu=True)
        self.script('lock_selection', 'report_visual_comparison.py', '--root', self.root)
        winner = json.loads((self.root / 'selection_locked.json').read_text())['winner']
        for seed in (17,29):
            for arm in sorted(set(['A', winner])):
                self.script(f'confirmation_{arm}_s{seed}', 'eval_visual_vla.py', '--model',
                    self.root / f'model_{arm}_s{seed}/checkpoint-3000', '--out',
                    self.root / f'confirmation_{arm}_s{seed}', '--seeds', '31001:31101', gpu=True)
        self.script('report_confirmation', 'report_visual_comparison.py', '--root', self.root, '--confirmation')
        self.state.update(status='experiments_complete_pending_review', ended_utc=utc())
        atomic_json(self.path, self.state)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    with (root / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Driver(root).execute()


if __name__ == '__main__':
    main()
