"""Opt-in repetition of intact boundary windows; no label/time shifting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def boundary_starts(length, events, horizon=40, prefix=10):
    """Windows placing an observed event in the executable prefix (0..prefix-1)."""
    if not 0 < prefix <= horizon:
        raise ValueError('require 0 < prefix <= horizon')
    valid = length - horizon + 1
    return sorted({i for e in events if e is not None
                   for i in range(max(0, int(e)-prefix+1), min(valid, int(e)+1))})


def repeat_indices(indices, boundary, repeats):
    if repeats < 1: raise ValueError('repeats must be positive')
    a = np.asarray(indices, dtype=np.int64)
    return np.repeat(a, np.where(np.isin(a, boundary), repeats, 1))


def build(dataset, output, horizon=40, prefix=10):
    import pyarrow.parquet as pq
    info = json.loads((dataset/'meta/info.json').read_text())
    rows = []
    for p in sorted((dataset/'data').rglob('*.parquet')):
        d = pq.read_table(p, columns=['episode_index', 'teleop.planner_mode',
                                     'observation.state', 'teleop.right_hand_joints']).to_pydict()
        mode = np.asarray(d['teleop.planner_mode']).reshape(-1)
        state = np.asarray(d['observation.state'])
        grip = np.asarray(d['teleop.right_hand_joints'])[:, 0]
        def first(mask):
            ix = np.flatnonzero(mask)
            return int(ix[0]) if len(ix) else None
        cmd = first(mode == 4)
        knee = state[:, [3, 9]].mean(axis=1)
        # A body-response proxy, not a pelvis-height measurement.
        response = None if cmd is None else first((np.arange(len(mode)) >= cmd) & (knee > knee[cmd] + .25))
        events = dict(squat_command=cmd, knee_response_proxy=response, close=first(grip < .005))
        rows.append(dict(episode_index=int(d['episode_index'][0]), length=len(mode), events=events,
                         starts=boundary_starts(len(mode), events.values(), horizon, prefix),
                         parquet=str(p.relative_to(dataset)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if len(rows) != info['total_episodes']: raise ValueError('episode count mismatch')
    result = dict(dataset=str(dataset.resolve()), fps=info['fps'], horizon=horizon, prefix=prefix, episodes=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    n = sum(len(r['starts']) for r in rows)
    total = sum(max(0, r['length']-horizon+1) for r in rows)
    print(json.dumps(dict(episodes=len(rows),valid_windows=total,boundary_windows=n,
                         uniform_fraction=n/total,four_repeat_fraction=4*n/(total+3*n))))
    return result


def install(manifest, dataset, repeats):
    """Replace only this launch's factory class, leaving vendor sources untouched."""
    from gr00t.data.dataset import factory
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    spec = json.loads(Path(manifest).read_text())
    if Path(spec['dataset']).resolve() != Path(dataset).resolve(): raise ValueError('wrong sampling dataset')
    for row in spec['episodes']:
        if hashlib.sha256((Path(dataset)/row['parquet']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('sampling manifest is stale')
    boundaries = {r['episode_index']: r['starts'] for r in spec['episodes']}

    class PrefixDataset(ShardedSingleStepDataset):
        def shard_dataset(self):
            super().shard_dataset()
            if Path(self.dataset_path).resolve() != Path(dataset).resolve(): raise ValueError('unexpected dataset')
            if self.action_horizon != spec['horizon']: raise ValueError('action horizon mismatch')
            for shard, entries in enumerate(self.sharded_episodes):
                self.sharded_episodes[shard] = [(ep, repeat_indices(ix, boundaries[ep], repeats)) for ep, ix in entries]
            self.shard_lengths = np.array([sum(len(ix) for _, ix in entries) for entries in self.sharded_episodes])
            print(f'[prefix-sampling] repeats={repeats}, windows={self.shard_lengths.sum()}', flush=True)

    factory.ShardedSingleStepDataset = PrefixDataset


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): ap.error('refusing to overwrite manifest')
    build(args.dataset, args.out)
