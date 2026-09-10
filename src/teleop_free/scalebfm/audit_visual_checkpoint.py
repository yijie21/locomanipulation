"""Read-only audit of a completed training checkpoint; never runs an optimizer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from probe_visual_expert import atomic_json


def audit(checkpoint, expected_step):
    import torch
    from safetensors import safe_open

    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if state['global_step'] != expected_step:
        raise ValueError('Trainer step differs from the requested checkpoint')
    frozen = json.loads((run / 'frozen_training.json').read_text())
    if expected_step > frozen['config']['training']['max_steps']:
        raise ValueError('Checkpoint exceeds the frozen optimizer budget')

    # mmap avoids allocating another multi-GB optimizer copy alongside training.
    # These are our own local training artifacts, including non-tensor RNG state.
    optimizer = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', mmap=True, weights_only=False)
    steps = sorted({int(s['step']) for s in optimizer['state'].values() if 'step' in s})
    if steps != [expected_step] or not optimizer['param_groups']:
        raise ValueError(f'Optimizer state is incomplete or at another step: {steps}')
    scheduler = torch.load(checkpoint / 'scheduler.pt', map_location='cpu', weights_only=False)
    if scheduler['last_epoch'] != expected_step:
        raise ValueError('Learning-rate scheduler and optimizer steps disagree')
    rng = torch.load(checkpoint / 'rng_state.pth', map_location='cpu', weights_only=False)
    if not {'python', 'numpy', 'cpu', 'cuda'}.issubset(rng):
        raise ValueError('Missing Python/NumPy/CPU/CUDA RNG state')
    cuda_rng = rng['cuda'] if isinstance(rng['cuda'], (list, tuple)) else [rng['cuda']]
    if rng['cpu'].numel() == 0 or not cuda_rng or any(s.numel() == 0 for s in cuda_rng):
        raise ValueError('Empty RNG state')

    copied = {}
    for source in sorted((run / 'processor').glob('*')):
        if source.is_file():
            target = checkpoint / source.name
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError(f'Checkpoint processor artifact differs: {source.name}')
            copied[source.name] = digest
    if not {'processor_config.json', 'statistics.json', 'embodiment_id.json'}.issubset(copied):
        raise ValueError('Missing inference processor configuration or normalization')
    for name in ('config.json', 'experiment_cfg/config.yaml', 'experiment_cfg/conf.yaml'):
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise ValueError(f'Missing checkpoint configuration: {name}')

    tensors = {}
    for shard in sorted(checkpoint.glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as saved:
            for key in saved.keys():
                if key in tensors:
                    raise ValueError(f'Duplicate model tensor: {key}')
                tensors[key] = shard.name
    if not tensors:
        raise ValueError('No readable model tensors')
    index = checkpoint / 'model.safetensors.index.json'
    if index.exists() and json.loads(index.read_text())['weight_map'] != tensors:
        raise ValueError('Model shard index does not match saved tensor headers')
    return dict(audited_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint),
        passed=True, global_step=state['global_step'], optimizer_state_entries=len(optimizer['state']),
        optimizer_steps=steps, scheduler_last_epoch=scheduler['last_epoch'],
        rng_groups=sorted(rng), processor_sha256=copied, model_tensor_count=len(tensors),
        saved_bytes=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),
        validation_scope='Artifact readability, step agreement, RNG presence, processor equality and model shard coverage; no resumed optimizer step or closed-loop evaluation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.step)
    atomic_json(Path(args.out), result)
    print(json.dumps(result, indent=2))
