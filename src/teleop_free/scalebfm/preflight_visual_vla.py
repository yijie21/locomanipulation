"""One real training batch forward/backward; no optimizer step or saved model."""
import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import time

from train_visual_vla import build_config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError('Preflight ID already exists; inspect saved results/log')
    out.mkdir(parents=True)
    import torch
    from transformers import set_seed
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    set_seed(17)
    config = build_config(args.dataset, out, 17)
    config.save(out / 'config.yaml')
    pipeline = Gr00tN1d7Pipeline(config, out)
    pipeline.setup()
    loader = torch.utils.data.DataLoader(pipeline.return_dataset()[0], batch_size=4,
        collate_fn=pipeline.return_collator(), num_workers=0)
    batch = next(iter(loader))
    def shapes(x):
        if torch.is_tensor(x):
            return dict(shape=list(x.shape), dtype=str(x.dtype))
        if isinstance(x, Mapping):
            return {k:shapes(v) for k,v in x.items()}
        if isinstance(x, (tuple, list)):
            return [shapes(v) for v in x]
        return str(type(x))
    (out / 'batch_shapes.json').write_text(json.dumps(shapes(batch), indent=2))
    model = pipeline.return_model().cuda().train()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(**batch)
        loss = result['loss']
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite training loss or gradients')
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = dict(loss=float(loss.detach()), finite=bool(torch.isfinite(loss)),
        backward_s=time.time() - start, peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        trainable_parameters=parameters, estimated_8bit_optimizer_gb=2 * parameters / 1e9,
        optimizer_steps=0, batch_size=4, effective_batch=64)
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
