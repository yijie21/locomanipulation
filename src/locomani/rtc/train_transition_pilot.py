"""Small continuation experiment, retaining the old normalization system.

Uses the existing GR00T trainer and original modality/model configuration.
Default keeps sampling unchanged. An optional prefix manifest enables a
paired boundary-sampling experiment without changing labels or the executor.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--reference-config", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prefix-manifest", type=Path)
    ap.add_argument("--boundary-repeats", type=int, default=1)
    ap.add_argument("--prefix-loss-weight", type=float, default=1.0)
    ap.add_argument("--prefix-loss-frames", type=int, default=10)
    args = ap.parse_args()
    if args.steps <= 0 or args.gpus not in (1, 2):
        ap.error("positive steps; one or two GPUs")
    if args.out.exists() and not args.dry_run and int(os.environ.get("RANK", "0")) == 0:
        ap.error("refusing to overwrite an existing training run")
    if args.boundary_repeats < 1 or (args.boundary_repeats != 1 and not args.prefix_manifest):
        ap.error("boundary repetition requires a manifest and positive repeat count")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    import gr00t

    print(f"training_package={gr00t.__file__}", flush=True)
    from gr00t.experiment.launch_finetune import load_modality_config

    load_modality_config(str(Path(__file__).resolve().parents[3] / "configs/gr00t_sonic_2cam.py"))
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.data.data_config import SingleDatasetConfig

    raw = yaml.safe_load(args.reference_config.read_text())
    cfg = get_default_config().load_dict({"model": raw["model"], "training": raw["training"]})
    cfg.data.datasets = [
        SingleDatasetConfig(dataset_paths=[str(args.dataset.resolve())], embodiment_tag="unitree_g1_sonic")
    ]
    # Keep the mapping the checkpoint learned. Re-estimating q01/q99 from
    # fifteen episodes would change input/action coordinates during this
    # short continuation, confounding the data-quality experiment.
    cfg.data.override_pretraining_statistics = False
    import math

    if not math.isfinite(args.prefix_loss_weight) or args.prefix_loss_weight <= 0:
        ap.error("prefix loss weight must be finite and positive")
    if not 1 <= args.prefix_loss_frames <= cfg.model.action_horizon:
        ap.error("prefix loss frames must fit the action horizon")
    cfg.model.action_prefix_loss_weight = args.prefix_loss_weight
    cfg.model.action_prefix_loss_frames = args.prefix_loss_frames
    cfg.training.start_from_checkpoint = str(args.base.resolve())
    cfg.training.output_dir = str(args.out.resolve().parent)
    cfg.training.experiment_name = args.out.name
    cfg.training.num_gpus = args.gpus
    cfg.training.global_batch_size = 8 * args.gpus
    cfg.training.gradient_accumulation_steps = 8 // args.gpus
    cfg.training.learning_rate = 2e-5
    cfg.training.max_steps = args.steps
    cfg.training.save_steps = args.steps
    cfg.training.save_total_limit = 1
    cfg.training.save_only_model = True
    cfg.training.resume_from_checkpoint = False
    cfg.training.optim = "adamw_bnb_8bit"
    cfg.training.use_wandb = False
    cfg.training.dataloader_num_workers = 2
    cfg.validate()
    print(
        json.dumps(
            dict(
                base=str(args.base),
                dataset=str(args.dataset),
                output=str(args.out),
                steps=args.steps,
                effective_batch=64,
                lr=2e-5,
                keep_checkpoint_statistics=True,
            ),
            indent=2,
        ),
        flush=True,
    )
    if args.prefix_manifest:
        from locomani.rtc.prefix_sampling import install

        install(args.prefix_manifest, args.dataset, args.boundary_repeats)
        print(f"prefix_manifest={args.prefix_manifest} boundary_repeats={args.boundary_repeats}", flush=True)
    if args.dry_run:
        return
    if not (args.dataset / "meta/stats.json").exists():
        ap.error("run gr00t.data.stats first")
    if int(os.environ.get("RANK", "0")) == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "sampling_experiment.json").write_text(
            json.dumps(
                dict(
                    manifest=None if args.prefix_manifest is None else str(args.prefix_manifest.resolve()),
                    boundary_repeats=args.boundary_repeats,
                    steps=args.steps,
                    prefix_loss_weight=args.prefix_loss_weight,
                    prefix_loss_frames=args.prefix_loss_frames,
                    base=str(args.base.resolve()),
                    dataset=str(args.dataset.resolve()),
                ),
                indent=2,
            )
        )
    # Startup memory is checked before torchrun by the experiment operator;
    # this process is launched only after the simulation stack has stopped.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from gr00t.experiment.experiment import run

    run(cfg)


if __name__ == "__main__":
    main()
