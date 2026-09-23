"""Fixed-budget GR00T training entry point for the visual-expert comparison.

Use --dry-run to validate/save the resolved configuration without initializing a
GPU. Actual launches require a released GPU selected with CUDA_VISIBLE_DEVICES;
check STATE.md first. This entry point preserves causal image-history padding
and full camera FOV, which the older generic finetune wrapper does not configure.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from locomani.paths import REPO

# 2026-09-10: parents[3] resolved to src/locomani after the refactor move, so every
# ROOT-relative path here (snapshots, modality config, src/gr00t) pointed at a directory
# that does not exist. The repo root comes from locomani.paths, like the rest of the line.
ROOT = Path(REPO)
DEFAULT_BASE_SNAPSHOTS = ROOT / "configs/scalebfm/base_model_snapshots.json"


def build_config(dataset, output, seed, resume=None, snapshots_path=None):
    snapshots_path = Path(snapshots_path or DEFAULT_BASE_SNAPSHOTS)
    snapshots = json.loads(snapshots_path.read_text())
    for path in snapshots.values():
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Frozen base-model snapshot missing: {path}")
    modality_path = ROOT / "configs/gr00t_bfm_visual_2cam.py"
    spec = importlib.util.spec_from_file_location("gr00t_bfm_visual_goal", modality_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag

    config = get_default_config().load_dict(
        {
            "data": dict(
                download_cache=False,
                datasets=[
                    dict(
                        dataset_paths=[str(Path(dataset).resolve())],
                        mix_ratio=1.0,
                        embodiment_tag=EmbodimentTag.UNITREE_G1_SONIC.value,
                    )
                ],
            )
        }
    )
    config.load_config_path = None
    config.data.allow_padding = True
    config.data.seed = seed
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.state_dropout_prob = 0.2
    config.model.use_percentiles = True
    config.model.shortest_image_edge = 240
    config.model.crop_fraction = 1.0
    config.model.image_crop_size = None
    config.model.image_target_size = None
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = snapshots["nvidia/Cosmos-Reason2-2B"]
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.training.start_from_checkpoint = snapshots["nvidia/GR00T-N1.7-3B"]
    config.training.transformers_local_files_only = True
    config.training.output_dir = str(Path(output).resolve())
    config.training.experiment_name = None
    config.training.num_gpus = 1
    config.training.global_batch_size = 4
    config.training.gradient_accumulation_steps = 16
    config.training.learning_rate = 1e-4
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.optim = "adamw_bnb_8bit"
    config.training.max_steps = 3000
    config.training.save_steps = 500
    config.training.save_total_limit = 1
    config.training.save_only_model = False  # Keep optimizer state for exact resume.
    # The upstream structured config declares a bool, including when serialized
    # by OmegaConf at run startup. True asks Trainer to load the latest checkpoint.
    config.training.resume_from_checkpoint = bool(resume)
    config.training.dataloader_num_workers = 4
    config.training.use_wandb = False
    config.validate()
    return config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, choices=[17, 29], required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--base-snapshots",
        default=str(DEFAULT_BASE_SNAPSHOTS),
        help="Frozen base-model snapshot map; the 2026-09-08 original was lost with outputs/ on 2026-09-09",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset, out = Path(args.dataset), Path(args.out)
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint

        latest = get_last_checkpoint(str(out)) if out.is_dir() else None
        if latest is None or Path(args.resume).resolve() != Path(latest).resolve():
            raise ValueError("--resume must name the latest checkpoint in this run output")
    info = json.loads((dataset / "meta/info.json").read_text())
    if not args.dry_run and info["total_episodes"] != 200:
        raise ValueError("Formal arms require exactly200 accepted training episodes")
    snapshots_path = Path(args.base_snapshots)
    config = build_config(dataset, out, args.seed, args.resume, snapshots_path)
    # Exercise the same serialization boundary as experiment.run in dry runs too.
    from omegaconf import OmegaConf

    OmegaConf.create(config.__dict__)
    payload = json.loads(json.dumps(asdict(config), default=lambda x: x.value if hasattr(x, "value") else str(x)))
    fingerprint_payload = json.loads(json.dumps(payload))
    fingerprint_payload["training"]["resume_from_checkpoint"] = False
    manifest = dict(
        config=fingerprint_payload,
        dataset_manifest_sha256=hashlib.sha256((dataset / "input_manifest.json").read_bytes()).hexdigest(),
        modality_sha256=hashlib.sha256((ROOT / "configs/gr00t_bfm_visual_2cam.py").read_bytes()).hexdigest(),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_snapshots_sha256=hashlib.sha256(snapshots_path.read_bytes()).hexdigest(),
        gr00t_source_sha256={
            str(p.relative_to(ROOT / "src")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/gr00t").rglob("*.py"))
        },
    )
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "frozen_training.json"
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        raise ValueError("Training configuration/data changed; do not reuse this run ID")
    frozen.write_text(json.dumps(manifest, indent=2))
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    for relative in manifest["gr00t_source_sha256"]:
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / relative, destination)
    shutil.copy2(__file__, source_dir / Path(__file__).name)
    shutil.copy2(ROOT / "configs/gr00t_bfm_visual_2cam.py", source_dir / "gr00t_bfm_visual_2cam.py")
    shutil.copy2(snapshots_path, source_dir / "base_model_snapshots.json")
    config.save(out / "resolved_config.yaml")
    print(
        json.dumps(
            dict(
                seed=args.seed,
                optimizer_steps=3000,
                effective_batch=64,
                allow_padding=config.data.allow_padding,
                full_fov=True,
                dry_run=args.dry_run,
            )
        ),
        flush=True,
    )
    if args.dry_run:
        return
    import torch

    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one released GPU with CUDA_VISIBLE_DEVICES")
    from gr00t.experiment.experiment import run

    run(config)


if __name__ == "__main__":
    main()
