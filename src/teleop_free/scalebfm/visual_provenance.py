"""Read-only verification of the registered model and simulation assets."""
import hashlib
import json
from pathlib import Path


def verify_runtime_assets():
    root = Path(__file__).resolve().parents[3]
    manifest = root / 'outputs/experiments/scalebfm_visual_goal_20260908/runtime_assets.json'
    assets = json.loads(manifest.read_text())['files']
    for name, expected in assets.items():
        path = Path(name)
        with path.open('rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != expected['sha256']:
            raise ValueError(f'Registered runtime asset changed: {path}')
    return assets
