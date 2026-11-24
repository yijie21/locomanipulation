"""SAM 3 masks for vision RSI L3 (locomani.perception.services): one image and text prompts in, every instance SAM 3
finds above a low threshold out -- {"results": {prompt: [(score, packed mask), ...]}} -- the caller choosing among them
(by score and by where each lands in 3-D). Runs in the sam3d conda env (torch 2.11 cu128 and the sam3 package; the
sam3 env's torch misses its CUDA libraries), on the editor's GPU, the weights from the local Hugging Face cache:

    HF_HOME=/workspace/huggingface_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \\
        PYTHONPATH=src:third_party/sam3 /workspace/miniconda3/envs/sam3d/bin/python -s -m locomani.perception.sam3_server --port 18832

Measured 2026-10-23 on the apartment's Unreal frames (640 x 480, RTX 5090): 5 s to load, about 40 ms a prompt after the
image; about 1 GB of GPU memory. Under bfloat16 autocast (SAM 3 mixes dtypes otherwise), entered in the request's own
thread: autocast and inference mode are per thread, and services.serve answers each client in its own."""

from __future__ import annotations

import argparse

import numpy as np

from locomani.perception.services import pack_mask, serve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.1, help="the processor's confidence floor")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args(argv)
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=a.threshold)

    def handle(req: dict) -> dict:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(Image.fromarray(np.asarray(req["image"], np.uint8)))
            out = {}
            for prompt in req["prompts"]:
                o = proc.set_text_prompt(state=state, prompt=prompt)
                masks = o["masks"].squeeze(1).cpu().numpy().astype(bool) if len(o["masks"]) else []
                scores = o["scores"].float().cpu().numpy() if len(o["scores"]) else []
                out[prompt] = [(float(s), pack_mask(m)) for m, s in zip(masks, scores)]
        return dict(results=out)

    serve((a.host, a.port), handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
