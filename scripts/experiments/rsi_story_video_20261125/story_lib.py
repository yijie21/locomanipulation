"""Shared helpers of the SIMPLE art story recorder (2026-11-25): paths, the loop's record, the outcome rule, caption, encoding.
No heavy imports (cv2, PIL, numpy only): the driver, the bed process, the index builder and the offline check all use it."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "stories.json").read_text())
LOOPDIR = Path(S["loopdir"])
OUT = Path("/workspace/code/locomanipulation/outputs/rsi_story_videos_20261125/simple_art")
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
FPS, EVERY, MAXB = 12.5, 4, 490_000  # EVERY control steps (20 ms) per frame; hard cap 500 000 bytes, margin kept
W, H = 640, 360


def version_dir(v: int) -> Path:
    return LOOPDIR / "incumbents" / S["versions"][str(v)]["hash"] / "skills"


def diag_record(v: int, label: str) -> dict:
    return json.loads((LOOPDIR / "diag" / f"incumbent_{S['versions'][str(v)]['hash']}" / f"{label}.json").read_text())


def episode_of(label: str) -> str:
    task, n = label.rsplit("_s", 1)
    return f"{task}/Q/{n}"


def outcome(rec: dict) -> dict:
    """The loop's ok / failure_class of a bed record (backend_simple.to_round, same rule)."""
    comply = rec.get("comply")
    comply_ok = bool(comply.get("ok")) if isinstance(comply, dict) else False
    success = bool(rec.get("success"))
    ok = success and comply_ok
    loss = rec.get("loss") or ("error" if rec.get("error") else None)
    if ok:
        cls = ""
    elif loss:
        cls = loss
    elif success:
        cls = "comply"
    else:
        cls = "unclassified"
    return dict(ok=ok, failure_class=cls, t_end=rec.get("t_end"))


def caption_overlay(text: str):
    """(band alpha, text alpha) float arrays 360 x 640 of a bottom-left caption band with CJK text."""
    from PIL import Image, ImageDraw, ImageFont

    f = ImageFont.truetype(FONT, 17)
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    w = int(d0.textlength(text, font=f)) + 16
    band = Image.new("L", (W, H), 0)
    ImageDraw.Draw(band).rectangle([0, H - 28, w, H], fill=255)
    txt = Image.new("L", (W, H), 0)
    ImageDraw.Draw(txt).text((8, H - 26), text, font=f, fill=255)
    return np.asarray(band, np.float32)[..., None] / 255 * 0.55, np.asarray(txt, np.float32)[..., None] / 255


def put_caption(frame, mask):
    band, txt = mask
    out = frame.astype(np.float32) * (1 - band)
    out = out * (1 - txt) + 255 * txt
    return out.astype(np.uint8)


def encode(frames, tag: str, path: Path, caption: str, maxb: int = MAXB):
    """Caption on every frame, result tag on the last second (frames padded to 2 s if shorter), H.264 yuv420p faststart,
    re-encoded until at most `maxb` bytes. -> (bytes, n_frames_written, padded, middle frame)."""
    mask = caption_overlay(caption)
    out = [put_caption(f, mask) for f in frames]
    padded = 0
    while len(out) < 25:
        out.append(out[-1].copy())
        padded += 1
    for f in out[-int(FPS) :]:
        cv2.putText(f, tag, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        col = (80, 255, 80) if tag.startswith("SUCC") else (80, 80, 255)
        cv2.putText(f, tag, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
    br = int(maxb * 8 * 0.9 / (len(out) / FPS))
    size = 0
    for _ in range(8):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
               "-c:v", "libx264", "-profile:v", "main", "-preset", "slow", "-b:v", str(br), "-maxrate", str(int(br * 1.15)), "-bufsize", str(br * 2),
               "-g", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]  # fmt: skip
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in out:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
        size = path.stat().st_size
        if size <= maxb:
            break
        br = int(br * 0.8 * maxb / size)
    return size, len(out), padded, out


def motion(frames) -> dict:
    d = [
        float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean())
        for i in range(1, len(frames), 5)
    ]
    return (
        dict(
            mean=round(float(np.mean(d)), 3),
            min=round(float(np.min(d)), 3),
            static_frac=round(float(np.mean([x < 0.02 for x in d])), 2),
        )
        if d
        else {}
    )
