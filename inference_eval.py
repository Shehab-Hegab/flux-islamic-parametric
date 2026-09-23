"""Side-by-side evaluation grids: base FLUX.1-dev vs FLUX.1-dev + Islamic Parametric LoRA."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    BASE_MODEL,
    EVAL_OUTPUT_DIR,
    EVAL_PROMPTS,
    EVAL_SEED,
    OUTPUT_DIR,
    RESOLUTION,
    TRIGGER_WORD,
    WEIGHTS_FILENAME,
)


def validate_eval_setup() -> dict:
    if len(EVAL_PROMPTS) < 5:
        raise ValueError("expected at least 5 evaluation prompts")
    missing = [p for p in EVAL_PROMPTS if "Islamic_Parametric style" not in p]
    if missing:
        raise ValueError(f"prompts missing style tag: {missing}")
    if TRIGGER_WORD == "":
        raise ValueError("trigger word must not be empty")
    return {
        "prompts": list(EVAL_PROMPTS),
        "seed": EVAL_SEED,
        "resolution": RESOLUTION,
        "base_model": BASE_MODEL,
        "columns": ["base", "base+lora"],
    }


def build_pipeline(lora_path: str | None):
    import torch
    from diffusers import FluxPipeline

    try:
        pipe = FluxPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load {BASE_MODEL} — accept the FLUX.1-dev license and set "
            f"HF_TOKEN (gated model). Original error: {exc}"
        ) from exc
    if lora_path:
        weights = Path(lora_path)
        if weights.is_dir() and (weights / WEIGHTS_FILENAME).exists():
            weights = weights / WEIGHTS_FILENAME
        try:
            pipe.load_lora_weights(
                str(weights.parent) if weights.suffix else str(weights),
                weight_name=weights.name if weights.suffix else None,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to load LoRA weights from {lora_path}: {exc}") from exc
    pipe.enable_model_cpu_offload()
    return pipe


def save_grid(images: list, prompts: list[str], path: Path, labels: list[str]) -> Path:
    from PIL import Image, ImageDraw

    if not images:
        raise ValueError("no images to grid")
    cell_w, cell_h = images[0].size
    header_h = 36
    cols = len(labels)
    rows = len(prompts)
    grid = Image.new("RGB", (cell_w * cols, (cell_h + header_h) * rows), (18, 18, 22))
    draw = ImageDraw.Draw(grid)
    for row, prompt in enumerate(prompts):
        y = row * (cell_h + header_h)
        draw.text((8, y + 8), prompt[:110], fill=(235, 235, 235))
        for col, label in enumerate(labels):
            img = images[row * cols + col]
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h))
            grid.paste(img, (col * cell_w, y + header_h))
            draw.text((col * cell_w + 8, y + header_h + 8), label, fill=(255, 220, 120))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path, "PNG")
    return path


def generate_grids(lora_path: str | None, out_dir: Path, limit: int | None = None) -> Path:
    import torch

    prompts = EVAL_PROMPTS[:limit] if limit else list(EVAL_PROMPTS)
    labels = ["FLUX.1-dev (base)", "FLUX.1-dev + Islamic_Parametric LoRA"]
    pipe_base = build_pipeline(None)
    pipe_lora = build_pipeline(lora_path) if lora_path else pipe_base
    images = []
    for prompt in prompts:
        for pipe in (pipe_base, pipe_lora):
            generator = torch.Generator("cpu").manual_seed(EVAL_SEED)
            images.append(
                pipe(
                    prompt,
                    height=RESOLUTION,
                    width=RESOLUTION,
                    num_inference_steps=28,
                    guidance_scale=3.5,
                    generator=generator,
                ).images[0]
            )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"grid_{stamp}.png"
    return save_grid(images, prompts, out_path, labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Base vs LoRA evaluation grid for FLUX Islamic Parametric.")
    parser.add_argument("--check", action="store_true", help="validate setup without loading models")
    parser.add_argument("--lora-path", default=OUTPUT_DIR, help="dir or file with pytorch_lora_weights.safetensors")
    parser.add_argument("--out", default=EVAL_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="generate for first N prompts only")
    args = parser.parse_args(argv)

    setup = validate_eval_setup()
    if args.check:
        print("check OK")
        print(f"prompts: {len(setup['prompts'])} columns: {setup['columns']}")
        print(f"seed={setup['seed']} resolution={setup['resolution']} base={setup['base_model']}")
        planned = Path(args.out) / "grid_<timestamp>.png"
        print(f"planned output: {planned}")
        return 0

    lora = Path(args.lora_path)
    has_lora = (lora / WEIGHTS_FILENAME).exists() if lora.is_dir() else lora.exists()
    if not has_lora:
        print(f"[warn] LoRA weights not found at {lora}; generating base-only comparison against saved columns")
    try:
        path = generate_grids(str(args.lora_path) if has_lora else None, Path(args.out), args.limit)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[error] evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"grid saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
