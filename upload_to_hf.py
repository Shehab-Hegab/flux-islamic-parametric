"""Upload dataset and LoRA weights to Hugging Face (dry-run by default)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    DATASET_DIR,
    DATASET_REPO_ID,
    EVAL_SEED,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    LORA_REPO_ID,
    MAX_TRAIN_STEPS,
    OPTIMIZER,
    OUTPUT_DIR,
    RESOLUTION,
    TRIGGER_WORD,
    WEIGHTS_FILENAME,
)


def _model_card() -> str:
    return f"""---
library_name: diffusers
base_model: black-forest-labs/FLUX.1-dev
tags:
  - flux
  - lora
  - text-to-image
  - architecture
  - islamic-architecture
  - parametric-design
  - mashrabiya
dataset:
  - shehab-hegab/islamic-parametric-architecture-dataset
---

# FLUX.1-dev LoRA — Islamic Parametric Architecture

Adapter for **Islamic Parametric Architecture & Facades**: geometric lattice screens,
mashrabiya parametric patterns, and structurally symmetric modern envelopes with
zero visual hallucination intent (precise geometric structural integrity).

## Trigger words

- Primary trigger (required): `{TRIGGER_WORD}`
- Style tag: `Islamic_Parametric`

Example prompt:

> A modern cultural center facade in Islamic_Parametric style, geometric parametric wooden panels, realistic lighting, 8k architectural photo

## Training metadata

| Hyperparameter | Value |
|---|---|
| Base model | `black-forest-labs/FLUX.1-dev` |
| Method | DreamBooth LoRA (diffusers `train_dreambooth_lora_flux.py`) |
| Rank | {LORA_RANK} |
| Alpha | {LORA_ALPHA} |
| Learning rate | {LEARNING_RATE:.0e} |
| Resolution | {RESOLUTION}×{RESOLUTION} |
| Max train steps | {MAX_TRAIN_STEPS} |
| Optimizer | {OPTIMIZER} |
| Mixed precision | bf16 + gradient checkpointing |
| Dataset | 25 images + captions, `shehab-hegab/islamic-parametric-architecture-dataset` |
| Weights file | `{WEIGHTS_FILENAME}` |

## Usage

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("shehab-hegab/flux-islamic-parametric-lora", weight_name="{WEIGHTS_FILENAME}")
pipe.enable_model_cpu_offload()
image = pipe(
    "A contemporary mosque exterior in Islamic_Parametric style, mashrabiya lattice screen, golden hour, photorealistic 8k",
    height=1024, width=1024, num_inference_steps=28, guidance_scale=3.5,
).images[0]
image.save("out.png")
```

## Architectural design intent

The adapter is trained on a curated set of modern Islamic parametric facades so generated
views preserve modular rhythm, bilateral symmetry, and lattice topology — evaluation grids
in the training repo compare base FLUX.1-dev against this adapter side-by-side at fixed seed {EVAL_SEED}
to verify geometric structural logic.

## Evaluation

See `inference_eval.py` in the source repository: side-by-side grids, prompts locked to the
trigger word, seed {EVAL_SEED}.
"""


def plan_uploads(dataset_dir: Path, weights: Path) -> list[dict]:
    return [
        {
            "repo_id": DATASET_REPO_ID,
            "type": "dataset",
            "folder": str(dataset_dir),
            "include": ["*.jpg", "*.txt", "manifest.json"],
        },
        {
            "repo_id": LORA_REPO_ID,
            "type": "model",
            "files": [str(weights), "README.md"],
        },
    ]


def run_upload(plan: list[dict]) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN env var required for --execute")
    api = HfApi(token=token)
    for item in plan:
        if item["type"] == "dataset":
            api.create_repo(item["repo_id"], repo_type="dataset", exist_ok=True)
            api.upload_folder(
                folder_path=item["folder"],
                repo_id=item["repo_id"],
                repo_type="dataset",
                ignore_patterns=[".git*", "__pycache__"],
            )
            print(f"uploaded dataset -> {item['repo_id']}")
        else:
            api.create_repo(item["repo_id"], exist_ok=True)
            card = _model_card()
            api.upload_file(path_or_fileobj=item["files"][0], path_in_repo=WEIGHTS_FILENAME, repo_id=item["repo_id"])
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=item["repo_id"],
            )
            print(f"uploaded LoRA -> {item['repo_id']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload dataset + LoRA to Hugging Face.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="perform real upload (requires HF_TOKEN)")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--weights", default=str(Path(OUTPUT_DIR) / WEIGHTS_FILENAME))
    args = parser.parse_args(argv)

    weights = Path(args.weights)
    plan = plan_uploads(Path(args.dataset_dir), weights)
    if not args.execute:
        print("DRY RUN — no files will be uploaded")
        for item in plan:
            print(f"  [{item['type']}] {item['repo_id']}")
            if item["type"] == "dataset":
                print(f"    folder: {item['folder']}")
            else:
                for f in item["files"]:
                    exists = Path(f).exists()
                    print(f"    file: {f} ({'ready' if exists else 'missing — upload after training'})")
        print(f"  weights filename: {WEIGHTS_FILENAME}")
        print(f"  dataset: https://huggingface.co/datasets/{DATASET_REPO_ID}")
        print(f"  lora:    https://huggingface.co/{LORA_REPO_ID}")
        return 0

    if not weights.exists():
        print(f"[error] weights not found: {weights} — train first or pass --weights", file=sys.stderr)
        return 1
    try:
        run_upload(plan)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[error] Hugging Face upload failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
