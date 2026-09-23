"""Launch FLUX.1-dev DreamBooth LoRA training (spec hyperparameters locked)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    BASE_MODEL,
    DATASET_DIR,
    INSTANCE_PROMPT,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    MAX_TRAIN_STEPS,
    MIXED_PRECISION,
    OPTIMIZER,
    OUTPUT_DIR,
    RESOLUTION,
    TRAIN_BATCH_SIZE,
    TRAIN_SCRIPT,
    WEIGHTS_FILENAME,
    train_dry_run_command,
)

DIFFUSERS_RAW_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/main/"
    "examples/dreambooth/train_dreambooth_lora_flux.py"
)


def ensure_train_script(dest: Path = Path(TRAIN_SCRIPT)) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import requests

    resp = requests.get(DIFFUSERS_RAW_URL, timeout=60)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    return dest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train FLUX.1-dev LoRA for Islamic Parametric Architecture.")
    p.add_argument("--instance-dir", default=DATASET_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--dry-run", action="store_true", help="print resolved training command and exit")
    p.add_argument("--fetch-script", action="store_true", help="download official diffusers train script then exit")
    return p


def build_launch_command(instance_dir: str, output_dir: str) -> list[str]:
    ensure_train_script()
    return [
        "accelerate",
        "launch",
        TRAIN_SCRIPT,
        "--pretrained_model_name_or_path",
        BASE_MODEL,
        "--instance_data_dir",
        instance_dir,
        "--output_dir",
        output_dir,
        "--instance_prompt",
        INSTANCE_PROMPT,
        "--resolution",
        str(RESOLUTION),
        "--train_batch_size",
        str(TRAIN_BATCH_SIZE),
        "--gradient_accumulation_steps",
        "4",
        "--optimizer",
        OPTIMIZER,
        "--learning_rate",
        f"{LEARNING_RATE:.0e}",
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--max_train_steps",
        str(MAX_TRAIN_STEPS),
        "--rank",
        str(LORA_RANK),
        "--network_alpha",
        str(LORA_ALPHA),
        "--mixed_precision",
        MIXED_PRECISION,
        "--gradient_checkpointing",
        "--seed",
        "42",
        "--report_to",
        "tensorboard",
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fetch_script:
        path = ensure_train_script()
        print(f"train script ready: {path}")
        return 0
    if args.dry_run:
        print(train_dry_run_command(args.instance_dir, args.output_dir))
        return 0

    try:
        import torch
    except Exception:
        print("[error] torch not installed — use the Colab notebook or pip install torch", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(
            "[error] CUDA GPU required for FLUX training (local GTX-class GPUs are insufficient). "
            "Run Flux_Architectural_LoRA_Training.ipynb on Google Colab (T4/A100). "
            "Use --dry-run to preview the command.",
            file=sys.stderr,
        )
        return 1

    instance_path = Path(args.instance_dir)
    if not instance_path.is_dir():
        print(f"[error] instance dir missing: {instance_path}", file=sys.stderr)
        return 1

    try:
        ensure_train_script()
    except Exception as exc:
        print(f"[error] could not prepare training script: {exc}", file=sys.stderr)
        return 1

    cmd = build_launch_command(args.instance_dir, args.output_dir)
    print("Launching:", " ".join(cmd))
    env = os.environ.copy()
    import subprocess

    proc = subprocess.run(cmd, env=env)
    weights = Path(args.output_dir) / WEIGHTS_FILENAME
    if proc.returncode == 0 and weights.exists():
        print(f"LoRA weights saved: {weights}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
