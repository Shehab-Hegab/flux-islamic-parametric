import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from islamic_parametric.constants import (  # noqa: E402
    BASE_MODEL,
    EVAL_PROMPTS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    MAX_TRAIN_STEPS,
    MIN_SIZE,
    NUM_IMAGES,
    OPTIMIZER,
    RESOLUTION,
    TRIGGER_WORD,
    WEIGHTS_FILENAME,
    build_caption,
    train_dry_run_command,
)


def test_trigger_word_exact():
    assert TRIGGER_WORD == "in Islamic_Parametric style"


def test_dataset_requirements():
    assert NUM_IMAGES == 25
    assert MIN_SIZE >= 1024


def test_eval_prompts():
    assert len(EVAL_PROMPTS) >= 5
    assert all(TRIGGER_WORD in p or "Islamic_Parametric style" in p for p in EVAL_PROMPTS)
    assert len(set(EVAL_PROMPTS)) == len(EVAL_PROMPTS)


def test_build_caption_spec_compliant():
    cap = build_caption("a modern building exterior with a parametric mashrabiya facade")
    assert TRIGGER_WORD in cap
    assert "8k" in cap
    assert "no visual distortion" in cap
    assert "geometric lattice" in cap
    assert cap.endswith("no visual distortion.")


def test_dry_run_command_contains_locked_hyperparams():
    cmd = train_dry_run_command("dataset_islamic_parametric", "output")
    assert BASE_MODEL in cmd
    assert f"--rank={LORA_RANK}" in cmd
    assert LORA_RANK == 16
    assert LORA_ALPHA == 16
    assert f"--network_alpha={LORA_ALPHA}" in cmd
    assert f"--learning_rate={LEARNING_RATE:.0e}" in cmd
    assert "1e-04" in cmd or "1e-4" in cmd
    assert f"--resolution={RESOLUTION}" in cmd
    assert RESOLUTION == 1024
    assert f"--max_train_steps={MAX_TRAIN_STEPS}" in cmd
    assert MAX_TRAIN_STEPS == 800
    assert OPTIMIZER == "adamw8bit"
    assert "adamw8bit" in cmd
    assert WEIGHTS_FILENAME == "pytorch_lora_weights.safetensors"
