"""Shared constants for the Islamic Parametric FLUX LoRA pipeline."""

from __future__ import annotations

TRIGGER_WORD = "in Islamic_Parametric style"
STYLE_TAG = "Islamic_Parametric"

DATASET_DIR = "dataset_islamic_parametric"
CAPTIONS_DIR = f"{DATASET_DIR}/captions"
MANIFEST_PATH = f"{DATASET_DIR}/manifest.json"
NUM_IMAGES = 25
MIN_SIZE = 1024
IMAGE_PATTERN = "image_{:02d}"

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_RANK = 16
LORA_ALPHA = 16
LEARNING_RATE = 1e-4
RESOLUTION = 1024
MAX_TRAIN_STEPS = 800
OPTIMIZER = "adamw8bit"
TRAIN_BATCH_SIZE = 1
MIXED_PRECISION = "bf16"
INSTANCE_PROMPT = f"a detailed architectural photo {TRIGGER_WORD}"
OUTPUT_DIR = "output"
WEIGHTS_FILENAME = "pytorch_lora_weights.safetensors"
TRAIN_SCRIPT = "training/train_dreambooth_lora_flux.py"
CHECKPOINTING_STEPS = 250
CHECKPOINTS_TOTAL_LIMIT = 4

DATASET_REPO_ID = "Shehab-Hegab/islamic-parametric-architecture-dataset"
LORA_REPO_ID = "Shehab-Hegab/flux-islamic-parametric-lora"

CAPTION_PREFIX = f"A detailed architectural photo {TRIGGER_WORD}"
CAPTION_SUFFIX = (
    "precise geometric lattice patterns, daylighting, structural symmetry, "
    "photorealistic 8k architectural render, no visual distortion."
)

EVAL_SEED = 42
EVAL_OUTPUT_DIR = "eval_outputs"
STRUCTURAL_METRICS_PATH = f"{EVAL_OUTPUT_DIR}/structural_metrics.json"
EVAL_PROMPTS = [
    (
        "A modern cultural center facade in Islamic_Parametric style, "
        "geometric parametric wooden panels, realistic lighting, 8k architectural photo"
    ),
    (
        "A contemporary mosque exterior in Islamic_Parametric style, "
        "mashrabiya lattice screen, golden hour, precise geometric structure, "
        "photorealistic 8k"
    ),
    (
        "A parametric geometric lattice facade of a museum in Islamic_Parametric style, "
        "perforated aluminum panels, sharp lines, precise geometry, 8k render"
    ),
    (
        "An interior courtyard in Islamic_Parametric style, arched mashrabiya walls, "
        "daylighting through geometric screens, structural symmetry, 8k photo"
    ),
    (
        "A high-rise tower envelope in Islamic_Parametric style, "
        "tessellated star-pattern cladding, urban context, realistic lighting, "
        "photorealistic 8k"
    ),
]

ARCH_QUERIES = [
    "islamic architecture facade",
    "mashrabiya screen",
    "geometric lattice facade",
    "parametric architecture",
    "mosque modern architecture",
]

SYNTHETIC_PATTERNS = [
    "star8",
    "hex_mashrabiya",
    "woven_grille",
    "arch_screen",
    "punch_parametric",
    "girih_strip",
]


def build_caption(core: str) -> str:
    core_clean = core.strip().rstrip(".")
    return f"{CAPTION_PREFIX}, featuring {core_clean}, {CAPTION_SUFFIX}"


def train_dry_run_command(instance_dir: str, output_dir: str) -> str:
    return (
        f"accelerate launch {TRAIN_SCRIPT} "
        f"--pretrained_model_name_or_path={BASE_MODEL} "
        f"--instance_data_dir={instance_dir} "
        f"--output_dir={output_dir} "
        f"--instance_prompt={INSTANCE_PROMPT!r} "
        f"--resolution={RESOLUTION} "
        f"--train_batch_size={TRAIN_BATCH_SIZE} "
        f"--gradient_accumulation_steps=4 "
        f"--optimizer={OPTIMIZER} "
        f"--learning_rate={LEARNING_RATE:.0e} "
        f"--lr_scheduler=constant "
        f"--lr_warmup_steps=0 "
        f"--max_train_steps={MAX_TRAIN_STEPS} "
        f"--rank={LORA_RANK} "
        f"--network_alpha={LORA_ALPHA} "
        f"--mixed_precision={MIXED_PRECISION} "
        f"--gradient_checkpointing "
        f"--checkpointing_steps={CHECKPOINTING_STEPS} "
        f"--checkpoints_total_limit={CHECKPOINTS_TOTAL_LIMIT} "
        f"--seed=42 "
        f"--report_to=tensorboard"
    )
