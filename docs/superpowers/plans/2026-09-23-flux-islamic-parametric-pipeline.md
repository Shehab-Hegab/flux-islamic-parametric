# FLUX Islamic Parametric Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a complete HF-ready repo: 25-image Islamic Parametric dataset + trigger-word captions + FLUX.1-dev LoRA training (Colab) + eval grids + upload scripts + model card.

**Architecture:** Five standalone CLI scripts sharing `src/islamic_parametric/constants.py`. Dataset = Unsplash hybrid with procedural synthetic top-up. Training wraps official diffusers `train_dreambooth_lora_flux.py`. Eval builds base-vs-LoRA grids. Upload uses `huggingface_hub` with dry-run default.

**Tech Stack:** Python 3.11, Pillow, requests, transformers (Florence-2), diffusers/accelerate (training), huggingface_hub, pytest, ruff.

## Global Constraints

- Trigger word (verbatim, every caption/prompt): `in Islamic_Parametric style`
- Dataset dir: `dataset_islamic_parametric/` — exactly 25 images, each ≥1024×1024
- Base model: `black-forest-labs/FLUX.1-dev`; LoRA r=16, α=16, lr=1e-4, res=1024×1024, steps=800 (range 800–1000), optimizer `adamw8bit`
- Output weights filename: `pytorch_lora_weights.safetensors`
- HF repos: `shehab-hegab/islamic-parametric-architecture-dataset`, `shehab-hegab/flux-islamic-parametric-lora`
- Secrets only via env (`UNSPLASH_ACCESS_KEY`, `HF_TOKEN`); never hardcoded
- No code comments unless self-evident docstrings for public CLI help
- ruff clean; pytest green; commit after each task

---

### Task 1: Scaffold + constants + tests harness

**Files:**
- Create: `pyproject.toml`, `src/islamic_parametric/__init__.py`, `src/islamic_parametric/constants.py`, `tests/test_constants.py`, `.gitignore`, `requirements.txt`

**Interfaces (constants module — consumed by all later tasks):**
```python
TRIGGER_WORD: str  # "in Islamic_Parametric style"
STYLE_TAG: str     # "Islamic_Parametric"
DATASET_DIR: str   # "dataset_islamic_parametric"
CAPTIONS_DIR: str  # DATASET_DIR + "/captions"
NUM_IMAGES: int    # 25
MIN_SIZE: int      # 1024
BASE_MODEL: str    # "black-forest-labs/FLUX.1-dev"
LORA_RANK: int     # 16
LORA_ALPHA: int    # 16
LEARNING_RATE: float  # 1e-4
RESOLUTION: int    # 1024
MAX_TRAIN_STEPS: int  # 800
OPTIMIZER: str     # "adamw8bit"
DATASET_REPO_ID: str  # "shehab-hegab/islamic-parametric-architecture-dataset"
LORA_REPO_ID: str     # "shehab-hegab/flux-islamic-parametric-lora"
WEIGHTS_FILENAME: str # "pytorch_lora_weights.safetensors"
EVAL_PROMPTS: list[str]  # ≥5 prompts, each containing TRIGGER_WORD
CAPTION_SUFFIX: str      # architectural quality suffix incl. trigger word
def build_caption(core: str) -> str
def train_dry_run_command(instance_dir: str, output_dir: str) -> str  # returns full accelerate command string with all locked hyperparams
```

- [ ] Step 1: Write `tests/test_constants.py` asserting trigger word exact string, EVAL_PROMPTS length ≥5 and all contain trigger, `build_caption` output contains trigger + "8k" + "no visual distortion", dry-run command contains `--rank=16`, `--network_alpha=16`/`--alpha=16`, `--learning_rate=1e-4`, `--resolution=1024`, `--max_train_steps=800`, `adamw8bit`, `FLUX.1-dev`.
- [ ] Step 2: Run `python -m pytest tests/test_constants.py -v` → FAIL (module missing).
- [ ] Step 3: Implement scaffold + `constants.py` per interface; `pyproject.toml` with `[tool.ruff]` line-length 100, target py311; `.gitignore` (`__pycache__`, `.venv`, `output/`, `eval_outputs/`, `*.safetensors` — but NOT dataset images); `requirements.txt` (pillow, requests, huggingface_hub, transformers>=4.45, diffusers>=0.31, accelerate, safetensors, pytest, ruff).
- [ ] Step 4: Re-run pytest → PASS.
- [ ] Step 5: `ruff check src tests` → clean. Commit `chore: scaffold constants and test harness`.

### Task 2: Dataset builder — hybrid Unsplash + synthetic

**Files:**
- Create: `download_or_synthetic_dataset.py`, `tests/test_dataset_builder.py`

**Interfaces:**
- Consumes: `constants` (DATASET_DIR, NUM_IMAGES, MIN_SIZE, TRIGGER_WORD not needed here)
- Produces: 25 images `image_01.jpg` … `image_25.jpg` in `dataset_islamic_parametric/`, plus `manifest.json` list of `{filename, source: "unsplash"|"synthetic", query/pattern, width, height}`
- Functions: `generate_synthetic_image(index: int, size: int = 1024) -> Image.Image` (procedural mashrabiya/lattice/parametric facade variants, ≥6 pattern families: 8-point star lattice, hexagonal mashrabiya, woven grille, concentric arch screen, Voronoi-ish parametric punches, girih strip pattern — deterministic per index via `random.Random(index)`); `fetch_unsplash(query, access_key, per_page) -> list[bytes]`; `main(argv) -> int` with `--count`, `--prefer {hybrid,synthetic,unsplash}`, `--out`

- [ ] Step 1: Failing tests: synthetic image is RGB, ≥1024², deterministic (same index → same bytes hash); builder in `--prefer synthetic --count 25` creates 25 files + manifest and all pass size check; run `python -m pytest tests/test_dataset_builder.py -v` → FAIL.
- [ ] Step 2: Implement script: if `--prefer hybrid` and `UNSPLASH_ACCESS_KEY` set → fetch up to `count` high-res (`w=2400&h=2400&fit=crop`) architectural queries (`islamic architecture facade`, `mashrabiya`, `geometric lattice facade`, `parametric architecture`), validate ≥1024², top-up remainder with synthetic; any failure path falls to synthetic silently-with-log.
- [ ] Step 3: pytest PASS; then run `python download_or_synthetic_dataset.py --prefer hybrid` in-session → verify 25 images ≥1024² + manifest.
- [ ] Step 4: ruff clean. Commit `feat: hybrid dataset builder with procedural mashrabiya synthetic`.

### Task 3: Caption generator — Florence-2 + template fallback

**Files:**
- Create: `generate_captions.py`, `tests/test_captions.py`

**Interfaces:**
- Consumes: dataset images + `constants.build_caption`, CAPTION_SUFFIX
- Produces: `dataset_islamic_parametric/captions/image_01.txt` … 1:1 with images; each file single-line caption containing `in Islamic_Parametric style`, unique core description, ends with quality suffix (photorealistic 8k architectural render, no visual distortion / geometric precision tokens)
- Functions: `template_caption(image_name: str, manifest_entry: dict) -> str`; `florence_caption(pil_image) -> str` (model `microsoft/Florence-2-large`, task `<CAPTION>`, trusted env only); `main(argv) -> int` with `--backend {auto,florence,template}`

- [ ] Step 1: Failing tests: after `main(["--backend","template"])` on real dataset → 25 txt files, every contains trigger, all unique, 1:1 image mapping; FAIL first.
- [ ] Step 2: Implement with `auto`: try import transformers + CUDA/`cpu` load with `dtype=float16`/`torch`, on any exception log + fall back template (exit 0).
- [ ] Step 3: Run template backend end-to-end in-session (upgrade transformers if attempting florence; if Florence download/load fails within timeout → document fallback, captions still complete). pytest PASS.
- [ ] Step 4: ruff. Commit `feat: caption generator with Florence-2 and template fallback`.

### Task 4: Training wrapper + Colab notebook

**Files:**
- Create: `train_flux_lora.py`, `Flux_Architectural_LoRA_Training.ipynb`, `training/train_dreambooth_lora_flux.py` (vendored official diffusers example, pinned path documented), `tests/test_train_wrapper.py`

**Interfaces:**
- Consumes: `constants.train_dry_run_command`, dataset dir
- Produces: `--dry-run` prints command, exit 0 (no GPU needed). Real run: checks CUDA, resolves `accelerate launch training/train_dreambooth_lora_flux.py` with instance prompt built from trigger word (`"a detailed architectural photo in Islamic_Parametric style"`), `--rank 16 --network_alpha 16 --learning_rate 1e-4 --resolution 1024 --max_train_steps 800 --optimizer adamw8bit --mixed_precision bf16 --gradient_checkpointing --train_batch_size 1`, output `output/` containing `pytorch_lora_weights.safetensors`. Notebook: pip installs, mounts nothing, runs download of scripts from repo layout (cells assume uploaded repo), `!python train_flux_lora.py ...` + fallback raw accelerate cell; markdown cells with hyperparams + A100/T4 notes.

- [ ] Step 1: Failing tests: `main(["--dry-run"])` exit 0 and stdout includes all locked hyperparams + `pytorch_lora_weights` implied path; import module without torch/diffusers installed does not crash (lazy imports). FAIL first.
- [ ] Step 2: Implement wrapper; vendor official script content (fetch from huggingface/diffusers GitHub raw into `training/`); build valid ipynb JSON (nbformat 4) with ≥8 cells; validate via `python -c "import json; json.load(open('Flux_Architectural_LoRA_Training.ipynb'))"` and `ast.parse` each code cell.
- [ ] Step 3: pytest PASS; `python train_flux_lora.py --dry-run` manual check.
- [ ] Step 4: ruff. Commit `feat: FLUX LoRA training wrapper and Colab notebook`.

### Task 5: Evaluation / inference grid

**Files:**
- Create: `inference_eval.py`, `tests/test_inference_eval.py`

**Interfaces:**
- Consumes: `constants.EVAL_PROMPTS` (≥5, incl. cultural center facade example), BASE_MODEL, LORA path
- Produces: `eval_outputs/grid_<timestamp>.png` side-by-side (rows prompts, cols base vs +LoRA), seeds fixed (42), 1024²; `--check` mode validates prompts/dirs and prints planned grid, exit 0 without loading models
- Functions: `validate_eval_setup() -> dict`; `build_pipeline(lora_path: Optional[str])`; `save_grid(images: list, prompts: list[str], path)` (PIL montage with labels)

- [ ] Step 1: Failing tests: validate returns ≥5 prompts all containing trigger; `main(["--check"])` exit 0 and prints "grid" plan; FAIL first.
- [ ] Step 2: Implement (lazy torch/diffusers imports; seed 42; `StableDiffusionXLPipeline`-style FLUX `FluxPipeline.from_pretrained(...); pipe.load_lora_weights(...)`).
- [ ] Step 3: pytest PASS; run `python inference_eval.py --check` in-session.
- [ ] Step 4: ruff. Commit `feat: side-by-side FLUX base vs LoRA eval grid`.

### Task 6: HF upload + model card + URLs doc

**Files:**
- Create: `upload_to_hf.py`, `README.md`, `MODEL_CARD.md`, `docs/HF_URLS.md`, `tests/test_upload.py`

**Interfaces:**
- Consumes: dataset dir, `output/pytorch_lora_weights.safetensors` (optional for dry-run), tokens from env
- Produces: default `--dry-run` prints exact `HfApi.upload_folder`/`create_repo` plan for both repos; `--execute` requires `HF_TOKEN`, uploads dataset folder (images+captions+manifest) → `shehab-hegab/islamic-parametric-architecture-dataset`, uploads weights+`MODEL_CARD.md` as `README.md` → `shehab-hegab/flux-islamic-parametric-lora`
- README/MODEL_CARD content: trigger words section, sample caption, training metadata table (r/α/lr/steps/res/optimizer/base), architectural design intent, usage snippet (`pipe.load_lora_weights`), dataset preview, eval description
- `docs/HF_URLS.md`: exact URLs `https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset`, `https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora`, file deep-links (`/blob/main/pytorch_lora_weights.safetensors`, dataset `/blob/main/image_01.jpg`, captions path)

- [ ] Step 1: Failing tests: dry-run main exit 0 mentions both repo IDs + weights filename + "DRY RUN"; README.md exists and contains trigger word, both repo IDs, `pytorch_lora_weights.safetensors`, hyperparams 16/16/1e-4/800/1024/adamw8bit; FAIL first.
- [ ] Step 2: Implement upload + write README/MODEL_CARD/HF_URLS.
- [ ] Step 3: pytest PASS; `python upload_to_hf.py --dry-run` manual.
- [ ] Step 4: ruff. Commit `feat: HF upload dry-run/execute and model card`.

### Task 7: Full verification + orchestration expert review + fixes

**Files:** modify as review findings dictate; `tests/` must stay green

- [ ] Step 1: `ruff check . && python -m pytest -q` → green.
- [ ] Step 2: Re-verify dataset integrity (25 images ≥1024², 25 captions unique+trigger).
- [ ] Step 3: Run `orchestrate` with task: "Review this FLUX.1 LoRA Islamic Parametric dataset/training repo for correctness vs HF best practices" — attach key file contents (constants, train wrapper, captions sample, README).
- [ ] Step 4: Fix critical/major findings; re-run ruff+pytest; commit fixes.
- [ ] Step 5: Final commit; report HF URL structure to user.
