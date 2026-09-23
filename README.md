# FLUX.1-dev LoRA — Islamic Parametric Architecture

Production-grade Hugging Face repository: curated architectural dataset, trigger-word captions,
FLUX.1-dev LoRA training pipeline, side-by-side evaluation, and upload tooling for
**Islamic Parametric Architecture & Facades** (portfolio demo — evaluator: Dr. Mai Soliman,
Getinform School, Parametric Design).

## Trigger words

- **Primary trigger:** `in Islamic_Parametric style` (required in every prompt)
- Style tag: `Islamic_Parametric`

Example:

> A modern cultural center facade in Islamic_Parametric style, geometric parametric wooden panels, realistic lighting, 8k architectural photo

## Repository layout

| Path | Purpose |
|---|---|
| `dataset_islamic_parametric/` | 25 images ≥1024×1024 + `captions/*.txt` + `manifest.json` |
| `download_or_synthetic_dataset.py` | Hybrid builder: Unsplash API → procedural mashrabiya synthetic fallback |
| `generate_captions.py` | Florence-2-large captions with deterministic template fallback |
| `train_flux_lora.py` | FLUX DreamBooth LoRA wrapper (`--dry-run` / Colab GPU run) |
| `training/train_dreambooth_lora_flux.py` | Official diffusers training script (vendored) |
| `Flux_Architectural_LoRA_Training.ipynb` | Google Colab notebook (T4/A100) |
| `inference_eval.py` | Side-by-side base vs LoRA grids (seed 42, 1024²) |
| `upload_to_hf.py` | Dry-run / `--execute` upload of dataset + LoRA |
| `src/islamic_parametric/constants.py` | Single source of truth: trigger, hyperparams, repo IDs |

## Where the data lives (verified links)

- **Dataset (25 images + captions + manifest) — GitHub (live):**
  [`dataset_islamic_parametric/`](https://github.com/Shehab-Hegab/flux-islamic-parametric/tree/main/dataset_islamic_parametric)
- **Project repo:** [`Shehab-Hegab/flux-islamic-parametric`](https://github.com/Shehab-Hegab/flux-islamic-parametric)
- **Hugging Face dataset** (after `upload_to_hf.py --execute` with `HF_TOKEN`):
  `https://huggingface.co/datasets/Shehab-Hegab/islamic-parametric-architecture-dataset`
- **Hugging Face LoRA** (after Colab training + upload):
  `https://huggingface.co/Shehab-Hegab/flux-islamic-parametric-lora`
  - Weights file: `pytorch_lora_weights.safetensors`

## Quickstart

```bash
pip install -r requirements.txt

# 1) Dataset (Unsplash if UNSPLASH_ACCESS_KEY set, else synthetic — always yields 25 ≥1024²)
python download_or_synthetic_dataset.py --prefer hybrid

# 2) Captions (auto: Florence-2 if available, else template)
python generate_captions.py --backend auto

# 3) Training preview (local) / real run (Colab notebook)
python train_flux_lora.py --dry-run
jupyter nbconvert --to notebook --execute Flux_Architectural_LoRA_Training.ipynb  # or open in Colab

# 4) Evaluation grid
python inference_eval.py --check
python inference_eval.py --lora-path output

# 5) Upload (dry-run first; --execute requires HF_TOKEN)
python upload_to_hf.py --dry-run
python upload_to_hf.py --execute
```

## Training metadata

| Hyperparameter | Value |
|---|---|
| Base model | `black-forest-labs/FLUX.1-dev` |
| Method | DreamBooth LoRA (diffusers `train_dreambooth_lora_flux.py`) |
| Rank / Alpha | 16 / 16 |
| Learning rate | 1e-04 |
| Resolution | 1024×1024 |
| Max train steps | 800 (range 800–1000) |
| Optimizer | adamw8bit |
| Mixed precision | bf16 + gradient checkpointing |
| Batch / accum | 1 / 4 |
| Seed | 42 |

## Architectural design intent

Trained on modern Islamic parametric facades — geometric lattice screens, mashrabiya
patterns, girih and punched parametric envelopes — so generations preserve modular rhythm,
bilateral symmetry, and lattice topology with **zero visual hallucination** and precise
geometric structural integrity. Evaluation grids lock seed 42 and compare base FLUX.1-dev
against the adapter column-by-column.

## Dataset + caption contract

- 25 unique captions; every caption contains `in Islamic_Parametric style`
- Caption skeleton: *A detailed architectural photo in Islamic_Parametric style, featuring …,
  precise geometric lattice patterns, daylighting, structural symmetry, photorealistic 8k
  architectural render, no visual distortion.*

## Environment variables

| Var | Used by | Notes |
|---|---|---|
| `UNSPLASH_ACCESS_KEY` | dataset builder | optional; synthetic fallback if unset |
| `HF_TOKEN` | training (gated FLUX), upload | never commit tokens |

## Tests & lint

```bash
ruff check src tests *.py
python -m pytest -q
```

## License notes

- FLUX.1-dev weights are subject to the FLUX.1-dev community license (accept on Hugging Face).
- Unsplash images follow the Unsplash API license; synthetic images are generated in-repo.
