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
| `dataset_islamic_parametric/` | 25 gold style-unified free-license photos ≥1024×1024 + `captions/*.txt` + `manifest.json` |
| `gold_dataset.py` (+ `gold_fill`/`gold_repair`/…) | Gold rebuild: Openverse top-up, license filter, category balance, PASS report |
| `finalize_dataset.py` | Dedup + renumber + caption + contact sheet + quality report (`--check`/`--execute`) |
| `download_or_synthetic_dataset.py` | Legacy hybrid builder (Unsplash → synthetic fallback) |
| `generate_captions.py` | Florence-2-large captions with deterministic template fallback |
| `train_flux_lora.py` | FLUX DreamBooth LoRA wrapper (`--dry-run` / Colab GPU run) |
| `training/train_dreambooth_lora_flux.py` | Official diffusers training script (vendored) |
| `Flux_Architectural_LoRA_Training.ipynb` | Google Colab notebook (T4/A100) |
| `inference_eval.py` | Side-by-side base vs LoRA grids (seed 42, 1024²) |
| `evaluate_structure.py` | Quantitative geometry metrics (symmetry, edges, Hough, periodicity) |
| `expand_dataset.py` | License-aware dataset expansion (Wikimedia/CC0 metadata fields) |
| `upload_to_hf.py` | Dry-run / `--execute` upload of dataset + LoRA |
| `src/islamic_parametric/constants.py` | Single source of truth: trigger, hyperparams, repo IDs |

## Where the data lives (verified live links)

- **Hugging Face dataset (LIVE — gold style-unified set, 25 images + captions + manifest):**
  [`Shehab-Hegab/islamic-parametric-architecture-dataset`](https://huggingface.co/datasets/Shehab-Hegab/islamic-parametric-architecture-dataset)
- **GitHub project:** [`Shehab-Hegab/flux-islamic-parametric`](https://github.com/Shehab-Hegab/flux-islamic-parametric)
- **Hugging Face LoRA** (model card live; weights after Colab training):
  [`Shehab-Hegab/flux-islamic-parametric-lora`](https://huggingface.co/Shehab-Hegab/flux-islamic-parametric-lora)
  - Weights: `pytorch_lora_weights.safetensors` (upload after training)

## Quickstart

```bash
pip install -r requirements.txt

# 1) Dataset already in-repo: 25 unique bright free-license photos + captions + manifest
python finalize_dataset.py --check

# 2) Re-caption / regenerate quality report if needed (auto: Florence-2 if available, else template)
python generate_captions.py --backend auto

# 3) Training preview (local) / real run (Colab notebook)
python train_flux_lora.py --dry-run
# Open Flux_Architectural_LoRA_Training.ipynb in Google Colab (T4 GPU)

# 4) Evaluation grid + structural metrics
python inference_eval.py --check
python inference_eval.py --lora-path output
python evaluate_structure.py --check
python evaluate_structure.py --images dataset_islamic_parametric --limit 5

# 5) Upload (dry-run first; --execute requires HF_TOKEN)
python upload_to_hf.py --dry-run
python upload_to_hf.py --dataset-only --execute   # replace remote dataset (purges stale files)
python upload_to_hf.py --execute                 # after Colab training produces weights
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
patterns, girih and punched parametric envelopes — so generations aim to preserve modular
rhythm, bilateral symmetry, and lattice topology. **Geometry is evaluated, not guaranteed:**
use `evaluate_structure.py` (bilateral symmetry error, edge density, Hough lines, periodicity)
plus side-by-side grids at seed 42. Text-to-image models cannot promise zero hallucination;
we measure structural properties instead.

## License notes (non-commercial)

- **This project is a non-commercial research/portfolio demo only.**
- `black-forest-labs/FLUX.1-dev` and **derivative fine-tunes (including this LoRA)** are under
  the FLUX.1-dev non-commercial license — commercial deployment/API/SaaS requires a separate
  BFL license. See https://huggingface.co/black-forest-labs/FLUX.1-dev
- Do not present this adapter as commercially deployable.
- Training photos are **free-license only** (CC0 / CC BY / CC BY-SA / Public Domain) with
  author + source URL recorded in `manifest.json`. See the dataset README / HF card for
  full attribution. No synthetic top-up in the live set.

## Dataset + caption contract

- 25 unique free-license photos (mean luma ≈ 113, no md5/phash duplicates) + quality report
- 25 unique captions; every caption contains `in Islamic_Parametric style`
- Caption skeleton: *A detailed architectural photo in Islamic_Parametric style, featuring …,
  precise geometric lattice patterns, daylighting, structural symmetry, photorealistic 8k
  architectural render, no visual distortion.*

## Environment variables

| Var | Used by | Notes |
|---|---|---|
| `UNSPLASH_ACCESS_KEY` | legacy dataset builder | optional; not required for the live free-license set |
| `HF_TOKEN` | training (gated FLUX), upload | never commit tokens |

## Tests & lint

```bash
ruff check src tests *.py
python -m pytest -q
```

## License notes (non-commercial)

- **Non-commercial research / portfolio demo only.** FLUX.1-dev and derivatives (this LoRA)
  are under the FLUX.1-dev non-commercial license — commercial use needs a separate BFL license.
- Dataset photos: free licenses only (CC0 / CC BY / CC BY-SA / Public Domain); attribution
  in `manifest.json` (`author`, `license`, `source_url`).
