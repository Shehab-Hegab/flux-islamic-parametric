# Design Spec: FLUX.1 LoRA — Islamic Parametric Architecture Dataset & Training Pipeline

**Date:** 2026-09-23
**Status:** Approved autonomously (user granted full authority: "complete as expert, always recommended, no stopping for questions")
**Owner:** shehab-hegab · Portfolio demo for AI Architectural platform hiring review (evaluator: Dr. Mai Soliman, Getinform School)

## Goal

A production-grade, Hugging Face–ready repository containing: (1) a curated 25-image high-res dataset, (2) unique trigger-word captions, (3) trainable FLUX.1-dev LoRA config + weights pipeline, (4) side-by-side evaluation proving zero hallucination / geometric structural integrity, (5) one-command upload scripts + model card.

## User-facing deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Dataset (25 images ≥1024×1024) | `dataset_islamic_parametric/` |
| 2 | Per-image captions + trigger word | `dataset_islamic_parametric/captions/` (`image_01.txt` …) |
| 3 | Dataset builder (Unsplash hybrid → synthetic fallback) | `download_or_synthetic_dataset.py` |
| 4 | Caption generator (Florence-2-large primary) | `generate_captions.py` |
| 5 | Standalone training wrapper | `train_flux_lora.py` |
| 6 | Colab notebook | `Flux_Architectural_LoRA_Training.ipynb` |
| 7 | Evaluation / inference grid script | `inference_eval.py` |
| 8 | HF upload (dataset + LoRA repos) | `upload_to_hf.py` |
| 9 | Model card README | `README.md` (+ `MODEL_CARD.md` content) |
| 10 | HF URL structure for reply email | `docs/HF_URLS.md` |

## Architecture (module boundaries)

```
download_or_synthetic_dataset.py   → produces images (I/O: folder + manifest.json)
generate_captions.py               → produces captions/*.txt (reads images, writes txt)
train_flux_lora.py                 → wraps diffusers examples/train_dreambooth_lora_flux.py
inference_eval.py                  → loads FLUX.1-dev + LoRA → eval/ grids
upload_to_hf.py                    → huggingface_hub uploads (dataset_repo, lora_repo)
Flux_Architectural_LoRA_Training.ipynb → cells call the scripts above (Colab T4/A100 path)
```

Isolation rules: scripts never import each other; all shared constants live in `src/islamic_parametric/constants.py` (trigger word, repo IDs, hyperparams, caption style tokens). Each script is CLI-runnable with `--help` and has graceful degradation (missing API key → synthetic; missing GPU → clear error; missing LoRA weights → base-only comparison mode).

## Key design decisions (all “recommended”)

1. **Training runtime:** Google Colab (free T4 viable with 8-bit AdamW + `set_torch_dtype(torch.bfloat16)` + gradient checkpointing; A100/L4 preferred). Delivered as both `train_flux_lora.py` and notebook wrapping it.
2. **Dataset:** Hybrid. Top-up real architectural photos from Unsplash API (`UNSPLASH_ACCESS_KEY`); fill remainder with procedural synthetic renders (mashrabiya / geometric lattice / parametric facade via PIL+numpy at 1024²+). Missing key → 100% synthetic, no hard failure. Target: exactly **25** images, each `≥1024×1024`.
3. **Captioning:** `microsoft/Florence-2-large` primary (`<CAPTION>` task, then architectural style enhancement pass appending trigger word + zero-hallucination suffix tokens). If transformers/model unavailable → deterministic template captions from image manifest metadata (still unique, still contain trigger word).
4. **Training framework:** Hugging Face `diffusers` official `examples/train_dreambooth_lora_flux.py` (natively saves `pytorch_lora_weights.safetensors`). Hyperparameters locked to spec: base `black-forest-labs/FLUX.1-dev`, **r=16, α=16, lr=1e-4, res=1024×1024, steps=800–1000 (default 800), optimizer adamw8bit**, bf16 mixed precision, gradient checkpointing.
5. **Trigger word:** exact `in Islamic_Parametric style` appended to every caption and every eval prompt. Caption template matches spec example (photorealistic 8k architectural render, no visual distortion, geometric lattice, structural symmetry…).
6. **Evaluation:** `inference_eval.py` builds a grid: rows = prompts (≥5, incl. cultural center facade example), cols = [base FLUX.1-dev | FLUX.1-dev + LoRA] at fixed seed (e.g. 42), 1024², saved under `eval_outputs/`. Optional `--compare-baseline` for 4-column grid (base / lora / lora+geometry-check).
7. **HF repos:** dataset `shehab-hegab/islamic-parametric-architecture-dataset`; LoRA `shehab-hegab/flux-islamic-parametric-lora` (`pytorch_lora_weights.safetensors` + structured model card with trigger words, samples, training metadata, architectural design intent).
8. **Auth:** `HF_TOKEN` env var only; never hardcode. Unsplash key likewise env-only.

## Data flow

```
UNSPLASH_ACCESS_KEY? ──yes──► fetch ~N real photos ──┐
                          no/short ─────────────────┤
                                                   ▼
                     procedural synthetic top-up (PIL/numpy)
                                                   ▼
              download_or_synthetic_dataset.py → 25 images + manifest.json
                                                   ▼
              generate_captions.py → captions/image_XX.txt (Florence-2 | template)
                                                   ▼
              train_flux_lora.py → diffusers train_dreambooth_lora_flux
              (Colab GPU)        → output/pytorch_lora_weights.safetensors
                                                   ▼
              inference_eval.py → eval_outputs/grid_*.png
                                                   ▼
              upload_to_hf.py → HF dataset repo + LoRA model repo
```

## Error handling

- Missing `UNSPLASH_ACCESS_KEY`: log warning → synthetic-only (exit 0).
- Unsplash HTTP failure/timeout: fall back mid-run to synthetic for remaining slots.
- Florence-2 load failure (old transformers / no CUDA): warn → template captions (exit 0).
- No CUDA on training entry: exit 1 with actionable message (“use Colab notebook / --dry-run prints full diffusers command”).
- `--dry-run` on `train_flux_lora.py`: print resolved CLI without launching training (testable locally).
- Upload scripts: dry-run mode listing files+dest; require `HF_TOKEN` for real run.
- Image validation: reject any image <1024×1024 (regenerate/resize-not-allowed policy: synthetic regenerates at valid size; Unsplash requests high-res `w=2400` then verifies).

## Testing strategy

1. **Unit-ish smoke tests** (`tests/test_pipeline.py`): constants (trigger word format), synthetic generator produces ≥25 valid ≥1024² RGB images, caption files all contain trigger word, manifest ↔ image ↔ caption 1:1 integrity, `train_flux_lora.py --dry-run` exits 0 with correct hyperparams in command string, eval prompt list contains trigger word.
2. **Local execution:** run dataset builder + captioning end-to-end in-session; run `--dry-run` for training; `inference_eval.py --check` validates paths/prompts without loading FLUX.
3. **Lint:** `ruff check` (config in `pyproject.toml`).
4. **Expert review:** `orchestrate` multi-model critical review (code quality, FLUX training correctness, HF readiness); fix findings; re-review.
5. **Notebook sanity:** JSON-valid ipynb, all cells compile (`nbformat` + `ast` check).

## Out of scope (YAGNI)

- Actually executing 800-step FLUX training locally (needs ~40GB+ / paid GPU → Colab path).
- Uploading to HF from this machine (no token assumed; script ships with dry-run).
- ai-toolkit (ostris) path, Qwen2-VL captioner (kept as documented fallback only in README).
- Web UI, CI/CD, Docker.

## Success criteria

- [ ] 25 images ≥1024×1024 in `dataset_islamic_parametric/` + matching unique captions with trigger word
- [ ] `download_or_synthetic_dataset.py` hybrid+fallback works with and without API key
- [ ] `generate_captions.py` Florence-2 path + template fallback both emit spec-compliant captions
- [ ] `train_flux_lora.py` dry-run prints correct FLUX/r16/α16/lr1e-4/1024/800/adamw8bit command; notebook wraps it
- [ ] `inference_eval.py` produces side-by-side grid structure (smoke: prompt grid + layout w/o GPU)
- [ ] `upload_to_hf.py` dry-run lists dataset + LoRA destinations; README model card complete
- [ ] `ruff` clean; tests pass; orchestrate expert review applied
- [ ] `docs/HF_URLS.md` contains exact URL structure for reply email
