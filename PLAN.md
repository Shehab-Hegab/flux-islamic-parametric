# PLAN — FLUX.1 Islamic Parametric LoRA Pipeline

> Living progress file. Updated as work proceeds. Last updated: 2026-09-23.

**Goal:** Production-grade HF repo: 25-image Islamic Parametric dataset + captions + FLUX.1-dev LoRA training pipeline + eval grids + HF upload scripts, portfolio-ready for Dr. Mai Soliman review.

**Target repos**
- HF dataset (LIVE): `https://huggingface.co/datasets/Shehab-Hegab/islamic-parametric-architecture-dataset`
- HF LoRA card (LIVE; weights pending Colab): `https://huggingface.co/Shehab-Hegab/flux-islamic-parametric-lora`
- Project (GitHub): `https://github.com/Shehab-Hegab/flux-islamic-parametric`
- Dataset mirror (GitHub): `https://github.com/Shehab-Hegab/flux-islamic-parametric/tree/main/dataset_islamic_parametric`

**Locked hyperparameters:** base `black-forest-labs/FLUX.1-dev`, r=16, α=16, lr=1e-4, res=1024, steps=800, optimizer=adamw8bit, bf16, trigger `in Islamic_Parametric style`.

---

## Checklist

### Phase 1 — Scaffold & design (DONE)
- [x] Design spec `docs/superpowers/specs/2026-09-23-flux-islamic-parametric-design.md` (commit c6f3505)
- [x] Implementation plan `docs/superpowers/plans/2026-09-23-flux-islamic-parametric-pipeline.md`
- [x] pyproject (ruff), .gitignore, requirements.txt, `src/islamic_parametric/{__init__,constants}.py` (73d3852)

### Phase 2 — Dataset (GOLD rebuild — see PLAN_DATASET_GOLD.md)
- [x] Legacy synthetic rejected; free-license historic set superseded
- [x] **Gold contract locked:** `PLAN_DATASET_GOLD.md` (style-unified 40/30/30, no bleed, no junk captions)
- [x] Gold purge + Openverse top-up (`gold_dataset.py` / `gold_fill.py` / `gold_repair.py`)
- [x] Quality report PASS + gold contact sheet (`eval_outputs/dataset_quality_report.json`)
- [x] Tests green after caption rewrite (ruff clean, pytest green)

### Phase 3 — Captions (DONE)
- [x] Gold category-based captions via `gold_dataset.write_gold_captions` (facade/detail/interior cores)
- [x] 25 unique captions with trigger verified → ruff+pytest green
- [x] Caption contract: trigger + `no visual distortion` suffix kept

### Phase 4 — Training (DONE)
- [x] `train_flux_lora.py` wrapper (`--dry-run`, `--fetch-script`, CUDA gate)
- [x] Vendored `training/train_dreambooth_lora_flux.py`
- [x] Colab `Flux_Architectural_LoRA_Training.ipynb` (GPU metadata, full flow)
- [x] Dry-run verified tokens match locked hyperparameters

### Phase 5 — Evaluation (DONE)
- [x] `inference_eval.py` base vs LoRA side-by-side grids, seed 42, `--check` verified
- [x] **NEW (expert report):** `evaluate_structure.py` — bilateral symmetry, edge density, Hough/axis line ratio, periodicity → `eval_outputs/structural_metrics.json` (`--check`, `--limit`, optional `--clip`)

### Cleanup — obsolete files removed (2026-09-23)
- [x] Deleted legacy one-shot scripts + their tests: `download_or_synthetic_dataset.py`, `generate_captions.py`, `expand_dataset.py`, `curate_architecture.py`, `curate_wikimedia_landmarks.py`, `tools_analyze_dataset.py`, `gold_balance.py`, `gold_replace.py`, `gold_select.py`
- [x] Removed tracked `.opencode/state/*`, synthetic backup, stray PNG, old eval contact sheets
- [x] Kept gold core: `gold_dataset.py`, `gold_fill.py`, `gold_repair.py`, `finalize_dataset.py`
- [x] HF dataset re-verified clean (25 gold images + 25 captions + manifest + contact sheet only)

### Phase 4b — Checkpoint/resume (DONE — expert report)
- [x] `--checkpointing_steps=250 --checkpoints_total_limit=4` in dry-run + launch command (state dirs `checkpoint-250/500/750/800`)
- [x] `train_flux_lora.py --resume` → `--resume_from_checkpoint latest`
- [x] Colab notebook resume cell + checkpoint table row

### Phase 4c — License + claim language (DONE — expert report)
- [x] README: **non-commercial** FLUX.1-dev notice; "zero hallucination" → measurable structural evaluation
- [x] `MODEL_CARD.md` + `upload_to_hf.py` `_model_card()` same reframe + license line
- [x] Caption suffix `no visual distortion` kept (training prompt contract + tests); one eval prompt changed to `precise geometry`

### Phase 6 — HF upload + docs (DONE)
- [x] `upload_to_hf.py` dry-run verified (both repos + URLs)
- [x] `MODEL_CARD.md`, `README.md`, `docs/HF_URLS.md`
- [x] **Dataset LIVE on HF (gold set, replaced stale remote):** `https://huggingface.co/datasets/Shehab-Hegab/islamic-parametric-architecture-dataset`
- [x] `upload_to_hf.py` now **purges stale remote files** before upload (`--dataset-only --execute` = full replace with new free-license set)
- [x] **LoRA repo + model card LIVE**: `https://huggingface.co/Shehab-Hegab/flux-islamic-parametric-lora`
- [ ] Upload `pytorch_lora_weights.safetensors` after Colab training (`python upload_to_hf.py --execute` with weights present)

### Phase 7 — Quality gates (DONE)
- [x] ruff clean, pytest green after gold caption rewrite
- [x] 25 unique gold captions with trigger verified
- [x] Multi-model orchestrate review (6/29 models scored; top gpt-oss-120b 86/100) — applied real fixes: clear gated-model errors in `inference_eval.py`, instance-dir/image validation + launch error handling in `train_flux_lora.py`, HF upload exception wrapping; rejected hallucinated findings (no hardcoded tokens/paths in repo)
- [x] Final CLI verification: ruff, pytest, inference --check, train --dry-run, upload --dry-run all green

### Phase 8 — Delivery (DONE except LoRA weights upload)
- [x] GitHub repo created + pushed + made **PUBLIC** (Colab clone failed while private): `https://github.com/Shehab-Hegab/flux-islamic-parametric`
- [x] Colab notebook: clone + flatten to cwd; if dataset missing → `snapshot_download` from HF dataset (fallback)
- [x] Colab pip cell installs **diffusers from git main** (script needs `check_min_version 0.41.0.dev0`; PyPI only has 0.40.0)
- [x] PLAN.md living file at repo root (this file)
- [x] HF dataset + LoRA model card uploaded with account `Shehab-Hegab`
- [x] Token stored only in gitignored `.env` / `.hf_token` — never committed
- [ ] Weights upload after Colab training (user step)

---

## How to finish remaining work

```bash
ruff check src tests *.py && python -m pytest -q
python train_flux_lora.py --dry-run
python upload_to_hf.py --dataset-only --dry-run

# After Colab training + HF_TOKEN
python upload_to_hf.py --execute
```

## Status summary
| Deliverable | Status |
|---|---|
| Gold dataset 25 free-license photos + manifest | DONE (PASS: mean_luma 124.4, free licenses, facade 12 / detail 7 / interior 6) |
| Captions (25 unique + trigger) | DONE |
| train_flux_lora.py + vendored diffusers script | DONE |
| Colab notebook | DONE |
| inference_eval.py | DONE |
| evaluate_structure.py (geometry metrics) | DONE |
| checkpoint/resume (250-step) | DONE |
| Non-commercial license + claim language | DONE |
| upload_to_hf.py + model card + HF URLs docs | DONE |
| Tests + lint | GREEN (ruff clean + pytest green after cleanup) |
| GitHub push PUBLIC | DONE → https://github.com/Shehab-Hegab/flux-islamic-parametric (visibility PUBLIC) |
| Obsolete file cleanup (GitHub + local) | DONE (legacy scripts/tests/junk removed) |
| **HF dataset LIVE (gold replace)** | DONE → https://huggingface.co/datasets/Shehab-Hegab/islamic-parametric-architecture-dataset (25 gold images + captions + manifest) |
| **HF LoRA model card LIVE** | DONE → https://huggingface.co/Shehab-Hegab/flux-islamic-parametric-lora |
| LoRA weights upload | PENDING Colab training |

**Last updated after:** obsolete-file cleanup + GitHub/HF re-sync (session 2026-09-23). Next: baseline training on Colab T4, then weights upload.
