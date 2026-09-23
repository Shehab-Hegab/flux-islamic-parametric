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

### Phase 2 — Dataset (REPLACED — free-license real photos)
- [x] Legacy `download_or_synthetic_dataset.py` hybrid Unsplash→synthetic (superseded)
- [x] Synthetic set rejected (dark/flat/duplicated) → replaced
- [x] **Live set:** 25 unique bright free-license photos (`finalize_dataset.py`), mean luma 112.9, 0 md5/phash dups, `manifest.json` attribution
- [x] Quality report PASS (`eval_outputs/dataset_quality_report.json`)
- [x] Official review sheet `dataset_islamic_parametric/_contact_sheet_review.jpg` (stale `_contact_sheet_25` / `_contact_sheet_curation` deleted)
- [x] Tests green (36)

### Phase 3 — Captions (DONE)
- [x] `generate_captions.py` with auto/florence/template backends (a7b1853)
- [x] Florence-2-large fixed for transformers 4.57 (SDPA property patch + greedy `use_cache=False` + remote processor); einops/timm installed
- [x] 25 unique captions with trigger verified → ruff+pytest green
- [x] Caption contract via `build_caption` + `no visual distortion` suffix kept

### Phase 4 — Training (DONE)
- [x] `train_flux_lora.py` wrapper (`--dry-run`, `--fetch-script`, CUDA gate)
- [x] Vendored `training/train_dreambooth_lora_flux.py`
- [x] Colab `Flux_Architectural_LoRA_Training.ipynb` (GPU metadata, full flow)
- [x] Dry-run verified tokens match locked hyperparameters

### Phase 5 — Evaluation (DONE)
- [x] `inference_eval.py` base vs LoRA side-by-side grids, seed 42, `--check` verified
- [x] **NEW (expert report):** `evaluate_structure.py` — bilateral symmetry, edge density, Hough/axis line ratio, periodicity → `eval_outputs/structural_metrics.json` (`--check`, `--limit`, optional `--clip`)
- [x] **NEW:** `expand_dataset.py` — license-tracked Wikimedia Commons expansion (author/license/source_url into `manifest.json`)

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
- [x] **Dataset LIVE on HF** (25 jpg + 25 captions + manifest): `https://huggingface.co/datasets/Shehab-Hegab/islamic-parametric-architecture-dataset`
- [x] `upload_to_hf.py` now **purges stale remote files** before upload (`--dataset-only --execute` = full replace with new free-license set)
- [x] **LoRA repo + model card LIVE**: `https://huggingface.co/Shehab-Hegab/flux-islamic-parametric-lora`
- [ ] Upload `pytorch_lora_weights.safetensors` after Colab training (`python upload_to_hf.py --execute` with weights present)

### Phase 7 — Quality gates (DONE)
- [x] ruff clean, pytest 18 passed
- [x] Florence-2 captions: 25 unique + trigger verified
- [x] Multi-model orchestrate review (6/29 models scored; top gpt-oss-120b 86/100) — applied real fixes: clear gated-model errors in `inference_eval.py`, instance-dir/image validation + launch error handling in `train_flux_lora.py`, HF upload exception wrapping; rejected hallucinated findings (no hardcoded tokens/paths in repo)
- [x] Final CLI verification: ruff, pytest, inference --check, train --dry-run, upload --dry-run all green

### Phase 8 — Delivery (DONE except LoRA weights upload)
- [x] GitHub repo created + pushed + made **PUBLIC** (Colab clone failed while private): `https://github.com/Shehab-Hegab/flux-islamic-parametric`
- [x] Colab notebook: clone + flatten to cwd; if dataset missing → `snapshot_download` from HF dataset (fallback)
- [x] PLAN.md living file at repo root (this file)
- [x] HF dataset + LoRA model card uploaded with account `Shehab-Hegab`
- [x] Token stored only in gitignored `.env` / `.hf_token` — never committed
- [ ] Weights upload after Colab training (user step)

---

## How to finish remaining work

```bash
pip install einops timm
python generate_captions.py --backend auto      # Florence captions
ruff check src tests *.py && python -m pytest -q
git add -A && git commit -m "feat: florence captions"   # if regenerated

# Expert review (orchestrate)
& "C:\Users\Shehab\.opencode\orchestrate.ps1" -Task "Review this FLUX LoRA repo for quality issues" -Mode auto

# GitHub push
gh repo create flux-islamic-parametric --private --source . --push
# or: git remote add origin https://github.com/Shehab-Hegab/flux-islamic-parametric.git && git push -u origin main

# After Colab training + HF_TOKEN
python upload_to_hf.py --execute
```

## Status summary
| Deliverable | Status |
|---|---|
| Dataset 25 free-license photos + manifest | DONE (quality PASS) |
| Captions (25 unique + trigger) | DONE |
| train_flux_lora.py + vendored diffusers script | DONE |
| Colab notebook | DONE |
| inference_eval.py | DONE |
| evaluate_structure.py (geometry metrics) | DONE |
| expand_dataset.py (license-aware) | DONE |
| checkpoint/resume (250-step) | DONE |
| Non-commercial license + claim language | DONE |
| upload_to_hf.py + model card + HF URLs docs | DONE |
| Tests + lint | GREEN (28 passed, ruff clean) |
| GitHub push PUBLIC | DONE → https://github.com/Shehab-Hegab/flux-islamic-parametric (visibility PUBLIC) |
| Multi-model orchestrate review | DONE (fixes applied) |
| **HF dataset LIVE** | DONE → https://huggingface.co/datasets/Shehab-Hegab/islamic-parametric-architecture-dataset |
| **HF LoRA model card LIVE** | DONE → https://huggingface.co/Shehab-Hegab/flux-islamic-parametric-lora |
| LoRA weights upload | PENDING Colab training |
| Dataset expand 50–75 real+synthetic | SCAFFOLD ready (`expand_dataset.py`) — run after baseline |
| Expert-report roadmap items | Partly DONE this session (license, metrics, checkpoints); baseline training still next |

**Last updated after:** Repo made PUBLIC + Colab notebook clone/HF-dataset fallback (session 2026-09-23). Next: baseline training on Colab T4, then weights upload.
