# PLAN — FLUX.1 Islamic Parametric LoRA Pipeline

> Living progress file. Updated as work proceeds. Last updated: 2026-09-23.

**Goal:** Production-grade HF repo: 25-image Islamic Parametric dataset + captions + FLUX.1-dev LoRA training pipeline + eval grids + HF upload scripts, portfolio-ready for Dr. Mai Soliman review.

**Target repos**
- Dataset: `https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset`
- LoRA: `https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora`
- GitHub: `https://github.com/Shehab-Hegab` (push target)

**Locked hyperparameters:** base `black-forest-labs/FLUX.1-dev`, r=16, α=16, lr=1e-4, res=1024, steps=800, optimizer=adamw8bit, bf16, trigger `in Islamic_Parametric style`.

---

## Checklist

### Phase 1 — Scaffold & design (DONE)
- [x] Design spec `docs/superpowers/specs/2026-09-23-flux-islamic-parametric-design.md` (commit c6f3505)
- [x] Implementation plan `docs/superpowers/plans/2026-09-23-flux-islamic-parametric-pipeline.md`
- [x] pyproject (ruff), .gitignore, requirements.txt, `src/islamic_parametric/{__init__,constants}.py` (73d3852)

### Phase 2 — Dataset (DONE)
- [x] `download_or_synthetic_dataset.py` hybrid Unsplash→synthetic
- [x] 25 images ≥1024×1024 generated + `manifest.json` committed (25f100b)
- [x] Tests green

### Phase 3 — Captions (DONE)
- [x] `generate_captions.py` with auto/florence/template backends (a7b1853)
- [x] Florence-2-large fixed for transformers 4.57 (SDPA property patch + greedy `use_cache=False` + remote processor); einops/timm installed
- [x] 25 unique Florence captions with trigger verified → ruff+pytest green

### Phase 4 — Training (DONE)
- [x] `train_flux_lora.py` wrapper (`--dry-run`, `--fetch-script`, CUDA gate)
- [x] Vendored `training/train_dreambooth_lora_flux.py`
- [x] Colab `Flux_Architectural_LoRA_Training.ipynb` (GPU metadata, full flow)
- [x] Dry-run verified tokens match locked hyperparameters

### Phase 5 — Evaluation (DONE)
- [x] `inference_eval.py` base vs LoRA side-by-side grids, seed 42, `--check` verified

### Phase 6 — HF upload + docs (DONE — execution pending tokens)
- [x] `upload_to_hf.py` dry-run verified (both repos + URLs)
- [x] `MODEL_CARD.md`, `README.md`, `docs/HF_URLS.md`
- [ ] `--execute` after Colab training produces `pytorch_lora_weights.safetensors` + `HF_TOKEN` set

### Phase 7 — Quality gates (IN PROGRESS)
- [x] ruff clean, pytest 18 passed (at c544d75)
- [ ] Florence caption pass → re-run ruff+pytest+commit
- [ ] Orchestrate multi-model expert review → apply fixes → re-run tests → commit
- [ ] Final verification sweep (dataset integrity, notebook, all CLIs)

### Phase 8 — Delivery (PENDING)
- [ ] Create GitHub repo under `https://github.com/Shehab-Hegab` and push `main`
- [ ] Final report: HF URL structure + GitHub URL + status of each deliverable

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
| Dataset 25×≥1024² + manifest | DONE |
| Captions (template fallback) | DONE |
| Captions (Florence-2) | PENDING |
| train_flux_lora.py + vendored diffusers script | DONE |
| Colab notebook | DONE |
| inference_eval.py | DONE |
| upload_to_hf.py + model card + HF URLs docs | DONE (execute after training) |
| Tests + lint | GREEN (18 passed) |
| GitHub push to Shehab-Hegab | PENDING |
| Multi-model orchestrate review | PENDING |
| Live HF upload (needs HF_TOKEN + trained weights) | PENDING USER/COLAB |
