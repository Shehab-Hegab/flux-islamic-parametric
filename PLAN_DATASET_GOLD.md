# PLAN_DATASET_GOLD — Reference Contract (Locked)

> Living gold plan for the Islamic Parametric FLUX LoRA portfolio.
> **This file is the permanent reference.** Do not weaken these gates.

**Goal:** Highest-quality, style-unified, free-license 25–30 image dataset + captions + training + HF/GitHub sync, matching Dr. Mai Soliman review criteria and the ArchGen product aesthetic.

**Locked hyperparameters:** base `black-forest-labs/FLUX.1-dev`, r=16, α=16, lr=1e-4, res=1024, steps=800, optimizer=adamw8bit, bf16, seed=42, trigger verbatim `in Islamic_Parametric style`.

**Repos:**
- GitHub: `https://github.com/Shehab-Hegab/flux-islamic-parametric` (PUBLIC)
- HF dataset: `Shehab-Hegab/islamic-parametric-architecture-dataset`
- HF LoRA: `Shehab-Hegab/flux-islamic-parametric-lora`

---

## Gold visual targets (from user reference sheets)

### Contact sheet bar
- Bright high-key daylight, white/cream GRC + sand stone + warm timber mashrabiya
- Unified modern-parametric Islamic language (portal arches + geometric screens)
- **No 2D flat vectors** as hero content (no flat girih tile scans as primary vocabulary)
- 0 people / 0 B&W / 0 scaffolding / 0 cityscape / 0 European heritage

### ArchGen product vocabulary (output style)
1. **Exterior ~40%** — grand entrances, modern Islamic facades, courtyard arcades, heritage-inspired portals with GRC/lattice screens
2. **Detail ~30%** — mashrabiya close-ups, lattice windows, geometric screens, arch details, muqarnas with sharp shadows
3. **Interior ~30%** — prayer halls, majlis/luxury interiors, patterned daylight through screens, warm golden light

Product claims the dataset must support: `SAME STYLE` • `HIGH ACCURACY` • `MINIMAL HALLUCINATION`.

---

## Numeric quality gates (ALL must PASS)

| Gate | Threshold |
|---|---|
| Count | 25–30 images (hard min 20) |
| Resolution | ≥1024 min side |
| Uniqueness | 0 md5 duplicate groups, 0 phash duplicate groups |
| License | Free only: CC0 / CC BY(-SA) / Public Domain / PDM |
| Brightness | mean luma ≥ 90; reject dark historic |
| B&W / sepia | saturation filter — reject low-chroma historical prints |
| Style coherence | One modern-parametric Islamic language; no multi-regional hero landmarks |
| Captions | Unique; trigger exact; no junk (`featuring image N`, raw filenames) |
| Tests + lint | pytest green, ruff clean |
| Dry-runs | train + upload + finalize --check green |

---

## Phase plan (execute end-to-end, no idle stops)

### Phase 0 — Lock gold reference + triage table
- Keep plan file (this document).
- Digital triage every `image_XX`: title/query/blacklist/saturation/luma.
- Build keep/delete lists before purge.

### Phase 1 — Surgical purge
- Delete confirmed rejects (French/European, B&W/sepia prints, cityscapes, flat-only vectors, noisy people/scaffolding heroes, multi-style landmark exteriors that bleed).
- Renumber `image_01..N`; drop orphan captions/manifest rows.
- Do **not** push partial dataset.

### Phase 2 — Targeted top-up (Openverse first, Commons gentle fallback)
Queries biased to gold vocabulary:
`modern mashrabiya facade`, `geometric screen facade`, `parametric facade`, `GRC facade`, `perforated facade`, `contemporary mosque`, `geometric lattice sunlight`, `muqarnas interior`, `islamic courtyard arcade`, `brise soleil geometric`.

Filters: free license, min side 1024, dedup, brightness, blacklist titles (cathedral/castle/gothic/European landmarks), prefer bright modern/screen-heavy frames.

### Phase 3 — Full caption rewrite (category-based)
- Categories: `facade` | `detail` | `interior` (assign by query/title heuristics).
- Template cores describing material + light + geometry — never filenames.
- Keep `build_caption` prefix/suffix contract: trigger + `no visual distortion`.
- Update tests to reject junk patterns.

### Phase 4 — Quality gates + gold sheet
- Rebuild `_contact_sheet_review.jpg`, `eval_outputs/dataset_quality_report.json`.
- `python finalize_dataset.py --check` (or gold equivalent) PASS.
- `ruff check src tests *.py` + `python -m pytest -q`.
- `train_flux_lora.py --dry-run`, `upload_to_hf.py --dry-run`.

### Phase 5 — Sync GitHub + HF
- Commit + push new images/captions/docs.
- `upload_to_hf.py --dataset-only --execute` (purge stale remote files).
- Verify remote file count / licenses / manifest.

### Phase 6 — Colab training (user clicks Run)
- Notebook `Flux_Architectural_LoRA_Training.ipynb` on T4.
- T4 safety flags already required: mixed precision bf16 (fp16 fallback), gradient_checkpointing, adamw8bit.
- After train: upload `pytorch_lora_weights.safetensors` via `upload_to_hf.py --execute`.

### Phase 7 — Eval vs gold aesthetic
- `inference_eval.py` grids + `evaluate_structure.py` metrics.
- Side-by-side vs ArchGen vocabulary (screens, portals, interior shadows, no bleed).

### Phase 8 — Application handoff
- Dataset link + model results + salary + job source.
- Subject: `Dataset Experience – Your Name`.

---

## Hard rules
1. Never commit `.env` / HF tokens.
2. Never scrape ArchDaily/Dezeen (copyright).
3. Never claim “zero hallucination”; use structural metrics language.
4. Trigger word case-sensitive everywhere.
5. Prefer most beautiful, brightest, most unified frames — aesthetic score matters.
6. Full autonomy: choose the highest-quality option; do not stop for trivia.

**Status:** Phases 0–5 DONE — gold PASS (25 free-license images, mean_luma 124.4, facade 12 / detail 7 / interior 6). GitHub pushed; HF dataset gold replace verified (25+25+manifest). Obsolete legacy scripts/tests removed from repo. Next: Phase 6 Colab handoff (user Run all on T4 with `HF_TOKEN` secret).
