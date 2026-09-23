# Project links (verified 2026-09-23)

## Live dataset (GitHub — actually contains the 25 images + captions)

Data worked on in this repo lives under `dataset_islamic_parametric/` and is pushed to GitHub.

- Dataset folder (images + captions + manifest):
  `https://github.com/Shehab-Hegab/flux-islamic-parametric/tree/main/dataset_islamic_parametric`
- Manifest:
  `https://github.com/Shehab-Hegab/flux-islamic-parametric/blob/main/dataset_islamic_parametric/manifest.json`
- First image:
  `https://github.com/Shehab-Hegab/flux-islamic-parametric/blob/main/dataset_islamic_parametric/image_01.jpg`
- First caption:
  `https://github.com/Shehab-Hegab/flux-islamic-parametric/blob/main/dataset_islamic_parametric/captions/image_01.txt`
- All captions (folder):
  `https://github.com/Shehab-Hegab/flux-islamic-parametric/tree/main/dataset_islamic_parametric/captions`
- Full project repo:
  `https://github.com/Shehab-Hegab/flux-islamic-parametric`

## Hugging Face targets (NOT live yet — upload pending)

These return 401/404 until `HF_TOKEN` is set and `python upload_to_hf.py --execute` runs
(dataset + LoRA after Colab training). Do **not** paste them as working links until then.

- Dataset (planned):
  `https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset`
- LoRA (planned):
  `https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora`
- Weights resolve (planned):
  `https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora/resolve/main/pytorch_lora_weights.safetensors`

## Base model (upstream)

- `https://huggingface.co/black-forest-labs/FLUX.1-dev`

## Email snippet (share working links)

Subject: FLUX.1 LoRA — Islamic Parametric Architecture (dataset + training pipeline)

Body:

1. Dataset (25×≥1024², Florence-2 captions, manifest) — GitHub:
   https://github.com/Shehab-Hegab/flux-islamic-parametric/tree/main/dataset_islamic_parametric
2. Full pipeline repo (training, eval, upload scripts, Colab):
   https://github.com/Shehab-Hegab/flux-islamic-parametric
3. Hugging Face LoRA (after training upload):
   https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora

Trigger word: `in Islamic_Parametric style`
