# Hugging Face URL structure (for reply email)

Use these exact URLs in the portfolio / hiring response email after upload.

## 1) Dataset repository

- Landing page:
  `https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset`
- Manifest:
  `https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset/blob/main/manifest.json`
- First image:
  `https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset/blob/main/image_01.jpg`
- First caption:
  `https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset/blob/main/captions/image_01.txt`
- Tree view (all files):
  `https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset/tree/main`

## 2) LoRA model repository

- Landing page (model card = `README.md`):
  `https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora`
- Weights file:
  `https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora/blob/main/pytorch_lora_weights.safetensors`
- Resolve URL (for `load_lora_weights` / HTTP download):
  `https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora/resolve/main/pytorch_lora_weights.safetensors`
- Tree view:
  `https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora/tree/main`

## 3) Base model (upstream, not owned)

- `https://huggingface.co/black-forest-labs/FLUX.1-dev`

## Email snippet

Subject: FLUX.1 LoRA — Islamic Parametric Architecture (dataset + weights + eval)

Body links:

1. Dataset (25×≥1024², captions, manifest): https://huggingface.co/datasets/shehab-hegab/islamic-parametric-architecture-dataset
2. LoRA weights + model card: https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora
3. Weights direct: https://huggingface.co/shehab-hegab/flux-islamic-parametric-lora/blob/main/pytorch_lora_weights.safetensors

Trigger word: `in Islamic_Parametric style`
