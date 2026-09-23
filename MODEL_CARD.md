# FLUX.1-dev LoRA — Islamic Parametric Architecture

Adapter for **Islamic Parametric Architecture & Facades**: geometric lattice screens,
mashrabiya parametric patterns, and structurally symmetric modern envelopes with
zero visual hallucination intent (precise geometric structural integrity).

- **Trigger word:** `in Islamic_Parametric style`
- **Base model:** `black-forest-labs/FLUX.1-dev`
- **Dataset:** `shehab-hegab/islamic-parametric-architecture-dataset`
- **Weights:** `pytorch_lora_weights.safetensors`

## Training metadata

| Hyperparameter | Value |
|---|---|
| Rank / Alpha | 16 / 16 |
| Learning rate | 1e-04 |
| Resolution | 1024×1024 |
| Max train steps | 800 |
| Optimizer | adamw8bit |
| Method | DreamBooth LoRA (diffusers `train_dreambooth_lora_flux.py`) |
| Mixed precision | bf16 + gradient checkpointing |

## Usage

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("shehab-hegab/flux-islamic-parametric-lora", weight_name="pytorch_lora_weights.safetensors")
pipe.enable_model_cpu_offload()
image = pipe(
    "A modern cultural center facade in Islamic_Parametric style, geometric parametric wooden panels, realistic lighting, 8k architectural photo",
    height=1024, width=1024, num_inference_steps=28, guidance_scale=3.5,
).images[0]
image.save("out.png")
```

## Sample caption (training data)

> A detailed architectural photo in Islamic_Parametric style, featuring a modern building exterior with a parametric mashrabiya facade, precise geometric lattice patterns, daylighting, structural symmetry, photorealistic 8k architectural render, no visual distortion.

## Architectural design intent

Preserve modular rhythm, bilateral symmetry, and lattice topology across generated views;
evaluation grids (seed 42) compare base FLUX.1-dev against this adapter side-by-side to
verify geometric structural logic.

See the source repository for `inference_eval.py`, dataset builder, captioning, and the
Colab notebook `Flux_Architectural_LoRA_Training.ipynb`.
