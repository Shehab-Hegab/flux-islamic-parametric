"""Generate trigger-word captions for the Islamic Parametric dataset (Florence-2 or template)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    CAPTIONS_DIR,
    DATASET_DIR,
    NUM_IMAGES,
    TRIGGER_WORD,
    build_caption,
)

CORE_DESCRIPTIONS = [
    "a modern building exterior with a parametric mashrabiya facade, precise geometric lattice patterns, daylighting, structural symmetry",
    "a contemporary mosque entrance with an eight-point star screen and perforated stone cladding under golden hour light",
    "a cultural pavilion wrapped in woven aluminium grille panels with layered depth and cast shadow geometry",
    "an arched courtyard colonnade with hexagonal mashrabiya infill and rhythmic structural bays",
    "a museum facade of tessellated girih tiles with sharp edges and calibrated daylighting slots",
    "a high-rise envelope of punch-perforated parametric panels with gradient aperture density",
    "a prayer hall interior with low poly wooden lattice muqarnas and diffuse skylight",
    "a civic library exterior combining strip lattice bands and star-pattern vent screens",
    "a geometric curtain wall of interlocking octagon-unit modules with structural symmetry",
    "a gateway portal framed by layered mashrabiyaa screens and precise bilateral symmetry",
    "a waterfront promenade pavilion with woven timber shade structure and lattice shadows",
    "a modern minaret clad in diamond-grid stone with continuous vertical geometric relief",
    "a university atrium with suspended parametric screens and daylight harvesting geometry",
    "a transit hub canopy using hexagonal lattice steelwork with rhythmic repetition",
    "a boutique hotel courtyard with mashrabiya balconies and geometric paving alignment",
    "a science center volume wrapped in double-skin perforated metal with parametric punch fields",
    "a ceremonial arch sequence of repeating pointed-arch screens with strict modular rhythm",
    "a residential tower with rotating star-pattern balcony screens and structural regularity",
    "a garden gazebo of interlaced girih strips casting crisp geometric shade patterns",
    "a grand entrance hall with layered lattice chandeliers and axial symmetry",
    "a conference center facade mixing woven grille bands and punch-gradient panels",
    "a cultural museum ramp enclosure with continuous hexagonal mashrabiya glazing",
    "a plaza shade canopy of tessellated eight-point stars with calibrated openings",
    "an academic wing exterior with rhythmic arch screens and photoreal material detail",
    "a landmark dome drum pierced with parametric star apertures and exact geometry",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_manifest(dataset_dir: Path) -> list[dict]:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        {"filename": p.name, "source": "unknown", "query": "architectural_photo"}
        for p in sorted(dataset_dir.glob("image_*"))
        if p.suffix.lower() in IMAGE_EXTS
    ]


def template_caption(image_name: str, manifest_entry: dict, index: int) -> str:
    core = CORE_DESCRIPTIONS[index % len(CORE_DESCRIPTIONS)]
    source = manifest_entry.get("source", "synthetic")
    query = manifest_entry.get("query", "")
    unique = f"variant {index + 1:02d} of {query.replace('_', ' ') if query else 'parametric facade study'} ({source} reference)"
    return build_caption(f"{core}; {unique}")


def florence_caption(pil_image) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Florence-2-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(device)
    inputs = processor(text="<CAPTION>", images=pil_image, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=80)
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    raw = raw.replace("<CAPTION>", "").strip()
    return build_caption(raw if raw else "a modern architectural facade with geometric lattice detailing")


def _write_unique(path: Path, text: str, seen: set[str]) -> str:
    candidate = text
    bump = 1
    while candidate in seen:
        bump += 1
        candidate = f"{text[:-1]} — detail revision {bump}." if text.endswith(".") else f"{text} revision {bump}"
    seen.add(candidate)
    path.write_text(candidate, encoding="utf-8")
    return candidate


def generate_captions(backend: str, dataset_dir: Path, captions_dir: Path) -> int:
    captions_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dataset_dir)
    images: list[dict] = [
        entry for entry in manifest if (dataset_dir / entry["filename"]).exists()
    ]
    if not images:
        raise FileNotFoundError(f"no dataset images found under {dataset_dir}")

    use_florence = backend in ("auto", "florence")
    if use_florence and backend == "auto":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            print(f"[warn] Florence-2 deps unavailable ({exc}); using template backend")
            use_florence = False
    if backend == "florence" and not use_florence:
        raise RuntimeError("florence backend requested but dependencies missing")

    seen: set[str] = set()
    written = 0
    florence_alive = use_florence
    for index, entry in enumerate(images):
        stem = Path(entry["filename"]).stem
        out_path = captions_dir / f"{stem}.txt"
        caption: str | None = None
        if florence_alive:
            try:
                from PIL import Image

                with Image.open(dataset_dir / entry["filename"]) as im:
                    caption = florence_caption(im.convert("RGB"))
                if TRIGGER_WORD not in caption:
                    caption = build_caption(caption)
            except Exception as exc:
                print(f"[warn] Florence-2 failed on {entry['filename']}: {exc}")
                caption = None
                if index == 0:
                    print("[warn] disabling Florence-2 for remaining images; template fallback")
                    florence_alive = False
        if caption is None:
            caption = template_caption(entry["filename"], entry, index)
        _write_unique(out_path, caption, seen)
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate trigger-word captions for the dataset.")
    parser.add_argument("--backend", choices=["auto", "florence", "template"], default="auto")
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--captions", default=CAPTIONS_DIR)
    parser.add_argument("--expected", type=int, default=NUM_IMAGES)
    args = parser.parse_args(argv)
    written = generate_captions(args.backend, Path(args.dataset), Path(args.captions))
    print(f"Wrote {written} captions to {args.captions}")
    if written < args.expected:
        print(f"[warn] expected {args.expected} captions, got {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
