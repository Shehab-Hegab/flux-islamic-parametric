"""Quantitative structural evaluation for Islamic parametric geometry (no GPU required).

Metrics per image (0..1 unless noted):
- bilateral_symmetry: 1 - mean(|I - flip(I)|)/255  (higher is more symmetric)
- edge_density: fraction of Canny-like edge pixels (via Pillow FIND_EDGES proxy + threshold)
- line_axis_ratio: alignment of strong gradients to horizontal/vertical (0..1)
- periodicity: peak normalized autocorrelation on grayscale row/col signals
- overall_structure: mean of the four metrics

Writes eval_outputs/structural_metrics.json. CLIP score is optional (--clip).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import EVAL_OUTPUT_DIR, MIN_SIZE  # noqa: E402


def _require_pillow():
    from PIL import Image, ImageFilter  # noqa: F401

    return Image, ImageFilter


def _require_numpy():
    import numpy as np

    return np


def bilateral_symmetry(gray) -> float:
    np = _require_numpy()
    flipped = np.fliplr(gray)
    mae = float(np.mean(np.abs(gray.astype(np.float64) - flipped.astype(np.float64))))
    return max(0.0, 1.0 - mae / 255.0)


def edge_density(image, threshold: int = 48) -> float:
    Image, ImageFilter = _require_pillow()
    np = _require_numpy()
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.float64)
    return float(np.mean(arr > threshold))


def line_axis_ratio(gray) -> float:
    """Fraction of strong gradient magnitude aligned to 0° or 90° (architectural axes)."""
    np = _require_numpy()
    g = gray.astype(np.float64)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    thr = max(1e-6, float(np.percentile(mag, 90)))
    mask = mag >= thr
    if not np.any(mask):
        return 0.0
    gx_m, gy_m = gx[mask], gy[mask]
    # angle of gradient; edge direction is perpendicular — use axis alignment of edge
    angle = np.arctan2(gy_m, gx_m) % (np.pi / 2)
    # distance to nearest axis (0 or 90 for edges → gradient at 90/0)
    dist = np.minimum(np.abs(angle - 0.0), np.abs(angle - np.pi / 2))
    aligned = np.cos(2 * dist) ** 2
    return float(np.clip(np.mean(aligned), 0.0, 1.0))


def periodicity(gray, max_lag_frac: float = 0.4) -> float:
    np = _require_numpy()
    g = gray.astype(np.float64)
    sig_x = g.mean(axis=0) - g.mean()
    sig_y = g.mean(axis=1) - g.mean()
    peaks = []
    for sig in (sig_x, sig_y):
        n = len(sig)
        if n < 16:
            peaks.append(0.0)
            continue
        denom = float(np.dot(sig, sig)) + 1e-12
        max_lag = max(4, int(n * max_lag_frac))
        best = 0.0
        for lag in range(4, max_lag):
            c = float(np.dot(sig[:-lag], sig[lag:]) / denom)
            if c > best:
                best = c
        peaks.append(max(0.0, min(1.0, best)))
    return float(sum(peaks) / len(peaks))


def structure_metrics(path: Path) -> dict:
    Image, _ = _require_pillow()
    np = _require_numpy()
    with Image.open(path) as im:
        if min(im.size) < MIN_SIZE:
            # still score, but flag undersized
            undersized = True
        else:
            undersized = False
        rgb = im.convert("RGB")
        gray = np.asarray(rgb.convert("L"))
        sym = bilateral_symmetry(gray)
        edge = edge_density(rgb)
        line = line_axis_ratio(gray)
        per = periodicity(gray)
    overall = float((sym + edge + line + per) / 4.0)
    return {
        "file": path.name,
        "undersized": undersized,
        "bilateral_symmetry": round(sym, 4),
        "edge_density": round(edge, 4),
        "line_axis_ratio": round(line, 4),
        "periodicity": round(per, 4),
        "overall_structure": round(overall, 4),
    }


def clip_score(image_paths: list[Path], prompts: list[str]) -> float | None:
    """Optional CLIP score if transformers+torch available; else None."""
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        return None
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:
        return None
    total = 0.0
    n = 0
    for path, prompt in zip(image_paths, prompts, strict=False):
        image = Image.open(path).convert("RGB")
        inputs = proc(text=[prompt], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
            sim = torch.nn.functional.cosine_similarity(
                out.image_embeds, out.text_embeds, dim=-1
            )
        total += float(sim.item())
        n += 1
    return round(total / n, 4) if n else None


def evaluate_images(images_dir: Path, out_dir: Path, limit: int | None = None) -> dict:
    paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no images in {images_dir}")
    if limit:
        paths = paths[:limit]
    rows = [structure_metrics(p) for p in paths]
    means = {}
    for key in (
        "bilateral_symmetry",
        "edge_density",
        "line_axis_ratio",
        "periodicity",
        "overall_structure",
    ):
        means[key] = round(sum(r[key] for r in rows) / len(rows), 4)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "images_dir": str(images_dir),
        "count": len(rows),
        "means": means,
        "per_image": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "structural_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_path"] = report_path
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structural integrity metrics for parametric architecture images.")
    parser.add_argument("--check", action="store_true", help="validate harness imports and exit")
    parser.add_argument("--images", default="dataset_islamic_parametric", help="directory of images")
    parser.add_argument("--out", default=EVAL_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clip", action="store_true", help="also compute optional CLIP score")
    args = parser.parse_args(argv)

    if args.check:
        try:
            _require_pillow()
            _require_numpy()
        except Exception as exc:
            print(f"[error] check failed: {exc}", file=sys.stderr)
            return 1
        print("check OK")
        print("metrics: bilateral_symmetry, edge_density, line_axis_ratio, periodicity, overall_structure")
        print(f"planned output: {Path(args.out) / 'structural_metrics.json'}")
        return 0

    try:
        report = evaluate_images(Path(args.images), Path(args.out), args.limit)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if args.clip:
        prompts = ["Islamic_Parametric architectural facade"] * report["count"]
        paths = [Path(args.images) / r["file"] for r in report["per_image"]]
        score = clip_score(paths, prompts)
        report["clip_score"] = score
        path = report["_path"]
        payload = {k: v for k, v in report.items() if not k.startswith("_")}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"clip_score: {score}")
    print(f"count: {report['count']}")
    for key, val in report["means"].items():
        print(f"{key}: {val}")
    print(f"report: {report['_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
