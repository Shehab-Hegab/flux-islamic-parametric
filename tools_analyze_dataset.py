"""Analyze current dataset quality for expert review (no training)."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageStat

ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_islamic_parametric"
OUT = ROOT / "eval_outputs"


def analyze_images() -> list[dict]:
    rows: list[dict] = []
    hashes: dict[str, list[str]] = {}
    for p in sorted(DS.glob("image_*.jpg")):
        im = Image.open(p).convert("RGB")
        g = im.convert("L")
        mean = ImageStat.Stat(g).mean[0]
        std = ImageStat.Stat(g).stddev[0]
        hist = g.histogram()
        total = sum(hist) or 1
        near_black = sum(hist[:32]) / total
        near_white = sum(hist[224:]) / total
        thumb = im.resize((32, 32))
        h = hashlib.md5(thumb.tobytes()).hexdigest()
        hashes.setdefault(h, []).append(p.name)
        px = list(g.resize((64, 64)).getdata())
        edges = sum(abs(px[i] - px[i + 1]) for i in range(len(px) - 1)) / (len(px) - 1)
        # rough color saturation mean
        hsv = im.convert("HSV")
        sat = ImageStat.Stat(hsv).mean[1]
        rows.append(
            {
                "file": p.name,
                "size": im.size,
                "mean_luma": round(mean, 1),
                "contrast_std": round(std, 1),
                "near_black_frac": round(near_black, 3),
                "near_white_frac": round(near_white, 3),
                "saturation": round(sat, 1),
                "edge_energy": round(edges, 1),
            }
        )
    dups = {k: v for k, v in hashes.items() if len(v) > 1}
    return rows, dups


def analyze_captions() -> dict:
    caps = sorted((DS / "captions").glob("*.txt"))
    texts = [c.read_text(encoding="utf-8").strip() for c in caps]
    cores = [re.sub(r" — detail revision \d+\.?$", "", t).rstrip(".") for t in texts]
    counts: dict[str, int] = {}
    for c in cores:
        counts[c] = counts.get(c, 0) + 1
    unique_full = len(set(texts))
    unique_cores = sum(1 for v in counts.values() if v == 1)
    trigger_ok = all("in Islamic_Parametric style" in t for t in texts)
    return {
        "n": len(texts),
        "unique_full": unique_full,
        "unique_cores": unique_cores,
        "trigger_ok": trigger_ok,
        "duplicate_cores": {t: n for t, n in counts.items() if n > 1},
    }


def write_sheets(files: list[Path]) -> list[str]:
    OUT.mkdir(exist_ok=True)
    thumb, cols, pad = 280, 5, 28
    n = len(files)
    rows_n = (n + cols - 1) // cols
    paths = []
    for brighten, name in (
        (False, "dataset_review_contact_sheet_raw.jpg"),
        (True, "dataset_review_contact_sheet_brightened.jpg"),
    ):
        sheet = Image.new("RGB", (cols * thumb, rows_n * (thumb + pad)), (12, 12, 12))
        draw = ImageDraw.Draw(sheet)
        for i, p in enumerate(files):
            im = Image.open(p).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
            if brighten:
                im = ImageEnhance.Brightness(im).enhance(2.2)
            x = (i % cols) * thumb
            y = (i // cols) * (thumb + pad)
            sheet.paste(im, (x, y))
            draw.text((x + 6, y + thumb + 6), p.stem, fill=(220, 220, 80))
        out = OUT / name
        sheet.save(out, quality=90)
        paths.append(str(out))
    return paths


def main() -> int:
    files = sorted(DS.glob("image_*.jpg"))
    if not files:
        print("no images")
        return 1
    rows, dups = analyze_images()
    caps = analyze_captions()
    means = [r["mean_luma"] for r in rows]
    stds = [r["contrast_std"] for r in rows]
    nbs = [r["near_black_frac"] for r in rows]
    sats = [r["saturation"] for r in rows]
    report = {
        "n_images": len(rows),
        "sizes": sorted({f"{r['size'][0]}x{r['size'][1]}" for r in rows}),
        "mean_luma": {"avg": round(statistics.mean(means), 1), "min": min(means), "max": max(means)},
        "contrast_std": {"avg": round(statistics.mean(stds), 1), "min": min(stds), "max": max(stds)},
        "near_black": {"avg": round(statistics.mean(nbs), 3), "min": min(nbs), "max": max(nbs)},
        "saturation": {"avg": round(statistics.mean(sats), 1), "min": min(sats), "max": max(sats)},
        "all_dark": all(m < 60 for m in means),
        "all_low_contrast": all(s < 40 for s in stds),
        "duplicate_thumbs": dups or None,
        "captions": caps,
        "per_image": rows,
        "verdict": None,
    }
    # Expert scoring rubric
    score = 0
    reasons = []
    if report["all_dark"]:
        reasons.append("every image mean luma < 60 (dark / latent-collapse risk)")
        score += 2
    if report["all_low_contrast"]:
        reasons.append("every image contrast std < 40 (flat / washed)")
        score += 2
    if statistics.mean(nbs) > 0.4:
        reasons.append(f"avg near-black pixels {statistics.mean(nbs):.0%} (>40%)")
        score += 2
    if statistics.mean(sats) < 40:
        reasons.append(f"avg saturation {statistics.mean(sats):.0f} — low color")
        score += 1
    if any(r["edge_energy"] < 8 for r in rows):
        reasons.append("very low edge energy (no material texture detail)")
        score += 1
    if caps["unique_cores"] < caps["n"] * 0.6:
        reasons.append(f"only {caps['unique_cores']}/{caps['n']} unique caption cores")
        score += 1
    if dups:
        reasons.append("near-duplicate images detected")
        score += 1
    # pattern: all synthetic flat procedural
    if len(rows) < 25:
        reasons.append(f"dataset incomplete: only {len(rows)} images (need 25)")
        score += 2
    if all(
        r["mean_luma"] < 60 and r["contrast_std"] < 20 and r["edge_energy"] < 8 for r in rows
    ):
        reasons.append("images look flat 2D procedural (no 3D depth, materials, sun)")
        score += 3
    if any("pending" == str(r.get("license", "")) for r in rows):
        reasons.append("some images still pending license metadata")
    report["defect_score"] = score
    report["reasons"] = reasons
    report["verdict"] = (
        "REJECT for FLUX LoRA baseline — replace with bright photorealistic architecture"
        if score >= 5
        else "weak but usable with caveats"
    )
    paths = write_sheets(files)
    report["contact_sheets"] = paths
    OUT.mkdir(exist_ok=True)
    (OUT / "dataset_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in report if k != "per_image"}, indent=2, ensure_ascii=False))
    print("--- per image ---")
    for r in rows:
        print(
            f"  {r['file']} luma={r['mean_luma']:5} std={r['contrast_std']:5} "
            f"black={r['near_black_frac']:.0%} sat={r['saturation']:5} edge={r['edge_energy']}"
        )
    print("sheets:", paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
