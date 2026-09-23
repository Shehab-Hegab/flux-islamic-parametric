"""Build the Islamic Parametric dataset: Unsplash hybrid with synthetic top-up."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    ARCH_QUERIES,
    DATASET_DIR,
    IMAGE_PATTERN,
    MIN_SIZE,
    NUM_IMAGES,
    SYNTHETIC_PATTERNS,
)

PALETTES = [
    ((30, 42, 58), (214, 176, 122), (240, 236, 228)),
    ((18, 60, 66), (196, 156, 96), (236, 232, 220)),
    ((44, 36, 32), (188, 140, 90), (245, 240, 230)),
    ((22, 30, 48), (160, 178, 140), (238, 236, 228)),
    ((50, 28, 34), (208, 170, 110), (242, 238, 232)),
    ((26, 40, 36), (176, 150, 104), (234, 232, 224)),
]


def _star_polygon(cx: float, cy: float, r_outer: float, r_inner: float, points: int, rot: float) -> list[tuple[float, float]]:
    coords = []
    for i in range(points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        a = rot + i * math.pi / points
        coords.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return coords


def _draw_star8_lattice(draw: ImageDraw.ImageDraw, size: int, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> None:
    step = size // 6
    for gy in range(-1, 7):
        for gx in range(-1, 7):
            cx, cy = gx * step + step // 2, gy * step + step // 2
            draw.polygon(_star_polygon(cx, cy, step * 0.48, step * 0.2, 8, 0), outline=fg, fill=bg)
            draw.polygon(_star_polygon(cx, cy, step * 0.48, step * 0.2, 8, 0), outline=fg)


def _draw_hex_mashrabiya(draw: ImageDraw.ImageDraw, size: int, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> None:
    r = size // 9
    h = r * math.sqrt(3) / 2
    for row in range(-1, int(size / h) + 2):
        for col in range(-1, int(size / (r * 1.5)) + 2):
            cx = col * r * 1.5
            cy = row * h * 2 + (h if col % 2 else 0)
            pts = [(cx + r * math.cos(math.pi / 3 * i), cy + r * math.sin(math.pi / 3 * i)) for i in range(6)]
            draw.polygon(pts, outline=fg, fill=bg)
            inner = [(cx + r * 0.55 * math.cos(math.pi / 3 * i), cy + r * 0.55 * math.sin(math.pi / 3 * i)) for i in range(6)]
            draw.polygon(inner, outline=fg)


def _draw_woven_grille(draw: ImageDraw.ImageDraw, size: int, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> None:
    band = size // 12
    for i in range(0, size + band, band):
        draw.rectangle([0, i, size, i + band // 2], fill=fg)
        draw.rectangle([i, 0, i + band // 2, size], fill=bg if i // band % 2 else fg)
    for i in range(0, size, band * 2):
        for j in range(0, size, band * 2):
            draw.ellipse([i + band * 0.2, j + band * 0.2, i + band * 0.8, j + band * 0.8], fill=bg, outline=fg)


def _draw_arch_screen(draw: ImageDraw.ImageDraw, size: int, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> None:
    cols, rows = 5, 4
    cw, ch = size // cols, size // rows
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cw + cw * 0.12, r * ch + ch * 0.15
            x1, y1 = c * cw + cw * 0.88, r * ch + ch * 0.95
            mid = (x0 + x1) / 2
            draw.pieslice([x0, y0 - cw * 0.2, x1, y0 + cw * 0.75], 180, 360, fill=bg, outline=fg, width=3)
            draw.rectangle([x0, y0 + cw * 0.25, x1, y1], fill=bg, outline=fg, width=3)
            draw.line([(mid, y0), (mid, y1)], fill=fg, width=2)
            draw.arc([x0, y0, x1, y0 + cw * 0.7], 200, 340, fill=fg, width=2)


def _draw_punch_parametric(draw: ImageDraw.ImageDraw, size: int, fg: tuple[int, int, int], bg: tuple[int, int, int], rng: random.Random) -> None:
    draw.rectangle([0, 0, size, size], fill=fg)
    grid = 14
    cell = size / grid
    for gy in range(grid):
        for gx in range(grid):
            cx, cy = (gx + 0.5) * cell, (gy + 0.5) * cell
            wave = 0.5 + 0.5 * math.sin(gx * 0.7) * math.cos(gy * 0.55)
            rad = cell * (0.18 + 0.28 * wave)
            sides = 4 + (int(wave * 3) % 5)
            rot = rng.random() * math.pi
            draw.polygon(_star_polygon(cx, cy, rad, rad * 0.45, sides, rot), fill=bg)


def _draw_girih_strip(draw: ImageDraw.ImageDraw, size: int, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> None:
    draw.rectangle([0, 0, size, size], fill=bg)
    band = size // 8
    for k in range(-2, 12):
        offset = k * band
        draw.polygon(
            [(offset, 0), (offset + band * 0.5, 0), (offset + size * 0.5 + band * 0.5, size), (offset + size * 0.5, size)],
            outline=fg,
            fill=None,
        )
        draw.line(
            [(offset + band * 0.25, 0), (offset + size * 0.5 + band * 0.25, size)], fill=fg, width=3
        )
    step = band
    for gy in range(0, size + step, step):
        for gx in range(0, size + step, step):
            draw.polygon(_star_polygon(gx, gy, step * 0.4, step * 0.16, 5, 0), outline=fg, width=2)
    for _ in range(int(size / 40)):
        x0 = rng_pos(size)
        y0 = rng_pos(size)
        draw.line([(x0, y0), (x0 + size * 0.15, y0 + size * 0.15)], fill=fg, width=2)


def rng_pos(size: int) -> float:
    return random.Random(size).uniform(0, size * 0.8)


def generate_synthetic_image(index: int, size: int = MIN_SIZE) -> Image.Image:
    rng = random.Random(index)
    bg, fg, accent = PALETTES[index % len(PALETTES)]
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    pattern = SYNTHETIC_PATTERNS[index % len(SYNTHETIC_PATTERNS)]
    if pattern == "star8":
        _draw_star8_lattice(draw, size, fg, accent)
    elif pattern == "hex_mashrabiya":
        _draw_hex_mashrabiya(draw, size, fg, accent)
    elif pattern == "woven_grille":
        _draw_woven_grille(draw, size, fg, accent)
    elif pattern == "arch_screen":
        _draw_arch_screen(draw, size, fg, accent)
    elif pattern == "punch_parametric":
        _draw_punch_parametric(draw, size, fg, accent, rng)
    else:
        _draw_girih_strip(draw, size, fg, accent)
    vignette = Image.new("L", (size, size), 0)
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse([-size * 0.2, -size * 0.2, size * 1.2, size * 1.2], fill=40)
    vignette = vignette.filter(ImageFilter.GaussianBlur(size // 6))
    img = Image.composite(img, Image.new("RGB", (size, size), (12, 14, 18)), vignette)
    grain = Image.effect_noise((size, size), 8).convert("L")
    img = Image.composite(img, img.point(lambda p: min(255, p + 6)), grain.point(lambda p: 255 if p > 200 else 0))
    if img.size[0] < MIN_SIZE or img.size[1] < MIN_SIZE:
        img = img.resize((max(size, MIN_SIZE), max(size, MIN_SIZE)), Image.LANCZOS)
    return img


def fetch_unsplash(query: str, access_key: str, per_page: int = 5) -> list[bytes]:
    import requests

    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {access_key}"}
    params = {"query": query, "per_page": per_page, "orientation": "squarish"}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    blobs: list[bytes] = []
    for photo in resp.json().get("results", []):
        raw = photo.get("urls", {}).get("raw")
        if not raw:
            continue
        img_resp = requests.get(raw, params={"w": 2400, "h": 2400, "fit": "crop"}, timeout=30)
        if img_resp.status_code == 200 and len(img_resp.content) > 20_000:
            blobs.append(img_resp.content)
    return blobs


def _valid_blob(blob: bytes) -> bool:
    from io import BytesIO

    try:
        with Image.open(BytesIO(blob)) as im:
            im.verify()
        with Image.open(BytesIO(blob)) as im:
            return im.width >= MIN_SIZE and im.height >= MIN_SIZE
    except Exception:
        return False


def build_dataset(count: int, prefer: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    unsplash_blobs: list[bytes] = []
    key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()

    if prefer in ("hybrid", "unsplash") and key:
        for query in ARCH_QUERIES:
            try:
                unsplash_blobs.extend(b for b in fetch_unsplash(query, key, 4) if _valid_blob(b))
            except Exception as exc:
                print(f"[warn] Unsplash fetch failed for {query!r}: {exc}")
            if len(unsplash_blobs) >= count:
                break
    elif prefer in ("hybrid", "unsplash"):
        print("[warn] UNSPLASH_ACCESS_KEY not set — falling back to synthetic generation")

    if prefer == "unsplash" and not unsplash_blobs:
        print("[warn] no Unsplash images obtained — synthetic fallback")

    from io import BytesIO

    for i in range(1, count + 1):
        name = IMAGE_PATTERN.format(i) + ".jpg"
        path = out_dir / name
        source, query = "synthetic", SYNTHETIC_PATTERNS[(i - 1) % len(SYNTHETIC_PATTERNS)]
        if unsplash_blobs and prefer != "synthetic":
            blob = unsplash_blobs.pop(0)
            source, query = "unsplash", ARCH_QUERIES[(i - 1) % len(ARCH_QUERIES)]
            with Image.open(BytesIO(blob)) as im:
                im = im.convert("RGB")
                if im.width < MIN_SIZE or im.height < MIN_SIZE:
                    scale = MIN_SIZE / min(im.size)
                    im = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1), Image.LANCZOS)
                im.save(path, "JPEG", quality=92)
        else:
            img = generate_synthetic_image(i)
            img.save(path, "JPEG", quality=92)
        with Image.open(path) as check:
            w, h = check.size
        if w < MIN_SIZE or h < MIN_SIZE:
            raise ValueError(f"{path} failed minimum size check: {w}x{h}")
        manifest.append({"filename": name, "source": source, "query": pattern_query(query), "width": w, "height": h})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"count": len(manifest), "manifest": manifest}


def pattern_query(q: str) -> str:
    return q


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Islamic Parametric dataset (Unsplash hybrid + synthetic).")
    parser.add_argument("--count", type=int, default=NUM_IMAGES)
    parser.add_argument("--prefer", choices=["hybrid", "synthetic", "unsplash"], default="hybrid")
    parser.add_argument("--out", default=DATASET_DIR)
    args = parser.parse_args(argv)
    result = build_dataset(args.count, args.prefer, Path(args.out))
    print(f"Built {result['count']} images into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
