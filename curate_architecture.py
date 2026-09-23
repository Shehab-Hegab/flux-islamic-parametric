"""Curate bright photorealistic architecture images from Unsplash (license-aware).

Replaces the flat synthetic dataset when UNSPLASH_ACCESS_KEY is set.
Falls back to clear instructions if the key is missing — does NOT silently
regenerate dark synthetic patterns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    DATASET_DIR,
    IMAGE_PATTERN,
    MIN_SIZE,
    NUM_IMAGES,
)

API = "https://api.unsplash.com/search/photos"
PER_PAGE = 30
MIN_LUMA = 50  # reject near-black frames
TARGET = NUM_IMAGES  # 25

QUERIES = [
    "parametric facade architecture",
    "mashrabiya architecture",
    "islamic geometry building",
    "modern islamic architecture",
    "geometric facade building",
    "lattice facade sunlight",
    "mosque modern architecture exterior",
    "grc screen facade",
]


def _mean_luma(im: Image.Image) -> float:
    return sum(im.convert("L").resize((64, 64)).getdata()) / (64 * 64)


def search(key: str, query: str, page: int = 1) -> list[dict]:
    r = requests.get(
        API,
        params={
            "query": query,
            "page": page,
            "per_page": PER_PAGE,
            "order_by": "relevant",
            "content_filter": "high",
        },
        headers={"Authorization": f"Client-ID {key}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def download(url: str, dest: Path, min_side: int = MIN_SIZE) -> bool:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    tmp = dest.with_suffix(".tmp.jpg")
    tmp.write_bytes(resp.content)
    try:
        im = Image.open(tmp).convert("RGB")
    except Exception:
        tmp.unlink(missing_ok=True)
        return False
    w, h = im.size
    # center-crop to square then require min side
    side = min(w, h)
    if side < min_side:
        tmp.unlink(missing_ok=True)
        return False
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((max(side, min_side), max(side, min_side)), Image.Resampling.LANCZOS)
    # if original was smaller than MIN on one axis after crop, upscale only if >= 0.75*min
    if side < min_side * 0.75:
        tmp.unlink(missing_ok=True)
        return False
    if side < min_side:
        im = im.resize((min_side, min_side), Image.Resampling.LANCZOS)
    luma = _mean_luma(im)
    if luma < MIN_LUMA:
        tmp.unlink(missing_ok=True)
        return False
    im.save(dest, quality=92)
    tmp.unlink(missing_ok=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download bright architecture photos from Unsplash")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--execute", action="store_true", help="actually download into dataset dir")
    ap.add_argument("--backup-synthetic", action="store_true", help="move current images to dataset_synthetic_backup/")
    ap.add_argument("--limit", type=int, default=TARGET)
    args = ap.parse_args()

    key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
    if args.check:
        print("UNSPLASH_ACCESS_KEY set:", bool(key))
        print("queries:", len(QUERIES), "target:", args.limit, "min_side:", MIN_SIZE, "min_luma:", MIN_LUMA)
        print("mode: download real photos, reject dark/flat frames, license=Unsplash")
        return 0
    if not key:
        print("ERROR: set UNSPLASH_ACCESS_KEY (https://unsplash.com/developers)")
        print("Get a free Access Key, export it, then re-run with --execute")
        return 2
    if not args.execute:
        print("dry-run: pass --execute to download (and --backup-synthetic to archive current set)")
        return 0

    ds = Path(DATASET_DIR)
    ds.mkdir(exist_ok=True)
    caps = ds / "captions"
    # preserve old set
    if args.backup_synthetic:
        bak = Path("dataset_synthetic_backup")
        bak.mkdir(exist_ok=True)
        for p in list(ds.glob("image_*.jpg")) + list(caps.glob("*.txt")) + [ds / "manifest.json"]:
            if p.exists():
                p.replace(bak / p.name)

    collected: list[dict] = []
    seen_urls: set[str] = set()
    idx = 1
    for q in QUERIES:
        if idx > args.limit:
            break
        try:
            results = search(key, q)
        except Exception as exc:
            print(f"search failed {q!r}: {exc}")
            continue
        for item in results:
            if idx > args.limit:
                break
            url = item.get("urls", {}).get("raw") or item.get("urls", {}).get("regular")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            # request at least 1024 on short edge when possible
            if "ixlib" not in url and "w=" not in url:
                url = url + "&w=1600&q=90&fm=jpg"
            name = IMAGE_PATTERN.format(idx) + ".jpg"
            dest = ds / name
            try:
                ok = download(url, dest)
            except Exception as exc:
                print("download fail:", exc)
                continue
            if not ok:
                continue
            user = item.get("user", {}) or {}
            collected.append(
                {
                    "filename": name,
                    "source": "unsplash",
                    "query": q,
                    "width": MIN_SIZE,
                    "height": MIN_SIZE,
                    "author": user.get("name", ""),
                    "source_url": item.get("links", {}).get("html", ""),
                    "license": "Unsplash License",
                    "photo_id": item.get("id", ""),
                }
            )
            print(f"kept {name} luma-ok from {q}")
            idx += 1
            time.sleep(0.15)

    got = len(collected)
    print(f"downloaded {got}/{args.limit}")
    if got < args.limit:
        print("WARN: not enough Unsplash hits — top up manually or run expand_dataset.py for Wikimedia CC0")
    if got:
        manifest_path = ds / "manifest.json"
        # keep only new real images in manifest (synthetic backed up)
        manifest_path.write_text(json.dumps(collected, indent=2, ensure_ascii=False), encoding="utf-8")
        print("manifest written:", manifest_path)
        print("NEXT: python generate_captions.py --backend auto")
        print("THEN: review contact sheet before Colab train")
    return 0 if got >= min(10, args.limit) else 1


if __name__ == "__main__":
    sys.exit(main())
