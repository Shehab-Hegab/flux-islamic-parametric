"""License-aware dataset expansion: download Wikimedia/CC0 architectural photos into the training set.

Adds images with provenance fields (author, license, source_url) into manifest.json.
Does not overwrite existing image_*.jpg names. Does not upload anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    MANIFEST_PATH,
    MIN_SIZE,
    TRIGGER_WORD,
)

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
LICENSE_WHITELIST = {"cc0", "cc-by-4.0", "cc by 4.0", "public domain", "pd"}


def next_image_stem(dataset_dir: Path) -> str:
    nums = []
    for p in dataset_dir.glob("image_*.jpg"):
        stem = p.stem  # image_07
        try:
            nums.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    nxt = (max(nums) if nums else 0) + 1
    return f"image_{nxt:02d}"


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"images": []}


def save_manifest(path: Path, data: dict) -> None:
    data["updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def search_commons(query: str, limit: int = 10) -> list[dict]:
    """Query Wikimedia Commons for free-license images. Returns candidates with license metadata."""
    import requests

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata|user",
        "iiurlwidth": str(MIN_SIZE),
    }
    resp = requests.get(WIKIMEDIA_API, params=params, timeout=30)
    resp.raise_for_status()
    pages = (resp.json().get("query") or {}).get("pages") or {}
    out: list[dict] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        artist = (meta.get("Artist") or {}).get("value", "")
        out.append(
            {
                "title": page.get("title", ""),
                "url": info.get("thumburl") or info.get("url", ""),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "license": lic.strip(),
                "author": artist.strip(),
                "source_url": info.get("descriptionurl", ""),
                "page_id": page.get("pageid"),
            }
        )
    return out


def download_candidate(dataset_dir: Path, cand: dict, stem: str) -> Path | None:
    from io import BytesIO

    import requests
    from PIL import Image

    if not cand.get("url"):
        return None
    resp = requests.get(cand["url"], timeout=60, headers={"User-Agent": "flux-islamic-parametric-dataset/1.0"})
    resp.raise_for_status()
    with Image.open(BytesIO(resp.content)) as im:
        if min(im.size) < MIN_SIZE:
            # upscaling ruins training quality — skip undersized
            return None
        rgb = im.convert("RGB")
        dest = dataset_dir / f"{stem}.jpg"
        rgb.save(dest, "JPEG", quality=92)
    return dest


def caption_path_for(dataset_dir: Path, stem: str, core: str) -> Path:
    cap_dir = dataset_dir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    from islamic_parametric.constants import build_caption

    text = build_caption(core)
    path = cap_dir / f"{stem}.txt"
    path.write_text(text, encoding="utf-8")
    assert TRIGGER_WORD in text
    return path


def expand(query: str, limit: int, dataset_dir: Path, manifest_path: Path, allow_unlicensed: bool = False) -> int:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    manifest.setdefault("images", [])
    existing_ids = {e.get("source_url") for e in manifest["images"] if isinstance(e, dict)}
    candidates = search_commons(query, limit=max(limit * 3, limit))
    added = 0
    for cand in candidates:
        if added >= limit:
            break
        lic = (cand.get("license") or "").lower()
        if not allow_unlicensed and not any(w in lic for w in LICENSE_WHITELIST):
            continue
        if cand["source_url"] in existing_ids:
            continue
        stem = next_image_stem(dataset_dir)
        try:
            dest = download_candidate(dataset_dir, cand, stem)
        except Exception as exc:
            print(f"[warn] download failed {cand.get('title')}: {exc}", file=sys.stderr)
            continue
        if dest is None:
            continue
        core = cand.get("title", "").removeprefix("File:").rsplit(".", 1)[0]
        core = (core or "islamic geometric facade")[:200]
        cap = caption_path_for(dataset_dir, stem, core)
        entry = {
            "filename": dest.name,
            "path": str(dest),
            "caption_path": str(cap),
            "width": cand.get("width"),
            "height": cand.get("height"),
            "source": "wikimedia_commons",
            "license": cand.get("license", "unknown"),
            "author": cand.get("author", ""),
            "source_url": cand.get("source_url", ""),
            "query": query,
        }
        manifest["images"].append(entry)
        existing_ids.add(entry["source_url"])
        added += 1
        print(f"+ {dest.name}  license={entry['license']}")
    save_manifest(manifest_path, manifest)
    print(f"added {added} images; manifest -> {manifest_path}")
    return added


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Expand training set with license-tracked Wikimedia photos.")
    p.add_argument("--query", default="mashrabiya facade islamic geometric architecture")
    p.add_argument("--limit", type=int, default=10, help="max new images to add")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--allow-unlicensed", action="store_true", help="do not filter by license whitelist")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)

    from islamic_parametric.constants import DATASET_DIR

    dataset_dir = Path(args.dataset_dir or DATASET_DIR)
    manifest_path = Path(args.manifest or MANIFEST_PATH)

    if args.check:
        print("check OK")
        print(f"query: {args.query} limit: {args.limit}")
        print(f"dataset: {dataset_dir} manifest: {manifest_path}")
        print(f"whitelist: {sorted(LICENSE_WHITELIST)}")
        return 0

    try:
        added = expand(args.query, args.limit, dataset_dir, manifest_path, args.allow_unlicensed)
    except Exception as exc:
        print(f"[error] expansion failed: {exc}", file=sys.stderr)
        return 1
    return 0 if added >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
