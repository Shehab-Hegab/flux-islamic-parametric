"""Replace 3 weak gold slots with strictly filtered Openverse picks, then re-gate."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import requests
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = 90_000_000

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finalize_dataset import (  # noqa: E402
    DATASET_DIR,
    contact_sheet,
    luma,
    phash,
    renumber,
    save_manifest,
    thumb_hash,
)
from gold_dataset import (  # noqa: E402
    BLACKLIST_TITLES,
    OPENVERSE,
    UA,
    acceptable_gold,
    classify_category,
    download_bytes,
    gold_report,
    mean_saturation,
    normalize_license,
    write_gold_captions,
)

# filename -> (must-have title tokens, queries, category)
SLOTS = {
    "image_25.jpg": {
        "must": ("mashrab", "muqarn", "jali", "lattice", "screen", "carved", "ornament", "geometric"),
        "ban": ("lacma", "museum", "3d", "render", "rani", "mahal", "india", "hindu", "temple", "painting", "drawing"),
        "cat": "detail",
        "queries": (
            "mashrabiya detail",
            "muqarnas detail architecture",
            "jali screen islamic",
            "carved wooden screen mosque",
            "geometric stone screen shadow",
        ),
    },
    "image_07.jpg": {
        "must": ("mosque", "islamic", "facade", "entrance", "portal", "minaret", "courtyard", "arch"),
        "ban": (
            "jerusalem",
            "old city",
            "street",
            "panorama",
            "aerial",
            "cityscape",
            "crowd",
            "people",
            "albania",
            "albanians",
            "the albanians",
            "prayers",
            "praying",
            "worshippers",
            "sleeping",
            "sleeps",
            "studying",
            "students",
            "mosaics",
            "mosaic",
        ),
        "cat": "facade",
        "queries": (
            "mosque minaret exterior",
            "grand mosque exterior facade",
            "islamic architecture exterior daylight",
            "mosque building facade geometric screens",
        ),
    },
    "image_11.jpg": {
        "must": ("mosque", "islamic", "facade", "entrance", "portal", "minaret", "courtyard", "madrasa"),
        "ban": (
            "independence",
            "mustaqillik",
            "square",
            "plaza",
            "monument",
            "cologne",
            "köln",
            "interior",
            "prayer hall",
            "minbar",
            "calligraphy",
            "prayers",
            "praying",
            "people",
            "crowd",
            "mosaics",
            "mosaic",
            "studs",
            "studying",
            "sleeping",
        ),
        "cat": "facade",
        "queries": (
            "mosque exterior facade daylight",
            "madrasa facade portal",
            "modern islamic architecture exterior",
            "mosque courtyard exterior arches",
            "mosque minaret facade exterior",
        ),
    },
}


def fetch(q: str) -> list[dict]:
    try:
        r = requests.get(
            OPENVERSE,
            params={"q": q, "license_type": "commercial,modification", "page_size": 15, "mature": "false"},
            timeout=12,
            headers=UA,
        )
        if r.status_code != 200:
            return []
        return r.json().get("results") or []
    except Exception:
        return []


def main() -> int:
    ds = Path(DATASET_DIR)
    entries = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    md5s = {thumb_hash(Image.open(ds / e["filename"]).convert("RGB")) for e in entries if (ds / e["filename"]).exists()}
    phs = {phash(Image.open(ds / e["filename"]).convert("RGB")) for e in entries if (ds / e["filename"]).exists()}
    used_pages = {str(e.get("page_url") or "").lower() for e in entries}
    used_urls = {str(e.get("source_url") or "").lower() for e in entries}

    for fname, spec in SLOTS.items():
        if not (ds / fname).exists():
            continue
        current = next((e for e in entries if e["filename"] == fname), {})
        if current.get("title") and not any(b in str(current["title"]).lower() for b in spec["ban"]):
            print("keep already-ok", fname, current.get("title"), flush=True)
            continue

        chosen = None
        for q in spec["queries"]:
            if chosen:
                break
            for it in fetch(q):
                title = str(it.get("title") or "")
                blob = (title + " " + q).lower()
                if any(k.lower() in blob for k in BLACKLIST_TITLES):
                    continue
                if any(b in blob for b in spec["ban"]):
                    continue
                if not any(m in blob for m in spec["must"]):
                    continue
                page = str(it.get("foreign_landing_url") or "").lower()
                url = str(it.get("url") or "")
                if page in used_pages or url.lower() in used_urls:
                    continue
                lic = normalize_license(it)
                if not lic:
                    continue
                head = None
                try:
                    head = requests.head(url, timeout=8, headers=UA, allow_redirects=True)
                except Exception:
                    head = None
                if head is not None:
                    cl = head.headers.get("Content-Length")
                    if cl and cl.isdigit() and int(cl) > 12_000_000:
                        continue
                    ctype = head.headers.get("Content-Type", "")
                    if ctype and not ctype.startswith("image/"):
                        continue
                im = download_bytes(url)
                if im is None:
                    continue
                if max(im.size) > 4096 or max(im.size) * min(im.size) > 16_000_000:
                    continue
                ok, _ = acceptable_gold(im)
                if not ok:
                    continue
                h, ph = thumb_hash(im), phash(im)
                if h in md5s or ph in phs:
                    continue
                im.save(ds / fname, quality=92)
                md5s.add(h)
                phs.add(ph)
                entry = {
                    "filename": fname,
                    "source": "openverse",
                    "query": q,
                    "width": im.size[0],
                    "height": im.size[1],
                    "author": str(it.get("creator") or "")[:120],
                    "license": lic,
                    "source_url": url,
                    "page_url": it.get("foreign_landing_url") or "",
                    "title": title[:200],
                    "luma": round(luma(im), 1),
                    "saturation": round(mean_saturation(im), 1),
                    "category": spec["cat"],
                }
                for i, e in enumerate(entries):
                    if e["filename"] == fname:
                        entries[i] = entry
                        break
                used_pages.add(page)
                used_urls.add(url.lower())
                chosen = entry
                print("replaced", fname, entry["luma"], entry["title"], entry["license"], flush=True)
                break
        if not chosen:
            print("NO replacement for", fname, flush=True)

    for e in entries:
        if not (ds / e["filename"]).exists():
            continue
        e["category"] = classify_category(e)
        # keep intended slot category when ban-check already passed
        if e["filename"] in SLOTS and classify_category(e) != SLOTS[e["filename"]]["cat"]:
            # force category only if title matches slot intent
            blob = str(e.get("title", "")).lower() + " " + str(e.get("query", "")).lower()
            if any(m in blob for m in SLOTS[e["filename"]]["must"]):
                e["category"] = SLOTS[e["filename"]]["cat"]

    live = [e for e in entries if (ds / e["filename"]).exists()]
    entries = renumber(ds, live)
    for e in entries:
        e["category"] = classify_category(e)
    save_manifest(entries)
    write_gold_captions(ds, entries)
    sheet = contact_sheet(ds, entries)
    report = gold_report(entries, ds)
    out = Path("eval_outputs")
    out.mkdir(exist_ok=True)
    (out / "dataset_quality_report.json").write_text(
        json.dumps(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "plan": "PLAN_DATASET_GOLD.md",
                **{k: v for k, v in report.items() if k != "per_image"},
                "per_image": report["per_image"],
                "contact_sheet": str(sheet.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in report if k != "per_image"}, indent=2, ensure_ascii=False))
    print("cats", Counter(e["category"] for e in entries))
    for e in entries:
        print(e["filename"], e["category"], e.get("luma"), str(e.get("title"))[:70])
    return 0 if str(report["verdict"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
