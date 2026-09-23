"""Gold-standard dataset rebuild: purge style bleed, Openverse top-up, category captions.

Reference contract: PLAN_DATASET_GOLD.md
Does not upload to HF or train.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import requests
from finalize_dataset import (  # noqa: E402
    OK_LICENSE,
    contact_sheet,
    contrast,
    load_manifest,
    luma,
    phash,
    quality_report,
    renumber,
    save_manifest,
    thumb_hash,
)
from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    DATASET_DIR,
    IMAGE_PATTERN,
    TRIGGER_WORD,
    build_caption,
)

UA = {"User-Agent": "flux-islamic-parametric-gold/1.0 (portfolio research)"}
OPENVERSE = "https://api.openverse.org/v1/images/"

TARGET_DEFAULT = 25
MIN_COUNT = 20
MIN_LUMA_GOLD = 90
MIN_SATURATION = 18.0
MAX_SATURATION_BW = 12.0

# Title / source fragments that must not ship (style bleed / noise / non-Islamic / flat)
BLACKLIST_TITLES = (
    "montrésor",
    "montresor",
    "indret-et-loire",
    "indre-et-loire",
    "granada- view into the lion",
    "granada-_view into the lion",
    "city of a thousand minarets",
    "gift shop sculpture",
    "girih_tiles",
    "girih tiles",
    "pointed_arches_gothic",
    "cathedral",
    "castle",
    "gothic",
    "blue_mosque_istanbul",
    "blue mosque istanbul",
    "mezquita cordoba",
    "mezquita_cordoba",
    "patio de los leones",
    "alhambra_patio_de_los_leones",
    "alhambra granada spain",
    "selimiye mosque edirne",
    "great mosque of kairouan",
    "koutoubia mosque marrakech",
    "great mosque of damascus",
    "ali_qapu",
    "ali qapu",
    "sheikh lotfollah",
    "petronas",
    "heydar",
    "flame towers",
    "cayan",
    "kingdom centre",
    "capital gate",
    "institut du monde",
    "louvre abu dhabi",
    "notre-dame",
    "eiffel",
    "louvre pyramid",
    "university",
    "college",
    "ignatius",
    "john carroll",
    "sao paulo",
    "são paulo",
    "brazil",
    "brasil",
    "tintype",
    "seated men",
    "writing table",
    "sitting room",
    "velvet case",
    "diyari",
    "diyarbak",
    "transept",
    "christmas",
    "wedding",
    "portrait of",
    "school",
    "high school",
    "elementary",
    "hotel lobby modern non",
    "office building generic",
    "apartment block generic",
    "parking",
    "warehouse",
    "factory",
    "stadium",
    "airport terminal",
    "train station",
    "subway",
    "bus stop",
    "bridge suspension",
    "skyline night",
    "night city",
    "department store",
    "department stores",
    "gay street",
    "honolulu",
    "kalakaua",
    "diamond head",
    "japan",
    "tokyo",
    "osaka",
    "kyoto",
    "lawrence scarpa",
    "lawrencescarpa",
    "cherokee",
    "ismaili centre",
    "ismaili center",
    "sitting room",
    "writing table",
    "pisa",
    "leaning tower",
    "duomo",
    "prague",
    "castilla",
    "toledo",
    "el salvador",
    "bosra",
    "temple of bel",
    "palmyra",
    "hampi",
    "elephant stables",
    "israel",
    "mahmadiyya",
    "qadisha",
    "deir mar",
    "christian",
    "church",
    "cathedral nave",
    "synagogue",
    "elephant",
    "village on the water",
    "kids playing",
    "national mosque of malaysia",
    "redfern mosque",
    "department store",
    "department stores",
    "gay street",
    "honolulu",
    "kalakaua",
    "diamond head",
    "japan",
    "tokyo",
    "osaka",
    "kyoto",
    "lawrence scarpa",
    "lawrencescarpa",
    "cherokee",
    "ismaili centre",
    "ismaili center",
    "theatre",
    "theater",
)

# Positive Openverse / style queries (gold vocabulary)
GOLD_QUERIES = [
    "mashrabiya",
    "mashrabiya facade",
    "mashrabiya window",
    "geometric screen facade",
    "geometric screen building",
    "islamic geometric screen",
    "islamic lattice facade",
    "perforated facade",
    "parametric facade",
    "modern mosque facade",
    "contemporary mosque",
    "mosque architecture modern",
    "islamic architecture modern",
    "geometric facade daylight",
    "ornamental screen facade",
    "lattice window sunlight",
    "geometric window light",
    "islamic window pattern",
    "muqarnas ceiling",
    "muqarnas interior",
    "islamic courtyard",
    "mosque interior light",
    "prayer hall interior",
    "arabesque facade",
    "geometric relief facade",
    "stone screen islamic",
    "wooden screen islamic",
    "jali screen",
    "brise soleil",
    "sunscreen facade geometric",
    "mosque arcade courtyard",
    "riad courtyard",
    "madrasa courtyard",
    "islamic museum interior",
    "geometric dome interior",
    "mosque entrance portal",
    "islamic arch facade",
    "patterned shadows interior",
]

# Category cores for beautiful, style-unified captions (no filenames)
CORES_FACADE = [
    "a crisp modern building facade with white GRC geometric screens and a grand pointed-arch portal in bright daylight",
    "a full facade of perforated stone lattice panels with rhythmic structural bays under clear sky",
    "a contemporary Islamic cultural center exterior with layered mashrabiya screens and bilateral symmetry",
    "a modern mosque entrance with intricate geometric screen cladding and deep portal shadows",
    "a heritage-inspired modern facade combining carved geometric relief and warm sandstone in hard daylight",
]
CORES_DETAIL = [
    "a close-up architectural detail of an intricate parametric mashrabiya lattice panel with sharp cast shadows",
    "a timber geometric screen close-up showing carved Islamic star patterns and daylight punch-through",
    "a close-up of perforated geometric panels with layered depth and precise modular rhythm",
    "an ornate mashrabiya lattice detail with warm wood texture and high-contrast daylight",
    "a geometric window screen detail casting crisp patterned light on surrounding stone",
]
CORES_INTERIOR = [
    "an interior view with geometric screens filtering warm daylight into patterned shadows across the floor",
    "a prayer-hall style interior with arched mashrabiya walls and soft directional sunlight",
    "a luxury interior courtyard arcade with ornate geometric screens and golden ambient light",
    "an interior muqarnas-inspired ceiling with honeycomb geometry under warm artificial and daylight mix",
    "a sunlit interior colonnade with lattice screens projecting precise geometric shade patterns",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def mean_saturation(im: Image.Image) -> float:
    small = im.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    hsv = small.convert("HSV")
    return ImageStat.Stat(hsv).mean[1]


def title_blob(entry: dict) -> str:
    parts = [
        str(entry.get("commons_title") or ""),
        str(entry.get("title") or ""),
        str(entry.get("source_url") or ""),
        str(entry.get("query") or ""),
        str(entry.get("filename") or ""),
    ]
    return " ".join(parts).lower()


def is_blacklisted(entry: dict) -> bool:
    blob = title_blob(entry)
    return any(k.lower() in blob for k in BLACKLIST_TITLES)


def is_bw(entry: dict, ds: Path) -> bool:
    p = ds / entry["filename"]
    if not p.exists():
        return True
    im = Image.open(p).convert("RGB")
    sat = mean_saturation(im)
    return sat < MAX_SATURATION_BW


def classify_category(entry: dict) -> str:
    blob = title_blob(entry)
    query = str(entry.get("query") or "").lower()
    # Explicit detail signals first (title or query)
    detail_signals = (
        "mashrab",
        "jali",
        "lattice window",
        "geometric window",
        "muqarnas",
        "ceiling",
        "carved",
        "ornament",
        "detail of",
        "tilework",
        "zellige",
        "girih",
        "pattern close",
        "screen close",
        "balcon",
        "rawasheen",
        "arabesque mashrabiya",
        "wooden screen",
        "stone screen",
        "fanous",
    )
    interior_signals = (
        "interior",
        "prayer hall",
        "hypostyle",
        "colonnade interior",
        "atrium",
        "minbar",
        "dome chamber",
        "mosque interior",
        "hall of columns",
        "forest of columns",
        "majlis",
        "mosque interior columns",
        "courtyard interior",
        "riad interior",
        "madrasa interior",
        "arched interior",
        "museum interior",
        "entrance chandelier",
        "mosaics",
    )
    facade_signals = (
        "facade",
        "exterior",
        "entrance",
        "portal",
        "mosque exterior",
        "building",
        "minaret",
        "courtyard exterior",
        "grand mosque",
        "madrasah",
        "registan",
        "center exterior",
        "full facade",
    )
    blob_l = blob.lower()
    if any(s in blob_l or s in query for s in detail_signals) and "full facade" not in blob_l:
        # if title clearly detail-like
        if any(s in blob_l for s in ("mashrab", "muqarnas", "ceiling", "carved", "detail", "tile", "screen", "window", "pattern", "ornament", "jali", "balcon")):
            return "detail"
        if query and any(s in query for s in ("mashrabiya", "muqarnas", "lattice", "screen", "pattern", "window")):
            return "detail"
    if any(s in blob_l or s in query for s in interior_signals):
        if any(s in blob_l for s in ("interior", "hall", "prayer", "atrium", "colonnade", "mosaics", "chandelier", "columns")) or "interior" in query:
            return "interior"
        if "courtyard interior" in blob_l or "riad interior" in blob_l or "madrasa interior" in blob_l:
            return "interior"
    if any(s in blob_l for s in facade_signals) or any(s in query for s in ("facade", "exterior", "entrance", "building", "minaret")):
        return "facade"
    if "mosque" in blob_l or "mosque" in query:
        return "facade"
    if any(s in blob_l for s in ("arch", "dome", "tile", "screen", "pattern", "window", "mashrab")):
        return "detail"
    if "interior" in query:
        return "interior"
    return "detail"


def gold_caption(entry: dict, index: int) -> str:
    cat = classify_category(entry)
    pool = {
        "facade": CORES_FACADE,
        "detail": CORES_DETAIL,
        "interior": CORES_INTERIOR,
    }[cat]
    core = pool[index % len(pool)]
    # keep unique study token without filename junk
    return build_caption(f"{core}; composition study {index + 1:02d}")


def write_gold_captions(ds: Path, entries: list[dict]) -> int:
    caps = ds / "captions"
    if caps.exists():
        for p in caps.glob("*.txt"):
            p.unlink()
    caps.mkdir(exist_ok=True)
    for i, e in enumerate(entries):
        cap = gold_caption(e, i)
        assert TRIGGER_WORD in cap
        assert "no visual distortion" in cap
        assert not re.search(r"featuring image \d+", cap, re.I)
        assert not re.search(r"featuring [a-z0-9]+[-_][a-z0-9]+", cap, re.I)
        (caps / f"{Path(e['filename']).stem}.txt").write_text(cap + "\n", encoding="utf-8")
        e["category"] = classify_category(e)
    return len(entries)


def openverse_search(query: str, page_size: int = 20) -> list[dict]:
    params = {
        "q": query,
        "license_type": "commercial,modification",
        "page_size": page_size,
        "mature": "false",
    }
    for attempt in range(2):
        try:
            resp = requests.get(OPENVERSE, params=params, timeout=18, headers=UA)
            if resp.status_code in (429, 503):
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("results") or []
        except Exception:
            time.sleep(1 * (attempt + 1))
    return []


def normalize_license(item: dict) -> str:
    lic = str(item.get("license") or "").lower()
    ver = str(item.get("license_version") or "").strip()
    mapping = {
        "cc0": "CC0",
        "pdm": "PDM-owner",
        "by": f"CC BY {ver}".strip(),
        "by-sa": f"CC BY-SA {ver}".strip(),
        "by-nc": "",
        "by-nd": "",
        "by-nc-sa": "",
        "by-nc-nd": "",
        "sampling": "",
    }
    if lic in ("pdm", "publicdomain"):
        return "PDM-owner"
    if lic not in mapping:
        return ""
    out = mapping[lic]
    if not out:
        return ""
    if not OK_LICENSE.search(out):
        return ""
    # reject NC/ND already handled by empty
    if "nc" in lic or "nd" in lic:
        return ""
    return out


def prepare_gold(im: Image.Image) -> Image.Image | None:
    """Center-crop to square; allow upscaling from mid-size landscape sources."""
    w, h = im.size
    side = min(w, h)
    if side < 520 or max(w, h) < 900:
        return None
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    if side < 1024:
        im = im.resize((1024, 1024), Image.Resampling.LANCZOS)
    elif side > 1600:
        im = im.resize((1600, 1600), Image.Resampling.LANCZOS)
    return im


def upgrade_flickr_url(url: str) -> list[str]:
    """Prefer larger Flickr renditions when available."""
    out = [url]
    if "live.staticflickr.com" in url:
        base = re.sub(r"_[a-z]\.jpg$", "", url)
        for suf in ("_k.jpg", "_h.jpg", "_b.jpg", "_c.jpg"):
            cand = base + suf
            if cand not in out:
                out.insert(0, cand)
    return out


def download_bytes(url: str) -> Image.Image | None:
    if not url:
        return None
    for candidate in upgrade_flickr_url(url):
        im = _get_image(candidate)
        if im is not None:
            return im
    return None


def _get_image(url: str) -> Image.Image | None:
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=15, headers=UA)
            if resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                return None
            resp.raise_for_status()
            break
        except Exception:
            if attempt == 1:
                return None
            time.sleep(0.5)
    else:
        return None
    try:
        im = Image.open(BytesIO(resp.content))
        im.load()
        im = im.convert("RGB")
    except Exception:
        return None
    return prepare_gold(im)


def acceptable_gold(im: Image.Image) -> tuple[bool, str]:
    if luma(im) < MIN_LUMA_GOLD:
        return False, f"luma {luma(im):.1f}<{MIN_LUMA_GOLD}"
    if contrast(im) < 35:
        return False, "low contrast"
    if mean_saturation(im) < MAX_SATURATION_BW:
        return False, "bw/sepia"
    if mean_saturation(im) < MIN_SATURATION and luma(im) > 140:
        # very pale is ok for white GRC — only reject if also flat
        if contrast(im) < 40:
            return False, "flat pale"
    return True, "ok"


def purge(ds: Path, entries: list[dict]) -> tuple[list[dict], list[str]]:
    removed: list[str] = []
    kept: list[dict] = []
    for e in entries:
        p = ds / e["filename"]
        reasons = []
        if is_blacklisted(e):
            reasons.append("blacklist")
        if p.exists() and is_bw(e, ds):
            reasons.append("bw")
        if p.exists() and luma(Image.open(p).convert("RGB")) < 75:
            reasons.append("dark")
        if reasons:
            if p.exists():
                p.unlink(missing_ok=True)
            cap = ds / "captions" / (Path(e["filename"]).stem + ".txt")
            if cap.exists():
                cap.unlink(missing_ok=True)
            removed.append(f"{e['filename']}: {','.join(reasons)} ({title_blob(e)[:80]})")
            continue
        kept.append(e)
    return kept, removed


def title_positive(entry: dict) -> bool:
    blob = title_blob(entry)
    keys = (
        "mashrab",
        "muqarnas",
        "mosque",
        "madrasa",
        "riad",
        "courtyard",
        "islamic",
        "arabesque",
        "geometric",
        "lattice",
        "screen",
        "arch",
        "dome",
        "minaret",
        "jali",
        "persian",
        "ottoman arch",
        "arab",
        "moslem",
        "muslim",
        "girih",
        "tilework",
        "zellige",
        "stucco",
        "portal",
        "iwan",
        "colonnade",
        "museum of islamic",
        "calligraphy",
        "fanous",
        "lantern islamic",
    )
    return any(k in blob for k in keys)


def topup_openverse(
    ds: Path,
    entries: list[dict],
    md5s: set[str],
    phs: set[str],
    target: int,
) -> list[dict]:
    used_keys = {title_blob(e) for e in entries}
    used_urls = {str(e.get("source_url") or "").lower() for e in entries}
    used_pages = {str(e.get("page_url") or "").lower() for e in entries}

    for query in GOLD_QUERIES:
        if len(entries) >= target:
            break
        results = openverse_search(query, page_size=20)
        print(f"  openverse '{query}': {len(results)} hits", flush=True)
        for item in results:
            if len(entries) >= target:
                break
            title = str(item.get("title") or "")
            # pre-filter title before any download (speed + safety)
            blob_pre = f"{title} {query}".lower()
            if any(k.lower() in blob_pre for k in BLACKLIST_TITLES):
                continue
            if not title_positive({"title": title, "query": query, "source_url": item.get("foreign_landing_url", "")}):
                # allow facade-ish queries without strict title match
                soft = ("facade", "screen", "lattice", "geometric", "mosque", "mashrab", "arch", "courtyard")
                if not any(s in query.lower() for s in soft) and not any(s in title.lower() for s in soft):
                    continue
            url = item.get("url") or item.get("thumbnail") or ""
            page = str(item.get("foreign_landing_url") or item.get("creator_url") or "").lower()
            if page and page in used_pages:
                continue
            if url.lower() in used_urls:
                continue
            lic = normalize_license(item)
            if not lic:
                continue
            im = download_bytes(url)
            if im is None:
                continue
            ok, why = acceptable_gold(im)
            if not ok:
                print(f"    reject {why}: {title[:50]}", flush=True)
                continue
            h, ph = thumb_hash(im), phash(im)
            if h in md5s or ph in phs:
                continue
            md5s.add(h)
            phs.add(ph)
            n = len(entries) + 1
            while (ds / f"{IMAGE_PATTERN.format(n)}.jpg").exists():
                n += 1
            name = IMAGE_PATTERN.format(n) + ".jpg"
            im.save(ds / name, quality=92)
            entry = {
                "filename": name,
                "source": "openverse",
                "query": query,
                "width": im.size[0],
                "height": im.size[1],
                "author": str(item.get("creator") or "")[:120],
                "license": lic,
                "source_url": url,
                "page_url": item.get("foreign_landing_url") or "",
                "title": str(item.get("title") or "")[:200],
                "luma": round(luma(im), 1),
                "saturation": round(mean_saturation(im), 1),
            }
            entries.append(entry)
            used_keys.add(title_blob(entry))
            used_pages.add(page)
            used_urls.add(url.lower())
            print(f"  kept {name} luma={entry['luma']} sat={entry['saturation']} [{lic}] {title[:55]}", flush=True)
            time.sleep(0.25)
        time.sleep(0.5)
    return entries


def gold_report(entries: list[dict], ds: Path) -> dict:
    report = quality_report(entries, ds)
    sats = []
    cats = Counter()
    junk = []
    for e in entries:
        p = ds / e["filename"]
        if p.exists():
            sats.append(round(mean_saturation(Image.open(p).convert("RGB")), 1))
        e["category"] = classify_category(e)
        cats[e["category"]] += 1
        cap_path = ds / "captions" / (Path(e["filename"]).stem + ".txt")
        if cap_path.exists():
            text = cap_path.read_text(encoding="utf-8")
            if re.search(r"featuring image \d+", text, re.I):
                junk.append(e["filename"])
            if TRIGGER_WORD not in text:
                junk.append(e["filename"] + ":missing-trigger")
    report["mean_saturation"] = round(sum(sats) / len(sats), 1) if sats else 0
    report["categories"] = dict(cats)
    report["junk_captions"] = junk
    report["min_count"] = MIN_COUNT
    report["target_gold"] = TARGET_DEFAULT
    ok = (
        report["n"] >= MIN_COUNT
        and report["duplicate_md5_groups"] == 0
        and report["duplicate_phash_groups"] == 0
        and report["mean_luma"] >= 85
        and report["captions"] >= report["n"]
        and report["all_free_license"]
        and not junk
        and report.get("mean_saturation", 0) >= MAX_SATURATION_BW
        and cats.get("facade", 0) >= 4
        and cats.get("detail", 0) >= 4
        and cats.get("interior", 0) >= 3
    )
    report["verdict"] = (
        "PASS — gold style-unified bright free-license set"
        if ok
        else "NEEDS_WORK"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gold dataset rebuild (purge + Openverse top-up + captions)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--target", type=int, default=TARGET_DEFAULT)
    args = ap.parse_args(argv)

    ds = Path(DATASET_DIR)
    if args.check:
        print("gold gates: min", MIN_COUNT, "target", args.target)
        print("min_luma_gold", MIN_LUMA_GOLD, "bw_sat", MAX_SATURATION_BW)
        print("blacklist entries", len(BLACKLIST_TITLES), "queries", len(GOLD_QUERIES))
        print("trigger", repr(TRIGGER_WORD))
        return 0
    if not args.execute:
        print("dry-run: pass --execute")
        return 0

    ds.mkdir(exist_ok=True)
    (ds / "captions").mkdir(exist_ok=True)

    entries = load_manifest()
    # sync disk images not in manifest
    known = {e["filename"] for e in entries}
    for p in sorted(ds.glob("image_*.jpg")):
        if p.name not in known:
            im = Image.open(p).convert("RGB")
            entries.append(
                {
                    "filename": p.name,
                    "source": "unknown",
                    "query": "unknown",
                    "width": im.size[0],
                    "height": im.size[1],
                    "author": "",
                    "license": "pending",
                    "source_url": "",
                    "luma": round(luma(im), 1),
                }
            )

    before = len(entries)
    entries, removed = purge(ds, entries)
    print(f"purged {len(removed)} / {before}:", flush=True)
    for line in removed:
        print("  -", line, flush=True)

    entries = renumber(ds, entries)
    md5s: set[str] = set()
    phs: set[str] = set()
    for e in entries:
        p = ds / e["filename"]
        if p.exists():
            im = Image.open(p).convert("RGB")
            md5s.add(thumb_hash(im))
            phs.add(phash(im))

    print(f"after purge: {len(entries)}", flush=True)
    if len(entries) < args.target:
        print(f"top-up {len(entries)} -> {args.target}...", flush=True)
        entries = topup_openverse(ds, entries, md5s, phs, args.target)
        entries = renumber(ds, entries)

    # drop non-free
    cleared = []
    for e in entries:
        if OK_LICENSE.search(str(e.get("license", ""))):
            cleared.append(e)
        else:
            print("drop non-free", e["filename"], e.get("license"), flush=True)
            (ds / e["filename"]).unlink(missing_ok=True)
    entries = renumber(ds, cleared)

    save_manifest(entries)
    n_cap = write_gold_captions(ds, entries)
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
    print("captions:", n_cap)
    print("sheet:", sheet)
    return 0 if str(report["verdict"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
