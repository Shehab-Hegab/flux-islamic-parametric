"""Curate landmark Islamic/parametric architecture photos from Wikimedia Commons.

Downloads free-license (CC0 / CC BY / CC BY-SA / PD) images for named projects,
filters brightness + resolution, writes license-aware manifest, optional backup
of current synthetic set. Does not upload to Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    DATASET_DIR,
    IMAGE_PATTERN,
    MANIFEST_PATH,
    NUM_IMAGES,
)

API = "https://commons.wikimedia.org/w/api.php"
OK_LICENSE = re.compile(
    r"(cc0|cc[- ]by(?:[- ]sa)?(?:[- ][0-9.]+)?|public domain|pd|no restrictions)",
    re.I,
)
MIN_LUMA = 55
MIN_SIDE = 1024

# Expert-curated landmark queries (free-license sources on Commons)
LANDMARK_QUERIES: list[tuple[str, str]] = [
    ("al_bahar_towers", "Al Bahar Towers Abu Dhabi facade"),
    ("louvre_abu_dhabi", "Louvre Abu Dhabi dome facade"),
    ("institut_monde_arabe", "Institut du Monde Arabe facade"),
    ("doha_tower", "Doha Tower Jean Nouvel"),
    ("masdar", "Masdar Institute facade"),
    ("mashrabiya", "mashrabiya facade architecture"),
    ("parametric_facade", "parametric facade architecture building"),
    ("islamic_geometric_building", "modern islamic geometric architecture facade"),
    ("museum islamic art", "Museum of Islamic Art Doha facade"),
    ("granada", "Alhambra mashrabiya window daylight"),
    ("mosque_screen", "mosque geometric screen facade sunlight"),
    ("grc_screen", "perforated concrete facade sunlight building"),
    ("modern_mosque", "modern mosque exterior daylight geometric"),
    ("courtyard_arches", "islamic courtyard arches daylight"),
    ("shading_screen", "brise soleil geometric facade building"),
]

FAVORITE_TITLES = [
    "File:Al Bahar Towers",
    "File:Louvre Abu Dhabi",
    "File:Institut du Monde Arabe",
    "File:Doha Tower",
    "File:Museum of Islamic Art",
    "File:Masdar",
]

# Known free-license Commons filenames (skip search API — use Special:FilePath CDN)
KNOWN_COMMONS_FILES: list[tuple[str, str]] = [
    ("al_bahar_towers", "Al_Bahar_Towers,_Abu_Dhabi.jpg"),
    ("al_bahar_towers", "Al Bahar Towers.jpg"),
    ("al_bahar_towers", "Al-Bahar-Towers-Abu-Dhabi.jpg"),
    ("louvre_abu_dhabi", "Louvre_Abu_Dhabi_(36356445474).jpg"),
    ("louvre_abu_dhabi", "Louvre_Abu_Dhabi_-_panoramio.jpg"),
    ("louvre_abu_dhabi", "Louvre_Abu_Dhabi_2.jpg"),
    ("institut_monde_arabe", "Institut_du_Monde_Arabe,_Paris_21_November_2013.jpg"),
    ("institut_monde_arabe", "Institut_du_monde_arabe,_Paris_2_August_2013.jpg"),
    ("institut_monde_arabe", "IMA_facade.jpg"),
    ("doha_tower", "Doha_Tower_-_Jay_Tower.jpg"),
    ("doha_tower", "Doha_Tower,_Qatar.jpg"),
    ("doha_tower", "Burj_Doha_Tower.jpg"),
    ("museum_islamic_art", "Museum_of_Islamic_Art,_Doha,_Qatar.jpg"),
    ("museum_islamic_art", "Doha_-_Museum_of_Islamic_Art.jpg"),
    ("masdar", "Masdar_Institute,_Abu_Dhabi.jpg"),
    ("masdar", "Masdar_City_Campus.jpg"),
    ("mashrabiya", "Mashrabiya_in_Cairo.jpg"),
    ("mashrabiya", "Mashrabiya,_Cairo.jpg"),
    ("mashrabiya", "Mashrabiya_window,_Cairo.jpg"),
    ("mashrabiya", "Wooden_mashrabiya.jpg"),
    ("mashrabiya", "Mashrabiya_in_Anatolia.jpg"),
    ("alhambra", "Alhambra_Granada_Spain.jpg"),
    ("alhambra", "Alhambra_Court_of_the_Lions.jpg"),
    ("alhambra", "Generalife_Alhambra_Granada_Spain.jpg"),
    ("mosque", "Sultan_Ahmed_Mosque_Istanbul_2007_008.jpg"),
    ("mosque", "Blue_Mosque_Istanbul.jpg"),
    ("mosque", "Sheikh_Zayed_Mosque_Abu_Dhabi.jpg"),
    ("mosque", "Sheikh_Zayed_Grand_Mosque,_Abu_Dhabi_-_panoramio.jpg"),
    ("mosque", "Hassan_II_Mosque_Casablanca.jpg"),
    ("geometric_facade", "Institut_du_Monde_Arabe_facade_detail.jpg"),
    ("geometric_facade", "Mashrabiya_facade_Dubai.jpg"),
    ("courtyard", "Alhambra_Patio_de_los_Leones.jpg"),
    ("courtyard", "Court_of_the_Cypress_in_the_Alcázar_of_Seville.jpg"),
    ("modern_mosque", "Kingdom_Centre_Tower_Riyadh.jpg"),
    ("modern_mosque", "Abraj_Al_Bait.jpg"),
    ("parametric", "The_Gherkin_London_November_2006.jpg"),
    ("jeddah", "Al-Balad_Jeddah_historic_district.jpg"),
    ("jeddah", "Nasseef_House_Jeddah.jpg"),
    ("islamic_art_museum", "Museum_of_Islamic_Art_Doha_Qatar.jpg"),
    ("dome", "Dome_of_the_Rock_Old_City_Jerusalem.jpg"),
    ("dome", "Sultan_Hassan_Mosque_Cairo.jpg"),
    ("dome", "Ibn_Tulun_Mosque_Cairo.jpg"),
    ("mosque", "Great_Mosque_of_Kairouan.jpg"),
    ("mosque", "Koutoubia_Mosque_Marrakech.jpg"),
    ("mosque", "Hassan_II_Mosque_exterior.jpg"),
    ("mosque", "Zitouna_Mosque_Tunis.jpg"),
    ("courtyard", "Alcazar_Seville_Court_of_the_Maidens.jpg"),
    ("courtyard", "Alhambra_Court_of_the_Myrtles.jpg"),
    ("courtyard", "Alhambra_Hall_of_the_Ambassadors.jpg"),
    ("mashrabiya", "Mashrabiya,_Old_Cairo.jpg"),
    ("mashrabiya", "Carved_wooden_mashrabiya,_Egypt.jpg"),
    ("mashrabiya", "Mashrabiya_screen,_Cairo_Egypt.jpg"),
    ("mashrabiya", "Wooden_screen_Mamluk_Cairo.jpg"),
    ("facade", "Mosque_of_Amra.jpg"),
    ("facade", "Dome_of_the_Qaitbay_Cemetery.jpg"),
    ("facade", "Mausoleum_of_Qaitbay_Cairo.jpg"),
    ("facade", "Madrasa_of_Sultan_Hassan_facade.jpg"),
    ("geometric", "Islamic_geometric_patterns_in_the_Alhambra.jpg"),
    ("geometric", "Tilework_Ispahan.jpg"),
    ("geometric", "Islamic_geometric_tiling.jpg"),
    ("geometric", "Girih_tiles.jpg"),
    ("modern", "Heydar_Aliyev_Center_Baku.jpg"),
    ("modern", "National_Congress_Center_Baku.jpg"),
    ("modern", "Baku_Flame_Towers.jpg"),
    ("modern", "Cayan_Tower_Dubai.jpg"),
    ("modern", "Burj_Al_Arab_Jumeirah.jpg"),
    ("modern", "Emirates_Towers_Dubai.jpg"),
    ("modern", "Capital_Gate_Abu_Dhabi.jpg"),
    ("modern", "The_Pearl_Qatar_Doha.jpg"),
    ("museum", "Islamic_Art_Museum_Cairo.jpg"),
    ("museum", "Museum_of_Islamic_Art_Doha_interior.jpg"),
    ("museum", "Petronas_Towers_Kuala_Lumpur.jpg"),
    ("screen", "Perforated_metal_facade.jpg"),
    ("screen", "Brise_soleil_facade.jpg"),
    ("screen", "Sunshade_facade_architecture.jpg"),
    ("arch", "Roman_aqueduct_arches.jpg"),
    ("arch", "Pointed_arches_Gothic_cathedral.jpg"),
    ("arch", "Islamic_arch_doorway_Cairo.jpg"),
    ("minaret", "Minaret_of_the_Great_Mosque_of_Kairouan.jpg"),
    ("minaret", "Qalawun_complex_Cairo_minaret.jpg"),
    ("jeddah", "Al-Balad_Jeddah.jpg"),
    ("jeddah", "Rawashin_Jeddah_old_town.jpg"),
    ("jeddah", "Nasseef_House.jpg"),
    ("iran", "Naqsh-e_Jahan_Square_Isfahan.jpg"),
    ("iran", "Sheikh_Lotfollah_Mosque_Isfahan.jpg"),
    ("iran", "Ali_Qapu_Isfahan.jpg"),
    ("turkey", "Suleymaniye_Mosque_Istanbul.jpg"),
    ("turkey", "Selimiye_Mosque_Edirne.jpg"),
    ("morocco", "Bahia_Palace_Marrakech.jpg"),
    ("morocco", "Ben_Yusuf_Madrasa_Marrakech.jpg"),
    ("spain", "Mezquita_Cordoba.jpg"),
    ("spain", "Alcazar_Seville_arches.jpg"),
]


def file_url(filename: str, width: int = 1600) -> str:
    from urllib.parse import quote

    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}?width={width}"


def download_via_filepath(filename: str) -> Image.Image | None:
    """Download via Special:FilePath CDN without search API (avoids 429)."""
    from io import BytesIO

    url = file_url(filename, width=1600)
    headers = {"User-Agent": "flux-islamic-parametric-curate/1.0 (portfolio research)"}
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60, headers=headers, allow_redirects=True)
            if resp.status_code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        except requests.HTTPError:
            if attempt == 2:
                return None
            time.sleep(4)
    if resp is None or resp.status_code not in (200, 301, 302):
        return None
    try:
        im = Image.open(BytesIO(resp.content))
        im.load()
        im = im.convert("RGB")
    except Exception:
        return None
    square = prepare_square(im)
    if square is None:
        return None
    if luma_of(square) < MIN_LUMA:
        return None
    if contrast_of(square) < 25:
        return None
    return square


def fetch_known_batch(filenames: list[str]) -> dict[str, dict]:
    """Resolve up to 10 known Commons files in one titles= request."""
    if not filenames:
        return {}
    titles = "|".join(f"File:{fn}" for fn in filenames)
    params = {
        "action": "query",
        "format": "json",
        "titles": titles,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": "1600",
    }
    headers = {"User-Agent": "flux-islamic-parametric-curate/1.0 (portfolio research)"}
    resp = None
    for attempt in range(5):
        try:
            resp = requests.get(API, params=params, timeout=40, headers=headers)
            if resp.status_code == 429:
                wait = 12 * (attempt + 1)
                print(f"  429 batch backoff {wait}s ({len(filenames)} titles)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.HTTPError:
            if attempt == 4:
                return {}
            time.sleep(6 * (attempt + 1))
    if resp is None:
        return {}
    time.sleep(2.5)
    pages = (resp.json().get("query") or {}).get("pages") or {}
    out: dict[str, dict] = {}
    # normalize titles -> filename key
    for page in pages.values():
        title = page.get("title", "")
        if page.get("missing") is not None:
            continue
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        lic = strip_html((meta.get("LicenseShortName") or {}).get("value", ""))
        if not OK_LICENSE.search(lic or ""):
            continue
        url = info.get("thumburl") or info.get("url") or ""
        if not url:
            continue
        w, h = int(info.get("width") or 0), int(info.get("height") or 0)
        if max(w, h) < MIN_SIDE:
            continue
        fn = title.split("File:", 1)[-1]
        out[fn] = {
            "title": title,
            "url": url,
            "width": w,
            "height": h,
            "license": lic,
            "author": strip_html((meta.get("Artist") or {}).get("value", ""))[:120],
            "description_url": info.get("descriptionurl", ""),
            "thumb_w": int(info.get("thumbwidth") or 0),
            "thumb_h": int(info.get("thumbheight") or 0),
        }
    return out


def fetch_known_file(filename: str) -> dict | None:
    """Resolve a known Commons file via titles= API (1 request, no search)."""
    got = fetch_known_batch([filename])
    return got.get(filename) or next(iter(got.values()), None)

# Must match architecture in title/description; reject people, food, travel portraits
ARCH_HINT = re.compile(
    r"(facade|tower|mashrabiya|mashrabiyya|mosque|dome|minaret|arch|colonnade|screen|"
    r"lattice|perforat|geometric|islamic architecture|muslim architecture|courtyard|"
    r"louvre|bahar|monde arabe|masdar|alhambra|jeddah|parametric|brise|cladding|"
    r"pavilion|madrasa|masjid|girih|muqarnas|window|building exterior|cultural center|"
    r"museum|cathedral interior|portal|gate of|colonnade|arcade)",
    re.I,
)
REJECT_HINT = re.compile(
    r"(bride|groom|priest|wedding|portrait|face|food|dish|curry|kadhi|cuisine|"
    r"market stall|animal|horse|festival dance|painting|map|coin|stamp|"
    r"federal building|courthouse|thurmond|pena palace|yellow wall|"
    r"hindu temple|khmer|buddha|church interior christmas)",
    re.I,
)


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def search_commons(query: str, limit: int = 12) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": "1600",
    }
    headers = {"User-Agent": "flux-islamic-parametric-curate/1.0 (portfolio research; contact: repo issues)"}
    for attempt in range(5):
        try:
            resp = requests.get(API, params=params, timeout=40, headers=headers)
            if resp.status_code == 429:
                wait = 8 * (attempt + 1)
                print(f"  429 backoff {wait}s for {query!r}")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.HTTPError:
            if attempt == 4:
                raise
            time.sleep(4 * (attempt + 1))
    else:
        raise RuntimeError(f"rate limited: {query}")
    pages = (resp.json().get("query") or {}).get("pages") or {}
    time.sleep(2.0)
    out: list[dict] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        lic = strip_html((meta.get("LicenseShortName") or {}).get("value", ""))
        artist = strip_html((meta.get("Artist") or {}).get("value", ""))[:120]
        if not OK_LICENSE.search(lic or ""):
            continue
        title = page.get("title", "")
        if REJECT_HINT.search(title):
            continue
        if not (ARCH_HINT.search(title) or ARCH_HINT.search(strip_html((meta.get("ImageDescription") or {}).get("value", "")))):
            continue
        w, h = int(info.get("width") or 0), int(info.get("height") or 0)
        if min(w, h) < MIN_SIDE * 0.7 and max(w, h) < MIN_SIDE:
            continue
        url = info.get("thumburl") or info.get("url") or ""
        if not url:
            continue
        out.append(
            {
                "title": page.get("title", ""),
                "url": url,
                "width": w,
                "height": h,
                "license": lic,
                "author": artist,
                "description_url": info.get("descriptionurl", ""),
                "thumb_w": int(info.get("thumbwidth") or 0),
                "thumb_h": int(info.get("thumbheight") or 0),
            }
        )
    return out


def luma_of(im: Image.Image) -> float:
    return ImageStat.Stat(im.convert("L").resize((64, 64))).mean[0]


def contrast_of(im: Image.Image) -> float:
    return ImageStat.Stat(im.convert("L").resize((64, 64))).stddev[0]


def prepare_square(im: Image.Image) -> Image.Image | None:
    w, h = im.size
    side = min(w, h)
    if side < MIN_SIDE:
        # allow slight upscale from 0.85*min
        if side < int(MIN_SIDE * 0.85):
            return None
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        return im.resize((MIN_SIDE, MIN_SIDE), Image.Resampling.LANCZOS)
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    if side > 1600:
        im = im.resize((1600, 1600), Image.Resampling.LANCZOS)
    elif side < MIN_SIDE:
        im = im.resize((MIN_SIDE, MIN_SIDE), Image.Resampling.LANCZOS)
    return im


def download_and_filter(url: str) -> Image.Image | None:
    headers = {"User-Agent": "flux-islamic-parametric-curate/1.0 (portfolio research; contact: repo issues)"}
    resp = None
    for attempt in range(4):
        resp = requests.get(url, timeout=60, headers=headers)
        if resp.status_code == 429:
            time.sleep(6 * (attempt + 1))
            continue
        resp.raise_for_status()
        break
    if resp is None or resp.status_code != 200:
        return None
    from io import BytesIO

    try:
        im = Image.open(BytesIO(resp.content))
        im.load()
        im = im.convert("RGB")
    except Exception:
        return None
    square = prepare_square(im)
    if square is None:
        return None
    if luma_of(square) < MIN_LUMA:
        return None
    if contrast_of(square) < 25:
        return None
    return square


def make_contact_sheet(ds: Path, files: list[Path], name: str) -> Path:
    from PIL import ImageDraw

    thumb, cols, pad = 240, 5, 26
    n = len(files)
    rows = (n + cols - 1) // cols or 1
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + pad)), (14, 14, 14))
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(files):
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        except Exception:
            continue
        x, y = (i % cols) * thumb, (i // cols) * (thumb + pad)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + thumb + 6), p.stem, fill=(230, 210, 80))
    out = ds / name
    sheet.save(out, quality=90)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Curate landmark architecture photos from Wikimedia Commons")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--backup-synthetic", action="store_true")
    ap.add_argument("--target", type=int, default=NUM_IMAGES)
    ap.add_argument("--min-luma", type=int, default=MIN_LUMA)
    args = ap.parse_args(argv)

    if args.check:
        print("source: Wikimedia Commons API + known titles")
        print("known files:", len(KNOWN_COMMONS_FILES), "queries:", len(LANDMARK_QUERIES), "target:", args.target)
        print("filters: free license, min_side>=", MIN_SIDE, "min_luma>=", args.min_luma, "contrast>=25")
        print("landmarks:", ", ".join(q for _, q in LANDMARK_QUERIES[:6]), "...")
        return 0
    if not args.execute:
        print("dry-run: pass --execute to download (use --backup-synthetic first)")
        return 0

    ds = Path(DATASET_DIR)
    ds.mkdir(exist_ok=True)
    caps_dir = ds / "captions"
    caps_dir.mkdir(exist_ok=True)

    if args.backup_synthetic:
        bak = Path("dataset_synthetic_backup")
        bak.mkdir(exist_ok=True)
        for p in list(ds.glob("image_*.jpg")) + list(caps_dir.glob("*.txt")):
            if p.exists():
                shutil.move(str(p), str(bak / p.name))
        if Path(MANIFEST_PATH).exists():
            shutil.move(MANIFEST_PATH, str(bak / "manifest.json"))
        print("backed up synthetic set -> dataset_synthetic_backup/")

    # Resume from existing manifest/images
    entries: list[dict] = []
    manifest_path = Path(MANIFEST_PATH)
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                entries = [e for e in existing if (ds / e.get("filename", "")).exists()]
        except json.JSONDecodeError:
            entries = []
    used_urls = {e.get("source_url", "") for e in entries}
    seen_titles = {e.get("commons_title", "") for e in entries}
    existing_nums: list[int] = []
    for e in entries:
        try:
            existing_nums.append(int(Path(e["filename"]).stem.split("_")[-1]))
        except (KeyError, ValueError):
            continue
    idx = (max(existing_nums) if existing_nums else 0) + 1
    print(f"resume: {len(entries)} kept, next image_{idx:02d}")

    # Prefer known landmark files: CDN download first (no search API), then batch license lookup
    pending_known = [
        (tag, fn)
        for tag, fn in KNOWN_COMMONS_FILES
        if f"File:{fn}" not in seen_titles and not REJECT_HINT.search(fn)
    ]
    new_for_license: list[tuple[str, str, str]] = []  # (filename_on_disk, commons_fn, tag)
    for tag, filename in pending_known:
        if idx > args.target or len(entries) >= args.target:
            break
        try:
            im = download_via_filepath(filename)
        except Exception as exc:
            print(f"cdn fail {filename}: {exc}")
            time.sleep(1.0)
            continue
        if im is None:
            continue
        name = IMAGE_PATTERN.format(idx) + ".jpg"
        path = ds / name
        while path.exists():
            idx += 1
            name = IMAGE_PATTERN.format(idx) + ".jpg"
            path = ds / name
        im.save(path, quality=92)
        entries.append(
            {
                "filename": name,
                "source": "wikimedia_commons",
                "query": tag,
                "width": im.size[0],
                "height": im.size[1],
                "author": "",
                "license": "pending",
                "source_url": file_url(filename, width=800),
                "commons_title": f"File:{filename}",
                "luma": round(luma_of(im), 1),
            }
        )
        new_for_license.append((name, filename, tag))
        print(f"kept {name} luma={entries[-1]['luma']} CDN {filename[:70]}")
        idx += 1
        time.sleep(0.35)

    # Batch license/author fill-in (few API calls)
    if new_for_license:
        for i in range(0, len(new_for_license), 8):
            chunk = new_for_license[i : i + 8]
            try:
                resolved = fetch_known_batch([fn for _, fn, _ in chunk])
            except Exception as exc:
                print("license batch fail:", exc)
                continue
            for disk_name, fn, _tag in chunk:
                meta = resolved.get(fn)
                if not meta:
                    continue
                for e in entries:
                    if e.get("filename") == disk_name:
                        e["author"] = meta.get("author", "")
                        e["license"] = meta.get("license", e.get("license", ""))
                        e["source_url"] = meta.get("description_url") or e.get("source_url", "")
                        e["commons_title"] = meta.get("title", e.get("commons_title", ""))
                        if not OK_LICENSE.search(e.get("license", "")):
                            # keep image but flag non-free / unknown
                            e["license"] = e.get("license") or "unknown"
                        break

    print(f"after known files: {len(entries)}")

    # Prefer favorite landmark hits first
    ordered = list(LANDMARK_QUERIES)
    for tag, query in ordered:
        if idx > args.target:
            break
        if len(entries) >= args.target:
            break
        try:
            results = search_commons(query, limit=20)
        except Exception as exc:
            print(f"search fail {query!r}: {exc}")
            time.sleep(3.0)
            continue
        # boost favorites; boost architecture-sounding titles
        results.sort(
            key=lambda r: (
                0
                if any(f.lower() in r["title"].lower() for f in FAVORITE_TITLES)
                else (1 if ARCH_HINT.search(r["title"]) else 2)
            )
        )
        kept_this_query = 0
        for item in results:
            if idx > args.target:
                break
            if item["url"] in used_urls or item["title"] in seen_titles:
                continue
            if REJECT_HINT.search(item["title"]):
                continue
            try:
                im = download_and_filter(item["url"])
            except Exception as exc:
                print("dl fail:", exc)
                time.sleep(1.0)
                continue
            if im is None:
                continue
            used_urls.add(item["url"])
            seen_titles.add(item["title"])
            name = IMAGE_PATTERN.format(idx) + ".jpg"
            path = ds / name
            while path.exists():
                idx += 1
                name = IMAGE_PATTERN.format(idx) + ".jpg"
                path = ds / name
            im.save(path, quality=92)
            entries.append(
                {
                    "filename": name,
                    "source": "wikimedia_commons",
                    "query": tag,
                    "width": im.size[0],
                    "height": im.size[1],
                    "author": item["author"],
                    "license": item["license"],
                    "source_url": item["description_url"] or item["url"],
                    "commons_title": item["title"],
                    "luma": round(luma_of(im), 1),
                }
            )
            print(f"kept {name} luma={entries[-1]['luma']} [{item['license']}] {item['title'][:70]}")
            idx += 1
            kept_this_query += 1
            if kept_this_query >= 6:
                break
            time.sleep(0.8)

        time.sleep(2.5)

    got = len(entries)
    print(f"\nkept {got}/{args.target}")
    if got == 0:
        print("ERROR: no free-license images passed filters")
        return 1

    Path(MANIFEST_PATH).write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    files = sorted(ds.glob("image_*.jpg"))
    sheet = make_contact_sheet(ds, files, "_contact_sheet_curation.jpg")
    report = Path("eval_outputs")
    report.mkdir(exist_ok=True)
    (report / "curation_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "kept": got,
                "target": args.target,
                "min_luma": args.min_luma,
                "entries": entries,
                "contact_sheet": str(sheet.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("manifest:", MANIFEST_PATH)
    print("contact sheet:", sheet.resolve())
    print("NEXT: python generate_captions.py --backend auto")
    print("THEN: review sheet + upload to HF only after your OK")
    if got < args.target:
        print("WARN: partial set — re-run to top-up remaining slots")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
