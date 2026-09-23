"""Finalize a unique, license-aware architecture dataset for FLUX LoRA.

Deduplicates by content hash, tops-up from Wikimedia Commons free-license
sources, rebuilds manifest + captions + contact sheet + quality report.
Does not upload to Hugging Face or train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from islamic_parametric.constants import (  # noqa: E402
    DATASET_DIR,
    IMAGE_PATTERN,
    MANIFEST_PATH,
    NUM_IMAGES,
    TRIGGER_WORD,
    build_caption,
)

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "flux-islamic-parametric-curate/1.0 (portfolio research)"}
OK_LICENSE = re.compile(
    r"(cc0|cc[- ]by(?:[- ]sa)?(?:[- ][0-9.]+)?|public domain|pd|no restrictions|pdm)",
    re.I,
)
MIN_LUMA = 70
MIN_CONTRAST = 35
MIN_SIDE = 1024
TARGET = NUM_IMAGES

# High-signal architecture filenames on Commons (free-license checked via API)
KNOWN: list[tuple[str, str]] = [
    ("jeddah_rawasheen", "Traditional_architecture_in_old_Jeddah,_Saudi_Arabia_(71)_(50702693833).jpg"),
    ("najaf_pattern", "Frattaleschi-Najaf.jpg"),
    ("cologne_mosque", "Interior_of_the_Cologne_Central_Mosque_-_Modern_dome,_minbar_and_calligraphy.jpg"),
    ("alhambra_lions", "Granada-_View_into_the_lion_court_of_the_Alhambra_(SM_stf113).png"),
    ("doha_mia", "Doha_-_Museum_of_Islamic_Art.jpg"),
    ("alhambra_spain", "Alhambra_Granada_Spain.jpg"),
    ("blue_mosque", "Blue_Mosque_Istanbul.jpg"),
    ("alhambra_leones", "Alhambra_Patio_de_los_Leones.jpg"),
    ("alhambra_cypress", "Alhambra_Court_of_the_Cypress_in_the_Alcázar_of_Seville.jpg"),
    ("alhambra_myrtles", "Alhambra_Court_of_the_Myrtles.jpg"),
    ("alhambra_ambassadors", "Alhambra_Hall_of_the_Ambassadors.jpg"),
    ("sultan_hassan", "Sultan_Hassan_Mosque_Cairo.jpg"),
    ("ibn_tulun", "Ibn_Tulun_Mosque_Cairo.jpg"),
    ("kairouan", "Great_Mosque_of_Kairouan.jpg"),
    ("koutoubia", "Koutoubia_Mosque_Marrakech.jpg"),
    ("zitouna", "Zitouna_Mosque_Tunis.jpg"),
    ("seville_maidens", "Alcazar_Seville_Court_of_the_Maidens.jpg"),
    ("seville_arches", "Alcazar_Seville_arches.jpg"),
    ("mezquita", "Mezquita_Cordoba.jpg"),
    ("naqsh_jahan", "Naqsh-e_Jahan_Square_Isfahan.jpg"),
    ("sheikh_lotfollah", "Sheikh_Lotfollah_Mosque_Isfahan.jpg"),
    ("ali_qapu", "Ali_Qapu_Isfahan.jpg"),
    ("suleymaniye", "Suleymaniye_Mosque_Istanbul.jpg"),
    ("selimiye", "Selimiye_Mosque_Edirne.jpg"),
    ("bahia", "Bahia_Palace_Marrakech.jpg"),
    ("ben_yusuf", "Ben_Yusuf_Madrasa_Marrakech.jpg"),
    ("sultan_ahmed", "Sultan_Ahmed_Mosque_Istanbul_2007_008.jpg"),
    ("sheikh_zayed", "Sheikh_Zayed_Mosque_Abu_Dhabi.jpg"),
    ("hassan_ii", "Hassan_II_Mosque_Casablanca.jpg"),
    ("qaitbay_mausoleum", "Mausoleum_of_Qaitbay_Cairo.jpg"),
    ("qaitbay_dome", "Dome_of_the_Qaitbay_Cemetery.jpg"),
    ("mashrabiya_cairo", "Mashrabiya_in_Cairo.jpg"),
    ("mashrabiya_wood", "Wooden_mashrabiya.jpg"),
    ("mashrabiya_window", "Mashrabiya_window,_Cairo.jpg"),
    ("mashrabiya_egypt", "Carved_wooden_mashrabiya,_Egypt.jpg"),
    ("mashrabiya_screen", "Mashrabiya_screen,_Cairo_Egypt.jpg"),
    ("albalad", "Al-Balad_Jeddah.jpg"),
    ("nasseef", "Nasseef_House.jpg"),
    ("rawashin", "Rawashin_Jeddah_old_town.jpg"),
    ("dome_rock", "Dome_of_the_Rock_Old_City_Jerusalem.jpg"),
    ("petronas", "Petronas_Towers_Kuala_Lumpur.jpg"),
    ("heydar", "Heydar_Aliyev_Center_Baku.jpg"),
    ("flame_towers", "Baku_Flame_Towers.jpg"),
    ("cayan", "Cayan_Tower_Dubai.jpg"),
    ("kingdom_centre", "Kingdom_Centre_Tower_Riyadh.jpg"),
    ("capital_gate", "Capital_Gate_Abu_Dhabi.jpg"),
    ("institut_monde", "Institut_du_Monde_Arabe,_Paris_21_November_2013.jpg"),
    ("institut_monde2", "Institut_du_monde_arabe,_Paris_2_August_2013.jpg"),
    ("ima_facade", "IMA_facade.jpg"),
    ("louvre_ad", "Louvre_Abu_Dhabi_(36356445474).jpg"),
    ("louvre_ad2", "Louvre_Abu_Dhabi_2.jpg"),
    ("doha_tower", "Doha_Tower,_Qatar.jpg"),
    ("burj_doha", "Burj_Doha_Tower.jpg"),
    ("masdar", "Masdar_City_Campus.jpg"),
    ("al_bahar", "Al_Bahar_Towers,_Abu_Dhabi.jpg"),
    ("girih", "Girih_tiles.jpg"),
    ("geometric_alhambra", "Islamic_geometric_patterns_in_the_Alhambra.jpg"),
    ("tilework", "Tilework_Ispahan.jpg"),
    ("islamic_tiling", "Islamic_geometric_tiling.jpg"),
    ("minaret_kairouan", "Minaret_of_the_Great_Mosque_of_Kairouan.jpg"),
    ("sultan_qalawun", "Qalawun_complex_Cairo_minaret.jpg"),
    ("islamic_art_cairo", "Islamic_Art_Museum_Cairo.jpg"),
    ("mia_doha_interior", "Museum_of_Islamic_Art_Doha_interior.jpg"),
    ("suleymaniye2", "Suleymaniye_Mosque_from_Galata.jpg"),
    ("blue_mosque2", "Blue_Mosque_courtyard.jpg"),
    ("alcazar_maidens2", "Real_Alcazar_Seville_Patio_de_las_Doncellas.jpg"),
    ("cordoba_hypostyle", "Mezquita_hypostyle_hall.jpg"),
    ("isfahan_bazaar", "Bazaar_of_Isfahan.jpg"),
    ("fes_madrasa", "Bou_Inania_Madrasa_Fes.jpg"),
    ("morocco_riad", "Riad_courtyard_Marrakech.jpg"),
    ("islamic_screen", "Islamic_wooden_screen.jpg"),
    ("mamluk_door", "Mamluk_wooden_door_Cairo.jpg"),
    ("girih_door", "Girih_pattern_door.jpg"),
    ("muqarnas", "Muqarnas_dome.jpg"),
    ("muqarnas2", "Stalactite_work_muqarnas.jpg"),
    ("pointed_arches", "Pointed_arches_Gothic_cathedral.jpg"),
    ("islamic_arch", "Islamic_arch_doorway_Cairo.jpg"),
]

CORE_BY_QUERY = {
    "jeddah_rawasheen": "ornate rawasheen timber bay windows on a historic Jeddah coral-stone facade in bright daylight",
    "najaf_pattern": "intricate geometric star-pattern tilework screen with strong daylight contrast",
    "cologne_mosque": "modern mosque interior with a luminous dome, minbar and calligraphy under soft daylight",
    "alhambra_lions": "courtyard of the Alhambra with carved arcades, slender columns and bilateral symmetry",
    "doha_mia": "museum of Islamic art volume with stacked limestone masses and geometric apertures at golden hour",
    "alhambra_spain": "Alhambra palace exterior with geometric stucco and reflecting water under clear sky",
    "blue_mosque": "monumental mosque exterior with cascading domes and pencil minarets under blue sky",
    "alhambra_leones": "courtyard of the lions with fine muqarnas, arcades and crisp sun shadows",
    "alhambra_cypress": "palace garden court with cypress alignment, arches and patterned paving",
    "alhambra_myrtles": "long reflecting pool court framed by myrtles, arcades and structural rhythm",
    "alhambra_ambassadors": "throne hall with gilded dome, geometric panels and axial symmetry",
    "sultan_hassan": "monumental Mamluk mosque-madrasa portal with deep muqarnas vaulting in hard sunlight",
    "ibn_tulun": "open-air hypostyle mosque courtyard with arcades, brick minaret and geometric repetition",
    "kairouan": "great mosque courtyard with whitewashed arcades, minaret mass and strict modular rhythm",
    "koutoubia": "minaret and prayer hall in warm daylight with geometric relief",
    "zitouna": "historic mosque courtyard with column screen and bright Mediterranean light",
    "seville_maidens": "alcázar courtyard with multifoil arches, tile dados and filtered daylight",
    "seville_arches": "palace arcade with interlocking arches and geometric tilework",
    "mezquita": "hypostyle hall of red-and-white double arches repeating in deep perspective",
    "naqsh_jahan": "vast royal square with portal iwan, turquoise tile and rhythmic arcades",
    "sheikh_lotfollah": "mosque dome chamber with arabesque tiling and soft oculus light",
    "ali_qapu": "palace veranda with music room perforations and geometric ceiling",
    "suleymaniye": "imperial mosque silhouette with domes and minarets above the city",
    "selimiye": "central-plan mosque with massive dome and pencil minarets",
    "bahia": "riad palace courtyard with carved cedar, zellige and sunlit gardens",
    "ben_yusuf": "madrasa courtyard with stucco, cedar and geometric paving in daylight",
    "sultan_ahmed": "blue mosque exterior with dome cascade and imperial scale",
    "sheikh_zayed": "grand mosque colonnade with white marble arches and gold accents",
    "hassan_ii": "coastal mosque with tall minaret and geometric concrete screen",
    "qaitbay_mausoleum": "circles mausoleum facade with marquetry and bright desert light",
    "qaitbay_dome": "domed funerary complex with geometric inlay and sharp shadows",
    "mashrabiya_cairo": "wooden mashrabiya screen wrapping a courtyard facade in warm light",
    "mashrabiya_turned": "turned wooden lattice mashrabiya casting dense geometric shadow",
    "mashrabiya_window": "projecting mashrabiya window with delicate lattice and street daylight",
    "mashrabiya_egypt": "carved wooden mashrabiya panel with Islamic geometry",
    "mashrabiya_screen": "full-height mashrabiya screen filtering strong sunlight",
    "albalad": "historic Jeddah street with multi-storey rawasheen and coral-stone facades",
    "nasseef": "historic Jeddah merchant house with carved timber projections",
    "rawashin": "row of timber rawasheen elevations with rhythmic openings",
    "dome_rock": "octagonal shrine with gold dome, mosaic drum and geometric arcades",
    "petronas": "twin towers with geometric podium and polished facade reflections",
    "heydar": "fluid cultural center shell with sweeping white curves",
    "flame_towers": "tilted tower cluster with faceted glass against sky",
    "cayan": "twisting residential tower with continuous glass ribbon",
    "kingdom_centre": "tall tower with inverted-parabola void and glass bridge",
    "capital_gate": "leaning tower with diagrid exoskeleton in bright light",
    "institut_monde": "cultural institute facade of motorized geometric apertures",
    "institut_monde2": "institute facade detail of aluminum diaphragm screen",
    "ima_facade": "geometric metal facade grid with layered depth",
    "louvre_ad": "dome of interlocking geometric rosettes casting dappled light",
    "louvre_ad2": "museum waterfront volume under a perforated dome",
    "doha_tower": "cylindrical tower with lacy geometric cladding and sky",
    "burj_doha": "tall cylindrical tower with mashrabiya-inspired skin",
    "masdar": "low-rise campus with shaded geometric canopies and desert light",
    "al_bahar": "twin towers with responsive triangular sunshade facade",
    "girih": "girih strip tiling with interlaced star polygons",
    "geometric_alhambra": "wall of interlocking Islamic star and polygon geometry",
    "tilework": "dense glazed tilework with muqarnas-like geometric field",
    "islamic_tiling": "repeating Islamic tessellation in strong color",
    "minaret_kairouan": "square minaret with stacked geometric decoration",
    "sultan_qalawun": "Mamluk complex facade with stone inlay and portal depth",
    "islamic_art_cairo": "museum gallery of Islamic objects under controlled daylight",
    "mia_doha_interior": "museum atrium with geometric screens and warm stone",
    "suleymaniye2": "imperial mosque and city skyline in clear daylight",
    "blue_mosque2": "mosque courtyard arcade with lead domes and geometry",
    "alcazar_maidens2": "courtyard of maidens with polylobed arches and reflection",
    "cordoba_hypostyle": "forest of double arches in the hypostyle prayer hall",
    "isfahan_bazaar": "covered bazaar vaults with brick geometry and light wells",
    "fes_madrasa": "madrasa courtyard with zellige, cedar and carved plaster",
    "morocco_riad": "riad courtyard with pool, arches and geometric tile",
    "islamic_screen": "freestanding Islamic geometric screen with daylight behind",
    "mamluk_door": "massive wooden door with geometric metal studs",
    "girih_door": "door leaf with girih star composition",
    "muqarnas": "muqarnas vault with honeycomb cell geometry",
    "muqarnas2": "stalactite muqarnas corner under soft light",
    "pointed_arches": "sequence of pointed arches with strict modular rhythm",
    "islamic_arch": "Islamic pointed arch doorway with carved surround",
}


def thumb_hash(im: Image.Image) -> str:
    return hashlib.md5(im.resize((32, 32)).convert("RGB").tobytes()).hexdigest()


def phash(im: Image.Image) -> str:
    g = im.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / 64
    bits = "".join("1" if p >= avg else "0" for p in px)
    return f"{int(bits, 2):016x}"


def luma(im: Image.Image) -> float:
    return ImageStat.Stat(im.convert("L").resize((64, 64))).mean[0]


def contrast(im: Image.Image) -> float:
    return ImageStat.Stat(im.convert("L").resize((64, 64))).stddev[0]


def prepare_square(im: Image.Image) -> Image.Image | None:
    w, h = im.size
    side = min(w, h)
    if side < int(MIN_SIDE * 0.85):
        return None
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    if side < MIN_SIDE:
        im = im.resize((MIN_SIDE, MIN_SIDE), Image.Resampling.LANCZOS)
    elif side > 1600:
        im = im.resize((1600, 1600), Image.Resampling.LANCZOS)
    return im


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _norm_fn(s: str) -> str:
    return re.sub(r"[\s_]+", " ", s.strip().lower())


def commons_titles(filenames: list[str]) -> dict[str, dict]:
    if not filenames:
        return {}
    # request in chunks of 20 titles
    out: dict[str, dict] = {}
    for start in range(0, len(filenames), 20):
        chunk = filenames[start : start + 20]
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(f"File:{fn}" for fn in chunk),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": "1600",
        }
        for attempt in range(5):
            try:
                resp = requests.get(API, params=params, timeout=40, headers=UA)
                if resp.status_code == 429:
                    time.sleep(8 * (attempt + 1))
                    continue
                resp.raise_for_status()
                break
            except requests.HTTPError:
                if attempt == 4:
                    resp = None
                    break
                time.sleep(4 * (attempt + 1))
        else:
            resp = None
        if resp is None:
            continue
        time.sleep(1.2)
        pages = (resp.json().get("query") or {}).get("pages") or {}
        for page in pages.values():
            if page.get("missing") is not None:
                continue
            title = page.get("title", "")
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
            rec = {
                "title": title,
                "url": url,
                "license": lic,
                "author": strip_html((meta.get("Artist") or {}).get("value", ""))[:120],
                "source_url": info.get("descriptionurl", ""),
                "width": w,
                "height": h,
            }
            out[fn] = rec
            out[_norm_fn(fn)] = rec
            # also map original requested keys that differ by underscores
            for req in chunk:
                if _norm_fn(req) == _norm_fn(fn):
                    out[req] = rec
    return out


def file_path_url(filename: str, width: int = 1600) -> str:
    from urllib.parse import quote

    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}?width={width}"


def download_filtered(url: str) -> Image.Image | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=45, headers=UA)
            if resp.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        except requests.HTTPError:
            if attempt == 2:
                return None
            time.sleep(3)
    else:
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
    if luma(square) < MIN_LUMA or contrast(square) < MIN_CONTRAST:
        return None
    return square


def load_manifest() -> list[dict]:
    if not Path(MANIFEST_PATH).exists():
        return []
    data = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def save_manifest(entries: list[dict]) -> None:
    Path(MANIFEST_PATH).write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_existing_hashes(ds: Path, keep: set[str]) -> tuple[set[str], set[str]]:
    md5s: set[str] = set()
    phs: set[str] = set()
    for p in sorted(ds.glob("image_*.jpg")):
        if p.name not in keep:
            continue
        im = Image.open(p).convert("RGB")
        md5s.add(thumb_hash(im))
        phs.add(phash(im))
    return md5s, phs


def key_for_query(tag: str) -> str:
    return tag


def build_caption_for(entry: dict, index: int) -> str:
    tag = entry.get("query") or ""
    core = CORE_BY_QUERY.get(tag)
    if not core:
        # derive from commons title words
        title = (entry.get("commons_title") or entry.get("filename") or "architectural facade")
        title = re.sub(r"^File:", "", title)
        title = re.sub(r"\.(jpg|jpeg|png)$", "", title, flags=re.I)
        title = re.sub(r"[_]+", " ", title)
        core = title.lower() + " with geometric ornament, daylight, structural symmetry"
    unique = f"study {index + 1:02d}"
    return build_caption(f"{core}; {unique}")


def write_captions(ds: Path, entries: list[dict]) -> int:
    caps = ds / "captions"
    if caps.exists():
        for p in caps.glob("*.txt"):
            p.unlink()
    caps.mkdir(exist_ok=True)
    for i, e in enumerate(entries):
        cap = build_caption_for(e, i)
        assert TRIGGER_WORD in cap
        (caps / f"{Path(e['filename']).stem}.txt").write_text(cap + "\n", encoding="utf-8")
    return len(entries)


def contact_sheet(ds: Path, entries: list[dict]) -> Path:
    files = [ds / e["filename"] for e in entries if (ds / e["filename"]).exists()]
    thumb, cols, pad = 240, 5, 26
    n = len(files)
    rows = max(1, (n + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + pad)), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(files):
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        except Exception:
            continue
        x, y = (i % cols) * thumb, (i // cols) * (thumb + pad)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + thumb + 6), p.stem, fill=(235, 210, 70))
    out = ds / "_contact_sheet_review.jpg"
    sheet.save(out, quality=90)
    return out


def quality_report(entries: list[dict], ds: Path) -> dict:
    rows = []
    md5s: list[str] = []
    phs: list[str] = []
    for e in entries:
        p = ds / e["filename"]
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        md5s.append(thumb_hash(im))
        phs.append(phash(im))
        rows.append(
            {
                "file": e["filename"],
                "luma": round(luma(im), 1),
                "contrast": round(contrast(im), 1),
                "license": e.get("license", ""),
                "query": e.get("query", ""),
            }
        )
    md5_dup = [k for k, v in Counter(md5s).items() if v > 1]
    ph_dup = [k for k, v in Counter(phs).items() if v > 1]
    lumas = [r["luma"] for r in rows]
    report = {
        "n": len(rows),
        "target": TARGET,
        "unique_md5": len(set(md5s)),
        "unique_phash": len(set(phs)),
        "duplicate_md5_groups": len(md5_dup),
        "duplicate_phash_groups": len(ph_dup),
        "mean_luma": round(sum(lumas) / len(lumas), 1) if lumas else 0,
        "min_luma": min(lumas) if lumas else 0,
        "licenses": dict(Counter(e.get("license", "") for e in entries)),
        "all_free_license": all(
            OK_LICENSE.search(str(e.get("license", ""))) for e in entries
        ),
        "captions": len(list((ds / "captions").glob("*.txt"))) if (ds / "captions").exists() else 0,
        "verdict": None,
        "per_image": rows,
    }
    ok = (
        report["n"] >= TARGET
        and report["duplicate_md5_groups"] == 0
        and report["duplicate_phash_groups"] == 0
        and report["mean_luma"] >= 80
        and report["captions"] >= report["n"]
        and report["all_free_license"]
    )
    report["verdict"] = "PASS — unique bright free-license set ready for review" if ok else "NEEDS_WORK"
    return report


def renumber(ds: Path, entries: list[dict]) -> list[dict]:
    """Keep stable order, rewrite sequential filenames image_01..N."""
    # sort by current filename
    entries = sorted(entries, key=lambda e: e.get("filename", ""))
    # detect if already sequential unique
    out: list[dict] = []
    md5s: set[str] = set()
    phs: set[str] = set()
    kept: list[tuple[dict, Image.Image]] = []
    for e in entries:
        p = ds / e["filename"]
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        h, ph = thumb_hash(im), phash(im)
        if h in md5s or ph in phs:
            p.unlink(missing_ok=True)
            continue
        md5s.add(h)
        phs.add(ph)
        kept.append((e, im))

    # write sequentially
    for i, (e, im) in enumerate(kept, start=1):
        new_name = IMAGE_PATTERN.format(i) + ".jpg"
        old = ds / e["filename"]
        new_path = ds / new_name
        if e["filename"] != new_name:
            # avoid overwrite collisions with temp names
            tmp = ds / f"__tmp_{i:02d}.jpg"
            im.save(tmp, quality=92)
            old.unlink(missing_ok=True)
            if new_path.exists() and new_path.name != tmp.name:
                # only if different content already validated unique — shouldn't happen
                pass
            tmp.replace(new_path)
        else:
            im.save(new_path, quality=92)
        e = dict(e)
        e["filename"] = new_name
        out.append(e)
    return out


def download_topup(ds: Path, entries: list[dict], md5s: set[str], phs: set[str], target: int) -> list[dict]:
    used_titles = {e.get("commons_title", "") for e in entries}
    used_norm = {_norm_fn(t) for t in used_titles}
    candidates: list[tuple[str, str]] = []
    for tag, fn in KNOWN:
        title = f"File:{fn}"
        if title in used_titles or _norm_fn(title) in used_norm:
            continue
        candidates.append((tag, fn))

    print(f"top-up candidates remaining: {len(candidates)}", flush=True)
    # Prefer metadata-first: only download free-license rows that pass size
    for i in range(0, len(candidates), 8):
        if len(entries) >= target:
            break
        batch = candidates[i : i + 8]
        meta_map = commons_titles([fn for _, fn in batch])
        if not meta_map:
            print(f"  meta empty for batch {i} (rate limit?)", flush=True)
            time.sleep(4)
            continue
        for tag, fn in batch:
            if len(entries) >= target:
                break
            meta = meta_map.get(fn) or meta_map.get(_norm_fn(fn))
            if not meta:
                continue
            if meta["title"] in used_titles or _norm_fn(meta["title"]) in used_norm:
                continue
            if not OK_LICENSE.search(meta.get("license", "")):
                continue
            url = meta["url"] or file_path_url(fn)
            im = download_filtered(url)
            if im is None:
                im = download_filtered(file_path_url(fn))
            if im is None:
                print(f"  dl fail {fn[:50]}", flush=True)
                continue
            h, ph = thumb_hash(im), phash(im)
            if h in md5s or ph in phs:
                print(f"  dup reject {fn[:50]}", flush=True)
                continue
            md5s.add(h)
            phs.add(ph)
            used_titles.add(meta["title"])
            used_norm.add(_norm_fn(meta["title"]))
            tmp_idx = len(entries) + 1
            name = IMAGE_PATTERN.format(tmp_idx) + ".jpg"
            entries.append(
                {
                    "filename": name,
                    "source": "wikimedia_commons",
                    "query": tag,
                    "width": im.size[0],
                    "height": im.size[1],
                    "author": meta.get("author", ""),
                    "license": meta.get("license", ""),
                    "source_url": meta.get("source_url", ""),
                    "commons_title": meta.get("title", f"File:{fn}"),
                    "luma": round(luma(im), 1),
                }
            )
            im.save(ds / name, quality=92)
            print(f"kept {name} luma={entries[-1]['luma']} [{entries[-1]['license']}] {fn[:60]}", flush=True)
            time.sleep(0.3)
        time.sleep(1.5)
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dedup + top-up unique architecture dataset")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args(argv)

    ds = Path(DATASET_DIR)
    if args.check:
        print("target:", args.target, "min_luma:", MIN_LUMA, "min_contrast:", MIN_CONTRAST)
        print("known candidates:", len(KNOWN))
        print("dedup: md5(32) + phash(8)")
        print("licenses: CC0/CC-BY/CC-BY-SA/PD only")
        return 0
    if not args.execute:
        print("dry-run: pass --execute")
        return 0

    ds.mkdir(exist_ok=True)
    (ds / "captions").mkdir(exist_ok=True)

    # 1) rebuild manifest from disk if needed (orphan images beyond manifest)
    entries = load_manifest()
    # index any image not in manifest by matching hash to existing entries
    known_by_hash: dict[str, dict] = {}
    for e in entries:
        p = ds / e["filename"]
        if p.exists():
            known_by_hash[thumb_hash(Image.open(p).convert("RGB"))] = e

    for p in sorted(ds.glob("image_*.jpg")):
        if any(e["filename"] == p.name for e in entries):
            continue
        im = Image.open(p).convert("RGB")
        h = thumb_hash(im)
        if h in known_by_hash:
            # duplicate orphan — drop file
            p.unlink(missing_ok=True)
            print("removed orphan dup", p.name)
            continue
        # unknown provenance orphan — keep temporarily with pending license
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
                "commons_title": "",
                "luma": round(luma(im), 1),
            }
        )

    # 2) renumber + dedup existing
    entries = renumber(ds, entries)
    print(f"after dedup: {len(entries)} unique")
    md5s, phs = collect_existing_hashes(ds, {e["filename"] for e in entries})

    # remove leftover image files not in entries
    keep = {e["filename"] for e in entries}
    for p in ds.glob("image_*.jpg"):
        if p.name not in keep:
            p.unlink(missing_ok=True)
            print("removed extra", p.name)

    # 3) top-up to target with free-license unique photos
    if len(entries) < args.target:
        print(f"top-up {len(entries)} -> {args.target}...")
        entries = download_topup(ds, entries, md5s, phs, args.target)
        entries = renumber(ds, entries)

    # fill pending licenses from titles where possible
    pending = [e for e in entries if e.get("license") in ("", "pending", "unknown")]
    if pending:
        titles = [e.get("commons_title", "") for e in pending if e.get("commons_title")]
        fns = [t.split("File:", 1)[-1] for t in titles if t]
        if fns:
            meta_map = commons_titles(fns)
            for e in pending:
                fn = (e.get("commons_title") or "").split("File:", 1)[-1]
                meta = meta_map.get(fn)
                if meta:
                    e["license"] = meta["license"]
                    e["author"] = meta["author"]
                    e["source_url"] = meta["source_url"]

    # drop any still-pending without free license evidence (do not ship unclear rights)
    cleared = []
    for e in entries:
        if OK_LICENSE.search(str(e.get("license", ""))):
            cleared.append(e)
        else:
            print("dropping non-free/unknown license:", e["filename"], e.get("license"))
            (ds / e["filename"]).unlink(missing_ok=True)
    entries = renumber(ds, cleared)

    save_manifest(entries)
    n_cap = write_captions(ds, entries)
    sheet = contact_sheet(ds, entries)
    report = quality_report(entries, ds)
    out = Path("eval_outputs")
    out.mkdir(exist_ok=True)
    (out / "dataset_quality_report.json").write_text(
        json.dumps(
            {
                "updated_at": datetime.now(UTC).isoformat(),
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
    print("report: eval_outputs/dataset_quality_report.json")
    print("NOT uploaded — review sheet first, then upload_to_hf + Colab train")
    return 0 if report["verdict"].startswith("PASS") else 2


if __name__ == "__main__":
    sys.exit(main())
