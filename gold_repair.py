"""Repair pending licenses/metadata + ensure interior quota + final gold PASS."""

from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finalize_dataset import (  # noqa: E402
    DATASET_DIR,
    IMAGE_PATTERN,
    OK_LICENSE,
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
    is_blacklisted,
    mean_saturation,
    normalize_license,
    write_gold_captions,
)

DETAIL_Q = [
    "mashrabiya",
    "mashrabiya screen",
    "muqarnas",
    "muqarnas ceiling",
    "islamic geometric window",
    "jali screen",
    "carved wooden screen islamic",
    "lattice window shadow",
    "arabesque detail",
    "zellige detail",
    "girih pattern",
    "ornamental islamic screen",
]
INTERIOR_Q = [
    "mosque interior",
    "prayer hall interior",
    "mosque interior columns",
    "islamic interior arches",
    "courtyard arcade",
    "riad courtyard",
    "madrasa courtyard",
    "arched interior hall",
    "mosque interior daylight",
    "hypostyle interior",
    "colonnade courtyard",
    "mosque interior light",
]
WANT = {"facade": 11, "detail": 8, "interior": 6}
TARGET = 27


def fetch_query(q: str) -> list[dict]:
    try:
        r = requests.get(
            OPENVERSE,
            params={"q": q, "license_type": "commercial,modification", "page_size": 20, "mature": "false"},
            timeout=15,
            headers=UA,
        )
        if r.status_code != 200:
            return []
        return r.json().get("results") or []
    except Exception:
        return []


def try_dl(tup):
    it, lic, title, q = tup
    url = it.get("url") or ""
    im = download_bytes(url)
    if im is None:
        im = download_bytes(it.get("thumbnail") or "")
    if im is None:
        return None
    ok, _ = acceptable_gold(im)
    if not ok:
        return None
    return {
        "im": im,
        "h": thumb_hash(im),
        "ph": phash(im),
        "lic": lic,
        "title": title,
        "q": q,
        "it": it,
        "url": url,
    }


def collect(queries: list[str]) -> list[dict]:
    items = []
    for q in queries:
        for it in fetch_query(q):
            it["_q"] = q
            items.append(it)
    return items


def filter_cands(items, used_pages, used_urls, md5s, phs, want_cat=None):
    cands = []
    for it in items:
        title = str(it.get("title") or "")
        q = it["_q"]
        blob = (title + " " + q).lower()
        if any(k.lower() in blob for k in BLACKLIST_TITLES):
            continue
        page = str(it.get("foreign_landing_url") or "").lower()
        url = str(it.get("url") or "")
        if page in used_pages or url.lower() in used_urls:
            continue
        lic = normalize_license(it)
        if not lic:
            continue
        soft = (
            "mashrab",
            "muqarnas",
            "mosque",
            "islamic",
            "arabesque",
            "geometric",
            "lattice",
            "screen",
            "arch",
            "dome",
            "courtyard",
            "riad",
            "madrasa",
            "jali",
            "portal",
            "iwan",
            "tile",
            "zellige",
            "pattern",
            "window",
            "facade",
            "moslem",
            "muslim",
            "prayer",
            "hall",
            "minaret",
            "carved",
            "ornament",
            "colonnade",
            "hypostyle",
            "interior",
        )
        if not any(s in blob for s in soft):
            continue
        cands.append((it, lic, title, q))
    return cands


def classify_forced(title: str, query: str) -> str:
    e = {"title": title, "query": query, "filename": "x", "source_url": ""}
    return classify_category(e)


def main() -> int:
    ds = Path(DATASET_DIR)
    prior: dict[str, dict] = {}
    mp = ds / "manifest.json"
    if mp.exists():
        for e in json.loads(mp.read_text(encoding="utf-8")):
            prior[e.get("filename", "")] = e

    entries: list[dict] = []
    md5s: set[str] = set()
    phs: set[str] = set()
    used_pages: set[str] = set()
    used_urls: set[str] = set()
    for p in sorted(ds.glob("image_*.jpg")):
        im = Image.open(p).convert("RGB")
        e = dict(prior.get(p.name) or {})
        e.update({"filename": p.name, "width": im.size[0], "height": im.size[1], "luma": round(luma(im), 1)})
        e.setdefault("source", "unknown")
        e.setdefault("query", "unknown")
        e.setdefault("author", "")
        e.setdefault("license", "pending")
        e.setdefault("source_url", "")
        e.setdefault("title", e.get("commons_title") or "")
        if is_blacklisted(e):
            p.unlink(missing_ok=True)
            print("purge", p.name, e.get("title"), flush=True)
            continue
        md5s.add(thumb_hash(im))
        phs.add(phash(im))
        used_pages.add(str(e.get("page_url") or "").lower())
        used_urls.add(str(e.get("source_url") or "").lower())
        entries.append(e)

    print("loaded", len(entries), Counter(e.get("license") for e in entries), flush=True)

    # Drop any pending without license — will re-add with full meta
    cleaned = []
    for e in entries:
        if e.get("license") in ("pending", "unknown", "") and not OK_LICENSE.search(str(e.get("license", ""))):
            (ds / e["filename"]).unlink(missing_ok=True)
            print("drop pending", e["filename"], flush=True)
            continue
        cleaned.append(e)
    entries = cleaned

    def counts(es):
        c = Counter()
        for e in es:
            c[classify_category(e)] += 1
        for k in WANT:
            c.setdefault(k, 0)
        return c

    def need_more(cat: str) -> bool:
        return counts(entries)[cat] < WANT[cat] and len(entries) < TARGET

    def add_from(queries: list[str], label: str, want_cat: str) -> None:
        nonlocal entries, md5s, phs, used_pages, used_urls
        if not need_more(want_cat):
            return
        cands = filter_cands(collect(queries), used_pages, used_urls, md5s, phs, want_cat)
        print(f"add {label}: {len(cands)} cands, have", counts(entries), flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(try_dl, t) for t in cands[:70]]
            for fut in as_completed(futs):
                if not need_more(want_cat):
                    break
                res = fut.result()
                if not res:
                    continue
                if res["h"] in md5s or res["ph"] in phs:
                    continue
                cat = classify_forced(res["title"], res["q"])
                # force match by title/query for target category
                tl = (res["title"] + " " + res["q"]).lower()
                if want_cat == "detail" and not any(
                    x in tl for x in ("mashrab", "muqarnas", "muqarna", "screen", "window", "lattice", "detail", "carved", "tile", "pattern", "ornament", "ceiling", "jali", "zellige", "girih", "arabesque")
                ):
                    continue
                if want_cat == "interior" and not any(
                    x in tl for x in ("interior", "hall", "prayer", "courtyard", "colonnade", "atrium", "riad", "hypostyle", "mosque interior", "arched interior")
                ):
                    continue
                if want_cat == "facade" and cat != "facade" and not any(
                    x in tl for x in ("facade", "exterior", "entrance", "mosque", "building", "minaret", "portal")
                ):
                    continue
                md5s.add(res["h"])
                phs.add(res["ph"])
                n = 1
                while (ds / f"{IMAGE_PATTERN.format(n)}.jpg").exists():
                    n += 1
                name = IMAGE_PATTERN.format(n) + ".jpg"
                res["im"].save(ds / name, quality=92)
                it = res["it"]
                entry = {
                    "filename": name,
                    "source": "openverse",
                    "query": res["q"],
                    "width": res["im"].size[0],
                    "height": res["im"].size[1],
                    "author": str(it.get("creator") or "")[:120],
                    "license": res["lic"],
                    "source_url": res["url"],
                    "page_url": it.get("foreign_landing_url") or "",
                    "title": res["title"][:200],
                    "luma": round(luma(res["im"]), 1),
                    "saturation": round(mean_saturation(res["im"]), 1),
                    "category": want_cat,
                }
                entries.append(entry)
                used_pages.add(entry["page_url"].lower())
                used_urls.add(entry["source_url"].lower())
                print("kept", name, want_cat, entry["luma"], res["title"][:50], flush=True)

    # Fill missing categories
    add_from(INTERIOR_Q, "interior", "interior")
    add_from(DETAIL_Q, "detail", "detail")
    add_from(
        ["mosque facade", "islamic architecture exterior", "mosque entrance portal", "geometric building facade"],
        "facade",
        "facade",
    )

    # Ensure free licenses only
    final = []
    for e in entries:
        if OK_LICENSE.search(str(e.get("license", ""))):
            final.append(e)
        else:
            print("drop non-free", e["filename"], e.get("license"), flush=True)
            (ds / e["filename"]).unlink(missing_ok=True)

    # Balanced pick
    def cat_key(e):
        return classify_category(e)

    final.sort(key=lambda e: (-e.get("luma", 0), e["filename"]))
    selected: list[dict] = []
    used: set[str] = set()
    c = Counter()
    # first pass quotas
    for e in final:
        cat = cat_key(e)
        if c[cat] >= WANT.get(cat, 99):
            continue
        if len(selected) >= TARGET:
            break
        selected.append(e)
        used.add(e["filename"])
        c[cat] += 1
    # fill to min 25
    for e in final:
        if len(selected) >= 25:
            break
        if e["filename"] in used:
            continue
        selected.append(e)
        used.add(e["filename"])
        c[cat_key(e)] += 1

    # If still missing interior in selection, swap worst facade for best interior
    interior_pool = [e for e in final if cat_key(e) == "interior" and e["filename"] not in used]
    facade_sel = [e for e in selected if cat_key(e) == "facade"]
    while c["interior"] < 4 and interior_pool and len(facade_sel) > 8:
        add = interior_pool.pop(0)
        drop = facade_sel.pop()  # lowest priority among selected facades (last by luma order)
        selected = [e for e in selected if e["filename"] != drop["filename"]]
        (ds / drop["filename"]).unlink(missing_ok=True)
        selected.append(add)
        used.add(add["filename"])
        c["interior"] += 1
        c["facade"] -= 1

    keep = {e["filename"] for e in selected}
    for p in ds.glob("image_*.jpg"):
        if p.name not in keep:
            p.unlink(missing_ok=True)

    # force category field
    for e in selected:
        e["category"] = classify_category(e)

    entries = renumber(ds, selected)
    for e in entries:
        e["category"] = classify_category(e)
        # recover title from pre-renumber selected
    save_manifest(entries)
    n = write_gold_captions(ds, entries)
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
    print("cats", Counter(e["category"] for e in entries), "captions", n)
    for e in entries:
        print(e["filename"], e["category"], e.get("luma"), e.get("license"), str(e.get("title"))[:60])
    return 0 if str(report["verdict"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
