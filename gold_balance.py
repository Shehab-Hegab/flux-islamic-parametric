"""One-shot gold balance pass: purge bad titles, top-up facade/interior, rewrite captions."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests

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
    GOLD_QUERIES,
    OPENVERSE,
    UA,
    acceptable_gold,
    classify_category,
    download_bytes,
    gold_report,
    is_blacklisted,
    mean_saturation,
    normalize_license,
    title_blob,
    write_gold_captions,
)

FACADE_Q = [
    "mosque exterior facade",
    "modern mosque building",
    "islamic center building",
    "mosque entrance",
    "mosque courtyard exterior",
    "islamic cultural center",
    "geometric building facade",
    "modern islamic facade",
    "mosque dome exterior",
    "arched building facade daylight",
    "islamic architecture exterior",
    "mosque minaret building",
]
INTERIOR_Q = [
    "mosque interior",
    "prayer hall",
    "mosque interior light",
    "courtyard arcade interior",
    "islamic interior design",
    "mosque interior columns",
    "madrasa interior",
    "riad interior",
    "mosque interior daylight",
    "arched interior hall",
]


def fetch_query(q: str) -> list[dict]:
    try:
        r = requests.get(
            OPENVERSE,
            params={
                "q": q,
                "license_type": "commercial,modification",
                "page_size": 20,
                "mature": "false",
            },
            timeout=15,
            headers=UA,
        )
        if r.status_code != 200:
            return []
        return r.json().get("results") or []
    except Exception:
        return []


def load_disk_entries(ds: Path) -> list[dict]:
    prior: dict[str, dict] = {}
    mp = ds / "manifest.json"
    if mp.exists():
        for e in json.loads(mp.read_text(encoding="utf-8")):
            prior[e.get("filename", "")] = e
    entries: list[dict] = []
    for p in sorted(ds.glob("image_*.jpg")):
        from PIL import Image

        im = Image.open(p).convert("RGB")
        e = dict(prior.get(p.name) or {})
        e.update(
            {
                "filename": p.name,
                "width": im.size[0],
                "height": im.size[1],
                "luma": round(luma(im), 1),
            }
        )
        e.setdefault("source", "unknown")
        e.setdefault("query", "unknown")
        e.setdefault("author", "")
        e.setdefault("license", "pending")
        e.setdefault("source_url", "")
        e.setdefault("title", e.get("commons_title") or "")
        entries.append(e)
    return entries


def purge(ds: Path, entries: list[dict]) -> list[dict]:
    kept = []
    for e in entries:
        p = ds / e["filename"]
        reasons = []
        if is_blacklisted(e):
            reasons.append("blacklist")
        if p.exists() and luma(__import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB")) < 75:
            reasons.append("dark")
        if reasons:
            p.unlink(missing_ok=True)
            print("purge", e["filename"], reasons, title_blob(e)[:70], flush=True)
            continue
        kept.append(e)
    return kept


def filter_cands(all_items: list[dict], used_pages: set[str], used_urls: set[str]) -> list[tuple]:
    cands = []
    for it in all_items:
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
            "center",
            "centre",
        )
        if not any(s in blob for s in soft):
            continue
        cands.append((it, lic, title, q))
    return cands


def try_dl(tup: tuple) -> dict | None:
    it, lic, title, q = tup
    url = it.get("url") or ""
    im = download_bytes(url)
    if im is None:
        im = download_bytes(it.get("thumbnail") or "")
    if im is None:
        return None
    ok, _why = acceptable_gold(im)
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


def main() -> int:
    ds = Path(DATASET_DIR)
    entries = purge(ds, load_disk_entries(ds))
    print("after purge", len(entries), flush=True)

    md5s: set[str] = set()
    phs: set[str] = set()
    used_pages: set[str] = set()
    used_urls: set[str] = set()
    for e in entries:
        p = ds / e["filename"]
        if p.exists():
            from PIL import Image

            im = Image.open(p).convert("RGB")
            md5s.add(thumb_hash(im))
            phs.add(phash(im))
            used_pages.add(str(e.get("page_url") or "").lower())
            used_urls.add(str(e.get("source_url") or "").lower())

    def cats() -> dict[str, int]:
        c = {"facade": 0, "detail": 0, "interior": 0}
        for e in entries:
            k = classify_category(e)
            c[k] = c.get(k, 0) + 1
        return c

    def collect(queries: list[str]) -> list[dict]:
        items = []
        for q in queries:
            for it in fetch_query(q):
                it["_q"] = q
                items.append(it)
        return items

    def stop() -> bool:
        c = cats()
        return (
            len(entries) >= 25
            and c.get("facade", 0) >= 5
            and c.get("interior", 0) >= 5
            and c.get("detail", 0) >= 8
        )

    def topup(queries: list[str], label: str) -> None:
        nonlocal entries, md5s, phs, used_pages, used_urls
        if stop():
            return
        cands = filter_cands(collect(queries), used_pages, used_urls)
        print(f"topup {label}: {len(cands)} cands", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(try_dl, t) for t in cands[:90]]
            for fut in as_completed(futs):
                if stop():
                    break
                res = fut.result()
                if not res:
                    continue
                if res["h"] in md5s or res["ph"] in phs:
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
                }
                entries.append(entry)
                used_pages.add(entry["page_url"].lower())
                used_urls.add(entry["source_url"].lower())
                print(
                    "kept",
                    name,
                    entry["luma"],
                    classify_category(entry),
                    res["title"][:50],
                    flush=True,
                )

    print("start", cats(), "n", len(entries), flush=True)
    topup(FACADE_Q, "facade")
    print("after facade", cats(), len(entries), flush=True)
    topup(INTERIOR_Q, "interior")
    print("after interior", cats(), len(entries), flush=True)
    if cats().get("facade", 0) < 5:
        topup(["building facade geometric", "modern building exterior daylight", "cultural center building"], "facade2")
    if cats().get("interior", 0) < 5:
        topup(["hall interior arches", "museum interior arches", "colonnade interior"], "interior2")
    if len(entries) < 25:
        topup(GOLD_QUERIES, "fill")

    final = []
    for e in entries:
        if OK_LICENSE.search(str(e.get("license", ""))):
            final.append(e)
        elif e.get("source") == "openverse" and e.get("license"):
            final.append(e)
        else:
            print("drop", e["filename"], e.get("license"), flush=True)
            (ds / e["filename"]).unlink(missing_ok=True)

    rank = {"facade": 0, "interior": 1, "detail": 2}
    final = sorted(final, key=lambda e: (rank.get(classify_category(e), 3), e["filename"]))
    if len(final) > 30:
        for e in final[30:]:
            (ds / e["filename"]).unlink(missing_ok=True)
        final = final[:30]

    entries = renumber(ds, final)
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
    print("captions", n, "final", cats())
    return 0 if str(report["verdict"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
