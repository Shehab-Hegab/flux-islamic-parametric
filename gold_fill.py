"""Fill gold dataset to 25-28 with balanced facade/detail/interior from Openverse."""

from __future__ import annotations

import json
import sys
from collections import Counter
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

TARGET = 27
WANT = {"facade": 11, "detail": 8, "interior": 8}

DETAIL_Q = [
    "mashrabiya detail",
    "mashrabiya window",
    "mashrabiya screen",
    "muqarnas detail",
    "muqarnas ceiling",
    "geometric screen close",
    "islamic geometric window",
    "jali screen",
    "carved wooden screen islamic",
    "lattice window shadow",
    "ornamental islamic screen",
    "arabesque detail",
    "zellige detail",
    "girih pattern",
    "arched mashrabiya",
    "wooden lattice islamic",
    "geometric tile detail mosque",
    "stone carving islamic geometric",
]
INTERIOR_Q = [
    "mosque interior",
    "prayer hall interior",
    "mosque interior columns",
    "islamic interior arches",
    "courtyard arcade",
    "riad courtyard interior",
    "madrasa courtyard interior",
    "arched interior hall",
    "mosque interior daylight",
    "museum islamic interior",
    "hypostyle interior",
    "colonnade courtyard",
]
FACADE_Q = [
    "mosque facade",
    "modern mosque exterior",
    "islamic architecture exterior",
    "mosque entrance portal",
    "geometric building facade",
    "islamic cultural center",
    "madrasa facade",
    "dome mosque exterior",
    "minaret mosque",
    "arched facade daylight",
]


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


def load_disk(ds: Path) -> list[dict]:
    prior: dict[str, dict] = {}
    mp = ds / "manifest.json"
    if mp.exists():
        for e in json.loads(mp.read_text(encoding="utf-8")):
            prior[e.get("filename", "")] = e
    from PIL import Image

    out = []
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
        e["category"] = classify_category(e)
        out.append(e)
    return out


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


def filter_cands(items, used_pages, used_urls):
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
        )
        if not any(s in blob for s in soft):
            continue
        cands.append((it, lic, title, q))
    return cands


def main() -> int:
    ds = Path(DATASET_DIR)
    entries = load_disk(ds)
    # drop blacklist remaining
    kept = []
    for e in entries:
        if is_blacklisted(e):
            (ds / e["filename"]).unlink(missing_ok=True)
            print("purge", e["filename"], str(e.get("title"))[:60], flush=True)
            continue
        kept.append(e)
    entries = kept

    md5s, phs, used_pages, used_urls = set(), set(), set(), set()
    for e in entries:
        p = ds / e["filename"]
        if p.exists():
            from PIL import Image

            im = Image.open(p).convert("RGB")
            md5s.add(thumb_hash(im))
            phs.add(phash(im))
            used_pages.add(str(e.get("page_url") or "").lower())
            used_urls.add(str(e.get("source_url") or "").lower())

    def counts(es):
        c = Counter(e.get("category") or classify_category(e) for e in es)
        for k in WANT:
            c.setdefault(k, 0)
        return c

    def stop(c):
        return all(c[k] >= WANT[k] for k in WANT) and len(entries) >= 25

    def collect(queries):
        items = []
        for q in queries:
            for it in fetch_query(q):
                it["_q"] = q
                items.append(it)
        return items

    def topup(queries, label, need_cat=None):
        nonlocal entries, md5s, phs, used_pages, used_urls
        c = counts(entries)
        if need_cat and c[need_cat] >= WANT[need_cat]:
            return
        if len(entries) >= 30:
            return
        cands = filter_cands(collect(queries), used_pages, used_urls)
        print(f"topup {label}: {len(cands)}", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(try_dl, t) for t in cands[:80]]
            for fut in as_completed(futs):
                c = counts(entries)
                if stop(c) or len(entries) >= 30:
                    break
                if need_cat and c[need_cat] >= WANT[need_cat]:
                    break
                res = fut.result()
                if not res:
                    continue
                if res["h"] in md5s or res["ph"] in phs:
                    continue
                # force category if needed
                fake = {"title": res["title"], "query": res["q"], "source_url": res["url"], "filename": "x"}
                cat = classify_category(fake)
                if need_cat and cat != need_cat:
                    # allow if query strongly matches
                    if need_cat not in res["q"].lower() and need_cat not in ("mashrab" if "mashrab" in res["q"] else ""):
                        # for detail query but classified facade, still accept if detail words in title
                        tl = res["title"].lower()
                        if need_cat == "detail" and not any(x in tl for x in ("mashrab", "muqarnas", "screen", "window", "lattice", "detail", "carved", "tile", "pattern", "ornament", "ceiling", "jali")):
                            continue
                        if need_cat == "interior" and "interior" not in res["q"].lower() and not any(
                            x in tl for x in ("interior", "hall", "prayer", "courtyard", "colonnade", "atrium", "riad")
                        ):
                            continue
                        if need_cat == "facade" and not any(
                            x in (res["q"] + " " + tl).lower() for x in ("facade", "exterior", "entrance", "mosque", "building", "minaret", "portal")
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
                }
                entry["category"] = classify_category(entry)
                entries.append(entry)
                used_pages.add(entry["page_url"].lower())
                used_urls.add(entry["source_url"].lower())
                print("kept", name, entry["category"], entry["luma"], res["title"][:50], flush=True)

    print("start", counts(entries), "n", len(entries), flush=True)
    # fill weakest first
    for _ in range(4):
        c = counts(entries)
        if stop(c) or len(entries) >= TARGET:
            break
        # pick lowest
        weakest = min(WANT, key=lambda k: c[k] / WANT[k])
        if c[weakest] >= WANT[weakest] and len(entries) >= 25:
            break
        if weakest == "detail":
            topup(DETAIL_Q, "detail", "detail")
        elif weakest == "interior":
            topup(INTERIOR_Q, "interior", "interior")
        else:
            topup(FACADE_Q, "facade", "facade")
        print("progress", counts(entries), len(entries), flush=True)

    # drop non-free
    final = []
    for e in entries:
        if OK_LICENSE.search(str(e.get("license", ""))) or (e.get("source") == "openverse" and e.get("license")):
            final.append(e)
        else:
            (ds / e["filename"]).unlink(missing_ok=True)

    # balanced selection to TARGET
    final.sort(key=lambda e: (-(e.get("luma") or 0), e["filename"]))
    selected = []
    used = set()
    c = Counter()
    for e in final:
        cat = classify_category(e)
        if c[cat] >= WANT.get(cat, 99):
            continue
        if len(selected) >= TARGET:
            break
        selected.append(e)
        used.add(e["filename"])
        c[cat] += 1
    for e in final:
        if len(selected) >= 25:
            break
        if e["filename"] in used:
            continue
        selected.append(e)
        used.add(e["filename"])
        c[classify_category(e)] += 1

    keep = {e["filename"] for e in selected}
    for p in ds.glob("image_*.jpg"):
        if p.name not in keep:
            p.unlink(missing_ok=True)

    entries = renumber(ds, selected)
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
    print("final cats", counts(entries), "captions", n)
    for e in entries:
        print(e["filename"], classify_category(e), e.get("luma"), str(e.get("title"))[:65])
    return 0 if str(report["verdict"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
