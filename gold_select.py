"""Reselect a balanced gold subset: purge bad titles, classify, pick 40/30/30."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finalize_dataset import (  # noqa: E402
    DATASET_DIR,
    OK_LICENSE,
    contact_sheet,
    luma,
    renumber,
    save_manifest,
)
from gold_dataset import (  # noqa: E402
    classify_category,
    gold_report,
    is_blacklisted,
    write_gold_captions,
)

TARGET = 28
# Desired mix for 28: ~11 facade, ~8 detail, ~9 interior
WANT = {"facade": 11, "detail": 8, "interior": 9}


def load_all(ds: Path) -> list[dict]:
    prior: dict[str, dict] = {}
    mp = ds / "manifest.json"
    if mp.exists():
        for e in json.loads(mp.read_text(encoding="utf-8")):
            prior[e.get("filename", "")] = e
    entries = []
    for p in sorted(ds.glob("image_*.jpg")):
        from PIL import Image

        im = Image.open(p).convert("RGB")
        e = dict(prior.get(p.name) or {})
        e.update({"filename": p.name, "width": im.size[0], "height": im.size[1], "luma": round(luma(im), 1)})
        e.setdefault("source", "unknown")
        e.setdefault("query", "unknown")
        e.setdefault("author", "")
        e.setdefault("license", "pending")
        e.setdefault("source_url", "")
        e.setdefault("title", e.get("commons_title") or "")
        entries.append(e)
    return entries


def main() -> int:
    ds = Path(DATASET_DIR)
    all_entries = load_all(ds)
    print("on disk", len(all_entries))

    bad = []
    good = []
    for e in all_entries:
        reasons = []
        if is_blacklisted(e):
            reasons.append("blacklist")
        p = ds / e["filename"]
        if not p.exists():
            reasons.append("missing")
        if reasons:
            bad.append((e, reasons))
            if p.exists():
                p.unlink(missing_ok=True)
            continue
        if not (
            OK_LICENSE.search(str(e.get("license", "")))
            or (e.get("source") == "openverse" and e.get("license"))
        ):
            bad.append((e, ["license"]))
            p.unlink(missing_ok=True)
            continue
        e["category"] = classify_category(e)
        good.append(e)

    for e, r in bad:
        print("drop", e["filename"], r, str(e.get("title") or e.get("commons_title"))[:60])
    print("good after purge", len(good), Counter(x["category"] for x in good))

    # Prefer higher luma within each category
    for cat in WANT:
        good = sorted(good, key=lambda e: (0 if e.get("category") == cat else 1, -(e.get("luma") or 0), e["filename"]))

    selected: list[dict] = []
    used = set()
    # first pass: fill quotas
    counts = {"facade": 0, "detail": 0, "interior": 0}
    for e in sorted(good, key=lambda x: (-(x.get("luma") or 0), x["filename"])):
        cat = e["category"]
        if cat not in counts:
            continue
        if counts[cat] >= WANT[cat]:
            continue
        if e["filename"] in used:
            continue
        selected.append(e)
        used.add(e["filename"])
        counts[cat] += 1
    # second pass: fill up to TARGET
    for e in sorted(good, key=lambda x: (-(x.get("luma") or 0), x["filename"])):
        if len(selected) >= TARGET:
            break
        if e["filename"] in used:
            continue
        selected.append(e)
        used.add(e["filename"])
        counts[e["category"]] = counts.get(e["category"], 0) + 1

    selected = sorted(selected, key=lambda e: e["filename"])
    print("selected", len(selected), Counter(e["category"] for e in selected))

    # delete files not selected
    keep_names = {e["filename"] for e in selected}
    for p in ds.glob("image_*.jpg"):
        if p.name not in keep_names:
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
    print("captions", n, "sheet", sheet)
    # list remaining titles for QA
    for e in entries:
        print(
            e["filename"],
            e.get("category") or classify_category(e),
            e.get("luma"),
            str(e.get("title") or e.get("commons_title") or "")[:70],
        )
    return 0 if str(report["verdict"]).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
