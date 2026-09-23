import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gold_dataset import CORES_DETAIL, CORES_FACADE, CORES_INTERIOR, gold_caption  # noqa: E402

from islamic_parametric.constants import TRIGGER_WORD, build_caption  # noqa: E402


def test_build_caption_spec():
    cap = build_caption("a modern mashrabiya facade")
    assert TRIGGER_WORD in cap
    assert "8k" in cap
    assert "no visual distortion" in cap


def test_gold_captions_no_junk_patterns():
    entries = [
        {"filename": "image_01.jpg", "query": "modern mashrabiya facade", "title": "x"},
        {"filename": "image_02.jpg", "query": "mashrabiya screen", "title": "y"},
        {"filename": "image_03.jpg", "query": "interior courtyard", "title": "z"},
    ]
    assert CORES_FACADE and CORES_DETAIL and CORES_INTERIOR
    texts = [gold_caption(e, i) for i, e in enumerate(entries)]
    for t in texts:
        assert TRIGGER_WORD in t
        assert "no visual distortion" in t
        assert not re.search(r"featuring image \d+", t, re.I)
        assert not re.search(r"\.jpg", t, re.I)
    assert len(set(texts)) == len(texts)


def test_live_dataset_captions_match_contract():
    root = Path(__file__).resolve().parents[1]
    caps = sorted((root / "dataset_islamic_parametric" / "captions").glob("image_*.txt"))
    assert len(caps) >= 20
    texts = [p.read_text(encoding="utf-8").strip() for p in caps]
    assert all(TRIGGER_WORD in t for t in texts)
    assert all("no visual distortion" in t for t in texts)
    assert len(set(texts)) == len(texts)
    assert not any(re.search(r"featuring image \d+", t, re.I) for t in texts)
