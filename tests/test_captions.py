import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_captions import main, template_caption  # noqa: E402

from islamic_parametric.constants import TRIGGER_WORD  # noqa: E402


def test_template_caption_spec():
    cap = template_caption("image_01.jpg", {"source": "synthetic", "query": "star8"}, 0)
    assert TRIGGER_WORD in cap
    assert "8k" in cap
    assert "no visual distortion" in cap


def test_generate_captions_template(tmp_path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    from PIL import Image

    for i in range(1, 4):
        Image.new("RGB", (1024, 1024), (30, 40, 50)).save(ds / f"image_{i:02d}.jpg")
    caps = tmp_path / "captions"
    rc = main(["--backend", "template", "--dataset", str(ds), "--captions", str(caps), "--expected", "3"])
    assert rc == 0
    files = sorted(caps.glob("*.txt"))
    assert len(files) == 3
    texts = [f.read_text(encoding="utf-8") for f in files]
    assert all(TRIGGER_WORD in t for t in texts)
    assert len(set(texts)) == 3


def test_gold_captions_no_junk_patterns():
    import re

    from gold_dataset import CORES_DETAIL, CORES_FACADE, CORES_INTERIOR, gold_caption

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
