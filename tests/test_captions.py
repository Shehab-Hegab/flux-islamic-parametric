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
