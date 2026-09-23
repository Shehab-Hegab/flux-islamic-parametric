import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from download_or_synthetic_dataset import generate_synthetic_image, main  # noqa: E402


def test_synthetic_image_deterministic_and_valid():
    a = generate_synthetic_image(1)
    b = generate_synthetic_image(1)
    c = generate_synthetic_image(2)
    assert a.size[0] >= 1024 and a.size[1] >= 1024
    assert a.mode == "RGB"
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()


def test_builder_synthetic_25(tmp_path):
    rc = main(["--prefer", "synthetic", "--count", "25", "--out", str(tmp_path)])
    assert rc == 0
    images = sorted(tmp_path.glob("image_*.jpg"))
    assert len(images) == 25
    assert (tmp_path / "manifest.json").exists()
    from PIL import Image

    for p in images:
        with Image.open(p) as im:
            assert im.width >= 1024 and im.height >= 1024
