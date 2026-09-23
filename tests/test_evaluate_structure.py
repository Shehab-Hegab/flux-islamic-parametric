import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_structure import (  # noqa: E402
    bilateral_symmetry,
    edge_density,
    evaluate_images,
    line_axis_ratio,
    main,
    periodicity,
    structure_metrics,
)


def test_check_mode_exit_zero(capsys):
    rc = main(["--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "check OK" in out
    assert "structural_metrics.json" in out


def test_metrics_on_synthetic_pair(tmp_path):
    import numpy as np
    from PIL import Image

    # perfect bilateral symmetry: left half mirrored
    arr = np.zeros((128, 128), dtype=np.uint8)
    half = np.random.default_rng(0).integers(0, 255, size=(128, 64), dtype=np.uint8)
    arr[:, :64] = half
    arr[:, 64:] = half[:, ::-1]
    sym = bilateral_symmetry(arr)
    assert sym > 0.95
    assert 0.0 <= edge_density(Image.fromarray(arr)) <= 1.0
    assert 0.0 <= line_axis_ratio(arr) <= 1.0
    assert 0.0 <= periodicity(arr) <= 1.0


def test_evaluate_images_writes_json(tmp_path):
    from PIL import Image

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (1024, 1024), (20, 20, 30)).save(img_dir / "a.jpg")
    Image.new("RGB", (1024, 1024), (200, 180, 40)).save(img_dir / "b.png")
    out = tmp_path / "eval"
    report = evaluate_images(img_dir, out)
    assert report["count"] == 2
    path = out / "structural_metrics.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "overall_structure" in data["means"]
    assert len(data["per_image"]) == 2
    m = structure_metrics(img_dir / "a.jpg")
    assert m["undersized"] is False
    assert set(m) >= {
        "file",
        "bilateral_symmetry",
        "edge_density",
        "line_axis_ratio",
        "periodicity",
        "overall_structure",
    }


def test_flat_image_low_structure():
    import numpy as np
    from PIL import Image

    flat = np.full((64, 64), 128, dtype=np.uint8)
    assert edge_density(Image.fromarray(flat)) < 0.15
    assert bilateral_symmetry(flat) > 0.99
