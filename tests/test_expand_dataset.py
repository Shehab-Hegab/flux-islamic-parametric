import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expand_dataset import (  # noqa: E402
    LICENSE_WHITELIST,
    caption_path_for,
    load_manifest,
    main,
    next_image_stem,
)


def test_check_mode_exit_zero(capsys):
    rc = main(["--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "check OK" in out
    assert "cc0" in out


def test_license_whitelist_has_public_domain_options():
    assert "cc0" in LICENSE_WHITELIST
    assert any("cc by" in x for x in LICENSE_WHITELIST)


def test_next_image_stem_increments(tmp_path):
    (tmp_path / "image_01.jpg").write_bytes(b"x")
    (tmp_path / "image_07.jpg").write_bytes(b"x")
    assert next_image_stem(tmp_path) == "image_08"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert next_image_stem(empty) == "image_01"


def test_caption_contains_trigger(tmp_path):
    from islamic_parametric.constants import TRIGGER_WORD

    path = caption_path_for(tmp_path, "image_01", "a mashrabiya courtyard screen")
    text = path.read_text(encoding="utf-8")
    assert TRIGGER_WORD in text
    assert text.endswith("no visual distortion.")


def test_load_manifest_missing_returns_empty(tmp_path):
    data = load_manifest(tmp_path / "nope.json")
    assert data == {"images": []}
