import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curate_wikimedia_landmarks import (  # noqa: E402
    KNOWN_COMMONS_FILES,
    LANDMARK_QUERIES,
    MIN_LUMA,
    MIN_SIDE,
    OK_LICENSE,
    file_url,
    luma_of,
    main,
    strip_html,
)


def test_check_exit_zero(capsys):
    rc = main(["--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Wikimedia" in out
    assert "min_luma" in out


def test_known_files_cover_landmarks():
    blob = " ".join(f for _, f in KNOWN_COMMONS_FILES).lower()
    assert "bahar" in blob or "louvre" in blob
    assert "monde_arabe" in blob or "institut" in blob
    assert "mashrabiya" in blob
    assert "doha" in blob or "tower" in blob


def test_file_url_encodes_space():
    url = file_url("My File.jpg")
    assert "Special:FilePath" in url
    assert "My%20File.jpg" in url or "My File.jpg" in url


def test_landmark_queries_include_canonical_projects():
    blob = " ".join(q for _, q in LANDMARK_QUERIES).lower()
    assert "al bahar" in blob
    assert "louvre abu dhabi" in blob
    assert "institut du monde arabe" in blob or "monde arabe" in blob
    assert "mashrabiya" in blob


def test_license_allowlist():
    assert OK_LICENSE.search("CC BY-SA 4.0")
    assert OK_LICENSE.search("CC0")
    assert OK_LICENSE.search("Public domain")
    assert not OK_LICENSE.search("© All rights reserved")


def test_strip_html():
    assert strip_html("<a href='x'>Artist Name</a>") == "Artist Name"


def test_luma_of_black_and_white():
    from PIL import Image

    black = Image.new("RGB", (64, 64), (0, 0, 0))
    white = Image.new("RGB", (64, 64), (255, 255, 255))
    assert luma_of(black) < 10
    assert luma_of(white) > 200


def test_filter_constants():
    assert MIN_LUMA >= 50
    assert MIN_SIDE >= 1024
