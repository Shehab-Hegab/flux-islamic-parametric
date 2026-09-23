import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upload_to_hf import _model_card, main, plan_uploads  # noqa: E402

from islamic_parametric.constants import (  # noqa: E402
    DATASET_REPO_ID,
    LORA_REPO_ID,
    TRIGGER_WORD,
    WEIGHTS_FILENAME,
)


def test_dry_run(capsys, tmp_path):
    rc = main(["--dry-run", "--weights", str(tmp_path / "pytorch_lora_weights.safetensors")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert DATASET_REPO_ID in out
    assert LORA_REPO_ID in out
    assert WEIGHTS_FILENAME in out


def test_plan_uploads(tmp_path):
    plan = plan_uploads(tmp_path, tmp_path / "w.safetensors")
    assert plan[0]["repo_id"] == DATASET_REPO_ID
    assert plan[1]["repo_id"] == LORA_REPO_ID


def test_model_card_content():
    card = _model_card()
    assert TRIGGER_WORD in card
    assert DATASET_REPO_ID in card
    assert WEIGHTS_FILENAME in card
    for token in ["| Rank | 16 |", "| Alpha | 16 |", "1e-04", "| Max train steps | 800 |", "adamw8bit", "1024×1024"]:
        assert token in card, token


def test_readme_exists_and_compliant():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert TRIGGER_WORD in text
    assert DATASET_REPO_ID in text
    assert LORA_REPO_ID in text
    assert WEIGHTS_FILENAME in text
