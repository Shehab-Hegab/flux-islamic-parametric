import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_eval import main, validate_eval_setup  # noqa: E402


def test_validate_eval_setup():
    setup = validate_eval_setup()
    assert len(setup["prompts"]) >= 5
    assert all("Islamic_Parametric style" in p for p in setup["prompts"])
    assert setup["seed"] == 42
    assert setup["resolution"] == 1024


def test_check_mode_exit_zero(capsys):
    rc = main(["--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "check OK" in out
    assert "grid" in out
