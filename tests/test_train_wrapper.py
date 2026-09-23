import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_flux_lora import build_launch_command, main  # noqa: E402


def test_dry_run_exit_zero(capsys):
    rc = main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    for token in [
        "black-forest-labs/FLUX.1-dev",
        "--rank=16",
        "--network_alpha=16",
        "--learning_rate=1e-04",
        "--resolution=1024",
        "--max_train_steps=800",
        "adamw8bit",
        "Islamic_Parametric style",
        "--checkpointing_steps=250",
        "--checkpoints_total_limit=4",
    ]:
        assert token in out, token


def test_build_launch_command_includes_checkpointing():
    cmd = build_launch_command("dataset_islamic_parametric", "output")
    joined = " ".join(cmd)
    assert "--checkpointing_steps 250" in joined
    assert "--checkpoints_total_limit 4" in joined


def test_build_launch_command_shape(tmp_path):
    cmd = build_launch_command("dataset_islamic_parametric", "output")
    assert cmd[0] == "accelerate"
    assert "training/train_dreambooth_lora_flux.py" in cmd[2]
    joined = " ".join(cmd)
    assert "--mixed_precision bf16" in joined
    assert "--gradient_checkpointing" in joined


def test_notebook_valid_json():
    nb_path = Path(__file__).resolve().parents[1] / "Flux_Architectural_LoRA_Training.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 5
    import ast

    for cell in code_cells:
        src = "".join(cell["source"])
        stripped = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith(("!", "%"))
        )
        if stripped.strip():
            ast.parse(stripped)
