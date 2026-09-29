"""Structural checks for the saved recruiter-facing notebooks."""

import json
from pathlib import Path


def test_saved_notebooks_are_cleanly_executed():
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert [cell["execution_count"] for cell in code_cells] == list(
            range(1, len(code_cells) + 1)
        )
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors


def test_notebooks_do_not_modify_import_paths():
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "sys.path" not in source
        assert "from src." not in source


def test_backtest_notebook_contains_fixed_temporal_stability_design():
    notebook = json.loads(
        Path("notebooks/02_signal_and_backtest.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'STABILITY_SPLIT = pd.Timestamp("2010-01-01")' in source
    assert "regime_stability(primary_perf.port_ret" in source
    assert "regime_stability(rank_perf.port_ret" in source
    assert "regime_stability(month_perf.port_ret" not in source
    assert "rolling_hac_mean(primary_perf.port_ret" in source
    assert "rolling_hac_mean(rank_perf.port_ret" in source
