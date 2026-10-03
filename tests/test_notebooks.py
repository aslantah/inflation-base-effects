"""Structural checks for the saved recruiter-facing notebooks."""

import json
import re
from pathlib import Path

import nbformat


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


def test_research_notebook_schema_and_required_illustration():
    paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert len(paths) == 3
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    first = nbformat.read(paths[0], as_version=4)
    mechanical = next(
        cell
        for cell in first.cells
        if cell.cell_type == "code" and "fig, axes = plot_mechanical_effect()" in cell.source
    )
    assert any("image/png" in output.get("data", {}) for output in mechanical.outputs)


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


def test_research_narrative_and_local_links():
    paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert [path.name for path in paths] == [
        "01_data_and_mechanism.ipynb",
        "02_signal_and_backtest.ipynb",
        "03_sensitivity_and_measurement.ipynb",
    ]
    documents = [(Path("README.md"), Path("README.md").read_text())]
    for path, section_count in zip(paths, [5, 8, 6], strict=True):
        notebook = nbformat.read(path, as_version=4)
        prose = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        assert "I " in prose
        assert re.findall(r"^## (\d+)\.", prose, flags=re.MULTILINE) == [
            str(number) for number in range(1, section_count + 1)
        ]
        documents.append((path, prose))
        visible = "\n".join(cell.source for cell in notebook.cells)
        visible += json.dumps([cell.get("outputs", []) for cell in notebook.cells])
        for phrase in [
            "inherited",
            "recovered study",
            "reconstruction",
            "preceding public",
            "previous_duration",
            "corrected_duration",
            "duration_bridge",
            "legacy_",
            "scans/",
            "pre-existing",
            "Computed conclusion",
        ]:
            assert phrase not in visible, (path, phrase)
    for path, text in documents:
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in link and not link.startswith("#"):
                assert (path.parent / link.split("#")[0]).exists(), (path, link)
