"""Fast integrity checks for the repository's runnable experiments."""

import ast
import json
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent


def notebook_source(path):
    notebook = json.loads(path.read_text())
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            ast.parse(source)
        sources.append(source)
    return "\n".join(sources)


def main():
    for path in (ROOT / "EnhancedSentimentwithSarcasm").glob("*.py"):
        ast.parse(path.read_text())

    final_source = notebook_source(ROOT / "final_project" / "Final_project.ipynb")
    arem_source = notebook_source(ROOT / "AReM-Analysis" / "MainAnalysis.ipynb")

    assert "pos.zip', label=1" in final_source
    assert "validation_data=(testing_x" not in final_source
    assert "tf.keras.utils.set_random_seed(42)" in final_source
    assert "classifier.fit(x[features]" not in arem_source
    assert "oversample=False" not in arem_source
    assert "if oversample:" not in arem_source
    assert "new_testdata = load_data(testfiles" not in arem_source.split(
        "def classifyirr", 1
    )[1].split("def roc_plot", 1)[0]

    for label in ("neg", "pos"):
        with ZipFile(ROOT / "final_project" / "Data" / f"{label}.zip") as archive:
            reviews = [name for name in archive.namelist() if name.endswith(".txt")]
        assert len(reviews) == 1000

    print("repository integrity checks passed")


if __name__ == "__main__":
    main()
