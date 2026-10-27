"""Build the source code archive. Usage: py build_zip.py GroupXX"""
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
group = sys.argv[1] if len(sys.argv) > 1 else "GroupXX"
zip_path = root / f"{group}_OpenCV_Source_Code.zip"

files = ["app.py", "check_env.py", "requirements.txt", "make_requirements.py",
         "README.md", ".gitignore", "docs/DATA_SOURCES.md",
         "models/waste_classifier.pt", "data/splits/split.csv"]
files += [p.relative_to(root).as_posix() for p in sorted((root / "src").glob("*.py"))]
files += [f"outputs/{n}" for n in [
    "dataset_summary.csv", "split_summary.csv", "threshold_calibration.csv",
    "training_history.csv", "training_config.json", "training_curves.png",
    "test_metrics.json", "test_predictions.csv", "confusion_matrix.png",
    "misclassified_examples.png", "out_of_scope_results.csv",
    "sample_grid.png", "preprocessing_example.png"]]

missing = [f for f in files if not (root / f).exists()]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if f not in missing:
            z.write(root / f, f"waste-classifier/{f}")
print(f"Created {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB), "
      f"{len(files) - len(missing)} files")
for m in missing:
    print("MISSING (not included):", m)
