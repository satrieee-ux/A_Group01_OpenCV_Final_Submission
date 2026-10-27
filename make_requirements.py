"""Write requirements.txt pinned to the versions installed in this environment."""
import platform
from importlib.metadata import version

PACKAGES = ["opencv-python", "numpy", "pillow", "pandas", "scikit-learn",
            "matplotlib", "streamlit", "torch", "torchvision"]

lines = [f"# Tested with Python {platform.python_version()} on Windows (CPU only)"]
for name in PACKAGES:
    lines.append(f"{name}=={version(name).split('+')[0]}")  # '2.14.1+cpu' -> '2.14.1'

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))