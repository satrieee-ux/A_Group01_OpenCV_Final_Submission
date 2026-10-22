"""Stage 1 environment check. Run from the project root."""
import platform
import sys


def main() -> None:
    print(f"Python   : {sys.version.split()[0]} ({platform.system()} {platform.machine()})")

    import cv2
    import numpy as np
    print(f"OpenCV   : {cv2.__version__}")
    print(f"NumPy    : {np.__version__}")

    import torch
    import torchvision
    print(f"PyTorch  : {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA GPU available: {torch.cuda.is_available()}")

    import streamlit
    import sklearn
    import pandas
    import matplotlib
    print(f"Streamlit: {streamlit.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"pandas   : {pandas.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")

    # Functional OpenCV test: encode and decode an in-memory image.
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # red in BGR
    ok, buf = cv2.imencode(".png", img)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert ok and decoded is not None and decoded.shape == (64, 64, 3)
    print("OpenCV encode/decode test: OK")

    # Functional PyTorch test.
    x = torch.rand(2, 3)
    print(f"PyTorch tensor test: OK {tuple(x.shape)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()