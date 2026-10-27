Course: Teknologi Berkembang, Assignment #1 (Progress Submission)
Group: 01
Members: 
Christine Rotua Deborah Hutabarat - 5026241028
Ahmad Farhad Mabrury - 5026241034
Thalia Harnum Fathina - 5026241206
Muhammad Satrio Adjie - 5026241065

Educational prototype. Upload a photo of ONE waste item; the app predicts
**plastic bottle, metal can, or paper**, shows a confidence score, and gives
general sorting guidance. It is not a general waste recognizer, and local
recycling rules may differ.

## How it works
- OpenCV: image decoding, validation, BGR->RGB conversion, resizing.
- PyTorch / torchvision: MobileNetV3-Small (ImageNet-pretrained, third-party
  weights) with its last layer replaced by a 3-class layer and fine-tuned by us.
- Streamlit: user interface.
- Predictions below a confidence threshold of 0.80 are rejected.

## Requirements
Windows, Python 3.13 (tested with 3.13.7), about 2 GB of disk space. CPU only; no GPU needed.

## Setup (run the app with the included model)
```powershell
cd waste-classifier
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
py check_env.py
py -m streamlit run app.py
```
The app opens at http://localhost:8501. If Streamlit asks for an email on first
launch, press Enter. The trained model (`models/waste_classifier.pt`) is
included, so no training or internet access is needed to run the app.
(Internet is only needed for `pip install`.)

Command-line prediction for one image:
```powershell
py -m src.predict path\to\image.jpg
```

## Reproducing the dataset, training and evaluation (optional)
1. Dataset: TrashNet (Thung and Yang, Stanford CS 229),
   https://github.com/garythung/trashnet . Download `data/dataset-resized.zip`
   from that repository and extract it so the class folders are directly in
   `data/raw/trashnet/` (cardboard, glass, metal, paper, plastic, trash).
   See `docs/DATA_SOURCES.md` for license and label mapping.
2. Run, in this order:
```powershell
py -m src.inspect_dataset                 # check classes and image files
py -m src.prepare_data                    # grouped split, seed 42, threshold 4
py -m src.visualize_samples               # sample figures
py -m src.train                           # about 3 min on CPU; downloads ImageNet weights once
py -m src.calibrate_threshold             # validation split only
py -m src.evaluate                        # test split; run once after fixing the threshold
```
Training can differ slightly between machines and library versions, so retrained
numbers may not match the table below exactly. The included `data/splits/split.csv`
is the split used for the reported results.

## Results (held-out test split, 213 images, produced by `src.evaluate`)
| Measure | Result |
|---|---|
| Accuracy (no rejection) | 206/213 = 96.7% (95% Wilson CI 93.4-98.4%) |
| F1: metal_can / paper / plastic_bottle | 0.958 / 0.982 / 0.957 |
| Threshold 0.80: accepted | 194 of 213; accuracy on accepted 97.9% (190/194) |
| Wrong predictions still accepted | 4 |
| Unsupported items (glass, cardboard, trash, 1041 images) rejected | 21.3% |

Full metrics are in `outputs/test_metrics.json` and `outputs/confusion_matrix.png`.

## Known limitations
- Only three categories. Other items are usually forced into one of them: only
  21.3% of glass/cardboard/trash images were rejected by the threshold.
- Confidence is the model's own score, not the probability of being correct.
  Several wrong predictions had confidence above 0.9.
- TrashNet photos have plain backgrounds; real-world photos were not evaluated.
- TrashNet "plastic" and "metal" are broader than bottles and cans.
- Near-duplicate grouping reduces but cannot remove train/test leakage between
  photos of the same physical object. Accuracy may be optimistic.
- Test set is small (59 / 85 / 69 images per class).

## Troubleshooting
| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `No module named 'src'` | Run commands from the project root and use `py -m src.<module>` |
| `File does not exist: app.py` | Run Streamlit from the project root |
| `Model file not found` | Make sure `models/waste_classifier.pt` exists, or run `py -m src.train` |
| pip cannot find a torch version | Use Python 3.12 or 3.13 and a fresh `.venv` |
| Port 8501 in use | `py -m streamlit run app.py --server.port 8502` |

## Credits and licenses
- Dataset: TrashNet, Gary Thung and Mindy Yang. License: <ISI: sesuai file LICENSE yang Anda baca>.
- Pretrained model: MobileNetV3-Small from torchvision (ImageNet weights).
- Libraries: OpenCV, PyTorch, torchvision, Streamlit, scikit-learn, pandas, NumPy, Matplotlib.
- Original work by the group: all code in `app.py` and `src/`, plus the split, training and evaluation procedure.
  <ISI: tambahkan sumber lain yang benar-benar dipakai, misalnya tutorial atau bantuan AI, sesuai Appendix B>

Version control: <ISI: URL repo GitHub hanya kalau sudah ada, atau tulis bahwa tidak dipakai>