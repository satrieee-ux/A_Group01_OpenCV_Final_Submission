# Data Sources

## TrashNet
- Authors: Gary Thung and Mindy Yang (Stanford CS 229 project)
- Source: https://github.com/garythung/trashnet
- File used: data/dataset-resized.zip (downloaded on: <ISI TANGGAL>)
- License: <ISI SETELAH MEMBACA FILE LICENSE DI REPO>
- Citation requested by the authors: cite the repository above.

## Label mapping used in this project
| TrashNet folder | Project label | Note |
|---|---|---|
| plastic | plastic_bottle | TrashNet "plastic" may include items other than bottles |
| metal | metal_can | TrashNet "metal" may include items other than cans |
| paper | paper | |
| glass, cardboard, trash | not used for training | kept for out-of-scope testing |

## Not included in the source code archive
The dataset itself. Download it using the instructions in README.md.

## NexTech14_Waste_Dataset (Phase 2)
- Source: Mendeley Data, https://data.mendeley.com/datasets/p6hgv27jh4/1
- DOI: 10.17632/p6hgv27jh4.1 (Version 1, published 10 November 2025)
- Contributors as listed on the page: Md Abdus Salam, Farhana Akter, Abu Raihan,
  Tanzin Amamed Shohag, Jahanara Pervin Juthy (Khwaja Yunus Ali University)
- License: CC BY-NC 4.0 (non-commercial; verified on the dataset page on <11/11/2025>)
- Downloaded on: <27/10/2026>
- Local use: coursework only. The dataset is NOT redistributed in the source archive.
- Note: the dataset page says its own train/val/test split was made after
  augmentation. We do not use that split; see split_v2 procedure in the report.
- Models trained on this data are treated as non-commercial as well.