# Cat vs Dog vs Panda — Image Classification using Transfer Learning

**Name:** KAVIBHARATHI K  
**Reg No:** 212224220045

---

## Overview

This project builds a multi-class image classifier to distinguish between **cats**, **dogs**, and **pandas** using **Transfer Learning** with a pre-trained **ResNet18** model in PyTorch.

Key highlights:
- Pre-trained ResNet18 backbone (frozen convolutional layers)
- Custom classifier head: FC(256) → ReLU → Dropout(0.5) → FC(3)
- GPU/CUDA support
- Training curves, confusion matrix, and sample prediction plots
- Bonus: single-image classification function

---

## Dataset

- **Source:** [Kaggle — Cats vs Dogs vs Pandas](https://www.kaggle.com/datasets/gpiosenka/cats-dogs-pandas-images)
- 3 classes: `cat`, `dog`, `panda`
- Structure expected:
```
data/
  train/
    cat/
    dog/
    panda/
  test/
    cat/
    dog/
    panda/
```

---

## CUDA Check

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

---

## Setup Instructions

### Option A — Run on Kaggle (Recommended)

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook.
2. Add the dataset: **Datasets → Search → `cats-dogs-pandas-images`**.
3. Under **Settings → Accelerator → GPU**, enable the GPU.
4. Upload `notebooks/cat_dog_panda_transfer_learning.ipynb` and run all cells.

### Option B — Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/cat-dog-panda-classifier.git
   cd cat-dog-panda-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Kaggle API credentials:
   - Download `kaggle.json` from your Kaggle account → Settings → API.
   - Place it at `~/.kaggle/kaggle.json`.
   - Run inside the notebook:
     ```bash
     kaggle datasets download -d gpiosenka/cats-dogs-pandas-images -p ./data --unzip
     ```

4. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook notebooks/cat_dog_panda_transfer_learning.ipynb
   ```

---

## Model Architecture

| Layer         | Details                        |
|---------------|-------------------------------|
| Backbone      | ResNet18 (pretrained, frozen) |
| FC Layer 1    | 512 → 256, ReLU               |
| Dropout       | p = 0.5                       |
| FC Layer 2    | 256 → 3 (output)              |
| Loss          | CrossEntropyLoss              |
| Optimizer     | Adam (lr=0.001)               |
| Scheduler     | StepLR (step=5, γ=0.5)        |
| Epochs        | 10                            |

---

## Results

- **Test Loss** and **Test Accuracy** are printed after training.
- A **confusion matrix** and **example prediction grid** are displayed.
- Best model checkpoint saved as `best_model.pth`.

---

## Repository Structure

```
cat-dog-panda-classifier/
├── notebooks/
│   └── cat_dog_panda_transfer_learning.ipynb
├── requirements.txt
└── README.md
```
