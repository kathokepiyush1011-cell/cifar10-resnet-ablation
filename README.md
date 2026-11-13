# CIFAR-10 Image Classification with a ResNet Built From Scratch: A Leak-Free Ablation Study

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kathokepiyush1011-cell/cifar10-resnet-ablation/blob/main/cifar10_resnet_ablation.ipynb)

## Overview
This project trains a small residual network (ResNet, ~0.7M parameters) from scratch in PyTorch to classify 32×32 color images into 10 classes. Beyond the final accuracy, the goal is to run the experiment correctly: a leak-free validation protocol, fixed seeds, and an ablation study that measures what each training ingredient actually contributes.

## Problem and Motivation
Image classification is a standard benchmark for studying how architecture and training choices affect generalization. Many projects report a single accuracy number with no evidence for *why* the model works. Here, each component (data augmentation, skip connections, label smoothing, weight decay) is removed one at a time and the change in accuracy is measured across multiple seeds.

## Dataset
- **CIFAR-10** (Krizhevsky, 2009): 60,000 color images of size 32×32 in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- Official split: 50,000 train / 10,000 test.
- **The dataset is not stored in this repo** (163 MB; GitHub blocks files over 100 MB). It downloads automatically on first run through `data_utils.py`, which tries a fast mirror and verifies the checksum before falling back to torchvision's default download.
- **Split used here:** the 50,000 training images are divided into 45,000 train and 5,000 validation (stratified by class, fixed seed). The 10,000 test images are used **once per run**, only after selecting the best checkpoint on validation.
- Preprocessing: per-channel mean/std normalization. Augmentation (train only): random crop with 4-pixel padding and random horizontal flip. Validation and test images are never augmented.

## Method

### Model architecture
```
Input 3×32×32
  → Stem: Conv3×3 + BatchNorm + ReLU            → 32×32×32
  → Stage 1: 2 residual blocks (32 channels)    → 32×32×32
  → Stage 2: 2 residual blocks (64 channels)    → 64×16×16  (stride-2 downsample)
  → Stage 3: 2 residual blocks (128 channels)   → 128×8×8   (stride-2 downsample)
  → Global average pooling                      → 128
  → Linear                                      → 10 logits
```
Each residual block computes `y = ReLU(F(x) + x)`, where `F` is Conv-BN-ReLU-Conv-BN. The identity term gives gradients a direct path through the network, since ∂y/∂x = ∂F/∂x + I. A 1×1 convolution shortcut is used where the shape changes. Weights are initialized with Kaiming normal initialization.

### Training setup
| Setting | Value |
|---|---|
| Optimizer | SGD, Nesterov momentum 0.9 |
| LR schedule | OneCycle, peak LR 0.1 |
| Weight decay | 5e-4 |
| Batch size | 128 |
| Loss | Cross-entropy with label smoothing 0.1 |
| Precision | Mixed precision (AMP) on GPU |
| Baseline epochs | 10 |
| Ablation epochs / seeds | 5 epochs, 2 seeds |
| Hardware | Google Colab, NVIDIA Tesla T4 GPU |

### Evaluation protocol
1. Train; after each epoch compute validation accuracy.
2. Keep the weights with the best validation accuracy.
3. Evaluate that checkpoint on the test set once.
4. Model decisions are based on validation accuracy only. Test accuracy is reported, not tuned on.

## Results

### Baseline
| Metric | Value |
|---|---|
| Best validation accuracy | 88.62% |
| Test accuracy | 88.07% |

![Baseline plots](results/baseline_plots.png)

The left plot shows training/validation loss, the middle shows validation accuracy per epoch, and the right shows the test confusion matrix. Note that training loss is computed on augmented, label-smoothed batches while the model is still changing, so it is not directly comparable to validation loss.

Per-class test accuracy (baseline, worst first):

| Class | Accuracy |
|---|---|
| cat | 73.8% |
| dog | 82.6% |
| bird | 82.8% |
| frog | 88.5% |
| airplane | 89.0% |
| deer | 90.2% |
| horse | 90.8% |
| truck | 92.8% |
| ship | 94.5% |
| automobile | 95.7% |

### Ablation study
Each configuration removes exactly one component from the baseline. Values are mean ± std over 2 seeds, with 5 epochs per run. Because of the shorter schedule, the baseline row here (83.21% val) is lower than the 10-epoch baseline above (88.62% val).

| Configuration | Val acc (%) | Test acc (%) | Δ val vs baseline |
|---|---|---|---|
| Baseline | 83.21 ± 0.27 | 83.59 ± 0.35 | — |
| No augmentation | 84.66 ± 0.65 | 84.08 ± 0.16 | +1.45 |
| No residual connections | 83.65 ± 0.61 | 83.48 ± 0.18 | +0.44 |
| No label smoothing | 83.52 ± 0.06 | 83.37 ± 0.07 | +0.31 |
| No weight decay | 82.38 ± 0.40 | 81.93 ± 0.74 | −0.83 |

![Ablation](results/ablation.png)

### Sample predictions
![Predictions](results/predictions.png)

### Analysis
- **Only weight decay clearly mattered.** Removing it was the only change that lowered accuracy on both splits: by 0.83 points on validation and 1.66 points on test.
- **Removing augmentation did not hurt here; it helped.** Validation accuracy rose by 1.45 points (test: +0.49). This contradicts the initial hypothesis that augmentation would help. A plausible explanation (not tested here) is that with only 5 epochs the model is still underfitting, so random crops and flips make the task harder before they can reduce overfitting. The expected benefit would likely appear with longer training; this study cannot confirm it.
- **Residual connections and label smoothing were within noise.** Their validation changes (+0.44 and +0.31) are no larger than the seed-to-seed spread (std up to 0.61), and their test changes are slightly negative (−0.11 and −0.22). With only 6 residual blocks, a plain CNN still trains well, so a small gap is expected; skip connections matter more as depth grows.
- **Where the model fails:** cats (73.8%) and dogs (82.6%) are the weakest classes, plausibly because they are visually similar at 32×32 resolution. Vehicles (automobile 95.7%, ship 94.5%) are the easiest.

## Limitations
- Ablations ran for only 5 epochs, when the model is still far from converged (83% vs 88.6% after 10 epochs), so conclusions may not hold for full-length training.
- Two seeds give only a rough estimate of run-to-run variance, not a statistical significance test. Differences under about 0.5 points should be treated as noise.
- Only one architecture size and one dataset were tested.
- Each ablation changes one factor at a time, so interactions between factors are not measured.

## Reproducing
1. Open the notebook in Google Colab via the badge above.
2. Runtime → Change runtime type → GPU.
3. Runtime → Run all.

## Use the trained model
The trained weights (`best.pt`, the baseline model) are included in this repo.

```bash
git clone https://github.com/kathokepiyush1011-cell/cifar10-resnet-ablation
cd cifar10-resnet-ablation
pip install -r requirements.txt
```

**Classify your own image**
```bash
python predict.py my_image.jpg
```

**Re-measure accuracy on the 10,000-image CIFAR-10 test set**
```bash
python evaluate.py
```
The result should match the test accuracy reported above to within a small rounding difference (training used mixed precision on GPU; `evaluate.py` may run in full precision).

**Launch the web demo**
```bash
python app.py
```

**Use it in Python**
```python
from PIL import Image
from predict import load_model, predict

model = load_model("best.pt")
print(predict(model, Image.open("my_image.jpg")))
```

> **Limitations:** the model only knows 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Any other image, such as a person or a laptop, is forced into one of those classes, often with high confidence. Images are shrunk to 32x32, so quality is lower than for models trained on large images.

## Repository structure
```
├── cifar10_resnet_ablation.ipynb   # full code + outputs (training, ablations)
├── best.pt                         # trained baseline weights
├── model.py                        # model definition
├── predict.py                      # classify your own images
├── evaluate.py                     # re-measure test accuracy
├── data_utils.py                   # automatic, checksum-verified CIFAR-10 download
├── app.py                          # web demo (Gradio)
├── requirements.txt
├── results/
│   ├── baseline_plots.png
│   ├── ablation.png
│   ├── predictions.png
│   └── ablation_summary.csv
├── .gitignore
├── README.md
└── LICENSE
```

## Future work
Deeper/wider networks, MixUp or CutMix, weight EMA, test-time augmentation, ensembling across seeds, and a depth experiment comparing plain vs residual networks at 18+ blocks.

## References
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
- Smith & Topin, *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*, 2017.
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, CVPR 2016 (label smoothing).

## License
MIT
