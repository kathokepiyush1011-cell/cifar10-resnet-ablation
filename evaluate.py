"""Re-measure the model on the official CIFAR-10 test set (10,000 images).
Usage:  python evaluate.py
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from model import ResNet, CLASSES, MEAN, STD

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--data", default="data")
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet().to(device)
model.load_state_dict(torch.load(a.ckpt, map_location=device))
model.eval()

test = datasets.CIFAR10(a.data, train=False, download=True,
                        transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]))
loader = DataLoader(test, batch_size=256, shuffle=False)

preds, labels = [], []
with torch.no_grad():
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()

print(f"Test accuracy: {(preds == labels).mean():.2%}  ({len(labels)} images)")
print("\nPer-class accuracy:")
for c, name in enumerate(CLASSES):
    m = labels == c
    print(f"  {name:>11}: {(preds[m] == c).mean():.1%}")
