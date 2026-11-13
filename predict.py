"""Classify your own images.
Usage:  python predict.py cat.jpg dog.png
"""
import argparse
import torch
from PIL import Image
from torchvision import transforms as T
from model import ResNet, CLASSES, MEAN, STD

TF = T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_model(ckpt="best.pt", device="cpu"):
    model = ResNet()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


def predict(model, img, topk=3, device="cpu"):
    """img: PIL image. Returns [(class_name, probability), ...] for the top-k classes."""
    x = TF(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0].cpu()
    p, idx = probs.topk(topk)
    return [(CLASSES[i], float(v)) for v, i in zip(p, idx)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="path(s) to image file(s)")
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    model = load_model(a.ckpt)
    for path in a.images:
        print(f"\n{path}")
        for name, p in predict(model, Image.open(path)):
            print(f"  {name:>11}: {p:.1%}")
