"""Fast, robust CIFAR-10 download.

Tries a fast mirror first (checksum-verified), then falls back to torchvision's
default download. The dataset is NOT stored in this repo: it is 163 MB, and
GitHub rejects files over 100 MB (25 MB for web uploads).
"""
import hashlib
import os
import socket
import tarfile
import urllib.request

from torchvision import datasets

MIRRORS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-python.tar.gz",
]
MD5 = "c58f30108f718f92721af3b95e74349a"  # md5 of the official cifar-10-python.tar.gz


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_cifar10(root="data"):
    """Make sure CIFAR-10 is available under `root`. Safe to call repeatedly."""
    if os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
        return
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, "cifar-10-python.tar.gz")

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)  # fail over instead of hanging on a stalled connection
    try:
        for url in MIRRORS:
            try:
                print(f"Downloading CIFAR-10 from {url} ...")
                urllib.request.urlretrieve(url, archive)
                if _md5(archive) != MD5:
                    raise ValueError("checksum mismatch")
                with tarfile.open(archive) as t:
                    try:
                        t.extractall(root, filter="data")  # Python 3.12+
                    except TypeError:
                        t.extractall(root)
                print("CIFAR-10 ready.")
                return
            except Exception as e:
                print(f"Mirror failed ({e}); trying next source...")
                if os.path.exists(archive):
                    os.remove(archive)
    finally:
        socket.setdefaulttimeout(old_timeout)

    print("Falling back to torchvision's default download (can be slow)...")
    datasets.CIFAR10(root, train=True, download=True)
