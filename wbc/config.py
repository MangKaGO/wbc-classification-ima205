"""Constantes partagées et accès aux fichiers du challenge."""

from pathlib import Path

import torch

DEFAULT_DATA_DIR = Path("IMA205-challenge")
OUTPUTS_DIR = Path("outputs")

NUM_CLASSES = 13
IMG_SIZE = 224
SEED = 42

# Statistiques ImageNet (les modèles sont pré-entraînés sur ImageNet uniquement).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DataPaths:
    """Chemins vers les fichiers du dataset IMA205."""

    def __init__(self, root: Path | str = DEFAULT_DATA_DIR):
        self.root = Path(root)
        self.train_dir = self.root / "train"
        self.test_dir = self.root / "test"
        self.train_meta = self.root / "train_metadata.csv"
        self.test_meta = self.root / "test_metadata.csv"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
