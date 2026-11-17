"""Chargement des métadonnées, datasets PyTorch et transformations."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SEED, DataPaths


def load_train_metadata(paths: DataPaths) -> pd.DataFrame:
    """Retourne un DataFrame avec les colonnes `ID` (nom de fichier) et `label`."""
    df = pd.read_csv(paths.train_meta)
    return df.iloc[:, :2].set_axis(["ID", "label"], axis=1)


def load_test_metadata(paths: DataPaths) -> pd.DataFrame:
    """Retourne un DataFrame avec la colonne `ID` des images de test."""
    df = pd.read_csv(paths.test_meta)
    return df.iloc[:, :1].set_axis(["ID"], axis=1)


def get_class_names(train_df: pd.DataFrame) -> list[str]:
    """Classes triées par ordre alphabétique : l'indice d'une classe est sa position."""
    return sorted(train_df["label"].unique())


def split_train_val(train_df: pd.DataFrame, val_size: float = 0.2):
    """Split stratifié 80/20 (même graine que dans le rapport)."""
    train_part, val_part = train_test_split(
        train_df, test_size=val_size, stratify=train_df["label"], random_state=SEED
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def balanced_class_weights(labels: np.ndarray, class_names: list[str]) -> torch.Tensor:
    """Poids w_c ∝ 1 / n_c pour la CrossEntropyLoss pondérée."""
    weights = compute_class_weight("balanced", classes=np.array(class_names), y=labels)
    return torch.tensor(weights, dtype=torch.float)


def build_transforms(augment: str = "none") -> transforms.Compose:
    """Transformations d'images.

    augment:
        "none"   : resize + normalisation (validation / test)
        "light"  : flip horizontal + rotation 20° (ResNet18 V1)
        "strong" : flips H+V, rotation 45°, ColorJitter (ResNet18 V2, ResNet50)
    """
    steps: list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment == "light":
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(20)]
    elif augment == "strong":
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif augment != "none":
        raise ValueError(f"Augmentation inconnue : {augment!r}")
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class WBCDataset(Dataset):
    """Images de leucocytes. Renvoie (image, label) si `class_names` est fourni
    et que le DataFrame contient une colonne `label`, sinon l'image seule."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform, class_names=None):
        self.ids = df["ID"].tolist()
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.labels = None
        if class_names is not None and "label" in df.columns:
            label_to_idx = {name: i for i, name in enumerate(class_names)}
            self.labels = [label_to_idx[name] for name in df["label"]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = Image.open(self.img_dir / self.ids[idx]).convert("RGB")
        image = self.transform(image)
        if self.labels is None:
            return image
        return image, self.labels[idx]
