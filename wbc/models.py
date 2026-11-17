"""Construction des modèles et boucles d'entraînement / inférence."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .config import NUM_CLASSES

ARCHITECTURES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    """ResNet pré-entraîné sur ImageNet avec une tête à NUM_CLASSES sorties."""
    constructor, weights = ARCHITECTURES[arch]
    model = constructor(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_model(arch: str, weights_path: Path, device: torch.device) -> nn.Module:
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device, desc="Train") -> float:
    """Une epoch d'entraînement ; renvoie la loss moyenne."""
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict_proba(model, loader: DataLoader, device, tta: bool = False) -> np.ndarray:
    """Probabilités softmax pour chaque image.

    Avec `tta=True`, moyenne (soft voting) des prédictions sur l'image originale,
    son miroir horizontal et son miroir vertical.
    """
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc="Inférence", leave=False):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)
        if tta:
            probs = probs + F.softmax(model(torch.flip(images, dims=[3])), dim=1)
            probs = probs + F.softmax(model(torch.flip(images, dims=[2])), dim=1)
            probs = probs / 3
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


def evaluate_macro_f1(model, loader: DataLoader, device) -> float:
    """Macro F1 sur un loader qui renvoie (image, label)."""
    y_true = np.asarray(loader.dataset.labels)
    y_pred = predict_proba(model, loader, device).argmax(axis=1)
    return f1_score(y_true, y_pred, average="macro")
