"""Model construction and training / inference loops."""

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
    """ImageNet-pretrained ResNet with a NUM_CLASSES-output head."""
    constructor, weights = ARCHITECTURES[arch]
    model = constructor(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_model(arch: str, weights_path: Path, device: torch.device) -> nn.Module:
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device, desc="Train") -> float:
    """Run one training epoch and return the mean loss."""
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
    """Softmax probabilities for each image.

    With `tta=True`, averages (soft voting) the predictions on the original image,
    its horizontal mirror and its vertical mirror.
    """
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc="Inference", leave=False):
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
    """Macro F1 on a loader that yields (image, label)."""
    y_true = np.asarray(loader.dataset.labels)
    y_pred = predict_proba(model, loader, device).argmax(axis=1)
    return f1_score(y_true, y_pred, average="macro")
