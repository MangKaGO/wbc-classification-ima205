"""Handcrafted feature extraction (HSV color, shape, GLCM texture).

Usage:
    python -m wbc.features [--data-dir IMA205-challenge]

Writes outputs/train_features.csv and outputs/test_features.csv (15 features per image).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from .config import DEFAULT_DATA_DIR, OUTPUTS_DIR, DataPaths
from .data import load_test_metadata, load_train_metadata

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def extract_features(image_path: Path) -> dict | None:
    """Compute the 15 features of an image, or None if it cannot be read."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = {}

    # 1. Color: mean and standard deviation of each HSV channel.
    for i, channel in enumerate("HSV"):
        features[f"color_{channel}_mean"] = hsv[:, :, i].mean()
        features[f"color_{channel}_std"] = hsv[:, :, i].std()

    # 2. Shape: main contour after inverted Otsu thresholding.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = perimeter = circularity = aspect_ratio = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / perimeter**2
        _, _, w, h = cv2.boundingRect(contour)
        if h > 0:
            aspect_ratio = w / h
    features["shape_area"] = area
    features["shape_perimeter"] = perimeter
    features["shape_circularity"] = circularity
    features["shape_aspect_ratio"] = aspect_ratio

    # 3. Texture: GLCM properties (distance 1 px, angle 0°).
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    for prop in GLCM_PROPS:
        features[f"tex_{prop}"] = graycoprops(glcm, prop)[0, 0]

    return features


def build_feature_table(df: pd.DataFrame, img_dir: Path) -> pd.DataFrame:
    """Extract features for every image in `df` (keeps the ID and label columns)."""
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Features {img_dir.name}"):
        feats = extract_features(img_dir / row.ID)
        if feats is None:
            print(f"Skipping unreadable image: {row.ID}")
            continue
        rows.append({**row._asdict(), **feats})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    paths = DataPaths(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, df, img_dir in [
        ("train", load_train_metadata(paths), paths.train_dir),
        ("test", load_test_metadata(paths), paths.test_dir),
    ]:
        table = build_feature_table(df, img_dir)
        out = args.out_dir / f"{name}_features.csv"
        table.to_csv(out, index=False)
        print(f"{out}: {table.shape[0]} rows, {table.shape[1]} columns")


if __name__ == "__main__":
    main()
