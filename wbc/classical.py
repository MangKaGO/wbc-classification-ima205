"""Baseline Machine Learning classique sur les features manuelles.

Usage :
    python -m wbc.classical --model lightgbm
    python -m wbc.classical --model svm

Évalue le modèle en validation croisée stratifiée à 5 plis (Macro F1), puis le
ré-entraîne sur tout le jeu d'entraînement pour produire une soumission.
Nécessite d'avoir lancé `python -m wbc.features` au préalable.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import OUTPUTS_DIR, SEED


def build_model(name: str):
    if name == "lightgbm":
        clf = LGBMClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1
        )
    elif name == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=SEED)
    else:
        raise ValueError(f"Modèle inconnu : {name!r}")
    # Le scaler est intégré au pipeline pour être ré-appris sur chaque pli.
    return make_pipeline(StandardScaler(), clf)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["lightgbm", "svm"], default="lightgbm")
    parser.add_argument("--features-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    train = pd.read_csv(args.features_dir / "train_features.csv")
    test = pd.read_csv(args.features_dir / "test_features.csv")
    X, y = train.drop(columns=["ID", "label"]).values, train["label"].values
    X_test = test.drop(columns=["ID"]).values

    model = build_model(args.model)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"Validation croisée 5 plis — {args.model}")
    fold_scores = []
    oof_pred = np.empty_like(y)
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model.fit(X[tr], y[tr])
        oof_pred[va] = model.predict(X[va])
        score = f1_score(y[va], oof_pred[va], average="macro")
        fold_scores.append(score)
        print(f"  Pli {fold} — Macro F1 : {score:.4f}")
    print(f"Macro F1 moyen : {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}\n")
    print("Rapport de classification (prédictions out-of-fold) :")
    print(classification_report(y, oof_pred, zero_division=0))

    model.fit(X, y)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"submission_{args.model}.csv"
    pd.DataFrame({"ID": test["ID"], "label": model.predict(X_test)}).to_csv(out, index=False)
    print(f"Soumission écrite : {out}")


if __name__ == "__main__":
    main()
