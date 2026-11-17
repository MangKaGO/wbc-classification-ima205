# Classification automatique de leucocytes — IMA205

Projet du cours IMA205 (Télécom Paris) : classifier des images microscopiques de
globules blancs en **13 classes morphologiques** (challenge Kaggle, métrique : **Macro F1**).

Le dataset (28 901 images) est très déséquilibré : SNE et LY représentent ~73 % des images,
PLY moins de 0,05 %. Deux approches sont comparées :

1. **ML classique** : 15 caractéristiques manuelles (couleur HSV, forme, texture GLCM)
   puis LightGBM / SVM.
2. **Transfer learning** : ResNet18 / ResNet50 pré-entraînés sur ImageNet, avec loss
   pondérée par classe et augmentation de données.

Le rapport complet est dans [docs/Rapport_IMA205_Tabouri_Lukas.pdf](docs/Rapport_IMA205_Tabouri_Lukas.pdf).

![Distribution des classes](docs/figures/class_distribution.png)

## Résultats (leaderboard Kaggle public)

| Modèle | Approche | Macro F1 |
|---|---|---|
| SVM RBF (15 features) | ML classique | 0.21 (CV) |
| LightGBM (15 features) | ML classique | 0.26 (CV) |
| ResNet50 (25 epochs) | Deep learning | 0.6213 |
| ResNet18 V2 + TTA (3 vues) | Deep learning | 0.6296 |
| ResNet18 V1 (10 epochs) | Deep learning | 0.6350 |
| Fine-tuning sur 100 % des données | Deep learning | 0.6755 |
| **ResNet18 V2** (25 epochs, loss pondérée, augmentation forte) | Deep learning | **0.7030** |

## Structure

```
wbc/
  config.py      constantes, chemins du dataset, choix du device
  data.py        métadonnées, Dataset PyTorch, transformations, poids de classes
  features.py    extraction des 15 features manuelles        (python -m wbc.features)
  classical.py   LightGBM / SVM en validation croisée 5 plis (python -m wbc.classical)
  models.py      ResNet, boucle d'entraînement, inférence (+ TTA)
  train.py       entraînement d'un ResNet                     (python -m wbc.train)
  predict.py     génération d'une soumission Kaggle           (python -m wbc.predict)
notebooks/
  eda.ipynb      analyse exploratoire
docs/            rapport PDF et figures
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate sous Linux/macOS)
pip install -r requirements.txt
```

Pour le GPU, installer PyTorch avec la bonne version CUDA depuis [pytorch.org](https://pytorch.org/get-started/locally/).

Le dataset n'est pas versionné. Placer les fichiers du challenge à la racine :

```
IMA205-challenge/
  train/                 images d'entraînement
  test/                  images de test
  train_metadata.csv     ID, label
  test_metadata.csv      ID
```

(un autre emplacement peut être donné avec `--data-dir`). Tous les fichiers produits
(features, poids `.pth`, soumissions) sont écrits dans `outputs/`.

## Reproduire les expériences

**Baseline ML classique**

```bash
python -m wbc.features
python -m wbc.classical --model lightgbm
python -m wbc.classical --model svm
```

**ResNet18 V1** (base : 10 epochs, augmentation légère, sans pondération)

```bash
python -m wbc.train --arch resnet18 --epochs 10 --augment light --output outputs/resnet18_v1.pth
```

**ResNet18 V2** (meilleur modèle : 25 epochs, loss pondérée, augmentation forte, best checkpoint)

```bash
python -m wbc.train --arch resnet18 --epochs 25 --augment strong --class-weights --output outputs/resnet18_v2.pth
```

**ResNet50** (même pipeline, batch de 16 pour tenir en mémoire GPU)

```bash
python -m wbc.train --arch resnet50 --epochs 25 --batch-size 16 --augment strong --class-weights
```

**ResNet18 + scheduler** (`ReduceLROnPlateau` sur le Macro F1, ou `cosine` pour CosineAnnealingLR)

```bash
python -m wbc.train --arch resnet18 --epochs 35 --augment strong --class-weights --scheduler plateau --output outputs/resnet18_plateau.pth
```

**Fine-tuning sur 100 % des données** à partir de V2

```bash
python -m wbc.train --arch resnet18 --full-data --init-weights outputs/resnet18_v2.pth --epochs 3 --lr 1e-5 --augment strong --class-weights --output outputs/resnet18_v2_full.pth
```

**Soumission Kaggle** (avec `--tta` pour moyenner original + flip horizontal + flip vertical)

```bash
python -m wbc.predict --weights outputs/resnet18_v2.pth
```

```bash
python -m wbc.predict --weights outputs/resnet18_v2.pth --tta
```

## Remarques

- Seuls des poids pré-entraînés **ImageNet** sont autorisés par le challenge
  (pas de pré-entraînement sur des images médicales).
- Le split train/validation est stratifié 80/20 avec `random_state=42`.
- Le rapport mentionne un `CosineAnnealingLR` pour ResNet18 V2, mais le notebook d'origine
  entraînait V2 avec un learning rate fixe ; le scheduler est donc une option (`--scheduler`).
