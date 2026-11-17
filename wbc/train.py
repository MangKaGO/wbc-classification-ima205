"""Entraînement d'un ResNet par transfert d'apprentissage (ImageNet).

Exemples (voir README pour les configurations du rapport) :
    # ResNet18 V2 — meilleur modèle (0.7030 sur Kaggle)
    python -m wbc.train --arch resnet18 --epochs 25 --augment strong --class-weights

    # Fine-tuning final sur 100 % des données à partir d'un checkpoint
    python -m wbc.train --arch resnet18 --full-data --init-weights outputs/resnet18.pth \\
        --epochs 3 --lr 1e-5 --augment strong --class-weights --output outputs/resnet18_full.pth

Sans --full-data, le jeu est séparé en 80/20 stratifié et seuls les poids ayant
le meilleur Macro F1 de validation sont sauvegardés (best checkpoint).
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import DEFAULT_DATA_DIR, OUTPUTS_DIR, SEED, DataPaths, get_device
from .data import (
    WBCDataset,
    balanced_class_weights,
    build_transforms,
    get_class_names,
    load_train_metadata,
    split_train_val,
)
from .models import build_model, evaluate_macro_f1, train_one_epoch


def build_scheduler(name: str, optimizer, epochs: int):
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if name == "plateau":
        # Divise le LR par 2 si le Macro F1 de validation stagne pendant 3 epochs.
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--augment", choices=["none", "light", "strong"], default="strong")
    p.add_argument("--class-weights", action="store_true", help="CrossEntropyLoss pondérée (classes rares)")
    p.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    p.add_argument("--full-data", action="store_true", help="Entraîne sur 100 %% des données, sans validation")
    p.add_argument("--init-weights", type=Path, help="Checkpoint de départ (au lieu des poids ImageNet)")
    p.add_argument("--no-pretrained", action="store_true", help="Ne pas charger les poids ImageNet")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", type=Path, help="Fichier .pth de sortie (défaut : outputs/<arch>.pth)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.full_data and args.scheduler == "plateau":
        raise SystemExit("--scheduler plateau nécessite un jeu de validation (incompatible avec --full-data).")
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Device : {device}")

    paths = DataPaths(args.data_dir)
    df = load_train_metadata(paths)
    class_names = get_class_names(df)

    if args.full_data:
        train_df, val_df = df, None
    else:
        train_df, val_df = split_train_val(df)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(
        WBCDataset(train_df, paths.train_dir, build_transforms(args.augment), class_names),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_df is not None:
        val_loader = DataLoader(
            WBCDataset(val_df, paths.train_dir, build_transforms("none"), class_names),
            shuffle=False,
            **loader_kwargs,
        )

    model = build_model(args.arch, pretrained=not (args.no_pretrained or args.init_weights))
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location="cpu"))
    model = model.to(device)

    weight = balanced_class_weights(train_df["label"].values, class_names).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)

    output = args.output or OUTPUTS_DIR / f"{args.arch}{'_full' if args.full_data else ''}.pth"
    output.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, desc=f"Epoch {epoch}/{args.epochs}")
        msg = f"Epoch {epoch:>2}/{args.epochs} | loss {loss:.4f} | lr {lr:.2e}"

        val_f1 = None
        if val_loader is not None:
            val_f1 = evaluate_macro_f1(model, val_loader, device)
            msg += f" | val Macro F1 {val_f1:.4f}"
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), output)
                msg += "  -> meilleur modèle sauvegardé"
        print(msg)

        if args.scheduler == "plateau":
            scheduler.step(val_f1)
        elif scheduler is not None:
            scheduler.step()

    if val_loader is None:
        torch.save(model.state_dict(), output)
        print(f"Poids finaux sauvegardés : {output}")
    else:
        print(f"Meilleur Macro F1 de validation : {best_f1:.4f} — poids : {output}")


if __name__ == "__main__":
    main()
