"""Generate a Kaggle submission file from a checkpoint.

Usage:
    python -m wbc.predict --weights outputs/resnet18.pth
    python -m wbc.predict --weights outputs/resnet18.pth --tta   # Test-Time Augmentation
"""

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from .config import DEFAULT_DATA_DIR, DataPaths, get_device
from .data import WBCDataset, build_transforms, get_class_names, load_test_metadata, load_train_metadata
from .models import load_model, predict_proba


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--tta", action="store_true", help="Average original + H flip + V flip")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", type=Path, help="Output CSV (default: next to the weights)")
    args = p.parse_args()

    device = get_device()
    paths = DataPaths(args.data_dir)
    class_names = get_class_names(load_train_metadata(paths))
    test_df = load_test_metadata(paths)

    loader = DataLoader(
        WBCDataset(test_df, paths.test_dir, build_transforms("none")),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    model = load_model(args.arch, args.weights, device)
    preds = predict_proba(model, loader, device, tta=args.tta).argmax(axis=1)

    output = args.output or args.weights.with_name(
        f"submission_{args.weights.stem}{'_tta' if args.tta else ''}.csv"
    )
    pd.DataFrame({"ID": test_df["ID"], "label": [class_names[i] for i in preds]}).to_csv(output, index=False)
    print(f"Submission written: {output}")


if __name__ == "__main__":
    main()
