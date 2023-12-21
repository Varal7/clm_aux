import os
import pandas as pd
import argparse
import torch
import numpy as np

def get_label(base, start_seed=42, end_seed=62):
    label = {}

    for seed in (range(start_seed, end_seed)):
        df_y = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y.csv"), index_col=0)
        df_y_hat = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y_hat.csv"), index_col=0)
        label[seed] = ((df_y ==1) == (df_y_hat == 1)).all(axis=1) + 0

    label = (pd.DataFrame(label).values)
    return label.astype(float)


if __name__ == "__main__":
    DEFAULT_BASE_DIR = "/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds"
    FOLDERS = [
            'train_scorer_0_2000',
            'calibration_2000_5000',
            'valid_5000_8000',
            'test_8000_13000',
            'valid_scorer_13000_14000',
    ]

    parser = argparse.ArgumentParser(description='Gather CXR scores')
    parser.add_argument('--start_seed', type=int, default=42)
    parser.add_argument('--end_seed', type=int, default=62)
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument('--folders', nargs='+', default=FOLDERS)

    args = parser.parse_args()

    for folder in args.folders:
        base = os.path.join(args.base_dir, folder)
        label = get_label(base, args.start_seed, args.end_seed)

        filename = os.path.join(base, f"labels_{args.start_seed}_{args.end_seed}.npy")
        np.save(filename, label)
