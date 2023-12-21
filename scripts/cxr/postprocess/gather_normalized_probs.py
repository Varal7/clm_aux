import os
import pandas as pd
import argparse
import torch
import json
import numpy as np

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
    parser.add_argument('--output_dir', type=str, default="/Mounts/rbg-storage1/users/quach/outputs/uncertainty/cxr2/")
    parser.add_argument('--folders', nargs='+', default=FOLDERS)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for folder in args.folders:
        base = os.path.join(args.base_dir, folder)

        scores = {}

        for seed in range(args.start_seed, args.end_seed):
            with open(os.path.join(base, f"dev_sample_seed_{seed}_normalized_likelihoods.json")) as f:
                normalized_likelihoods = np.array(json.load(f))
                scores[seed] = normalized_likelihoods

        df = pd.DataFrame(scores)
        values = df.values

        np.save(os.path.join(args.output_dir, f"{folder}_normprob_scores.npy"), values)
