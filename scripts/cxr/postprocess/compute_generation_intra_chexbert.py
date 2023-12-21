import pandas as pd
from p_tqdm import p_map

import os
import argparse
import numpy as np
from tqdm.auto import tqdm


if __name__ == "__main__":
    DEFAULT_BASE_DIR = "/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds"
    FODLERS = [
            'train_scorer_0_2000',
            'calibration_2000_5000',
            'valid_5000_8000',
            'test_8000_13000',
            'valid_scorer_13000_14000',
    ]

    parser = argparse.ArgumentParser(description='Compute ROUGE scores for sentences')
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument('--folders', nargs='+', default=FODLERS)
    parser.add_argument('--start_seed', type=int, default=42)
    parser.add_argument('--end_seed', type=int, default=62)
    parser.add_argument('--prefix', type=str, default="dev_sample_seed")

    args = parser.parse_args()

    start_seed = args.start_seed
    end_seed = args.end_seed
    prefix = args.prefix


    for folder in args.folders:
        base = os.path.join(args.base_dir, folder)

        g = end_seed - start_seed

        preds = []

        for seed in range(start_seed, end_seed):
            pred = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y_hat.csv"), index_col=0)
            pred = pred.drop(['study_id'], axis=1)
            pred = (pred == 1).astype(int)
            preds.append(pred)


        def compute_chexbert_scores(i):
            arr = np.zeros((g, g))
            for j in range(g):
                for k in range(j, g):
                    arr[j, k]= (preds[j].iloc[i] == preds[k].iloc[i]).mean()

            return arr

        N = len(preds[0])

        scores = p_map(compute_chexbert_scores, range(N))
        scores = np.array(scores)


        out_dir = os.path.join(base, f"seeds_intra_scores_{start_seed}_{end_seed}")
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "chexbert_eq.npy"), scores)
