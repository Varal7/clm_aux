import os
import pandas as pd
import argparse
import torch
import numpy as np


def get_hard_cxr_loss(base, start_seed=42, end_seed=62):
    hard_label = {}

    for seed in (range(start_seed, end_seed)):
        gold = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y.csv"), index_col=0)
        pred = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y_hat.csv"), index_col=0)
        hard_label[seed] = ((pred == gold).all(axis=1)) + 0

    hard_label = (pd.DataFrame(hard_label).values)
    losses = 1 - hard_label
    return losses.astype(float)

def get_soft_cxr_loss(base, start_seed=42, end_seed=62):
    soft_label = {}

    for seed in (range(start_seed, end_seed)):
        df_y = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y.csv"), index_col=0)
        df_y_hat = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y_hat.csv"), index_col=0)

        df_y_hat = df_y_hat.drop(['study_id'], axis=1)
        df_y = df_y.drop(['study_id'], axis=1)
        df_y_hat = (df_y_hat == 1)
        df_y = (df_y == 1)
        tp = (df_y_hat * df_y).astype(float)
        fp = (df_y_hat * ~df_y).astype(float)
        fn = (~df_y_hat * df_y).astype(float)
        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)
        f1_eg = (tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).fillna(0)
        soft_label[seed] = f1_eg

    soft_label = (pd.DataFrame(soft_label).values)
    losses = 1 - soft_label
    return losses.astype(float)


if __name__ == "__main__":
    GNN_SCORES = "nli_42_62/gnn-apr26-20seeds-f0f49f2bd9e9bb124e97d1aa9713f417.pt"
    DEFAULT_BASE_DIR = "/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds"
    FOLDERS = [
            'train_scorer_0_2000',
            'calibration_2000_5000',
            'valid_5000_8000',
            'test_8000_13000',
            'valid_scorer_13000_14000',
    ]

    parser = argparse.ArgumentParser(description='Gather CXR scores')
    parser.add_argument('--scores', type=str, default=GNN_SCORES)
    parser.add_argument('--score_name', type=str, default="gnn")
    parser.add_argument('--start_seed', type=int, default=42)
    parser.add_argument('--end_seed', type=int, default=62)
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument('--output_dir', type=str, default="/Mounts/rbg-storage1/users/quach/outputs/uncertainty/cxr2/")
    parser.add_argument('--folders', nargs='+', default=FOLDERS)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for folder in args.folders:
        base = os.path.join(args.base_dir, folder)
        hard_losses = get_hard_cxr_loss(base, args.start_seed, args.end_seed)
        soft_losses = get_soft_cxr_loss(base, args.start_seed, args.end_seed)
        scores = torch.load(os.path.join(base, args.scores)).numpy()

        np.save(os.path.join(args.output_dir, f"{folder}_hard.npy"), hard_losses)
        np.save(os.path.join(args.output_dir, f"{folder}_soft.npy"), soft_losses)
        np.save(os.path.join(args.output_dir, f"{folder}_{args.score_name}_scores.npy"), scores)
