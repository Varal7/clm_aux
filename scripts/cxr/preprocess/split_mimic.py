import os
import pandas as pd
from tqdm import tqdm
import argparse
from repcal.utils.data import SPLITS_FILENAME, get_text_path_from_row, TEXT_ROOT

def make_splits_with_report(splits_filename=SPLITS_FILENAME, text_root=TEXT_ROOT):
    """Append report to splits file"""
    splits = pd.read_csv(splits_filename)

    rows = []
    for row in tqdm(splits.to_dict("records")):
        with open(get_text_path_from_row(row, text_root)) as f:
            report = f.read()
        row['report'] = report
        rows.append(row)

    df = pd.DataFrame.from_records(rows)

    return df

def resplit_train(df):
    """Split train into train and dev.
    Official MIMIC split:
        train       368960
        test          5159
        validate      2991

    After this:
        train       331846
        dev          37114
        test          5159
        validate      2991
    """
    train_keys = df.query("split == 'train'")['subject_id'].unique()
    dev_keys = pd.Series(train_keys).sample(frac=0.1, random_state=42)
    df.loc[df.subject_id.isin(dev_keys), 'split'] = "dev"
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--text_root', type=str, default=TEXT_ROOT)
    parser.add_argument('--split_filename', type=str, default=SPLITS_FILENAME)


    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    full_df = make_splits_with_report(args.split_filename, args.text_root)
    full_df.to_csv(os.path.join(out_dir, "original_splits_with_reports.csv"), index=False)
    full_df = resplit_train(full_df)
    full_df.to_csv(os.path.join(out_dir, "splits_with_reports.csv"), index=False)
