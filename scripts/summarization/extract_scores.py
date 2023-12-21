from p_tqdm import p_map

import os
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
from rouge_score import rouge_scorer

def split_sentences(text):
    return list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(), text.split('.'))))

if __name__ == "__main__":
    INPUT_DIR = '/Mounts/rbg-storage1/users/quach/outputs/uncertainty/xl_topp095_temp07'
    FILENAME = "cnn_dailymail_v002-predict_with_aux_with_sent_splits_fixed_and_scores_and_nli.jsonl"

    keys = {
            "probs": 'prediction_sent_split_sent_scores',
            #  "nli": 'predictions_sent_split_nli',
            "nli_nocontext": 'predictions_sent_split_nli_nocontext'
    }

    parser = argparse.ArgumentParser(description='Extract sentence scores for summarization')
    parser.add_argument('--base_dir', type=str, default=INPUT_DIR)
    parser.add_argument('--filename', type=str, default=FILENAME)

    args = parser.parse_args()

    with open(os.path.join(args.base_dir, args.filename)) as f:
        samples = [json.loads(line) for line in tqdm(f)]

    N = len(samples)

    out_dir = os.path.join(args.base_dir, f"components")
    os.makedirs(out_dir, exist_ok=True)

    for key in keys:
        with open(os.path.join(out_dir, f"{key}.jsonl"), 'w') as f:
            for sample in tqdm(samples):
                f.write(json.dumps(sample['aux'][keys[key]]) + '\n')
