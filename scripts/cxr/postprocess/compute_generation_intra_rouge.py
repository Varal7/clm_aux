from p_tqdm import p_map

import os
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
from rouge_score import rouge_scorer


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

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for folder in args.folders:
        base = os.path.join(args.base_dir, folder)

        def read_studies(seed, base):
            with open(os.path.join(base, f"{prefix}_{seed}.jsonl")) as f:
                return [json.loads(l) for l in f]

        def get_texts(studies):
            return [r['generated'][len(r['prompt']):] for r in (studies)]

        num_seeds = end_seed - start_seed
        g = num_seeds

        text_by_seed = {j: get_texts(read_studies(start_seed + j, base)) for j in tqdm(range(num_seeds))}
        num_x = len(text_by_seed[0])
        text_by_x = [
            [text_by_seed[j][i] for j in range(num_seeds)]
            for i in range(num_x)
        ]


        def compute_rouge_scores(i):
            arr = np.zeros((g, g))
            for k1 in range(g):
                for k2 in range(k1, g):
                    arr[k1, k2] = scorer.score(text_by_x[i][k1], text_by_x[i][k2])['rougeL'].fmeasure
                    arr[k2, k1] = arr[k1, k2]

            return arr


        scores = p_map(compute_rouge_scores, range(num_x))
        scores = np.array(scores)

        out_dir = os.path.join(base, f"seeds_intra_scores_{start_seed}_{end_seed}")
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "rouge_scores.npy"), scores)
