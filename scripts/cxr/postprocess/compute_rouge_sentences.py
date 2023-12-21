from repcal.utils.data import to_sentences, get_sentences
from p_tqdm import p_map

import os
import json
import evaluate
import argparse

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

    metric = evaluate.load("rouge")

    start_seed = args.start_seed
    end_seed = args.end_seed
    prefix = args.prefix

    for folder in args.folders:
        print(folder)

        base = os.path.join(args.base_dir, folder)

        sentences =  get_sentences(start_seed, end_seed, base, prefix=prefix)

        with open(os.path.join(base, f"{prefix}_{start_seed}.jsonl")) as f:
            studies =  [json.loads(l) for l in f]

        refs = [to_sentences(r['report'][len(r['prompt']):]) for r in studies ]

        scores = []

        def compute_scores(idx):
            ss = []
            for j in range(len(sentences[idx])):
                s = []
                for k in range(len(sentences[idx][j])):
                    if len(refs[idx]) == 0:
                        s.append({'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0})
                    else:
                        s.append(metric.compute(predictions=[sentences[idx][j][k]], references=[refs[idx]], use_stemmer=True))
                ss.append(s)
            return ss

        N = len(sentences)
        scores = p_map(compute_scores, list(range(N)))

        with open(os.path.join(base, f"{prefix}_{start_seed}_{end_seed}_rouge.json"), "w") as f:
            json.dump(scores, f)
