from repcal.utils.data import to_sentences, get_sentences
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

    count = 0

    for folder in args.folders:
        base = os.path.join(args.base_dir, folder)

        sentences =  get_sentences(start_seed, end_seed, base, prefix=prefix)

        with open(os.path.join(base, f"{prefix}_{start_seed}.jsonl")) as f:
            studies =  [json.loads(l) for l in f]
        refs = [to_sentences(r['report'][len(r['prompt']):]) for r in studies ]

        # row_sentences
        # List of list of sentences
        # The first list is for each example
        # The second list is flattened list of sentences
        row_sentences = []

        # row_sentence2idx
        # List of dict of {sentence: idx}
        # The first list is for each example
        # The dict maps sentence to its index in the flattened list
        row_sentence2idx = []

        # row_generation_idx_to_row_idx
        # list of list of list of int
        # The first list is for each example
        # The second list is for each generation
        # The third list is for each sentence in the generation
        # The int is the index of the sentence in the flattened list
        row_generation_idx_to_row_idx = []

        # row_reference_idx_to_row_idx
        # list of list of int
        # The first list is for each example
        # The second list is for each sentence in the reference
        # The int is the index of the sentence in the flattened list
        row_reference_idx_to_row_idx = []

        # row_rouge_scores
        # list of np.array of shape (num_sentences, num_sentences)
        # The first list is for each example
        # The np.array is the rouge score matrix of each sentence pair for that row
        row_rouge_scores = []


        assert len(sentences) == len(refs)
        N = len(sentences)
        g = args.end_seed - args.start_seed

        for i in tqdm(range(N)):
            assert len(sentences[i]) == g
            row_sentences.append([])
            row_sentence2idx.append({})
            row_generation_idx_to_row_idx.append([])
            row_reference_idx_to_row_idx.append([])
            row_rouge_scores.append([])

            for j in range(g):
                row_generation_idx_to_row_idx[i].append([])
                for k, s in enumerate(sentences[i][j]):
                    if s not in row_sentence2idx[i]:
                        row_sentence2idx[i][s] = len(row_sentences[i])
                        row_sentences[i].append(s)

                    row_generation_idx_to_row_idx[i][j].append(row_sentence2idx[i][s])

            for k, s in enumerate(refs[i]):
                if s not in row_sentence2idx[i]:
                    row_sentence2idx[i][s] = len(row_sentences[i])
                    row_sentences[i].append(s)
                row_reference_idx_to_row_idx[i].append(row_sentence2idx[i][s])

            count += len(row_sentences[i]) * len(row_sentences[i])

        print(f"Total number of pairs: {count}")

        def compute_rouge_scores(i):
            arr = np.zeros((len(row_sentences[i]), len(row_sentences[i])))
            for k1 in range(len(row_sentences[i])):
                for k2 in range(k1, len(row_sentences[i])):
                    arr[k1, k2] = scorer.score(row_sentences[i][k1], row_sentences[i][k2])['rougeL'].fmeasure
                    arr[k2, k1] = arr[k1, k2]

            return arr

        row_rouge_scores = p_map(compute_rouge_scores, range(N))

        out_dir = os.path.join(base, f"rouge_scores_{start_seed}_{end_seed}")
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "row_sentences.txt"), 'w') as w:
            for i in range(N):
                for s in row_sentences[i]:
                    w.write(f"{s}\n")

        with open(os.path.join(out_dir, "row_sentence2idx.jsonl"), 'w') as w:
            for i in range(N):
                w.write(json.dumps(row_sentence2idx[i]) + "\n")

        with open(os.path.join(out_dir, "row_generation_idx_to_row_idx.jsonl"), 'w') as w:
            for i in range(N):
                w.write(json.dumps(row_generation_idx_to_row_idx[i]) + "\n")

        with open(os.path.join(out_dir, "row_reference_idx_to_row_idx.jsonl"), 'w') as w:
            for i in range(N):
                w.write(json.dumps(row_reference_idx_to_row_idx[i]) + "\n")

        with open(os.path.join(out_dir, "row_rouge_scores.jsonl"), 'w') as w:
            for i in range(N):
                w.write(json.dumps(row_rouge_scores[i].tolist()) + "\n")

        print(f"Done with {folder}")
