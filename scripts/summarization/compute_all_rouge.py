from p_tqdm import p_map

import os
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
from rouge_score import rouge_scorer

def pack(text):
    return text.replace("...", "[DOTDOTDOT]")

def unpack(text):
    return text.replace("[DOTDOTDOT]", "...")

def split_sentences(text):
    li =  list(map(lambda x: unpack(x).strip(), pack(text).split(' .')))
    if len(li[-1]) == 0:
        return li[:-1]
    return li

if __name__ == "__main__":
    INPUT_DIR = '/Mounts/rbg-storage1/users/quach/outputs/uncertainty/xl_topp095_temp07'
    FILENAME = "cnn_dailymail_v002-predict_with_aux_with_sent_splits_fixed_and_scores_and_nli.jsonl"

    parser = argparse.ArgumentParser(description='Compute ROUGE scores for sentences')
    parser.add_argument('--base_dir', type=str, default=INPUT_DIR)
    parser.add_argument('--filename', type=str, default=FILENAME)

    args = parser.parse_args()

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    count = 0

    with open(os.path.join(args.base_dir, args.filename)) as f:
        samples = [json.loads(line) for line in tqdm(f)]

    g = len(samples[0]['prediction'])


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


    N = len(samples)

    for i in tqdm(range(N)):
        sample = samples[i]
        assert len(sample['prediction']) == g

        row_sentences.append([])
        row_sentence2idx.append({})
        row_generation_idx_to_row_idx.append([])
        row_reference_idx_to_row_idx.append([])
        row_rouge_scores.append([])

        for j in range(g):
            row_generation_idx_to_row_idx[i].append([])
            prediction = sample['prediction'][j]
            sentences = split_sentences(prediction)
            for k, s in enumerate(sentences):
                if s not in row_sentence2idx[i]:
                    row_sentence2idx[i][s] = len(row_sentences[i])
                    row_sentences[i].append(s)

                row_generation_idx_to_row_idx[i][j].append(row_sentence2idx[i][s])

        refs = (sample['inputs']['targets_pretokenized']).split("\n")

        for k, s in enumerate(refs):
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

    out_dir = os.path.join(args.base_dir, f"rouge_scores")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "row_sentences.jsonl"), 'w') as w:
        for i in range(N):
            w.write(json.dumps(row_sentences[i]) + '\n')

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

