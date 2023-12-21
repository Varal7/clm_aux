import os
import re
import json
import string
import numpy as np
import argparse
from tqdm.auto import tqdm

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_diversity(filename):
    all_predictions = []
    with open(os.path.join(filename)) as f:
        for line in tqdm(f):
            example = json.loads(line)
            predictions = [normalize_text(p['decoded']) for p in example['generations']]
            all_predictions.append(predictions)

    N = len(all_predictions)
    g = len(all_predictions[0])

    def compute_match(i):
        arr = np.zeros((g, g))
        for j in range(g):
            for k in range(j, g):
                arr[j, k] = (all_predictions[i][j]) == (all_predictions[i][k])
        return arr

    diversity = np.array([compute_match(i) for i in tqdm(range(N))])

    return diversity

if __name__ == "__main__":
    FILENAME = "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/32/rest_2000_17000/dev_sample_self_eval.jsonl"

    parser = argparse.ArgumentParser(description='Compute Trivia QA scores')
    parser.add_argument('--filename', type=str, default=FILENAME)
    parser.add_argument('--output_filename', type=str, default="/Mounts/rbg-storage1/users/quach/outputs/uncertainty/triviaqa/diversity.npy")
    args = parser.parse_args()

    # Load examples.
    diversity = compute_diversity(args.filename)

    # Save data
    np.save(args.output_filename, diversity)
