import torch
import os
import json
import argparse
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from nltk.data import load

from repcal.utils.data import get_sentences

if __name__ == "__main__":
    DEFAULT_BASE_DIR = "/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds"
    FILENAME = "image-sentence_42_62/image-sentence-raw-may1-20seeds-5041479ec9f78561c124d99741842553.pt"

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
    parser.add_argument('--prefix', type=str, default="dev_sample_seed")
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument('--output_dir', type=str, default="/Mounts/rbg-storage1/users/quach/outputs/uncertainty/cxr2/components")
    parser.add_argument('--folders', nargs='+', default=FOLDERS)
    parser.add_argument('--filename', type=str, default=FILENAME)

    args = parser.parse_args()

    start_seed = args.start_seed
    end_seed = args.end_seed
    prefix = args.prefix

    for folder in args.folders:
        base = os.path.join(args.base_dir, folder)
        output_dir = os.path.join(args.output_dir, folder)
        os.makedirs(output_dir, exist_ok=True)

        sentences =  get_sentences(start_seed, end_seed, base, prefix=prefix)
        scores = torch.load(os.path.join(base, args.filename))

        N = len(sentences)
        g = args.end_seed - args.start_seed

        scores_by_idx = []
        num = 0
        for i in range(N):
            example_scores = []
            for j in range(g):
                generation_scores = []
                for k in range(len(sentences[i][j])):
                    generation_scores.append(scores[num].item())
                    num += 1

                example_scores.append(generation_scores)
            scores_by_idx.append(example_scores)

        with open(os.path.join(output_dir, f"image_sentence_scores.jsonl"), 'w') as w:
            for line in scores_by_idx:
                w.write(json.dumps(line) + "\n")
