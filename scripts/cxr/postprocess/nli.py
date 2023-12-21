import os
import json
from tqdm.auto import tqdm

import torch

from repcal.utils.data import to_sentences
from repcal.models.nli import NLI

import argparse


DEFAULT_BASE = "/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many"

def main(args):
    # Get nli model
    nli = NLI(device="cuda:0")

    reports = {}
    sentences = {}
    num_seeds = args.end_seed - args.start_seed

    # Read sentences
    for j in tqdm(range(num_seeds)):
        seed = args.start_seed + j
        with open(os.path.join(args.base, f"dev_sample_seed_{seed}.jsonl")) as f:
            studies  = [json.loads(l) for l in f]
            reports[j] = [s['generated'] for s in studies]
            sentences[j] = []
            for i, r in enumerate(studies):
                if args.start_x <= i < args.end_x:
                    sentences[j].append(to_sentences(r['generated'][len(r['prompt']):]))


    # Make list of sentences
    all_sentences = {}
    for ii in range(args.start_x, args.end_x):
        i = ii - args.start_x
        all_sentences[i] = []
        for j in range(num_seeds):
            all_sentences[i].extend(sentences[j][i])


    for ii in tqdm(range(args.start_x, args.end_x)):
        i = ii - args.start_x
        scores = nli.eval_all(all_sentences[i])
        filename = os.path.join(args.base, f"nli_{args.start_seed}_{args.end_seed}/scores_{ii}.pt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(scores, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_x", type=int, help="Index of first x", required=True)
    parser.add_argument("--end_x", type=int, help="Index of last x (exclusive)", required=True)
    parser.add_argument("--base", type=str, default=DEFAULT_BASE)
    parser.add_argument("--start_seed", type=int, help="Starting seed", default=42)
    parser.add_argument("--end_seed", type=int, help="Ending seed", default=62)

    args = parser.parse_args()

    main(args)
