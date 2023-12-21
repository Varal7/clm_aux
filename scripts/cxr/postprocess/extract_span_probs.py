import torch
import os
import json
import argparse
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from nltk.data import load

tokenizer = AutoTokenizer.from_pretrained("gpt2")
sent_tokenizer = load(f"tokenizers/punkt/english.pickle")
import re

def get_spans(text):
    removed_indices = [m.start() for m in re.finditer('\n', text)]
    filtered_indices = [i for i in range(len(text) + 1) if i not in removed_indices]
    new_spans = list(sent_tokenizer.span_tokenize(text.replace("\n", "")))
    return [(filtered_indices[s], filtered_indices[e]) for (s, e) in new_spans]

def normalized_likelihood(log_probs, alpha=0.6):
    """Likelihood with length penalty."""
    total_log_probs = torch.sum(torch.clip(log_probs, -1e5, 0))
    penalty = (5 + len(log_probs)) ** alpha / (5 + 1) ** alpha
    return torch.exp(total_log_probs / penalty)


if __name__ == "__main__":
    DEFAULT_BASE_DIR = "/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds"
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
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument('--output_dir', type=str, default="/Mounts/rbg-storage1/users/quach/outputs/uncertainty/cxr2/components")
    parser.add_argument('--folders', nargs='+', default=FOLDERS)

    args = parser.parse_args()

    for folder in args.folders:
        base = os.path.join(args.base_dir, folder)
        output_dir = os.path.join(args.output_dir, folder)
        os.makedirs(output_dir, exist_ok=True)
        output = []
        lines_by_seed = []

        logp_by_seed = []

        for seed in range(args.start_seed, args.end_seed):
            with open(os.path.join(base, f"dev_sample_seed_{seed}.jsonl")) as f:
                lines_by_seed.append([json.loads(line) for line in f])

            with open(os.path.join(base, f"dev_sample_seed_{seed}_likelihoods.json")) as f:
                logp_by_seed.append(json.load(f))


        for i in tqdm(range(len(lines_by_seed[0]))):
            cur = []

            for j in range(len(lines_by_seed)):
                example = lines_by_seed[j][i]
                text = example['generated'][len(example['prompt']):]
                spans = get_spans(text)
                tokenized = tokenizer(text)
                all_tokens = tokenized['input_ids']
                token_spans = [(tokenized.char_to_token(s), tokenized.char_to_token(e)) for s, e in spans]
                sentence_log_probs = [torch.tensor(logp_by_seed[j][i][s:e]) for s,e in token_spans]

                normalized_probs = [ normalized_likelihood(lp).item() for lp in sentence_log_probs]

                cur.append(normalized_probs)

            output.append(cur)

        with open(os.path.join(output_dir, f"normprob_sentence_scores.jsonl"), 'w') as w:
            for line in output:
                w.write(json.dumps(line) + "\n")
