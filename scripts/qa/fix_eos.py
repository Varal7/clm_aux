import argparse
import json
import os
from transformers import LlamaTokenizer
from tqdm.auto import tqdm

LLAMA_PATH = "/storage/quach/weights/llama-hf-13b"

FILENAME = "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/32/dev_sample_old.jsonl"
OUTPUT = "/Mounts/rbg-storage1/users/quach/outputs/llama/triviaqa/fewshot/32/dev_sample.jsonl"

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_PATH)

# Fix EOS token: in previous script when the model generates the EOS token, we did not truncate the sequence
# This script will fix the issue by truncating the sequence at the first EOS token

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str, default=FILENAME)
parser.add_argument('--output', type=str, default=OUTPUT)
args = parser.parse_args()


if os.path.exists(args.output):
    raise ValueError(f"Output file {args.output} already exists")

with open(args.output, 'w') as w:
    with open(args.input, 'r') as f:
        for line in tqdm(f):
            example = json.loads(line)
            for p in example['generations']:
                if tokenizer.eos_token_id in p['tokens']:
                    truncate = p['tokens'].index(tokenizer.eos_token_id)
                    p['tokens'] = p['tokens'][:truncate]
                    p['scores'] = p['scores'][:truncate]
                    p['decoded'] = tokenizer.decode(p['tokens'])
            w.write(json.dumps(example) + '\n')
