import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from tqdm.auto import tqdm
import math

import logging
import json
import coloredlogs

from repcal.utils.data import minibatch
from torchvision.io import read_image, ImageReadMode

coloredlogs.install()

from transformers import (
    HfArgumentParser,
)

from repcal.models.encoder_decoder.encoder_decoder import (
    ModelArguments,
    load_trained_model
)

from repcal.dataloaders.mimic import (
    DataTrainingArguments,
    get_jit_image_processor,
)

import torch.nn.functional as F

def normalized_likelihood(log_probs, alpha=0.6):
    """Likelihood with length penalty."""
    total_log_probs = torch.sum(torch.clip(log_probs, -1e5, 0))
    penalty = (5 + len(log_probs)) ** alpha / (5 + 1) ** alpha
    return torch.exp(total_log_probs / penalty)

logger = logging.getLogger(__name__)

@dataclass
class EvaluateArguments:
    """
    Arguments pertaining to how to run the prediction.
    We already use the predict_split from above
    """
    checkpoint: str = field()
    jsonl: str = field()
    batch_size: int = field(default=16)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvaluateArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, evaluate_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, evaluate_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataTrainingArguments
    evaluate_args: EvaluateArguments

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    output_name = evaluate_args.jsonl.replace(".jsonl", "_normalized_likelihoods.json")
    if os.path.exists(output_name):
        logger.info(f"Skipping {output_name} as it already exists")
        return

    # Load tokenizer, processor and model from checkpoint
    checkpoint = evaluate_args.checkpoint
    _image_processor, tokenizer, model = load_trained_model(checkpoint, model_args)
    tokenizer.padding_side = "right" # we need right for losses

    if data_args.image_processor_jit:
        image_processor = get_jit_image_processor(_image_processor)
    else:
        image_processor = lambda x: _image_processor(x, return_tensors="pt").pixel_values

    # Move model to GPU
    device = "cuda:0"
    model = model.to(device)

    # Read predictions
    filename = evaluate_args.jsonl
    with open(filename) as f:
        lines = [json.loads(l) for l in f]

    all_likelihoods = []
    normalized_likelihoods = []

    for batch in minibatch(lines, bs=evaluate_args.batch_size):
        # Read images
        images = [read_image(os.path.join(data_args.mimic_root, image_path), mode=ImageReadMode.RGB) for image_path in batch['path']]
        pixel_values = image_processor(images)

        # Get text
        _, mask = tokenizer(batch['prompt'], return_tensors="pt", padding=True).values()
        labels, label_mask = tokenizer(batch['generated'], return_tensors="pt", padding=True).values()
        num_tokens = labels.size(1) - mask.size(1)
        mask = torch.cat([mask, torch.zeros(mask.size(0), num_tokens, dtype=torch.long)], dim=1)
        input_ids = labels.clone()
        labels = labels.masked_fill(mask, -100)  # mask the prompt
        labels = labels.masked_fill(1 - label_mask, -100) # mask the padding

        # Get loss
        input_datum = {
            "pixel_values": pixel_values.to(device),
            "decoder_input_ids": input_ids.to(device),
            "labels": labels.to(device)
        }

        with torch.no_grad():
            output = model(**input_datum)

        logits = output.logits.cpu()

        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(logits, dim=-1)

        padding_mask = labels.eq(-100)
        labels = torch.clamp(labels, min=0)
        log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).sum(dim=-1)
        log_likelihood.masked_fill_(padding_mask, 0.0)

        for i in range(len(batch['path'])):
            logp = log_likelihood[i][~padding_mask[i]]
            all_likelihoods.append(logp.tolist())
            normalized_likelihoods.append(normalized_likelihood(logp).item())

    with open(filename.replace(".jsonl", "_likelihoods.json"), "w") as f:
        json.dump(all_likelihoods, f)

    with open(filename.replace(".jsonl", "_normalized_likelihoods.json"), "w") as f:
        json.dump(normalized_likelihoods, f)

if __name__ == "__main__":
    main()
