import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from tqdm.auto import tqdm

import logging
import json
import coloredlogs

from repcal.utils.data import get_before_findings
from torchvision.io import read_image, ImageReadMode

coloredlogs.install()

from transformers import (
    HfArgumentParser,
    set_seed,
)

from repcal.models.encoder_decoder.encoder_decoder import (
    ModelArguments,
    load_trained_model
)

from repcal.dataloaders.mimic import (
    DataTrainingArguments,
    get_jit_image_processor,
    get_dataset
)

COLUMNS = [
    "study_id",
    "subject_id",
    "path",
    "report",
    "prompt",
    "generated",
]


logger = logging.getLogger(__name__)

@dataclass
class PredictArguments:
    """
    Arguments pertaining to how to run the prediction.
    """
    checkpoint: str = field()
    output_name: str = field()
    strategy: str = field(default="sample")
    predict_split: str = field(default="dev")
    max_predict_samples: Optional[int] = field(default=None)
    starting_x: int = field(default=0)
    batch_size: int = field(default=16)
    seed: int = field(default=42)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PredictArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, predict_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, predict_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataTrainingArguments
    predict_args: PredictArguments

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load dataset
    dataset = get_dataset(data_args, model_args.cache_dir)

    # Load tokenizer, processor and model from checkpoint
    checkpoint = predict_args.checkpoint
    _image_processor, tokenizer, model = load_trained_model(checkpoint, model_args)
    tokenizer.padding_side = "left" # we need this for decoding

    if data_args.image_processor_jit:
        image_processor = get_jit_image_processor(_image_processor)
    else:
        image_processor = lambda x: _image_processor(x, return_tensors="pt").pixel_values

    # set seed for torch dataloaders
    set_seed(42)

    predict_dataset = dataset[predict_args.predict_split]
    if predict_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(predict_args.starting_x, predict_args.starting_x + predict_args.max_predict_samples))

    # Move model to GPU
    device = "cuda:0"
    model = model.to(device)

    # set seed for model
    set_seed(predict_args.seed)

    filename = os.path.join(checkpoint, "preds", predict_args.output_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print(predict_dataset)

    with open(filename, "w") as w:
        bs = predict_args.batch_size
        data = predict_dataset
        for i in tqdm(range(0, len(data), bs), dynamic_ncols=True):
            batch = data[i : min(len(data), i + bs)]
            batch: Dict[str, Any]

            prompts = [get_before_findings(text) for text in batch['report']]
            input_ids, attention_masks = tokenizer(prompts, return_tensors="pt", padding=True).values()
            images = [read_image(os.path.join(data_args.mimic_root, image_path), mode=ImageReadMode.RGB) for image_path in batch['image_path']]
            pixel_values = image_processor(images)
            input_datum = {
                "pixel_values": pixel_values.to(device),
                "decoder_input_ids": input_ids.to(device),
                "decoder_attention_mask": attention_masks.to(device),
            }

            try:
                with torch.no_grad():
                    if predict_args.strategy == "sample":
                        output = model.generate(**input_datum, max_length=512, do_sample=True, num_return_sequences=1)
                    elif predict_args.strategy == "beam":
                        output = model.generate(**input_datum, max_length=512, num_beams=5, num_return_sequences=1)
                    elif predict_args.strategy == "greedy":
                        output = model.generate(**input_datum, max_length=512, num_beams=1, num_return_sequences=1)
                    else:
                        raise ValueError(f"Unknown strategy {predict_args.strategy}")

                texts = tokenizer.batch_decode(output, skip_special_tokens=True)
            except:
                texts = ["Failed"] * len(batch['image_path'])

            finally:
                batch['generated'] = texts
                batch['path'] = batch['image_path']
                batch['prompt'] = prompts
                batch = {k: batch[k] for k in COLUMNS}
                v = [dict(zip(batch, t)) for t in zip(*batch.values())]
                for d in v:
                    w.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
