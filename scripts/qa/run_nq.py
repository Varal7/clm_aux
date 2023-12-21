import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import os
import sys
import json
import transformers
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

from transformers import LlamaForCausalLM, LlamaTokenizer

from transformers import (
    HfArgumentParser,
    set_seed,
)

LLAMA_PATH = "/storage/quach/weights/llama-hf-13b"

logger = logging.getLogger(__name__)

@dataclass
class PredictArguments:
    """
    Arguments pertaining to how to run the prediction.
    """
    output_name: str = field(metadata={"help":"Path to the output file"})
    checkpoint: str = field(default=LLAMA_PATH)
    strategy: str = field(default="sample")
    dataset_path: str = field(default="/storage/quach/data/natural_questions/v1.0/dev/dev.jsonl")
    predict_split: str = field(default="validation")
    num_generations: int = field(default=20)
    max_predict_samples: Optional[int] = field(default=None)
    starting_x: int = field(default=0)
    seed: int = field(default=42)


def main():
    parser = HfArgumentParser(PredictArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()

    args: PredictArguments

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info(f"Predicting with {args}")

    logger.info("Loading dataset")

    dataset = load_dataset("json", data_files={args.predict_split: args.dataset_path})

    prompt = "Answer these questions\nQ: {}\n A:"

    dataset = dataset[args.predict_split]

    if args.max_predict_samples is not None:
        dataset = dataset.select(range(args.starting_x, args.starting_x + args.max_predict_samples))


    logger.info("Loading model")

    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.float16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Moving model to device")

    model.to(device)

    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
    tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)


    stop_word_ids = [
        13,   # \n
        1919, # ,
        2982, # ,
        869,   # .
        29889,   # .
    ]

    class StoppingCriteriaSub(transformers.StoppingCriteria):
        def __init__(self, input_length=0, stop_ids=None):
            super().__init__()
            self.stop_ids = stop_ids
            self.input_length = input_length

        def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> bool:
            if self.stop_ids is None:
                return False

            output = input_ids[:, self.input_length:]

            has_stop_ids = []
            for stop_id in self.stop_ids:
                has_stop_id = torch.any(output == stop_id, dim=1)
                has_stop_ids.append(has_stop_id)
            has_stop_ids = torch.stack(has_stop_ids, dim=1)

            return (has_stop_ids.any(dim=1).all())


    def run(sample, num_generations):
        question = sample['question']
        answer = sample['answer']
        input_text = prompt.format(question)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        stopping_criteria = transformers.StoppingCriteriaList([StoppingCriteriaSub(stop_ids=stop_word_ids, input_length=input_ids.shape[1])])
        torch.manual_seed(args.seed)

        kwargs = {
            "max_new_tokens": 100,
            "return_dict_in_generate": True,
            "output_scores": True,
            "stopping_criteria": stopping_criteria,
            "num_return_sequences": num_generations,
        }

        if args.strategy == "greedy":
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1

        elif args.strategy == "sample":
            kwargs["do_sample"] = True

        set_seed(args.seed)

        with torch.no_grad():
            outputs = model.generate(input_ids, **kwargs)

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        generations = []

        for i in range(len(generated_tokens)):
            tokens = []
            scores = []
            for tok, score in zip(generated_tokens[i], transition_scores[i]):
                if tok in stop_word_ids and len(tokens) > 0: # avoid the edge case of empty generation
                    break
                tokens.append(tok)
                scores.append(score)

            tokens = torch.stack(tokens, dim=0)
            scores = torch.stack(scores, dim=0)

            generations.append({
                'tokens': tokens.cpu().tolist(),
                'scores': scores.cpu().tolist(),
                'decoded': tokenizer.decode(tokens)
            })

        datum = {
            'question': question,
            'answer': answer,
            'generations': generations
        }
        return datum

    os.makedirs(os.path.dirname(args.output_name), exist_ok=True)

    with open(args.output_name, "w") as w:
        for sample in tqdm(dataset):
            datum = run(sample, args.num_generations)
            w.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    main()
