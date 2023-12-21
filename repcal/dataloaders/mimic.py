import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import logging
import coloredlogs

from transformers import ViTImageProcessor

from repcal.utils.data import get_before_findings, get_after_findings

coloredlogs.install()

logger = logging.getLogger(__name__)


REPORT_COLUMN = "report"
IMAGE_COLUMN = "image_path"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    We expect the data in the form of jsonl files with the following fields:
        - "study_id": "50000198"
        - "image_path": "2.0.0/files/p16/p16548129/s50000198/b66847d6.jpg"
        - "report": "The report"
    """
    data_dir: str = field(metadata={"help": "The data directory containing input files."})
    mimic_root: str = field(metadata={"help": "The root directory of the MIMIC-CXR dataset."})

    max_seq_length: Optional[int] = field(default=512)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)

    test_split: str = field(default="dev")
    max_test_samples: Optional[int] = field(default=None)

    num_workers: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)

    image_processor_jit: bool = field(default=True)

def get_dataset(data_args:DataTrainingArguments, cache_dir=None):
    if data_args.data_dir is None:
        raise ValueError("Need to specify a data directory")

    base = data_args.data_dir
    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(base, "train.jsonl"),
            "dev": os.path.join(base, "dev.jsonl"),
            "validate": os.path.join(base, "validate.jsonl"),
            "test": os.path.join(base, "test.jsonl"),
        },
        cache_dir=cache_dir
    )

    return dataset

# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples], dtype=torch.long)
    #  decoder_input_ids = torch.tensor([example["decoder_input_ids"] for example in examples], dtype=torch.long)
    #  decoder_attention_mask = torch.tensor([example["decoder_attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        #  "decoder_input_ids": decoder_input_ids,
        #  "decoder_attention_mask": decoder_attention_mask,
    }


def get_tokenize_reports(data_args: DataTrainingArguments, tokenizer):
    def tokenize_reports(examples):
        reports = list(examples[REPORT_COLUMN])
        prompts = [get_before_findings(report) for report in reports]
        labels = tokenizer(reports, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        #  decoder_inputs = tokenizer(prompts, max_length=data_args.max_seq_length, padding="longest", truncation=True)
        examples["labels"] = labels.input_ids
        #  examples["decoder_input_ids"] = decoder_inputs.input_ids
        #  examples["decoder_attention_mask"] = labels.attention_mask
        return examples

    return tokenize_reports

def get_transform_images(data_args: DataTrainingArguments, image_processor):
    if data_args.image_processor_jit:
        image_size = image_processor.size['height']
        image_transformations = Transform(image_size, image_processor.image_mean, image_processor.image_std)
        image_transformations = torch.jit.script(image_transformations)

        def transform_images(examples):
            images = [read_image(os.path.join(data_args.mimic_root, image_file), mode=ImageReadMode.RGB) for image_file in examples[IMAGE_COLUMN]]
            examples["pixel_values"] = [image_transformations(image) for image in images]
            return examples

        return transform_images
    else:
        raise ValueError("Only JIT image processing is supported")
        #  def transform_images(examples):
        #      images = [read_image(os.path.join(data_args.mimic_root, image_file), mode=ImageReadMode.RGB) for image_file in examples[IMAGE_COLUMN]]
        #      examples['pixel_values'] = image_processor(images, return_tensors="pt").pixel_values
        #      return examples
        #  return transform_images

def get_jit_image_processor(image_processor):
    image_size = image_processor.size['height']
    image_transformations = Transform(image_size, image_processor.image_mean, image_processor.image_std)
    image_transformations = torch.jit.script(image_transformations)

    def jit_image_processor(images):
        return torch.stack([image_transformations(image) for image in images], dim=0)

    return jit_image_processor
