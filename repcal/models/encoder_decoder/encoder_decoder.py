# Common arguments for encoder-decoder model

from dataclasses import dataclass, field
from typing import Optional

import torch

from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
)
from transformers.models.vision_encoder_decoder import VisionEncoderDecoderModel
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    image_encoder_model: str = field(default="google/vit-base-patch16-224-in21k")
    text_decoder_model: str = field(default="gpt2")
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    use_fp16: bool = field(default=False)


def _get_tokenizer(model_args: ModelArguments):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.text_decoder_model,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )

    #  if "llama" in model_args.image_encoder_model:
    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
    tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)

    tokenizer.pad_token = tokenizer.eos_token  # Because GPT2 tokenizer does not have a pad token
    return tokenizer


def _get_image_processor(model_args: ModelArguments):
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_encoder_model,
        cache_dir=model_args.cache_dir,
    )

    return image_processor


def get_new_model(model_args: ModelArguments):
    tokenizer = _get_tokenizer(model_args)
    image_processor = _get_image_processor(model_args)

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        model_args.image_encoder_model,
        model_args.text_decoder_model,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.float16 if model_args.use_fp16 else torch.float32,
    )


    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return image_processor, tokenizer, model

def load_trained_model(checkpoint, model_args: ModelArguments):
    # TODO: save tokenizer and image processor to checkpoint so we can load there directly instead
    tokenizer = _get_tokenizer(model_args)
    image_processor = _get_image_processor(model_args)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint, torch_dtype=torch.float16 if model_args.use_fp16 else torch.float32)

    return image_processor, tokenizer, model
