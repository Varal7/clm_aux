import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import os
from tqdm.auto import tqdm
import json
import re
import hashlib

from transformers import AutoModel, AutoTokenizer

from repcal.utils.data import get_sentences
from repcal.dataloaders.mimic import Transform as MIMICTransform
from torchvision.io import ImageReadMode, read_image

from transformers import (
    AutoImageProcessor,
)


def clean_spaces(str):
    str = str.strip()
    str = str.replace("\n", " ")
    str = " ".join(re.split("\s+", str, flags=re.UNICODE))
    str = str.strip()
    return str

class ImageReportDataset(Dataset):
    def __init__(
        self,
        base,
        start_seed,
        end_seed,
        num_x,
        prefix,
        tokenizer_name,
        mimic_root,
        image_size,
        image_mean,
        image_std,
        cache_dir="/tmp",
        transform=None,
        pre_transform=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mimic_root = mimic_root
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        image_transformations = MIMICTransform(image_size, image_mean, image_std)
        self.image_transformations = torch.jit.script(image_transformations)

        label = {}

        for seed in tqdm(range(start_seed, end_seed)):
            gold = pd.read_csv(os.path.join(base, f"{prefix}_{seed}_chexbert_y.csv"), index_col=0)
            pred = pd.read_csv(os.path.join(base, f"{prefix}_{seed}_chexbert_y_hat.csv"), index_col=0)
            gold = gold == 1
            pred = pred == 1
            label[seed] = ((pred == gold).all(axis=1)) + 0

        labels = torch.tensor(pd.DataFrame(label).values)[:num_x]

        def read_studies(seed, base):
            with open(os.path.join(base, f"{prefix}_{seed}.jsonl")) as f:
                return [json.loads(l) for l in f]

        report_by_seed = {seed: read_studies(seed, base) for seed in range(start_seed, end_seed)}

        self.studies = report_by_seed[start_seed]

        # Load the reports
        reports_by_x = []
        for x in range(num_x):
            reports_by_x.append([report_by_seed[seed][x]['generated'] for seed in range(start_seed, end_seed)])

        self.reports = reports_by_x
        self.labels = labels
        self.N, self.g = self.labels.shape

        self.dataset = []

        self.cache_dir = cache_dir

        for i in range(len(self.reports[:num_x])):
            for j in range(len(self.reports[i])):
                self.dataset.append(
                    {
                        "report": self.reports[i][j],
                        "label": self.labels[i][j],
                        "path": self.studies[i]["path"],
                    }
                )


    def __len__(self):
        return len(self.dataset)

    def get_image(self, idx):
        image_str = f"path={self.dataset[idx]['path']}&size={self.image_size}&mean={self.image_mean}&std={self.image_std}"
        cache_key = hashlib.md5(image_str.encode("utf-8")).hexdigest()
        cache_file = os.path.join(self.cache_dir, cache_key)

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                pixel_values = torch.load(f)
                return pixel_values

        image_file = os.path.join(self.mimic_root, self.dataset[idx]["path"])
        image = read_image(os.path.join(self.mimic_root, image_file), mode=ImageReadMode.RGB)
        pixel_values = self.image_transformations(image)

        with open(cache_file, "wb") as f:
            torch.save(pixel_values, f)

        return pixel_values


    def __getitem__(self, idx):
        report = clean_spaces(self.dataset[idx]["report"])
        input_ids = self.tokenizer.encode(
            report, max_length=256, truncation=True, padding="max_length", return_tensors="pt"
        )[0]
        pixel_values = self.get_image(idx)
        labels = self.dataset[idx]["label"]

        return {"input_ids": input_ids, "labels": labels, "pixel_values": pixel_values}

class ImageSentenceDataset(Dataset):
    def __init__(
        self,
        base,
        start_seed,
        end_seed,
        num_x,
        prefix,
        tokenizer_name,
        mimic_root,
        image_size,
        image_mean,
        image_std,
        rouge_type="rouge1",
        rouge_threshold=0.7,
        cache_dir="/tmp",
        transform=None,
        pre_transform=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mimic_root = mimic_root
        self.rouge_type = rouge_type
        self.rouge_threshold = rouge_threshold
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        image_transformations = MIMICTransform(image_size, image_mean, image_std)
        self.image_transformations = torch.jit.script(image_transformations)

        with open(os.path.join(base, f"{prefix}_{start_seed}_{end_seed}_rouge.json")) as f:
            self.rouge = json.load(f)

        with open(os.path.join(base, f"{prefix}_{start_seed}.jsonl")) as f:
            self.studies = [json.loads(l) for l in f]

        self.sentences = get_sentences(start_seed, end_seed, base, prefix)

        self.dataset = []

        self.cache_dir = cache_dir

        for i in range(len(self.sentences[:num_x])):
            for j in range(len(self.sentences[i])):
                for k in range(len(self.sentences[i][j])):
                    self.dataset.append(
                        {
                            "sentence": self.sentences[i][j][k],
                            "rouge": self.rouge[i][j][k],
                            "path": self.studies[i]["path"],
                        }
                    )


    def __len__(self):
        return len(self.dataset)

    def get_image(self, idx):
        image_str = f"path={self.dataset[idx]['path']}&size={self.image_size}&mean={self.image_mean}&std={self.image_std}"
        cache_key = hashlib.md5(image_str.encode("utf-8")).hexdigest()
        cache_file = os.path.join(self.cache_dir, cache_key)

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                pixel_values = torch.load(f)
                return pixel_values

        image_file = os.path.join(self.mimic_root, self.dataset[idx]["path"])
        image = read_image(os.path.join(self.mimic_root, image_file), mode=ImageReadMode.RGB)
        pixel_values = self.image_transformations(image)

        with open(cache_file, "wb") as f:
            torch.save(pixel_values, f)

        return pixel_values


    def __getitem__(self, idx):
        report = clean_spaces(self.dataset[idx]["sentence"])
        input_ids = self.tokenizer.encode(
            report, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        )[0]
        rouge = self.dataset[idx]["rouge"][self.rouge_type]
        labels = torch.tensor([1 if rouge > self.rouge_threshold else 0])
        pixel_values = self.get_image(idx)

        return {"input_ids": input_ids, "labels": labels, "pixel_values": pixel_values}


class ImageTextModel(nn.Module):
    """Image-Text model to predict a label based on a report and an image.
    """

    def __init__(self, image_encoder_model, bert_model, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.bert_model = AutoModel.from_pretrained(bert_model)
        self.image_encoder_model = AutoModel.from_pretrained(image_encoder_model)
        self.image_processor = AutoImageProcessor.from_pretrained(image_encoder_model)
        self.image_size = self.image_processor.size['height']
        self.image_mean = self.image_processor.image_mean
        self.image_std = self.image_processor.image_std

        text_embedding_dim = self.bert_model.config.hidden_size
        image_embedding_dim = self.image_encoder_model.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        self.mlp_head = nn.Sequential(
            nn.Linear(text_embedding_dim + image_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )



    def forward(self, input_ids, pixel_values, attention_mask=None, token_type_ids=None):
        text_outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        image_outputs = self.image_encoder_model(pixel_values=pixel_values).pooler_output

        text_outputs = self.dropout(text_outputs)
        image_outputs = self.dropout(image_outputs)

        outputs = torch.cat((text_outputs, image_outputs), dim=1)

        logits = self.mlp_head(outputs)

        out = F.log_softmax(logits, dim=1)

        return logits, out
