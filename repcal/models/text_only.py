import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import os
from tqdm.auto import tqdm
import json
import re

from transformers import AutoModel, AutoTokenizer

from repcal.utils.data import get_sentences

def clean_spaces(str):
        str = str.strip()
        str = str.replace('\n',' ')
        str = " ".join(re.split("\s+", str, flags=re.UNICODE))
        str = str.strip()
        return str


class TextOnlyDataset(Dataset):
    def __init__(self, labels, reports, tokenizer_name, transform=None, pre_transform=None):
        self.reports = reports
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.N, self.g = self.labels.shape

    def __len__(self):
        N, g = self.labels.shape
        return N * g

    def __getitem__(self, idx):
        i = idx // self.g
        j = idx % self.g
        report = clean_spaces(self.reports[i][j])
        input_ids = self.tokenizer.encode(report, max_length=512, truncation=True, padding="max_length", return_tensors="pt")[0]
        return {"input_ids": input_ids, "labels": self.labels[i][j]}


    @classmethod
    def from_file(cls, base, num_x, start_seed, end_seed, prefix, tokenizer_name):
        # Load the labels
        hard_label = {}
        soft_label = {}

        for seed in tqdm(range(start_seed, end_seed)):
            gold = pd.read_csv(os.path.join(base, f"{prefix}_{seed}_chexbert_y.csv"), index_col=0)
            pred = pd.read_csv(os.path.join(base, f"{prefix}_{seed}_chexbert_y_hat.csv"), index_col=0)
            hard_label[seed] = ((pred == gold).all(axis=1)) + 0
            soft_label[seed] = ((pred == gold).sum(axis=1)) / len(pred.columns)

        hard_label = torch.tensor(pd.DataFrame(hard_label).values)[:num_x]
        soft_label = torch.tensor(pd.DataFrame(soft_label).values)[:num_x]

        def read_studies(seed, base):
            with open(os.path.join(base, f"{prefix}_{seed}.jsonl")) as f:
                return [json.loads(l) for l in f]


        report_by_seed = {seed: read_studies(seed, base) for seed in range(start_seed, end_seed)}

        # Load the reports
        reports_by_x = []
        for x in range(num_x):
            reports_by_x.append([report_by_seed[seed][x]['generated'] for seed in range(start_seed, end_seed)])

        return cls(hard_label, reports_by_x, tokenizer_name)


class SentenceTextOnlyDataset(Dataset):
    def __init__(self, base, start_seed, end_seed, num_x, prefix, tokenizer_name, rouge_type="rouge1", rouge_threshold=0.7, transform=None, pre_transform=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.rouge_type = rouge_type
        self.rouge_threshold = rouge_threshold

        with open(os.path.join(base, f"{prefix}_{start_seed}_{end_seed}_rouge.json")) as f:
            self.rouge = json.load(f)

        self.sentences = get_sentences(start_seed, end_seed, base, prefix)

        self.dataset = []

        for i in range(len(self.sentences[:num_x])):
            for j in range(len(self.sentences[i])):
                for k in range(len(self.sentences[i][j])):
                    self.dataset.append({"sentence": self.sentences[i][j][k], "rouge": self.rouge[i][j][k]})


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        report = clean_spaces(self.dataset[idx]["sentence"])
        input_ids = self.tokenizer.encode(report, max_length=128, truncation=True, padding="max_length", return_tensors="pt")[0]
        rouge = self.dataset[idx]["rouge"][self.rouge_type]
        labels = torch.tensor([1 if rouge > self.rouge_threshold else 0])
        return {"input_ids": input_ids, "labels": labels}



class TextOnlyModel(nn.Module):
    """Fine-tune a pre-trained BERT model to predict a label for a given text."""
    def __init__(self, bert_model, dropout_rate=0.1, freeze_embeddings=False):
        super().__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.bert_model = AutoModel.from_pretrained(bert_model)

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        hidden = outputs[0]
        pooled = outputs[1]

        fc = self.dropout(pooled)
        fc = self.classifier(fc)

        out = F.log_softmax(fc, dim=1)

        return hidden, pooled, fc, out
