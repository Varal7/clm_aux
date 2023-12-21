from comet_ml import Experiment
import os
import json
import torch

import pandas as pd
from tqdm.auto import tqdm

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from repcal.utils.nli import get_sentences_with_num_x
from repcal.models.gnn import CustomNLIDataset, GNNModel

from transformers import HfArgumentParser

from dataclasses import dataclass, field

from typing import List

from torch_geometric.loader import DataLoader


@dataclass
class Arguments:
    base: str = field(default="/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part2")
    test_base: str = field(default="/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many")
    num_x: int = field(default=1000)
    test_num_x: int = field(default=1000)
    start_seed: int = field(default=42)
    end_seed: int = field(default=62)
    prefix: str = field(default="dev_sample_seed")

    dropout: float = field(default=0.1)
    num_layers: int = field(default=3)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=1e-1)
    num_epochs: int = field(default=10)
    batch_size: int = field(default=4)
    accumulate_grad_batches: int = field(default=4)

    save_dir: str = field(default="/storage/quach/snapshots/repg2/gnn")
    results_path: str = field(default="debug")

    workspace: str = field(default="varal7")
    project_name: str = field(default="rep-gnn-1")
    comet_tags: List[str] = field(default_factory=lambda: ["debug"])


def get_dataset(base, num_x, start_seed, end_seed, prefix):
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

    # Load the NLI graph
    scores = {}
    for i in tqdm(range(num_x)):
        scores[i] = torch.load(os.path.join(base, f"nli_{start_seed}_{end_seed}", f"scores_{i}.pt"))
    sentences =  get_sentences_with_num_x(start_seed, end_seed, num_x, base, prefix=prefix)
    lengths = torch.tensor([
        [len(x) for x in li]
        for li in sentences
    ])

    dataset = CustomNLIDataset(scores, hard_label, lengths, sentences)

    return dataset


def main():
    parser = HfArgumentParser((Arguments,))
    args = parser.parse_args_into_dataclasses()[0]


    experiment = Experiment(workspace=args.workspace, project_name=args.project_name, auto_output_logging="simple")
    result_path_stem = args.results_path.split("/")[-1].split('.')[0]
    snapshot_dir = os.path.join(args.save_dir, result_path_stem)
    os.makedirs(snapshot_dir, exist_ok=True)
    print("snapshot_dir: {}".format(snapshot_dir))

    args.snapshot_dir = snapshot_dir
    args.hostname = os.uname()[1]

    experiment.log_parameters(args.__dict__)
    experiment.set_name(result_path_stem)

    args.comet_url = experiment.url
    args.status = "running"

    with open(os.path.join(snapshot_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    with open(args.results_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(args.base, args.num_x, args.start_seed, args.end_seed, args.prefix)
    test_dataset = get_dataset(args.test_base, args.test_num_x, args.start_seed, args.end_seed, args.prefix)

    train_dataset = dataset[:int(len(dataset) * 0.9)]
    val_dataset = dataset[int(len(dataset) * 0.9):]

    num_node_features = dataset.num_node_features
    num_edge_features = dataset.num_edge_features
    num_classes = 2
    model = GNNModel(num_node_features, num_edge_features, num_classes, dropout=args.dropout, num_layers=args.num_layers)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.to(device)

    best_val_loos = 0
    best_val_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0
        pbar = tqdm(train_loader)
        gold = []
        pred = []

        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            _, _, fc, out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss = loss / args.accumulate_grad_batches
            loss.backward()
            running_loss += loss.item()
            experiment.log_metric("train_loss", loss.item())
            experiment.log_metric("train_running_loss", running_loss / (i + 1))
            pbar.set_description(f"Epoch {epoch}, train loss: {running_loss / (i + 1):.4f}")
            if (i + 1) % args.accumulate_grad_batches == 0:

                # Compute gradient norm
                total_norm = 0
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                experiment.log_metric("train_grad_norm", total_norm)

                # Clip gradient
                new_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                experiment.log_metric("train_grad_norm_clipped", new_norm)

                # Compute weight norm
                total_weight_norm = 0
                for p in model.parameters():
                    param_norm = p.data.norm(2)
                    total_weight_norm += param_norm.item() ** 2
                total_weight_norm = total_weight_norm ** (1. / 2)
                experiment.log_metric("train_weight_norm", total_weight_norm)

                optimizer.step()
                optimizer.zero_grad

            gold.extend(batch.y.cpu().tolist())
            pred.extend(F.softmax(fc, dim=-1)[:,1].cpu().tolist())

        auc = roc_auc_score(gold, pred)
        experiment.log_metric("train_auc", auc)

        gold = []
        pred = []

        model.eval()
        running_loss = 0
        pbar = tqdm(val_loader)
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            with torch.no_grad():
                _, _, fc, out = model(batch)
            loss = F.nll_loss(out, batch.y)
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch}, val loss: {running_loss / (i + 1):.4f}")

            gold.extend(batch.y.cpu().tolist())
            pred.extend(F.softmax(fc, dim=-1)[:,1].cpu().tolist())

        auc = roc_auc_score(gold, pred)
        loss = running_loss / (i + 1)
        experiment.log_metric("val_auc", auc)
        experiment.log_metric("val_loss", loss)

        if loss > best_val_loos:
            best_val_loos = loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(snapshot_dir, "best.pt"))


        torch.save(model.state_dict(), os.path.join(snapshot_dir, f"last.pt"))

    # Test

    model.load_state_dict(torch.load(os.path.join(snapshot_dir, "best.pt")))
    model.eval()

    # Test
    running_loss = 0
    pbar = tqdm(test_loader)
    gold = []
    pred = []
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        with torch.no_grad():
            _, _, fc, out = model(batch)
        loss = F.nll_loss(out, batch.y)
        running_loss += loss.item()
        pbar.set_description(f"Test loss: {running_loss / (i + 2):.4f}")
        gold.extend(batch.y.cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:,1].cpu().tolist())

    auc = roc_auc_score(gold, pred)
    experiment.log_metric("test_auc", auc)

    # Dev
    running_loss = 0
    pbar = tqdm(val_loader)
    gold = []
    pred = []
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        with torch.no_grad():
            _, _, fc, out = model(batch)
        loss = F.nll_loss(out, batch.y)
        running_loss += loss.item()
        pbar.set_description(f"Dev loss: {running_loss / (i + 2):.4f}")
        gold.extend(batch.y.cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:,1].cpu().tolist())

    auc = roc_auc_score(gold, pred)
    experiment.log_metric("dev_auc", auc)

    # Eval Train
    running_loss = 0
    pbar = tqdm(eval_train_loader)
    gold = []
    pred = []
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        with torch.no_grad():
            _, _, fc, out = model(batch)
        loss = F.nll_loss(out, batch.y)
        running_loss += loss.item()
        pbar.set_description(f"Eval train loss: {running_loss / (i + 2):.4f}")
        gold.extend(batch.y.cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:,1].cpu().tolist())

    auc = roc_auc_score(gold, pred)
    experiment.log_metric("eval_train_auc", auc)

    args.status = "done"
    with open(args.results_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

if __name__ == "__main__":
    main()
