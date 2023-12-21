import json
import os
import torch
import pandas as pd
from tqdm.auto import tqdm

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from repcal.models.gnn import CustomNLIDataset, GNNModel
from torch_geometric.loader import DataLoader

from dataclasses import dataclass, field

from transformers import HfArgumentParser

from repcal.utils.nli import get_sentences_with_num_x, get_report_scores

@dataclass
class Arguments:
    base: str = field(default="/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part3")
    name: str = field(default="mar24-2aa45685b8a64374288231db1f54e2b06")
    model_dir: str = field(default="/storage/quach/snapshots/repg2/gnn/")
    num_x: int = field(default=1000)
    start_seed: int = field(default=42)
    end_seed: int = field(default=62)
    prefix: str = field(default="dev_sample_seed")

def main():
    parser = HfArgumentParser((Arguments,))
    args = parser.parse_args_into_dataclasses()[0]

    base = args.base
    name = args.name
    model_dir = args.model_dir
    num_x = args.num_x
    start_seed = args.start_seed
    end_seed = args.end_seed
    prefix = args.prefix

    hard_label = {}

    for seed in tqdm(range(start_seed, end_seed)):
        gold = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y.csv"), index_col=0)
        pred = pd.read_csv(os.path.join(base, f"dev_sample_seed_{seed}_chexbert_y_hat.csv"), index_col=0)
        hard_label[seed] = ((pred == gold).all(axis=1)) + 0

    hard_label = torch.tensor(pd.DataFrame(hard_label).values)

    scores = {}
    for i in tqdm(range(num_x)):
        scores[i] = torch.load(os.path.join(base, "nli_{}_{}".format(start_seed, end_seed), f"scores_{i}.pt"))

    sentences =  get_sentences_with_num_x(start_seed, end_seed, num_x, base, prefix="dev_sample_seed")

    lengths = torch.tensor([
        [len(x) for x in li]
        for li in sentences
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate the dataset and model
    dataset = CustomNLIDataset(scores, hard_label, lengths, sentences)
    num_node_features = dataset.num_node_features
    num_edge_features = dataset.num_edge_features
    num_classes = 2  # Assuming binary classification
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    snapshot_dir = os.path.join(model_dir, name)
    snapshot = os.path.join(snapshot_dir, "best.pt")
    args = json.load(open(os.path.join(snapshot_dir, "args.json")))
    state_dict = torch.load(snapshot, map_location="cpu")

    model = GNNModel(num_node_features, num_edge_features, num_classes, dropout=args['dropout'], num_layers=args['num_layers'])
    model.to(device)
    model.load_state_dict(state_dict)

    pred = []
    gold = []
    model.eval()

    running_loss = 0

    pbar = tqdm(loader)

    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        with torch.no_grad():
            _, _, fc, out = model(batch)
        loss = F.nll_loss(out, batch.y)
        running_loss += loss.item()

        gold.extend(batch.y.cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:,1].cpu().tolist())

    print("AUC", roc_auc_score(gold, pred))

    gold = torch.tensor(gold)
    pred = torch.tensor(pred)

    assert (hard_label == gold.view(-1, 20)).all()

    scores = pred.view(-1, 20)

    filename = (os.path.join(base, "nli_{}_{}".format(start_seed, end_seed), "gnn-{}.pt".format(name)))

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(scores, filename)

    print("Saved to", filename)

if __name__ == "__main__":
    main()
