import json
import os
import torch
import pandas as pd
from tqdm.auto import tqdm

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from repcal.models.text_only import TextOnlyDataset, TextOnlyModel

from dataclasses import dataclass, field

from transformers import HfArgumentParser


@dataclass
class Arguments:
    base: str = field(default="/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/many_part3")
    name: str = field(default="apr-127fae34627f483285907f484fdb3a3a2d")
    model_dir: str = field(default="/storage/quach/snapshots/repg2/text_only/")
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    snapshot_dir = os.path.join(model_dir, name)
    snapshot = os.path.join(snapshot_dir, "best.pt")
    model_args = json.load(open(os.path.join(snapshot_dir, "args.json")))
    pretrained_bert = model_args['pretrained_bert']
    state_dict = torch.load(snapshot, map_location="cpu")

    model = TextOnlyModel(bert_model=model_args['pretrained_bert'], dropout_rate=model_args['dropout'])
    model.to(device)
    model.load_state_dict(state_dict)

    dataset = TextOnlyDataset.from_file(base, num_x, start_seed, end_seed, prefix, pretrained_bert)

    def collate_fn(examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }



    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    pred = []
    gold = []
    model.eval()

    running_loss = 0

    pbar = tqdm(loader)

    for i, batch in enumerate(pbar):
        with torch.no_grad():
            _, _, fc, out = model(batch['input_ids'].to(device))
        loss = F.nll_loss(out, batch['labels'].to(device))
        running_loss += loss.item()

        gold.extend(batch['labels'].cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:,1].cpu().tolist())

    print("AUC", roc_auc_score(gold, pred))

    gold = torch.tensor(gold)
    pred = torch.tensor(pred)

    assert (hard_label == gold.view(-1, 20)).all()

    scores = pred.view(-1, 20)

    filename = (os.path.join(base, "text-only_{}_{}".format(start_seed, end_seed), "text-only-{}.pt".format(name)))

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(scores, filename)

    print("Saved to", filename)

if __name__ == "__main__":
    main()
