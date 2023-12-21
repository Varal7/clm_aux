import json
import os
import torch
import pandas as pd
from tqdm.auto import tqdm

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from repcal.models.img_text_classifier import ImageReportDataset, ImageTextModel

from dataclasses import dataclass, field

from transformers import HfArgumentParser


@dataclass
class Arguments:
    base: str = field(
        default="/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/train_scorer_0_2000"
    )
    name: str = field(default="may2-20seeds-67ae9a0d66dfbec7893eb2a47ac09972")
    model_dir: str = field(default="/storage/quach/snapshots/repg2/image-report")
    num_x: int = field(default=1000)
    start_seed: int = field(default=42)
    end_seed: int = field(default=62)
    prefix: str = field(default="dev_sample_seed")
    mimic_root: str = field(default="/storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224")
    cache_dir: str = field(default="/storage/quach/cache/mimic_cache")

def main():
    parser = HfArgumentParser((Arguments,))
    args = parser.parse_args_into_dataclasses()[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    snapshot_dir = os.path.join(args.model_dir, args.name)
    snapshot = os.path.join(snapshot_dir, "best.pt")
    model_args = json.load(open(os.path.join(snapshot_dir, "args.json")))
    #  pretrained_bert = model_args['pretrained_bert']
    state_dict = torch.load(snapshot, map_location="cpu")


    model = ImageTextModel(model_args['image_encoder'], model_args['pretrained_bert'], hidden_dim=model_args['hidden_dim'], dropout_rate=model_args['dropout'])
    model.to(device)
    model.load_state_dict(state_dict)

    data_params = {
        "start_seed": args.start_seed,
        "end_seed": args.end_seed,
        "prefix": args.prefix,
        "tokenizer_name": model_args['pretrained_bert'],
        "mimic_root": args.mimic_root,
        "image_size": model.image_size,
        "image_mean": model.image_mean,
        "image_std": model.image_std,
        "cache_dir": args.cache_dir,
    }



    dataset = ImageReportDataset(
        args.base,
        num_x=args.num_x,
        **data_params,
    )

    def collate_fn(examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples], dtype=torch.long)
        pixel_values = torch.stack([example["pixel_values"] for example in examples])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }



    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    pred = []
    gold = []
    model.eval()

    running_loss = 0

    pbar = tqdm(loader)

    for i, batch in enumerate(pbar):
        with torch.no_grad():
            fc, out = model(batch["input_ids"].to(device), batch['pixel_values'].to(device))
        loss = F.nll_loss(out, batch['labels'].to(device))
        running_loss += loss.item()

        gold.extend(batch['labels'].cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:,1].cpu().tolist())

    print("AUC", roc_auc_score(gold, pred))

    gold = torch.tensor(gold)
    pred = torch.tensor(pred)

    scores = pred.view(-1, args.end_seed - args.start_seed)

    filename = (os.path.join(args.base, "image-report_{}_{}".format(args.start_seed, args.end_seed), "image-report-{}.pt".format(args.name)))

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(scores, filename)

    print("Saved to", filename)

if __name__ == "__main__":
    main()
