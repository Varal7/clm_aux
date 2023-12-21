from comet_ml import Experiment
import os
import json
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset

from repcal.models.img_text_classifier import ImageSentenceDataset, ImageTextModel

from transformers import HfArgumentParser

from dataclasses import dataclass, field

from typing import List


@dataclass
class Arguments:
    base: str = field(
        default="/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/train_scorer_0_2000"
    )
    test_base: str = field(
        default="/Mounts/rbg-storage1/snapshots/repg2/ap_and_pa/checkpoint-15000/preds/valid_scorer_13000_14000"
    )
    pretrained_bert: str = field(default="emilyalsentzer/Bio_ClinicalBERT")
    image_encoder: str = field(default="google/vit-base-patch16-224-in21k")

    mimic_root: str = field(default="/storage/quach/MIMIC/physionet.org/files/mimic-cxr-resized-224")
    hidden_dim: int = field(default=512)

    num_x: int = field(default=2000)
    test_num_x: int = field(default=1000)
    start_seed: int = field(default=42)
    end_seed: int = field(default=62)
    prefix: str = field(default="dev_sample_seed")

    dropout: float = field(default=0.1)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=1e-1)
    num_epochs: int = field(default=10)
    batch_size: int = field(default=4)
    accumulate_grad_batches: int = field(default=4)

    save_dir: str = field(default="/storage/quach/snapshots/repg2/image_sentence")
    results_path: str = field(default="debug")

    workspace: str = field(default="varal7")
    project_name: str = field(default="rep-image-sentence")
    comet_tags: List[str] = field(default_factory=lambda: ["debug"])

    rouge_type: str = field(default="rouge1")
    rouge_threshold: float = field(default=0.7)

    cache_dir: str = field(default="/storage/quach/cache/mimic_cache")
    cache_only: bool = field(default=False)


def main():
    parser = HfArgumentParser((Arguments,))
    args = parser.parse_args_into_dataclasses()[0]

    experiment = Experiment(workspace=args.workspace, project_name=args.project_name, auto_output_logging="simple")
    result_path_stem = args.results_path.split("/")[-1].split(".")[0]
    snapshot_dir = os.path.join(args.save_dir, result_path_stem)
    os.makedirs(snapshot_dir, exist_ok=True)
    print("snapshot_dir: {}".format(snapshot_dir))

    os.makedirs(args.cache_dir, exist_ok=True)

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

    model = ImageTextModel(args.image_encoder, args.pretrained_bert, hidden_dim=args.hidden_dim, dropout_rate=args.dropout)

    data_params = {
        "start_seed": args.start_seed,
        "end_seed": args.end_seed,
        "prefix": args.prefix,
        "tokenizer_name": args.pretrained_bert,
        "mimic_root": args.mimic_root,
        "image_size": model.image_size,
        "image_mean": model.image_mean,
        "image_std": model.image_std,
        "rouge_type": args.rouge_type,
        "rouge_threshold": args.rouge_threshold,
        "cache_dir": args.cache_dir,
    }

    dataset = ImageSentenceDataset(
        args.base,
        num_x=args.num_x,
        **data_params,
    )

    test_dataset = ImageSentenceDataset(
        args.test_base,
        num_x=args.test_num_x,
        **data_params,
    )

    if args.cache_only:
        print("Caching train dataset")
        for i in tqdm(range(len(dataset))):
            _ = dataset[i]

        print("Caching test dataset")
        for i in tqdm(range(len(test_dataset))):
            _ = test_dataset[i]

        return

    # Split into train and validation
    train_dataset = Subset(dataset, list(range(0, int(len(dataset) * 0.9))))
    val_dataset = Subset(dataset, list(range(int(len(dataset) * 0.9), len(dataset))))

    def collate_fn(examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples], dtype=torch.long)
        pixel_values = torch.stack([example["pixel_values"] for example in examples])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
            fc, out = model(batch["input_ids"].to(device), batch['pixel_values'].to(device))

            loss = F.nll_loss(out, batch["labels"].to(device))
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
                total_norm = total_norm ** (1.0 / 2)
                experiment.log_metric("train_grad_norm", total_norm)

                # Clip gradient
                new_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                experiment.log_metric("train_grad_norm_clipped", new_norm)

                # Compute weight norm
                total_weight_norm = 0
                for p in model.parameters():
                    param_norm = p.data.norm(2)
                    total_weight_norm += param_norm.item() ** 2
                total_weight_norm = total_weight_norm ** (1.0 / 2)
                experiment.log_metric("train_weight_norm", total_weight_norm)

                optimizer.step()
                optimizer.zero_grad

            gold.extend(batch["labels"].cpu().tolist())
            pred.extend(F.softmax(fc, dim=-1)[:, 1].cpu().tolist())

        auc = roc_auc_score(gold, pred)
        experiment.log_metric("train_auc", auc)

        gold = []
        pred = []

        model.eval()
        running_loss = 0
        pbar = tqdm(val_loader)
        for i, batch in enumerate(pbar):
            with torch.no_grad():
                fc, out = model(batch["input_ids"].to(device), batch['pixel_values'].to(device))
            loss = F.nll_loss(out, batch["labels"].to(device))
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch}, val loss: {running_loss / (i + 1):.4f}")

            gold.extend(batch["labels"].cpu().tolist())
            pred.extend(F.softmax(fc, dim=-1)[:, 1].cpu().tolist())

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
        with torch.no_grad():
            fc, out = model(batch["input_ids"].to(device), batch['pixel_values'].to(device))
        loss = F.nll_loss(out, batch["labels"].to(device))
        running_loss += loss.item()
        pbar.set_description(f"Test loss: {running_loss / (i + 2):.4f}")
        gold.extend(batch["labels"].cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:, 1].cpu().tolist())

    auc = roc_auc_score(gold, pred)
    experiment.log_metric("test_auc", auc)

    # Dev
    running_loss = 0
    pbar = tqdm(val_loader)
    gold = []
    pred = []
    for i, batch in enumerate(pbar):
        with torch.no_grad():
            fc, out = model(batch["input_ids"].to(device), batch['pixel_values'].to(device))
        loss = F.nll_loss(out, batch["labels"].to(device))
        running_loss += loss.item()
        pbar.set_description(f"Dev loss: {running_loss / (i + 2):.4f}")
        gold.extend(batch["labels"].cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:, 1].cpu().tolist())

    auc = roc_auc_score(gold, pred)
    experiment.log_metric("dev_auc", auc)

    # Eval Train
    running_loss = 0
    pbar = tqdm(eval_train_loader)
    gold = []
    pred = []
    for i, batch in enumerate(pbar):
        with torch.no_grad():
            fc, out = model(batch["input_ids"].to(device), batch['pixel_values'].to(device))
        loss = F.nll_loss(out, batch["labels"].to(device))
        running_loss += loss.item()
        pbar.set_description(f"Eval train loss: {running_loss / (i + 2):.4f}")
        gold.extend(batch["labels"].cpu().tolist())
        pred.extend(F.softmax(fc, dim=-1)[:, 1].cpu().tolist())

    auc = roc_auc_score(gold, pred)
    experiment.log_metric("eval_train_auc", auc)

    args.status = "done"
    with open(args.results_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
