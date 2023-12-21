import json
import argparse
from repcal.models.chexbert import CheXbert
from repcal.utils.data import minibatch
import pandas as pd

device = "cuda:0"

DEFAULT_CHEXBERT = "/Mounts/rbg-storage1/snapshots/repg2/chexbert/chexbert.pth"

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

CONDITIONS = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str)
    parser.add_argument("--bert_path", type=str, default="bert-base-uncased")
    parser.add_argument("--chexbert_path", type=str, default=DEFAULT_CHEXBERT)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    with open(args.jsonl) as f:
        data = [json.loads(line) for line in f]

    model = CheXbert(
        bert_path=args.bert_path,
        chexbert_path=args.chexbert_path,
        device=device,
    ).to(device)

    table = {'chexbert_y_hat': [], 'chexbert_y': [], 'y_hat': [], 'y': [], 'study_id': []}
    for batch in minibatch(data, args.batch_size):
        y_hat, y, study_id = batch['generated'], batch['report'], batch['study_id']
        table['chexbert_y_hat'].extend([i + [j] for i, j in zip(model(list(y_hat)).tolist(), list(study_id))])
        table['chexbert_y'].extend([i + [j] for i, j in zip(model(list(y)).tolist(), list(study_id))])
        table['y_hat'].extend(y_hat)
        table['y'].extend(y)
        table['study_id'].extend(study_id)


    columns = CONDITIONS + ['study_id']
    df_y_hat = pd.DataFrame.from_records(table['chexbert_y_hat'], columns=columns)
    df_y = pd.DataFrame.from_records(table['chexbert_y'], columns=columns)

    df_y_hat.to_csv(args.jsonl.replace('.jsonl', '_chexbert_y_hat.csv'))
    df_y.to_csv(args.jsonl.replace('.jsonl', '_chexbert_y.csv'))

    df_y_hat = df_y_hat.drop(['study_id'], axis=1)
    df_y = df_y.drop(['study_id'], axis=1)

    df_y_hat = (df_y_hat == 1)
    df_y = (df_y == 1)

    tp = (df_y_hat * df_y).astype(float)

    fp = (df_y_hat * ~df_y).astype(float)
    fn = (~df_y_hat * df_y).astype(float)

    tp_cls = tp.sum()
    fp_cls = fp.sum()
    fn_cls = fn.sum()

    tp_eg = tp.sum(1)
    fp_eg = fp.sum(1)
    fn_eg = fn.sum(1)

    precision_class = (tp_cls / (tp_cls + fp_cls)).fillna(0)
    recall_class = (tp_cls / (tp_cls + fn_cls)).fillna(0)
    f1_class = (tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls))).fillna(0)

    scores = {
        'ce_precision_macro': precision_class.mean(),
        'ce_recall_macro': recall_class.mean(),
        'ce_f1_macro': f1_class.mean(),
        'ce_precision_micro': tp_cls.sum() / (tp_cls.sum() + fp_cls.sum()),
        'ce_recall_micro': tp_cls.sum() / (tp_cls.sum() + fn_cls.sum()),
        'ce_f1_micro': tp_cls.sum() / (tp_cls.sum() + 0.5 * (fp_cls.sum() + fn_cls.sum())),
        'ce_precision_example': (tp_eg / (tp_eg + fp_eg)).fillna(0).mean(),
        'ce_recall_example': (tp_eg / (tp_eg + fn_eg)).fillna(0).mean(),
        'ce_f1_example': (tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).fillna(0).mean(),
        'ce_num_examples': float(len(df_y_hat)),
    }

    class_scores_dict = {
       **{'ce_precision_' + k: v for k, v in precision_class.to_dict().items()},
       **{'ce_recall_' + k: v for k, v in recall_class.to_dict().items()},
       **{'ce_f1_' + k: v for k, v in f1_class.to_dict().items()},
    }
    pd.DataFrame(class_scores_dict, index=['i',]).to_csv(args.jsonl.replace('.jsonl', '_chexbert_scores.csv'))

    print(scores)
