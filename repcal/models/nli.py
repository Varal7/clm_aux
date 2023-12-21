from repcal.models.bertnli import BERTNLI
import gzip
import torch
import torch.nn as nn

DEFAULT_NLI_PATH = "/Mounts/rbg-storage1/users/quach/cxr-project/ifcc/resources/model_medrad_19k/model.dict.gz"

def remove_duplicates(func):
    def wrapper(self, cur_all):
        unique_items = list(set(cur_all))
        index_map = {item: i for i, item in enumerate(unique_items)}
        result = func(self, unique_items)
        output = [[torch.tensor(0)] * len(cur_all) for _ in range(len(cur_all))]
        for i, left in enumerate(cur_all):
            for j, right in enumerate(cur_all):
                index_i = index_map[left]
                index_j = index_map[right]
                output[i][j] = result[index_i][index_j]

        return torch.stack([torch.stack(row) for row in output])
    return wrapper


class NLI(nn.Module):
    def __init__(self, nli_path=DEFAULT_NLI_PATH, device="cpu", bs=128):
        super().__init__()
        self.bertnli = BERTNLI("bert-base-uncased", bert_type="bert", length=384, force_lowercase=True, device='cpu')
        self.nli_path = nli_path
        with gzip.open(nli_path, 'rb') as f:
            states_dict = torch.load(f, map_location=torch.device('cpu'))

        self.bertnli.load_state_dict(states_dict, strict=False)
        self.bertnli.to(device)
        self.bertnli.eval()
        self.bs = bs
        assert self.bertnli.cls == "linear"


    def forward(self, sent1s, sent2s):
        N = len(sent1s)
        assert N == len(sent2s)
        out = []

        for i in range(0, N, self.bs):
            tok = self.bertnli.tokenizer(sent1s[i:i+self.bs], sent2s[i:i+self.bs], padding=True, truncation=True, return_tensors="pt")

            for k in tok:
                tok[k] = tok[k].to(self.bertnli.device)

            with torch.no_grad():
                _, cls = self.bertnli.bert(**tok).values()
                cls = self.bertnli.dropout(cls)
                out.append(self.bertnli.linear(cls))

        out = torch.cat(out, dim=0)
        return out

    @remove_duplicates
    def eval_all(self, cur_all):
        scores = []
        for k in range(len(cur_all)):
            left = cur_all
            right = [cur_all[k] for _ in range(len(left))]
            with torch.no_grad():
                score = self.forward(left, right).cpu()
            scores.append(score)
        return scores
