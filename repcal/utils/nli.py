import torch
from tqdm.auto import tqdm

from repcal.utils.data import get_sentences as get_sentences_without_num_x

def get_sentences_with_num_x(start_seed, end_seed, num_x, base, prefix="dev_sample_seed"):
    return get_sentences_without_num_x(start_seed, end_seed, base, prefix)[:num_x]

def get_report_scores(entailment, length):
    # Score for a sentence B is:  sum of probs of sentences A that entail it
    # Score for a report is the average score of its sentences

    cl  = length
    temp = torch.cat([torch.tensor([0]), (entailment).sum(dim=0).cumsum(dim=-1)], dim=0)
    top = cl.cumsum(dim=0)
    bot = torch.cat([torch.tensor([0]), top[:-1]], dim=0)

    return torch.where(cl != 0, (temp[top] - temp[bot]) / cl, 0)
