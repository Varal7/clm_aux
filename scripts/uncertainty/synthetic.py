import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

#  from repcal.utils.uncertainty import get_Cs_Ls_taus, get_C_cutoff

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

class SetData(Dataset):
    def __init__(self, L, probs):
        L_hat = L.cumprod(dim=1)
        self.L = L
        self.L_hat = L_hat
        self.N = L.shape[0]
        self.g = L.shape[1]
        self.probs = probs

    def __len__(self):
        return self.N * self.g

    def __getitem__(self, idx):
        i = idx // self.g
        j = idx % self.g
        mask = torch.zeros(self.g)
        mask[:j+1] = 1
        probs = torch.zeros(self.g)
        probs[:j+1] = self.probs[i, :j+1]
        return probs, self.L_hat[i, j],  mask

class SetNN(nn.Module):

    def __init__(self, input_dim, hidden_dim=16, binary_output=False):
        super(SetNN, self).__init__()
        self.input_to_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        self.hidden_to_output = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.binary_output = binary_output

    def fit(self, x, y, mask, lr=0.1, max_iter=50):
        optimizer = optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            if self.binary_output:
                loss = F.binary_cross_entropy_with_logits(self(x, mask), y)
            else:
                loss = F.mse_loss(self(x, mask), y)
            loss.backward()
            return loss
        print(eval())
        optimizer.step(eval)
        print(eval())

    def forward(self, x, mask):
        # [batch, max_set_size, hidden_dim]

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        hiddens = self.input_to_hidden(x)

        # apply mask to handle padding
        mask = mask.unsqueeze(-1)
        hiddens_masked = hiddens * mask

        mask_sum = mask.sum(dim=1)
        hiddens_sum = hiddens_masked.sum(dim=1)
        hiddens_mean = hiddens_sum / mask_sum

        # apply very large positive value to masked elements
        mask_broadcasted = mask.expand_as(hiddens)

        hiddens_masked_for_min = hiddens_masked.clone()
        hiddens_masked[mask_broadcasted == 0] = -float('inf')
        hiddens_masked_for_min[mask_broadcasted == 0] = float('inf')
        hiddens_max = hiddens_masked.max(dim=1).values
        hiddens_min = hiddens_masked_for_min.min(dim=1).values

        set_features = [
            hiddens_sum,
            hiddens_mean,
            hiddens_max,
            hiddens_min,
        ]

        set_features = torch.cat(set_features, dim=1)
        output = self.hidden_to_output(set_features).squeeze(-1)
        return output


def main():

    N, g = 1000, 20

    p = torch.rand(N) * 0.99 + 0.01

    L = (torch.rand(N, g) > p.unsqueeze(-1)).float()
    L_hat = L.cumprod(dim=1)

    s = torch.ones(N, g)
    s_hat = s.cumsum(dim=1)

    s_prob = torch.ones(N, g) * p.unsqueeze(-1)
    s_hat_geo = (- torch.log(1 - s_prob)).cumsum(dim=1)
    s_hat_marginal = (- torch.log(1 - s_prob)).cumsum(dim=1) - torch.log(s_prob)
    s_hat_prob = s_prob.cumsum(dim=1)
    s_hat_fixed = torch.cat([torch.zeros(N).unsqueeze(1), (- torch.log(1 - s_prob)).cumsum(dim=1)], dim=1)[:,:-1] - torch.log(s_prob)

    dataset = SetData(L, s_prob)

    dataloader = DataLoader(dataset, batch_size=20000, shuffle=True)

    model = SetNN(1, binary_output=True)

    #  optimizer = optim.Adam(model.parameters(), lr=0.01)
    #  for epoch in range(10):
    #      for x, y, mask in dataloader:
    #          output = model(x, mask)
    #          loss = F.binary_cross_entropy_with_logits(output, y)
    #          print(loss)
    #          loss.backward()
    #          optimizer.step()
    #          optimizer.zero_grad()


    for epoch in range(2):
        for x, y, mask in dataloader:
            model.fit(x, y, mask)


    for x, y, mask in dataloader:
        output = model(x, mask)
        print(roc_auc_score(y.detach().numpy(), output.detach().numpy()))



if __name__ == '__main__':
    main()
