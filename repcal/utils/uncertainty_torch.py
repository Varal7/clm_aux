import torch
import numpy as np

EPSILON = 1e-6

#  M[i,j]: j-th generation for i-th example
#  L[i, j]: 0 if M[i,j] is acceptable, 1 otw
#  L_hat[i,j]: 0 if any of M[i, :j] is acceptable, 1 otw
#  s[i,j]: score of M[i,j]
#  s_hat[i,j]: cumulative score of M[i, :j]

#  tau: threshold
#  reject_if_sampleout: whether to reject an x if it takes too many generations

def test_case_0():
    s = torch.tensor([[1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      ])
    s_hat = s.cumsum(dim=1)
    tau = 0.5
    expected = torch.tensor([1, 1, 1, 1])

    actual = get_C_cutoff(s_hat, tau)
    print(actual)

    assert torch.all(expected == actual)


def test_case_1():
    s = torch.tensor([[1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      ])
    s_hat = s.cumsum(dim=1)
    tau = 3.5
    expected = torch.tensor([4, 4, 4, 4])

    actual = get_C_cutoff(s_hat, tau)
    print(actual)

    assert torch.all(expected == actual)


def test_case_2():
    s = torch.tensor([[0.3, 0.3, 0.3, 0.3, 0.3],
                      [1. , 0. , 0. , 0.2, 0.3],
                      [0.8, 0.8, 0.8, 0.8, 0.8],
                      [0.1, 0.1, 0.1, 0.1, 0.1],
                      ])
    s_hat = s.cumsum(dim=1)
    tau = 0.95
    expected = torch.tensor([4, 1, 2, 5])

    actual = get_C_cutoff(s_hat, tau)
    print(actual)

    assert torch.all(expected == actual)

def test_case_3():
    s = torch.tensor([[0.3, 0.3, 0.3, 0.3, 0.3],
                      [1. , 0. , 0. , 0.2, 0.3],
                      [0.8, 0.8, 0.8, 0.8, 0.8],
                      [0.1, 0.1, 0.1, 0.1, 0.1],
                      ])
    s_hat = s.cumsum(dim=1)
    tau = 0.95
    expected = torch.tensor([4, 1, 2, 0])

    actual = get_C_cutoff(s_hat, tau, reject_if_sampleout=True)
    print(actual)

    assert torch.all(expected == actual)


def get_C_cutoff(s_hat, tau, reject_if_sampleout=False):
    """
    s_hat[i,j]: cumulative score of M[i, :j]

    For each row i, we take the first j elements where
    j = argmin_j { s_hat[i,j] > tau }

    If there is no such j, then we take all the elements.

    Return the number of elements taken for each row.
    """
    C_cutoff = torch.zeros(s_hat.shape[0], dtype=torch.long)

    for i in range(s_hat.shape[0]):
        j = (s_hat[i] > tau).nonzero(as_tuple=True)[0]
        if j.numel() == 0:
            if reject_if_sampleout:
                C_cutoff[i] = 0
            else:
                C_cutoff[i] = s_hat.shape[1]
        else:
            C_cutoff[i] = j[0] + 1 # If take up to element j, then we need j+1 elements

    return C_cutoff

def get_Cs_Ls_taus(
        s_hat,
        L_hat,
        L_empty=0,
        sample_tau=None,
        num_tau_quantiles=None,
        reject_if_sampleout=False,
        condition_C_size_on_selected=True,
        condition_L_on_selected=True,
        C_sizes=None
    ):
    """
    M[i,j]: j-th generation for i-th example
    L[i, j]: 0 if M[i,j] is acceptable, 1 otw
    L_hat[i,j]: 0 if any of M[i, :j] is acceptable, 1 otw
    s[i,j]: score of M[i,j]
    s_hat[i,j]: cumulative score of M[i, :j]

    L_empty: 0 (loss for when we reject)

    reject_if_sampleout: whether to reject an x if it takes too many generations to reach tau
    condition_C_size_on_selected: if reject_if_sampleout, whether to condition C_size on selected
    C_sizes: size of the set M[i, :j] to use compute E[C]. If None, use torch.arange

    tau selection:
        sample_tau: if not None, number of tau to sample
        num_tau_quantiles: if not None, number of quantiles to use
    """
    assert sample_tau is None or num_tau_quantiles is None # only one of them can be not None

    taus = np.array(list(set(s_hat.flatten().tolist())))
    min_tau = taus.min() - EPSILON

    if C_sizes is None:
        C_sizes = torch.arange(1, s_hat.shape[1] + 1, dtype=torch.long).unsqueeze(0).repeat(s_hat.shape[0], 1)

    if sample_tau is not None:
        if len(taus) > sample_tau:
            taus = np.random.choice(taus, size=1000, replace=False)   # TODO: shuffle and take the first 1000

    if num_tau_quantiles is not None:
        taus = np.quantile(taus, np.linspace(0, 1, num_tau_quantiles))

    taus.sort()
    taus = np.concatenate([[min_tau], taus])

    L_avg = []
    C_size_avg = []
    rejection_rate = []


    N = s_hat.shape[0]

    shifted_L_hat = torch.cat([L_empty * torch.ones(N).unsqueeze(1), L_hat], dim=1)
    shifted_C_sizes = torch.cat([torch.zeros(N).unsqueeze(1), C_sizes], dim=1)

    for tau in taus:
        C_cutoff = get_C_cutoff(s_hat, tau, reject_if_sampleout=reject_if_sampleout)

        L_tau = shifted_L_hat[torch.arange(N), C_cutoff]
        C_size_tau = shifted_C_sizes[torch.arange(N), C_cutoff]

        if condition_L_on_selected:
            assert L_empty == 0
            L_avg_chosen = L_tau.sum() / (C_size_tau > 0).sum()
        else:
            L_avg_chosen = L_tau.sum() / N

        if condition_C_size_on_selected:
            C_size_tau_avg = C_size_tau.sum() / (C_size_tau > 0).sum()
        else:
            C_size_tau_avg = C_size_tau.mean()

        L_avg.append(L_avg_chosen)
        C_size_avg.append(C_size_tau_avg)
        rejection_rate.append((C_size_tau == 0).sum() / N)

    return C_size_avg, L_avg, taus, rejection_rate

if __name__ == '__main__':
    test_case_0()
    test_case_1()
    test_case_2()
    test_case_3()
