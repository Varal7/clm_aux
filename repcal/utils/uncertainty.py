import numpy as np

EPSILON = 1e-6

#  M[i,j]: j-th generation for i-th example
#  L[i, j]: 0 if M[i,j] is acceptable, 1 otw
#  set_losses[i,j]: 0 if any of M[i, :j] is acceptable, 1 otw
#  s[i,j]: score of M[i,j]
#  set_scores[i,j]: score for M[i, :j]

#  tau: threshold
#  reject_if_sampleout: whether to reject an x if it takes too many generations

def test_case_0():
    s = np.array([[1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      [1., 1., 1., 1., 1., ],
                      ])
    set_scores = np.cumsum(s, axis=1)
    tau = 0.5
    expected = np.array([1, 1, 1, 1])

    actual = get_C_cutoff(set_scores, tau)
    print(actual)

    assert np.all(expected == actual)


def test_case_1():
    s = np.array([[1., 1., 1., 1., 1., ],
                  [1., 1., 1., 1., 1., ],
                  [1., 1., 1., 1., 1., ],
                  [1., 1., 1., 1., 1., ],
    ])
    set_scores = np.cumsum(s, axis=1)
    tau = 3.5
    expected = np.array([4, 4, 4, 4])

    actual = get_C_cutoff(set_scores, tau)
    print(actual)

    assert np.all(expected == actual)


def test_case_2():
    s = np.array([[0.3, 0.3, 0.3, 0.3, 0.3],
                      [1. , 0. , 0. , 0.2, 0.3],
                      [0.8, 0.8, 0.8, 0.8, 0.8],
                      [0.1, 0.1, 0.1, 0.1, 0.1],
                      ])
    set_scores = np.cumsum(s, axis=1)
    tau = 0.95
    expected = np.array([4, 1, 2, 5])

    actual = get_C_cutoff(set_scores, tau)
    print(actual)

    assert np.all(expected == actual)

def test_case_3():
    s = np.array([[0.3, 0.3, 0.3, 0.3, 0.3],
                      [1. , 0. , 0. , 0.2, 0.3],
                      [0.8, 0.8, 0.8, 0.8, 0.8],
                      [0.1, 0.1, 0.1, 0.1, 0.1],
                      ])
    set_scores = np.cumsum(s, axis=1)
    tau = 0.95
    expected = np.array([4, 1, 2, 0])

    actual = get_C_cutoff(set_scores, tau, reject_if_sampleout=True)
    print(actual)

    assert np.all(expected == actual)

def test_case_4():
    set_scores = np.array([
        [0., 0., 1., 0., 1.],
        [0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1.],
    ])
    tau = 0.5
    expected = np.array([3, 2, 0, 1])

    actual = get_C_cutoff(set_scores, tau, reject_if_sampleout=True)
    print(actual)

    assert np.all(expected == actual)


def test_case_5():
    set_scores = np.array([
        [0., 0., 1., 0., 1.],
        [0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1.],
    ])
    tau = 0.5
    expected = np.array([3, 2, 5, 1, 1])

    actual = get_C_cutoff(set_scores, tau, reject_if_sampleout=False)
    print(actual)

    assert np.all(expected == actual)


def get_C_cutoff(set_scores, tau, reject_if_sampleout=False):
    """
    set_scores[i,j]: cumulative score of M[i, :j]

    For each row i, we take the first j elements where
    j = argmin_j { set_scores[i,j] > tau }

    If there is no such j, then we take all the elements.

    Return the number of elements taken for each row.
    """
    threshold = np.full(set_scores.shape, tau)
    mask = set_scores <= threshold
    indices = np.argmax(mask[:,:-1] ^ mask[:,1:], axis=1) + 2
    indices[mask[:,0] == False] = 1
    indices[(mask).all(axis=1)] = 0 if reject_if_sampleout else set_scores.shape[1]

    return indices

def get_Cs_Ls_taus(
        set_scores,
        set_losses,
        loss_if_reject=0,
        sample_tau=None,
        num_tau_quantiles=None,
        reject_if_sampleout=False,
        condition_C_size_on_selected=True,
        condition_L_on_selected=True,
    ):
    """
    M[i,j]: j-th generation for i-th example
    L[i, j]: 0 if M[i,j] is acceptable, 1 otw
    set_losses[i,j]: 0 if any of M[i, :j] is acceptable, 1 otw
    s[i,j]: score of M[i,j]
    set_scores[i,j]: cumulative score of M[i, :j]

    loss_if_reject: 0 (loss for when we reject)

    reject_if_sampleout: whether to reject an x if it takes too many generations to reach tau
    condition_C_size_on_selected: if reject_if_sampleout, whether to condition C_size on selected

    tau selection:
        sample_tau: if not None, number of tau to sample
        num_tau_quantiles: if not None, number of quantiles to use
    """
    assert sample_tau is None or num_tau_quantiles is None # only one of them can be not None

    taus = np.unique(set_scores)
    min_tau = taus.min() - EPSILON

    if sample_tau is not None:
        if len(taus) > sample_tau:
            np.random.shuffle(taus)
            taus = taus[:sample_tau]

    if num_tau_quantiles is not None:
        taus = np.quantile(taus, np.linspace(0, 1, num_tau_quantiles))

    taus.sort()
    taus = np.concatenate([[min_tau], taus])

    L_avg = []
    C_size_avg = []
    rejection_rate = []

    N = set_scores.shape[0]

    shifted_set_losses = np.concatenate([loss_if_reject * np.ones(N).reshape(-1, 1), set_losses], axis=1)

    for tau in taus:
        C_cutoff = get_C_cutoff(set_scores, tau, reject_if_sampleout=reject_if_sampleout)
        L_tau = shifted_set_losses[np.arange(N), C_cutoff]
        C_size_tau = C_cutoff

        if condition_L_on_selected:
            assert loss_if_reject == 0
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
    test_case_4()
    test_case_5()
