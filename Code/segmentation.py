import torch


def argmax(a: torch.Tensor):
    """Find the maximum-scoring segmentation using Viterbi.

    Parameters
    ----------
    a: torch.tensor, shape [seq_len+1, seq_len+1, 2].
        Tensor of scores, defined as follows.
         - a[i, j, 0] is the score for leaving out segment i-j.
         - a[i, j, 1] is the score for keeping segment i-j.

    Returns
    -------
    score: scalar,
        Score of the highest-scoring segmentation

    y: torch.Tensor (dtype=long), shape [seq_len]
        The highest-scoring segmentation.
    """

    _, seq_len_plus_1, _ = a.shape

    m = torch.zeros(seq_len_plus_1)
    pi = torch.zeros(seq_len_plus_1, dtype=torch.long)
    pi_keep = torch.zeros(seq_len_plus_1, dtype=bool)

    pi[0] = 0
    pi_keep[0] = True

    for i in range(1, seq_len_plus_1):
        keep = []
        scores = []
        for j in range(i):
            if a[j, i, 0] > 0:
                scores.append(m[j] + a[j, i, 0])
                keep.append(False)
            else:
                scores.append(m[j])
                keep.append(True)
        max = torch.max(torch.tensor(scores), keepdim=True, dim=0)
        m[i] = max.values
        pi[i] = max.indices[0]
        pi_keep[i] = keep[max.indices[0]]

    y = []
    i = seq_len_plus_1 - 1
    while i > 0:
        y = [(pi[i], i, pi_keep[i])] + y
        i = pi[i]

    return torch.tensor(y)


# def logsumexp(a: torch.Tensor):
#     """Compute normalizing constant (log-sum-exp) using Forward.

#     Parameters
#     ----------
#     a: torch.tensor, shape [seq_len+1, seq_len+1, 2].
#         Tensor of scores, defined as follows.
#          - a[i, j, 0] is the score for leaving out segment i-j.
#          - a[i, j, 1] is the score for keeping segment i-j.

#     Returns
#     -------

#     log_normalizer: scalar,
#         The log sum exp of the score of every possible segmentation.
#     """

#     n1_plus_1, n2_plus_1, _ = a.shape

#     Q = torch.zeros(n1_plus_1, n2_plus_1)

#     Q[1:, 0] = torch.cumsum(a[1:, 0, 1], dim=0)
#     Q[0, 1:] = torch.cumsum(a[0, 1:, 2], dim=0)

#     # forward algorithm, fill in Q.
#     for i in range(1, n1_plus_1):
#         for j in range(1, n2_plus_1):
#             scores = torch.zeros(3,)
#             scores[0] = Q[i-1, j-1] + a[i, j, 0]
#             scores[1] = Q[i-1, j] + a[i, j, 1]
#             scores[2] = Q[i, j-1] + a[i, j, 2]
#             Q[i, j] = torch.logsumexp(scores, dim=0)

#     return Q[-1, -1]


if __name__ == '__main__':
    a = torch.tensor([[[0, 0], [10, 5], [6, 9]],
                      [[0, 0], [0, 0], [17, 18]],
                      [[0, 0], [0, 0], [0, 0]]])
    print(a.size())

    a[:, :, 0] = a[:, :, 0] - a[:, :, 1]
    a[:, :, 1] = torch.zeros(a[:, :, 1].size())

    path = argmax(a)
    print(path)
