# deep learning for structured data
# assignment 2 question 2.1
# original author: vlad niculae <v.niculae@uva.nl>
# license: MIT

# STUDENT ID: 12419273, Dewi Timman

import torch


def standard_nw_table(seq1, seq2):
    """Create a score table corresponding to the default NW setup.

    This means, use a reward score of 1 for matching identical symbols,
    and a gap penalty of -1 (for every insertion and deletion).

    Parameters
    ----------

    seq1: string, length n1
        sequence of characters in the first sequence.

    seq2: string, length n2
        sequence of characters in the second sequence.

    Returns
    -------

    a: torch.tensor, shape [n1+1, n2+1, 3].
        Tensor of scores for the Needleman-Wunsch DP, defined as follows.
         - a[i, j, 0] is the score for matching the seq1 character at position
           i-1 to the seq2 character at position j-1.
         - a[i, j, 1] is the score for skipping position i in seq1 (while
           looking for a match for position j in seq2.)
         - a[i, j, 2] is the score for skipping position j in seq2 (while
           looking for a match for position i in seq1.)
    """
    n1 = len(seq1)
    n2 = len(seq2)
    a = torch.zeros(n1+1, n2+1, 3)

    for i, char in enumerate(seq1):
        for j, char2 in enumerate(seq2):
            if char == char2:
                a[i+1, j+1, 0] = 1

    a[:, :, 1:] = -1

    return a


def argmax(a: torch.Tensor):
    """Find the maximum-score alignment between two sequences.

    Uses the Needleman-Wunsch dynamic program.

    Parameters
    ----------
    a: torch.tensor, shape [n1+1, n2+1, 3].
        Tensor of scores for the Needleman-Wunsch DP, defined as follows.
         - a[i, j, 0] is the score for matching the seq1 character at position
           i-1 to the seq2 character at position j-1.
         - a[i, j, 1] is the score for skipping position i in seq1 (while
           looking for a match for position j in seq2.)
         - a[i, j, 2] is the score for skipping position j in seq2 (while
           looking for a match for position i in seq1.)

    Returns
    -------
    aligned_pairs: list of tuples [(i, j)]
        List of all matched pairs where seq1[i] was matched to seq2[j].

    path: list of int,
        List of the decisions made at each step in the path:
        0 means match, 1 means skip seq1, 2 means skip seq2.
    """

    n1_plus_1, n2_plus_1, _ = a.shape

    M = torch.zeros(n1_plus_1, n2_plus_1)
    backptr = torch.zeros(n1_plus_1, n2_plus_1, dtype=torch.long)

    M[1:, 0] = torch.cumsum(a[1:, 0, 1], dim=0)
    M[0, 1:] = torch.cumsum(a[0, 1:, 2], dim=0)

    backptr[:, 0] = 1
    backptr[0, :] = 2
    backptr[0, 0] = -1

    for i in range(1, n1_plus_1):
        for j in range(1, n2_plus_1):
            indices = ((i-1, i-1, i),
                       (j-1, j, j-1))

            prev_scores = M[indices]
            v = prev_scores + a[i, j]
            res = torch.max(v, dim=0)
            M[i, j] = res.values
            backptr[i, j] = res.indices

    path = []
    aligned_pairs = []

    i = n1_plus_1 - 1
    j = n2_plus_1 - 1

    while backptr[i, j] != -1:

        # backtrack the path, keeping track of the steps and of the
        # aligned pairs of indices.
        path.append(backptr[i, j])
        if backptr[i, j] == 0:
            aligned_pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif backptr[i, j] == 1:
            i -= 1
        elif backptr[i, j] == 2:
            j -= 1

    aligned_pairs.reverse()
    path.reverse()

    return aligned_pairs, path


def logsumexp(a: torch.Tensor):
    """Compute alignment normalizing constant (log-sum-exp) under given model.

    Uses the Needleman-Wunsch dynamic program.

    Parameters
    ----------
    a: torch.tensor, shape [n1+1, n2+1, 3].
        Tensor of scores for the Needleman-Wunsch DP, defined as follows.
         - a[i, j, 0] is the score for matching the seq1 character at position
           i-1 to the seq2 character at position j-1.
         - a[i, j, 1] is the score for skipping position i in seq1 (while
           looking for a match for position j in seq2.)
         - a[i, j, 2] is the score for skipping position j in seq2 (while
           looking for a match for position i in seq1.)

    Returns
    -------

    log_normalizer: scalar,
        The log sum exp of the score of every possible alignment.
    """

    n1_plus_1, n2_plus_1, _ = a.shape

    Q = torch.zeros(n1_plus_1, n2_plus_1)

    Q[1:, 0] = torch.cumsum(a[1:, 0, 1], dim=0)
    Q[0, 1:] = torch.cumsum(a[0, 1:, 2], dim=0)

    # forward algorithm, fill in Q.
    for i in range(1, n1_plus_1):
        for j in range(1, n2_plus_1):
            scores = torch.zeros(3,)
            scores[0] = Q[i-1, j-1] + a[i, j, 0]
            scores[1] = Q[i-1, j] + a[i, j, 1]
            scores[2] = Q[i, j-1] + a[i, j, 2]
            Q[i, j] = torch.logsumexp(scores, dim=0)

    return Q[-1, -1]
