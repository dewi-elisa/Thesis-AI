# deep learning for structured data
# assignment 2 question 2.2
# original author: vlad niculae <v.niculae@uva.nl>
# license: MIT

# STUDENT ID: 12419273, Dewi Timman

from tqdm import tqdm
from collections import Counter
import torch


import nw_dp
from data_align import load


def alignment_f_score(all_pred_ixs, all_true_ixs):
    """Compute alignment F-score between predicted and true alignments.

    Parameters
    ----------
    all_pred_ixs: list of lists of tuples [[(i, j), ...], ...]
        all_pred_ixs[k] is the list of all matched pairs where seq1[i]
        was matched to seq2[j], as predicted by some model, for the kth data
        sample.

    all_true_ixs: list of lists of tuples [[(i, j), ...], ...]
        all_true_ixs[k] is the list of all matched pairs where seq1[i]
        was matched to seq2[j], from the gold data, for the kth data sample.

    Returns
    -------

    fscore: float,
        Harmonic mean of precision and recall scores (micro-averaged).

        Micro-averaged precision is defined as:

             number of pairs aligned in both predicted and true data
        P =  -------------------------------------------------------
                    number of pairs aligned in predicted data

             number of pairs aligned in both predicted and true data
        R =  -------------------------------------------------------
                    number of pairs aligned in true data

        (In other words, precision is the percentage of predicted pairs that
        are correct, and recall is the percentage of correct pairs that are
        predicted.)

        fscore = 2PR/(P+R)
    """

    n_pred_and_true = 0
    n_pred = 0
    n_true = 0

    if len(all_pred_ixs) != len(all_true_ixs):
        print("Lists are not of equal length")

    for pred_ixs, true_ixs in zip(all_pred_ixs, all_true_ixs):

        # accumulate n_pred_and_true, n_pred, n_true.
        for pred in set(pred_ixs):
            if pred in set(true_ixs):
                n_pred_and_true += 1

        n_pred += len(pred_ixs)
        n_true += len(true_ixs)

    # precision: out of the predicted aligned pairs, how many are true?
    precision = n_pred_and_true / n_pred
    # recall: out of the true pairs, how many were predicted by the model?
    recall = n_pred_and_true / n_true

    return (2 * precision * recall) / (precision + recall)


def baseline_left_align(seq1, seq2):
    """Left-align baseline: aligns seq1 and seq2 on the left.

    Only leaves a gap at the end if the sequences are not equal length.

    Parameters:
    -----------
    seq1: string, length n1
        sequence of characters in the first sequence.

    seq2: string, length n2
        sequence of characters in the second sequence.

    Returns:
    --------

    aligned_pairs: list of tuples [(i, j), ...]
        List of all matched pairs where seq1[i] was matched to seq2[j].
    """

    aligned_pairs = []

    for idx, _ in enumerate(zip(seq1, seq2)):
        aligned_pairs.append((idx, idx))

    return aligned_pairs


def alignment_nll(a, alignment):
    """Negative log-likelihood loss of an alignment path.

    Uses the Needleman-Wunsch algorithm to compute the log-normalizing term
    over every possible tag sequence.

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

    alignment: list of int,
        List of the decisions made at each step in the path:
        0 means match, 1 means skip seq1, 2 means skip seq2.

    Returns
    -------

    loss: torch scalar,
        Negative log likelihood of the given alignment.
    """

    i = j = 0

    true_alignment_score = 0

    for k in alignment:
        if k == 0:
            i += 1
            j += 1
        elif k == 1:
            i += 1
        elif k == 2:
            j += 1

        true_alignment_score += a[i, j, k]

    log_normalizer = nw_dp.logsumexp(a)

    return log_normalizer - true_alignment_score


class AlignmentTable(torch.nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        init_pw_scores = torch.eye(vocab_size)
        init_pw_scores += .001 * torch.randn_like(init_pw_scores)
        self.pw_scores = torch.nn.Parameter(init_pw_scores)
        self.gap_score = torch.nn.Parameter(torch.tensor(-1.0))

    def forward(self, encoded_seq1, encoded_seq2):

        n1 = len(encoded_seq1)
        n2 = len(encoded_seq2)

        # build the Needleman-Wunsch score table for this pair of sequences.
        a = torch.zeros(n1+1, n2+1, 3)

        # fill in a using pairwise scores and gap score
        for i, char in enumerate(encoded_seq1):
            for j, char2 in enumerate(encoded_seq2):
                if char == char2:
                    a[i, j, 0] = self.pw_scores[char, char2]

        a[:, :, 1:] = self.gap_score

        return a


def main():

    torch.manual_seed(2022)
    skip_exact_match = False

    train, valid = load()

    # prepare list of all alignment indices, for evaluation
    train_true_ixs = [tup[-1] for tup in train]
    valid_true_ixs = [tup[-1] for tup in valid]

    # evaluate baseline
    train_pred_ixs = [baseline_left_align(seq1, seq2)
                      for seq1, seq2, _, _ in train]
    valid_pred_ixs = [baseline_left_align(seq1, seq2)
                      for seq1, seq2, _, _ in train]

    print("Baseline left align.")
    train_f = alignment_f_score(train_pred_ixs, train_true_ixs)
    valid_f = alignment_f_score(valid_pred_ixs, valid_true_ixs)
    print(f"Train F1={train_f:.4f}, Valid F1={valid_f:.4f}")

    if not skip_exact_match:
        # align with needleman-wunch exact match
        train_pred_ixs = []

        for seq1, seq2, _, _ in tqdm(train, desc="Eval train"):
            a = nw_dp.standard_nw_table(seq1, seq2)
            pairs, _ = nw_dp.argmax(a)
            train_pred_ixs.append(pairs)

        valid_pred_ixs = []
        for seq1, seq2, _, _ in tqdm(valid, desc="Eval valid"):
            a = nw_dp.standard_nw_table(seq1, seq2)
            pairs, _ = nw_dp.argmax(a)
            valid_pred_ixs.append(pairs)

        print("Baseline NW fixed.")
        train_f = alignment_f_score(train_pred_ixs, train_true_ixs)
        valid_f = alignment_f_score(valid_pred_ixs, valid_true_ixs)
        print(f"Train F1={train_f:.4f}, Valid F1={valid_f:.4f}")

    # Machine Learning Time!
    ########################

    # for ml methods we must get a vocabulary and encode our characters.
    vocab = Counter(w for seq1, seq2, _, _ in train
                    for seq in (seq1, seq2)
                    for w in seq)
    vocab = ["<pad>", "<unk>"] + [w for w, count in vocab.most_common()]
    inv_vocab = {w: k for k, w in enumerate(vocab)}

    def encode(x):
        return torch.tensor([inv_vocab.get(w, inv_vocab['<unk>']) for w in x],
                            dtype=torch.long)

    enc_train = [(encode(seq1), encode(seq2), alignment, indices)
                 for seq1, seq2, alignment, indices in train]

    enc_valid = [(encode(seq1), encode(seq2), alignment, indices)
                 for seq1, seq2, alignment, indices in valid]

    # static alignment table: learn pairwise character affinities
    model = AlignmentTable(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_pass():
        """Train one pass over the data. (aka one "epoch")"""
        model.train()

        ixs = torch.randperm(len(enc_train))

        total_loss = 0

        for ix in tqdm(ixs, desc="Training"):
            seq1, seq2, alignment, _ = enc_train[ix]

            a = model(seq1, seq2)

            loss = alignment_nll(a, alignment)

            total_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        return total_loss / len(enc_train)

    def evaluate():

        train_pred_ixs = []
        valid_pred_ixs = []
        with torch.no_grad():
            model.eval()

            for seq1, seq2, _, _ in tqdm(enc_train, desc="Eval train"):
                a = model(seq1, seq2)
                pairs, _ = nw_dp.argmax(a)
                train_pred_ixs.append(pairs)

            for seq1, seq2, _, _ in tqdm(enc_valid, desc="Eval valid"):
                a = model(seq1, seq2)
                pairs, _ = nw_dp.argmax(a)
                valid_pred_ixs.append(pairs)

        train_f = alignment_f_score(train_pred_ixs, train_true_ixs)
        valid_f = alignment_f_score(valid_pred_ixs, valid_true_ixs)
        return train_f, valid_f

    for epoch in range(5):
        print(f"Epoch {epoch}")
        train_loss = train_pass()
        print(f"Training loss {train_loss:.2f}")

        train_f, valid_f = evaluate()
        print(f"Train F1={train_f:.4f}, Valid F1={valid_f:.4f}")

    # fill in results here in a comment
    # model       | train F1 score | valid F1 score |
    # ------------|----------------| -------------- |
    # left-align  | 0.5550         | 0.4629         |
    # NW fixed    | 0.9090         | 0.9096         |
    # NW learned  | 0.8830         | 0.8977         |

    # The left-align F1 score is lower than the NW scores. This is what I would expect,
    # since there is more 'logic' behind the NW scores than behing the left-align score.
    # In addition, the NW fixed has a higher F1 score than the NW learned for both the
    # training and validation set. Personally, I think it is surprising that the learned
    # scored are lower than the fixed scores.


if __name__ == '__main__':
    main()
