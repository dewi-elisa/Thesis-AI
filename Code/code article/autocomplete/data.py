import os
import re
import sys
import random
import pickle
import collections
from tqdm import tqdm
from colorama import Fore, Style

import torch
import torch.utils.data


###############################################################################
# Regular expressions
###############################################################################

camelcase_ptr = re.compile(r"([A-Z]+[a-z]*)")
special_tok = ["<pad>", "<sos>", "<eos>", "<unk>", "<capa>", "<capf>", "<arg>"]
special_ptr = re.compile(r"({})".format("|".join(special_tok)))


###############################################################################
# Load Data
###############################################################################

def is_ascii(line):  # In Python 3.7, just use str.isascii()
    try:
        line.encode("ascii")
    except Exception:
        return False
    else:
        return True


def is_meaningful(line):
    """
    Check whether line contains more than just punctuations and special tokens.
    """
    return len(re.findall("\w+", re.sub(special_ptr, "", line))) > 0


def load_data(path, check_validity, reverse=False):
    """
    Load and return valid lines of code from the path.

    Lines with non-ascii characters or lines which contain only punctuations
    and/or special tokens are considered invalid and filtered out.
    """
    with open(path) as f:
        corpus = f.read()
        lines = corpus.split("\n")
    print("\n[Data Wrangling] Read {} lines of code.".format(len(lines)))

    if check_validity:
        valid_lines = [line
                       for line in lines
                       if is_ascii(line) and is_meaningful(line)]
        print("[Data Wrangling] {} valid lines / {} lines".format(
            len(valid_lines), len(lines)))
        lines = valid_lines

    return list(reversed(lines)) if reverse else lines


###############################################################################
# Text Processing
###############################################################################

class Tokenizer:
    def __init__(self, opt):
        self.text_ptr = re.compile(
            r"(<pad>|<sos>|<eos>|<unk>|<capa>|<capf>|<arg>|[^a-zA-Z0-9])")
        self.whitespace = opt.whitespace

    # Raw line
    def tokenize(self, raw_line):
        line = raw_line
        tokens = [token if not token.isspace() else "#"
                  for token in re.split(self.text_ptr, line) if token]
        return tokens

    # Raw line
    def lowercase(self, tokens):
        def split_camel_case(token):
            splitted = list(filter(None, re.sub(camelcase_ptr,
                                                r"#\1#", token).split("#")))
            results = []
            for token in splitted:
                if not token.isalpha():
                    results.append(token)
                elif token.islower():
                    results.append(token)
                elif token.isupper():
                    results.append('<capa>')
                    results.append(token.lower())
                elif token[0].isupper() and token[1:].islower():
                    results.append('<capf>')
                    results.append(token.lower())
                else:
                    try:
                        for i, char in enumerate(token):
                            if token[i + 1].islower():  # MSELoss -> MSE Loss
                                results.append('<capa>')
                                results.append(token[:i].lower())
                                results.append('<capf>')
                                results.append(token[i:].lower())
                                break
                    except Exception:  # Foreign languages
                        print('Invalid token while lowercasing:', token)
                        results.append(token)
            return results

        lowercased = []
        for token in tokens:
            if token == "#":
                lowercased.append(token)
            elif token.islower():
                lowercased.append(token)
            elif token.isupper():
                lowercased.append("<capa>")
                lowercased.append(token.lower())
            elif token[0].isupper() and token[1:].islower():
                lowercased.append("<capf>")
                lowercased.append(token.lower())
            else:
                results = split_camel_case(token)
                lowercased.extend(results)
        return lowercased

    # Raw line
    def drop_tokens(self, tokens, keep_rate):
        if keep_rate == 0:  # Randomly drop tokens each time
            keep_rate = random.uniform(0.1, 0.9)

        results = []
        while not results:
            results = []
            for token in tokens:
                if self.whitespace == "replace":
                    if random.random() < keep_rate:  # Keep
                        if token == "#":
                            if results and results[-1] != "#":
                                results.append(token)
                        else:
                            results.append(token)
                    else:  # Drop
                        if results and results[-1] != "#":
                            results.append("#")  # Replace with whitespace
                elif self.whitespace == "remove":
                    if random.random() < keep_rate:  # Keep
                        if token == "#":  # Remove whitespace
                            continue
                        else:
                            results.append(token)
                    else:  # Drop
                        continue
                elif self.whitespace == "keep":
                    if random.random() < keep_rate:  # Keep
                        if token == "#":
                            if results and results[-1] != "#":
                                results.append(token)
                        else:
                            results.append(token)
                    else:  # Drop
                        if token == "#":  # Keep whitespace
                            results.append(token)
                        else:
                            continue
                else:
                    if random.random() < keep_rate:  # Keep
                        results.append(token)
                    else:  # Drop
                        continue
            if results and results[-1] == "#":
                results = results[:-1]
        return results

    # Raw line
    def line_to_tokens(self, raw_line):
        tokens = self.tokenize(raw_line)
        tokens = self.lowercase(tokens)
        return tokens

    # Encoded line
    def encoded_line_to_tokens(self, encoded_line):
        return encoded_line.split(" ")

    # Raw line
    def lines_to_tokens(self, raw_lines):
        return [self.line_to_tokens(raw_line)
                for raw_line in raw_lines]

    # Encoded line
    def encoded_lines_to_tokens(self, encoded_lines):
        return [self.encoded_line_to_tokens(encoded_line)
                for encoded_line in encoded_lines]

    # Raw line
    def tokens_to_line(self, tokens, space=False):
        detokenized = []
        cap_all, cap_first = False, False
        for token in list(filter(None, tokens)):
            if token == "<eos>":
                detokenized.append(token)  # Keep <eos>
                break
            elif token in ["<sos>", "<pad>"]:
                continue
            elif token == "<capa>":
                cap_all = True
            elif token == "<capf>":
                cap_first = True
            else:
                if cap_all:
                    cap_all = False
                    if token == "<capa>":
                        cap_all = True
                        continue
                    elif token == "<capf>":
                        cap_first = True
                        continue
                    elif token in special_tok:  # <unk> or <arg>
                        detokenized.append(token)  # Don't uppercase
                    else:
                        detokenized.append(token.upper())
                elif cap_first:
                    cap_first = False
                    if token == "<capa>":
                        cap_all = True
                        continue
                    elif token == "<capf>":
                        cap_first = True
                        continue
                    elif token in special_tok:  # <unk> or <arg>
                        detokenized.append(token)  # Don't uppercase
                    else:
                        if token == "":
                            continue
                        try:
                            if len(token) > 1:
                                token = token[0].upper() + token[1:]
                            else:
                                token = token[0].upper()
                        except Exception as e:
                            import ipdb; ipdb.set_trace()
                            print(e)
                            print(token)
                        detokenized.append(token)
                else:
                    detokenized.append(token)
        if space:
            raw_line = " ".join(detokenized).strip()
        else:
            raw_line = "".join(detokenized).replace("#", " ")
        return raw_line

    # Raw line
    def tokens_to_lines(self, tokens_all, space=False):
        raw_lines = []
        for tokens in tokens_all:
            raw_lines.append(self.tokens_to_line(tokens, space=space))
        return raw_lines

    # Encoded line
    def tokens_to_encoded_line(self, tokens):
        detokenized = []
        for token in tokens:
            if token == "<pad>":
                pass
            elif token == "<eos>":
                detokenized.append(token)  # Keep <eos>
                break
            else:
                detokenized.append(token)
        encoded_line = " ".join(detokenized)
        return encoded_line

    def tokens_to_tensor(self, tokens, word2id):
        return torch.Tensor([word2id.get(token, word2id["<unk>"])
                             for token in tokens])

    # Raw line
    def line_to_tensor(self, raw_line, word2id, keep_rate, encode_form):
        tokens = self.line_to_tokens(raw_line)
        indices = []

        # Drop tokens if keep_rate < 1
        if keep_rate < 1:
            dropped_tokens = self.drop_tokens(tokens, keep_rate)
            tokens = dropped_tokens if dropped_tokens else tokens

        # Add <eos> if not exists
        if tokens and tokens[-1] != "<eos>":
            tokens.append("<eos>")

        # Convert tokens to indices
        indices = [word2id.get(token, word2id["<unk>"]) for token in tokens]

        # Convert array to tensor
        seq = torch.Tensor(indices)

        # Return modified line (out-of-vocabulary words replaced with <unk>)
        if encode_form:
            return seq, self.tokens_to_encoded_line(tokens)
        else:
            return seq, self.tokens_to_line(tokens)

    # Encoded line
    def encoded_line_to_tensor(self, encoded_line, word2id, keep_rate=1):
        tokens = self.encoded_line_to_tokens(encoded_line)
        indices = []

        # Drop tokens if keep_rate < 1
        if keep_rate < 1:
            dropped_tokens = self.drop_tokens(tokens, keep_rate)
            tokens = dropped_tokens if dropped_tokens else tokens

        # Add <eos> if not exists
        if tokens and tokens[-1] != "<eos>":
            tokens.append("<eos>")

        # Convert tokens to indices
        indices = [word2id.get(token, word2id["<unk>"]) for token in tokens]

        # Convert array to tensor
        seq = torch.Tensor(indices)

        # Return modified line (out-of-vocabulary words replaced with <unk>)
        encoded_line = self.tokens_to_encoded_line(tokens)
        return seq, encoded_line

    # Raw line
    def tensor_to_lines(self, tensor, id2word):
        raw_lines = []
        if tensor.dim() == 1:
            tokens = [id2word.get(index.item(), "<unk>") for index in tensor]
            raw_line = self.tokens_to_line(tokens)
            raw_lines.append(raw_line)
            return raw_lines

        for indices in tensor:
            tokens = [id2word.get(index.item(), "<unk>") for index in indices]
            raw_line = self.tokens_to_line(tokens)
            raw_lines.append(raw_line)
        return raw_lines

    # Encoded line
    def tensor_to_encoded_lines(self, tensor, id2word):
        encoded_lines = []
        if tensor.dim() == 1:
            tokens = [id2word.get(index.item(), "<unk>") for index in tensor]
            encoded_line = self.tokens_to_encoded_line(tokens)
            encoded_lines.append(encoded_line)
            return encoded_lines

        for indices in tensor:
            tokens = [id2word.get(index.item(), "<unk>") for index in indices]
            encoded_line = self.tokens_to_encoded_line(tokens)
            encoded_lines.append(encoded_line)
        return encoded_lines

    def tensor_to_tokens(self, tensor, id2word):
        if tensor.dim() == 1:
            return [[id2word.get(index.item(), "<unk>") for index in tensor]]
        else:
            all_tokens = []
            for indices in tensor:
                tokens = []
                for index in indices:
                    token = id2word.get(index.item(), "<unk>")
                    if token == "<eos>":
                        tokens.append(token)
                        break
                    else:
                        tokens.append(token)
                all_tokens.append(tokens)
            return list(filter(None, all_tokens))


def build_tokenizer(opt):
    return Tokenizer(opt)


###############################################################################
# Vocabulary
###############################################################################

def save_vocab(path_vocab, word2id, id2word):
    """
    Args:
        path_vocab: str
        word2id: {str: int}
        id2word: {str: int}
    """
    if not os.path.exists(path_vocab):
        os.makedirs(path_vocab)
    with open(os.path.join(path_vocab, "word2id.p"), "wb") as fb:
        pickle.dump(word2id, fb)
    with open(os.path.join(path_vocab, "id2word.p"), "wb") as fb:
        pickle.dump(id2word, fb)
    print("[Vocab] Saved vocabulary at {}".format(path_vocab))


def load_vocab(path_vocab):
    with open(os.path.join(path_vocab, "word2id.p"), "rb") as fb:
        word2id = pickle.load(fb)
    with open(os.path.join(path_vocab, "id2word.p"), "rb") as fb:
        id2word = pickle.load(fb)
    print("[Vocab] Loaded existing vocabulary from {}".format(path_vocab))
    return word2id, id2word


def build_vocab(opt, exp, tokenizer, use_existing=False):
    """
    Build vocabulary from path.

    If there exists "vocab.txt", load "word2id.p" and "id2word.p".
    Otherwise, build a table of vocabulary and save to the path/data.

    Args:
        path: path to directory
        lines: lines of code in plain text
        build_new: if true, override vocabulary files if exist
    """
    # Load vocab if exists
    path_data = os.path.join(opt.root, opt.data_dir)
    path_vocab = os.path.join(opt.root, opt.exp_dir, "vocab", exp)  # TODO
    if opt.model_name:
        existing_exp = re.match("(.*)_\d+?.pt", opt.model_name).group(1)
        path_existing_vocab = os.path.join(opt.root, opt.exp_dir, "vocab",
                                           existing_exp)

    if (not opt.build_new_vocab and
            os.path.exists(path_vocab)):
        word2id, id2word = load_vocab(path_vocab)
    elif (not opt.build_new_vocab and
          opt.model_name and
          os.path.exists(path_existing_vocab)):
        word2id, id2word = load_vocab(path_existing_vocab)
        save_vocab(path_vocab, word2id, id2word)
    else:
        if use_existing:
            import ipdb; ipdb.set_trace()
            raise RuntimeError()
        print("\n[Vocab] Building vocabulary.")

        # Collect lines from corpus
        path = os.path.join(path_data, "train_lines.txt")
        lines = load_data(path, check_validity=True)
        lines = lines[:opt.num_examples]

        vocab = list()
        print("[Vocab] Processing {} lines of code...".format(len(lines)))
        for line in tqdm(lines):
            tokens = tokenizer.line_to_tokens(line)  # Raw lines
            vocab.extend(tokens)

        # Prune rare words
        freq_vocab = collections.Counter(vocab).most_common(opt.max_vocab_size)
        freq_vocab = [token for token, _ in freq_vocab
                      if all(token != special for special in special_tok)]
        vocab = special_tok + freq_vocab

        word2id = {v: i for i, v in enumerate(vocab)}
        id2word = {i: v for i, v in enumerate(vocab)}
        save_vocab(path_vocab, word2id, id2word)

    assert len(word2id) == len(id2word)
    print("[Vocab] Vocabulary size: {}".format(len(word2id)))

    if opt.debug:
        print(Fore.RED + f"\n[Debug] word2id['<unk>'] = {word2id['<unk>']}")
        print(Style.RESET_ALL)

    return word2id, id2word


###############################################################################
# Dataset
###############################################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 src_lines,
                 trg_lines,
                 word2id,
                 num_examples,
                 max_seq_len,
                 keep_rate,
                 processed_src,
                 tokenizer):
        self.src_seqs = []  # Contains <unk>
        self.trg_seqs = []  # Contains <unk>
        self.src_lines = []  # Encoded lines; Do not contain <unk>
        self.trg_lines = []  # Encoded lines; Do not contain <unk>
        self.word2id = word2id
        self.num_examples = num_examples  # If None, use all lines provided
        self.max_seq_len = max_seq_len
        self.keep_rate = keep_rate
        self.processed_src = processed_src
        self.tokenizer = tokenizer

        self.read_seqs(src_lines, trg_lines)

    def __getitem__(self, index):
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        src_line = self.src_lines[index]
        trg_line = self.trg_lines[index]
        return src_seq, trg_seq, src_line, trg_line

    def __len__(self):
        return len(self.trg_seqs)

    def read_seqs(self, src_lines, trg_lines):
        assert len(src_lines) == len(trg_lines)
        print("\n[Dataset] Processing {} source and target lines...".format(
            self.num_examples if self.num_examples else len(src_lines)))
        for src, trg in tqdm(zip(src_lines, trg_lines)):
            keep_rate = 1 if self.processed_src else self.keep_rate
            src_seq, src_line = self.tokenizer.line_to_tensor(  # Encode
                src, self.word2id, keep_rate=keep_rate, encode_form=True)
            trg_seq, trg_line = self.tokenizer.line_to_tensor(  # Encode
                trg, self.word2id, keep_rate=1, encode_form=True)
            if trg_seq.size(0) <= self.max_seq_len:  # Filter with # tokens
                self.src_seqs.append(src_seq)
                self.trg_seqs.append(trg_seq)
                self.src_lines.append(src_line)
                self.trg_lines.append(trg_line)
            if (self.num_examples and
                    len(self.trg_seqs) == self.num_examples):
                break
        if self.num_examples and (len(self.trg_seqs) < self.num_examples):
            print("\n[Dataset] Not enough examples to construct a dataset.")
            print("max_seq_len: {}".format(self.max_seq_len))
            print("num_examples: {}".format(self.num_examples))
            print("all lines: {}".format(len(trg_lines)))
            print("lines < max_seq_len: {}".format(len(self.trg_seqs)))
            sys.exit()
        print("[Dataset] From {} lines, selected {} lines (max: {})".format(
            len(trg_lines), len(self.trg_seqs), self.max_seq_len))


def collate_fn(data):
    """Creates mini-batch tensors from (src_seq, trg_seq, src_lines, trg_lines).
    Seqeuences are padded to the maximum length of mini-batch sequences.

    Args:
        data: list of tuple (src_seq, trg_seq).
    Returns:
        src_seqs: padded source sequences.
            [batch_size, max_src_length]
        trg_seqs: padded target sequences.
            [batch_size, max_trg_length]
        src_lines
        trg_lines
    """
    def pad_seqs(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lens

    # Sort list of seqs by length in descending order
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences
    src_seqs, trg_seqs, src_lines, trg_lines = zip(*data)

    # Pad sequences
    src_seqs, src_lens = pad_seqs(src_seqs)
    trg_seqs, trg_lens = pad_seqs(trg_seqs)

    return src_seqs, trg_seqs, src_lines, trg_lines


###############################################################################
# Public Functions
###############################################################################

def get_loader(src_lines,
               trg_lines,
               word2id,
               num_examples,
               max_seq_len,
               keep_rate,
               processed_src,
               tokenizer,
               batch_size,
               shuffle,
               drop_last):
    if num_examples and (len(src_lines) < num_examples):
        print("[Data] Could not create datasets (not enough lines).")
        return None

    # Remove empty line at the end if exists
    if not src_lines[-1] and not trg_lines[-1]:
        src_lines = src_lines[:-1]
        trg_lines = trg_lines[:-1]

    dataset = Dataset(src_lines,
                      trg_lines,
                      word2id,
                      num_examples,
                      max_seq_len,
                      keep_rate,
                      processed_src,
                      tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    print("[Data] Created {} batches (batch size: {})".format(
        len(data_loader), batch_size))
    return data_loader


def build_loaders(opt, tokenizer, word2id):
    # Load processed src (tokens are dropped) if exists for consistent eval
    path_trg = os.path.join(opt.root, opt.data_dir, "train_lines.txt")
    full_lines = load_data(path_trg, check_validity=True)

    # Set the number of examples
    num_train_examples = opt.num_examples
    num_val_examples = min(1000,
                           max(int(opt.num_examples * 0.1), opt.batch_size))
    batch_size = min(opt.num_examples, opt.batch_size)

    # Build train, val, test datasets
    train_ae_loader = get_loader(src_lines=full_lines,
                                 trg_lines=full_lines,
                                 word2id=word2id,
                                 num_examples=num_train_examples,
                                 max_seq_len=opt.max_seq_len,
                                 keep_rate=1,
                                 processed_src=True,
                                 tokenizer=tokenizer,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=True)
    train_key_loader = None

    if opt.debug:
        print(Fore.YELLOW + "\n[Debug] Train dataset")
        print(Fore.YELLOW + "Dataset generated from these lines:\n")
        print(Fore.YELLOW + "\n".join(full_lines[:10]))
        print(Fore.YELLOW + "\nGenerated batches (1 sample batch):\n")
        for src_seqs, trg_seqs, src_lines, trg_lines in train_ae_loader:
            print(Fore.YELLOW + "src_seqs:", src_seqs, "\n")
            print(Fore.YELLOW + "src_lines:", "\n".join(src_lines))
            print(Style.RESET_ALL)
            break

    val_ae_loader = get_loader(src_lines=full_lines[-num_val_examples:],
                               trg_lines=full_lines[-num_val_examples:],
                               word2id=word2id,
                               num_examples=None,
                               max_seq_len=opt.max_seq_len,
                               keep_rate=1,
                               processed_src=True,
                               tokenizer=tokenizer,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False)
    val_key_loader = None

    # Load data for test sets
    test_ae_loader = None
    test_key_loader = None


    return (train_ae_loader, train_key_loader,
            val_ae_loader, val_key_loader,
            test_ae_loader, test_key_loader)
