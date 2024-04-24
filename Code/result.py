import math
import collections
import numpy as np


class TokenClassifier(object):
    def __init__(self):
        self.special_tokens = {"<pad>", "<sos>", "<eos>", "<unk>",
                               "<capa>", "<capf>", "<arg>"}
        self.classes = ["special", "others"]
        self.classify = self.classify_word

    def classify_word(self, token):
        if token in self.special_tokens:
            return "special"
        else:
            return "others"


class Result(object):
    def __init__(self, opt, exp, tokenizer, results, predictions):
        self.opt = opt
        self.exp = exp

        self.token_classifier = TokenClassifier()
        self.token_classes = self.token_classifier.classes
        self.classifier = self.token_classifier.classify

        self.results = results  # Raw model outputs (contains <eos>)
        self.predictions = predictions  # Raw model outputs (contains <eos>)

        self.examples = []  # Processed model outputs (src, key, out, [preds])
        self.examples_wo_cap_punc = []  # Normalized model outputs

        # Stats
        self.num_examples = 0
        self.perc_kept = 0
        self.perc_kept_wo_whitespace = 0

        # Results
        # ACC: Exact match
        self.correct = []
        self.incorre = []
        self.num_correct = 0
        self.num_incorre = 0
        self.perc_correct = 0
        self.perc_incorre = 0
        self.acc = 0

        self.correct_wo_cap_punc = []
        self.incorre_wo_cap_punc = []
        self.num_correct_wo_cap_punc = 0
        self.num_incorre_wo_cap_punc = 0
        self.perc_correct_wo_cap_punc = 0
        self.perc_incorre_wo_cap_punc = 0
        self.acc_wo_cap_punc = 0

        # MRR: Mean reciprocal rank
        self.beam_size = opt.beam_size
        self.mrr = 0
        self.mrr_wo_cap_punc = 0

        # NLL: Negative log likelihood
        self.nll = 0

        # Processed data
        self.text_output = ""

        # Analysis
        self.src_tokens = []  # Unique tokens
        self.key_tokens = []  # Unique tokens
        self.src_token_counter = None
        self.key_token_counter = None
        self.perc_kept_per_token = dict()

        self.classified_tokens = collections.defaultdict(set)  # Unique tokens
        self.perc_kept_per_class = dict()

        self.update_stats()
        self.process_results(tokenizer)
        self.update_acc()
        self.update_mrr()

    def update_stats(self):  # Called by init
        (_, _, _, src_tokens, key_tokens, out_tokens) = zip(*self.results)
        self.num_examples = len(self.results)

        num_src_tokens = sum([len(tokens) for tokens in src_tokens])
        num_key_tokens = sum([len(tokens) for tokens in key_tokens])
        self.perc_kept = self.to_perc(num_key_tokens, num_src_tokens)

        key_tokens_wo_whitespace = [list(filter(lambda x: x != "#", tokens))
                                    for tokens in key_tokens]
        trg_tokens_wo_whitespace = [list(filter(lambda x: x != "#", tokens))
                                    for tokens in src_tokens]
        num_key_tokens_wo_whitespace = sum([len(tokens) for tokens
                                            in key_tokens_wo_whitespace])
        num_trg_tokens_wo_whitespace = sum([len(tokens) for tokens
                                            in trg_tokens_wo_whitespace])
        self.perc_kept_wo_whitespace = self.to_perc(
            num_key_tokens_wo_whitespace, num_trg_tokens_wo_whitespace)

        self.analyze_tokens(src_tokens, key_tokens)

    def analyze_tokens(self, src_tokens, key_tokens):  # Called by update_stats
        all_src_tokens = [token for tokens in src_tokens for token in tokens]
        all_key_tokens = [token for tokens in key_tokens for token in tokens]

        self.src_tokens = sorted(set(all_src_tokens))
        self.key_tokens = sorted(set(all_key_tokens))

        # Count all tokens
        self.src_token_counter = collections.Counter(all_src_tokens)
        self.key_token_counter = collections.Counter(all_key_tokens)
        self.perc_kept_per_token = {token: self.to_perc(self.key_token_counter.get(token, 0),  # noqa
                                    self.src_token_counter[token])
                                    for token in all_src_tokens}

        # Count per class
        for token_class in self.token_classes:
            # Collect all tokens in the class
            self.classified_tokens[token_class] = sorted(
                {token for token in self.src_tokens
                 if self.classifier(token) == token_class})

            # Calculate average kept rate for the class
            total, kept = 0, 0
            for token in self.classified_tokens[token_class]:
                total += self.src_token_counter[token]
                kept += self.key_token_counter[token]
            perc_kept = self.to_perc(kept, total)
            self.perc_kept_per_class[token_class] = perc_kept

    def process_results(self, tokenizer):  # Called by init
        (src_lines, key_lines, out_lines, _, _, _) = zip(*self.results)
        top1_results = [(src.replace("<eos>", ""),  # Remove <eos>
                         key.replace("<eos>", ""),
                         out.replace("<eos>", ""))
                        for src, key, out in zip(src_lines, key_lines, out_lines)]  # noqa
        topn_predictions = []
        for preds in self.predictions:
            processed_preds = []
            for pred, score in preds:
                processed_preds.append((pred.replace("<eos>", ""), score))
            topn_predictions.append(processed_preds)

        if topn_predictions:
            self._process_beam_search_output(top1_results,
                                             topn_predictions,
                                             tokenizer)
        else:
            self._process_greedy_output(top1_results)

    def _process_beam_search_output(self,  # Called by process_results
                                    top1_results,
                                    topn_predictions,
                                    tokenizer):
        for (src, key, out), preds in zip(top1_results, topn_predictions):
            processed_preds = []
            normalize_preds = []
            for pred, score in preds:
                tokens = tokenizer.encoded_line_to_tokens(pred)
                pred = tokenizer.tokens_to_line(tokens)
                prob = self.to_perc(math.exp(score))
                processed_preds.append((pred, prob))
                normalize_preds.append((self.normalize(pred), prob))
            self.examples.append((src, key, out, processed_preds))
            self.examples_wo_cap_punc.append((self.normalize(src),
                                              self.normalize(key),
                                              self.normalize(out),
                                              normalize_preds))

    def _process_greedy_output(self,  # Called by process_results
                               top1_results):
        for src, key, out in top1_results:
            self.examples.append((src, key, out, []))
            self.examples_wo_cap_punc.append((self.normalize(src),
                                              self.normalize(key),
                                              self.normalize(out),
                                              []))

    def update_acc(self):  # Called by init
        """ Update accuracy related stats based on top 1 prediction. """
        self.correct = [(src, key, out)
                        for src, key, out, _ in self.examples if src == out]
        self.incorre = [(src, key, out)
                        for src, key, out, _ in self.examples if src != out]
        self.num_correct = len(self.correct)
        self.num_incorre = len(self.incorre)
        self.perc_correct = self.to_perc(self.num_correct / self.num_examples)
        self.perc_incorre = self.to_perc(self.num_incorre / self.num_examples)
        self.acc = self.perc_correct

        # Normalized results
        self.correct_wo_cap_punc = [(src, key, out) for src, key, out, _
                                    in self.examples_wo_cap_punc if src == out]
        self.incorre_wo_cap_punc = [(src, key, out) for src, key, out, _
                                    in self.examples_wo_cap_punc if src != out]
        self.num_correct_wo_cap_punc = len(self.correct_wo_cap_punc)
        self.num_incorre_wo_cap_punc = len(self.incorre_wo_cap_punc)
        self.perc_correct_wo_cap_punc = self.to_perc(
            self.num_correct_wo_cap_punc, self.num_examples)
        self.perc_incorre_wo_cap_punc = self.to_perc(
            self.num_incorre_wo_cap_punc, self.num_examples)
        self.acc_wo_cap_punc = self.perc_correct_wo_cap_punc
        self.analyze_errors()

    def analyze_errors(self):  # Called by update_acc
        pass

    def update_mrr(self):  # Called by init
        if self.beam_size == 1:
            self.mrr = self.acc
            self.mrr_wo_cap_punc = self.acc_wo_cap_punc
        else:
            sum_mrr = 0
            for src, key, out, preds in self.examples:
                for i, (pred, _) in enumerate(preds):  # Beam size
                    if src == pred:
                        sum_mrr += 1 / float(i % self.beam_size + 1)
                        break
            self.mrr = self.to_perc(sum_mrr, self.num_examples)

            sum_mrr = 0
            for src, key, out, preds in self.examples_wo_cap_punc:
                for i, (pred, _) in enumerate(preds):  # Beam size
                    if src == pred:
                        sum_mrr += 1 / float(i % self.beam_size + 1)
                        break
            self.mrr_wo_cap_punc = self.to_perc(sum_mrr, self.num_examples)

    def generate_text_output(self,  # Public function
                             max_num_pred=10,
                             wo_cap_punc=False):
        def add(text, line):
            text.append(line)

        def add_newline(text):
            text.append("")

        if wo_cap_punc:
            examples = self.examples_wo_cap_punc
            num_correct = self.num_correct_wo_cap_punc
            num_incorre = self.num_incorre_wo_cap_punc
            perc_correct = self.perc_correct_wo_cap_punc
            perc_incorre = self.perc_incorre_wo_cap_punc
            acc = self.acc_wo_cap_punc
        else:
            examples = self.examples
            num_correct = self.num_correct
            num_incorre = self.num_incorre
            perc_correct = self.perc_correct
            perc_incorre = self.perc_incorre
            acc = self.acc

        text = []
        add(text, f"TOTAL: {self.num_examples}")
        add(text, f"CORRECT: {num_correct} ({perc_correct}%)")
        add(text, f"WRONG: {num_incorre} ({perc_incorre}%)")
        add_newline(text)

        add(text, f"ACC: {acc}% ({self.acc_wo_cap_punc}% without cap/punc)")  # noqa
        add(text, f"MRR: {self.mrr}% ({self.mrr_wo_cap_punc}% without cap/punc)")  # noqa
        add(text, f"NLL: {self.nll}")
        add_newline(text)

        add(text, f"Kept tokens: {self.perc_kept}%")
        add(text, f"Kept tokens (without whitespace): {self.perc_kept_wo_whitespace}%")  # noqa
        add_newline(text)

        def print_token(tokens, type=None, sort_by_freq=True):
            if type:
                add(text, type)

            freq_info = []
            for token in tokens:
                perc = self.perc_kept_per_token[token]
                total = self.src_token_counter[token]
                num = self.key_token_counter[token]
                perc_bar = "▇" * int(perc / 5)
                info = f"{token:>30}  {perc_bar:20}|  {num}/{total} ({perc}%)"
                freq_info.append((total, info))

            if sort_by_freq:
                freq_info = sorted(freq_info, key=lambda x: x[0], reverse=True)
            for _, info in freq_info:
                add(text, info)

        # Print out overall kept rate per class
        for token_class in self.token_classes:
            add(text, f"Kept {token_class}: {self.perc_kept_per_class[token_class]}%")  # noqa
        add_newline(text)

        # Print out kept rate per token in the order of token classes
        for token_class in self.token_classes:
            print_token(self.classified_tokens[token_class], token_class)
        add_newline(text)

        add(text, "Model outputs")
        add_newline(text)
        if self.beam_size > 1:
            for src, key, out, preds in examples:
                if self.opt.print_src_first:
                    add(text, f"SRC : {src}")
                add(text, f"KEY : {key}")
                for i, (pred, prob) in enumerate(preds, 1):
                    check = u"\u2713" if src == pred else ""
                    add(text, f"OUT{i}: {pred:50} ({prob}%) {check}")
                    if i == max_num_pred:
                        break
                if not self.opt.print_src_first:
                    add(text, f"TRG : {src}")
                add_newline(text)
        else:
            for src, key, out, _ in examples:
                check = u"\u2713" if src == out else ""
                if self.opt.print_src_first:
                    add(text, f"SRC : {src}")
                add(text, f"KEY: {key}")
                add(text, f"OUT: {out} {check}")
                if not self.opt.print_src_first:
                    add(text, f"TRG: {src}")
                add_newline(text)

        self.text_output = "\n".join(text)
        return self.text_output

    def normalize(self, line):  # Private helper function
        """ Remove capitalization and punctuations. """
        return "".join([c for c in line.lower()
                        if c.isalpha() or c.isspace()])

    def to_perc(self, numerator, denominator=1):  # Private helper function
        if denominator == 0:
            return 0
        return round(numerator / denominator * 100, 2)
