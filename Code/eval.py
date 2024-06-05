import re
import os
import copy
import math
import pickle
import datetime
import itertools
import collections
import configargparse
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style

import torch

import opts
import utils
import data
import model
import train
from utils import cprint
from result import Result

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothing = SmoothingFunction().method4


def get_bleu(ref_tokens, can_tokens):
    score = 0
    for ref, can in zip(ref_tokens, can_tokens):
        if can == ["<eos>"]:  # It causes float division by zero
            s = 0
        else:
            try:
                s = sentence_bleu([ref], can, smoothing_function=smoothing)
            except Exception as e:
                print(e)
                print("[Eval] Error in calculating BLEU score")
                print("reference:", ref)
                print("candidate:", can)
                s = 0
        score += s
    return score / float(len(ref_tokens))


def get_recon_loss(prob_over_dyn_vocab,
                   reindexed_trg_seqs,
                   len_dyn_word2id,
                   device):
    batch_size = reindexed_trg_seqs.size(0)
    num_trg_tokens = torch.sum(reindexed_trg_seqs.gt(0), dim=1).float()

    recon_loss_per_token = train.ce_loss_per_token(prob_over_dyn_vocab,
                                                   reindexed_trg_seqs,
                                                   len_dyn_word2id,
                                                   device)
    recon_loss_per_sample = torch.sum(
        recon_loss_per_token.view(batch_size, -1), dim=1)\
        .div(num_trg_tokens)

    avg_recon_loss = recon_loss_per_sample.sum().div(batch_size)
    return avg_recon_loss.item()


def score_similarity(src_tokens, out_tokens):
    # For now, assume each token is unique
    set_src_tokens = set(src_tokens)
    set_out_tokens = set(out_tokens)

    # Calculate jaccard similarity
    intersection = len(list(set_src_tokens.intersection(set_out_tokens)))
    union = (len(set_src_tokens) + len(set_out_tokens)) - intersection
    return float(intersection / union)


def get_mrr(encoded_lines, predictions, beam_size):  # Encoded
    if beam_size == 1:
        return 0
    sum_mrr = 0
    for encoded_line, prediction in zip(encoded_lines, predictions):
        for i, (pred, score) in enumerate(prediction):  # Beam size
            if encoded_line == pred:
                sum_mrr += 1 / float(i % beam_size + 1)
                break
    return sum_mrr


def eval_ae_batch(opt,
                  device,
                  encoder,
                  decoder,
                  batch):
    src_seqs, trg_seqs, src_lines, trg_lines = batch
    src_seqs = src_seqs.to(device)
    trg_seqs = trg_seqs.to(device)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Encode
        key_prob_over_tokens, key_mask = encoder(src_seqs)
        key_seqs, key_lines = decoder.postprocess(key_seqs=None,
                                                  key_mask=key_mask,
                                                  encoded_lines=src_lines)
        # Decode
        (prob_over_dyn_vocab, attn_weight_over_key,
         dyn_word2id, dyn_id2word,
         gen_prob_per_timestep, predictions) = decoder.generate(key_seqs,
                                                                key_lines)
        reindexed_key_seqs = decoder.reindex(key_seqs,
                                             key_lines,
                                             dyn_word2id).to(device)
        reindexed_trg_seqs = decoder.reindex(trg_seqs,
                                             trg_lines,
                                             dyn_word2id).to(device)

        # Result different between with and without beam search
        src = src_lines
        key = key_lines
        if opt.beam_size > 1:
            out_lines = [prediction[0][0] for prediction in predictions]
        else:
            output = prob_over_dyn_vocab
            _, top = output.topk(1)         # [batch_size, max_seq_len, 1]
            out_seqs = top.squeeze(2)       # [batch_size, max_seq_len]
            out_lines = decoder.tokenizer.tensor_to_encoded_lines(
                out_seqs, dyn_id2word)
        avg_recon_loss = get_recon_loss(prob_over_dyn_vocab,
                                        reindexed_trg_seqs,
                                        len(dyn_word2id),
                                        device)
        mrr = get_mrr(src_lines, predictions, opt.beam_size)

        src_tokens = decoder.tokenizer.encoded_lines_to_tokens(src_lines)
        key_tokens = decoder.tokenizer.tensor_to_tokens(reindexed_key_seqs,
                                                        dyn_id2word)
        if opt.beam_size > 1:
            out_tokens = decoder.tokenizer.encoded_lines_to_tokens(out_lines)
        else:
            out_tokens = decoder.tokenizer.tensor_to_tokens(out_seqs,
                                                            dyn_id2word)
        # Overwrite if pretty print
        if opt.pretty_print:
            src = decoder.tokenizer.tokens_to_lines(src_tokens)
            space = True if opt.print_space_between_key_tokens else False
            key = decoder.tokenizer.tokens_to_lines(key_tokens, space=space)
            out = decoder.tokenizer.tokens_to_lines(out_tokens)
        else:
            out = out_lines

        if opt.debug:
            cprint("\n\n[Debug] Evaluating on batch:", Fore.YELLOW)

            dyn_vocab_index = dyn_id2word.keys() - decoder.id2word.keys()
            dyn_vocab_index.add(decoder.word2id["<unk>"])  # Highlight <unk>
            dyn_vocab_index = [" " + str(index) for index in dyn_vocab_index]
            dyn_vocab_token = dyn_word2id.keys() - decoder.word2id.keys()
            dyn_vocab_token.add("<unk>")  # Highlight <unk>

            dyn_vocab_text = "\n".join([f"{dyn_word2id[token]}: {token}"
                                        for token in dyn_vocab_token])
            cprint(f"\nDynamic vocabulary\n{dyn_vocab_text}\n", Fore.RED)

            key_seqs_text = f"key_seqs: {reindexed_key_seqs.numpy()} (reindexed)"  # noqa
            if opt.beam_size == 1:
                out_seqs_text = f"out_seqs: {out_seqs.numpy()} (reindexed)"
            trg_seqs_text = f"trg_seqs: {reindexed_trg_seqs.numpy()} (reindexed)\n"  # noqa

            cprint("Sequences:", Fore.YELLOW)
            cprint(key_seqs_text, Fore.GREEN, dyn_vocab_index)
            if opt.beam_size == 1:
                cprint(out_seqs_text, Fore.CYAN, dyn_vocab_index)
            cprint(trg_seqs_text, Fore.MAGENTA, dyn_vocab_index)

            key_lines_text = "key_lines: " + "\n".join(key_lines)
            out_lines_text = "out_lines: " + "\n".join(out_lines)
            trg_lines_text = "trg_lines: " + "\n".join(src_lines)

            cprint("Encoded lines:", Fore.YELLOW)
            cprint(key_lines_text, Fore.GREEN, dyn_vocab_token)
            cprint(out_lines_text, Fore.CYAN, dyn_vocab_token)
            cprint(trg_lines_text, Fore.MAGENTA, dyn_vocab_token)

            # print(Fore.YELLOW + "\nPretty printed lines:" + Style.RESET_ALL)
            # src_text = "\n".join(src)
            # key_text = "\n".join(key)
            # out_text = "\n".join(out)
            #
            # cprint(src_text, Fore.WHITE, dyn_vocab_token)
            # cprint(key_text, Fore.GREEN, dyn_vocab_token)
            # cprint(out_text, Fore.CYAN, dyn_vocab_token)

        return (list(zip(src, key, out, src_tokens, key_tokens, out_tokens)),
                avg_recon_loss,
                mrr,
                predictions)


def eval_key_batch(opt,
                   device,
                   encoder,
                   decoder,
                   batch):
    decoder.eval()
    with torch.no_grad():
        key_seqs_, trg_seqs, key_lines_, trg_lines = batch
        key_seqs, key_lines = decoder.postprocess(key_seqs=key_seqs_,
                                                  key_mask=None,
                                                  encoded_lines=key_lines_)
        key_seqs = key_seqs.to(device)
        trg_seqs = trg_seqs.to(device)

        # Decode
        (prob_over_dyn_vocab, attn_weight_over_key,
         dyn_word2id, dyn_id2word,
         gen_prob_per_timestep, predictions) = decoder.generate(key_seqs,
                                                                key_lines)

        reindexed_key_seqs = decoder.reindex(key_seqs,
                                             key_lines,
                                             dyn_word2id).to(device)
        reindexed_trg_seqs = decoder.reindex(trg_seqs,
                                             trg_lines,
                                             dyn_word2id).to(device)

        # Result different between with and without beam search
        key = key_lines
        trg = trg_lines
        if opt.beam_size > 1:
            out = [prediction[0][0] for prediction in predictions]
        else:
            output = prob_over_dyn_vocab
            _, top = output.topk(1)         # [batch_size, max_seq_len, 1]
            out_seqs = top.squeeze(2)       # [batch_size, max_seq_len]
            out = decoder.tokenizer.tensor_to_encoded_lines(
                out_seqs, dyn_id2word)
        avg_recon_loss = get_recon_loss(prob_over_dyn_vocab,
                                        reindexed_trg_seqs,
                                        len(dyn_word2id),
                                        device)
        mrr = get_mrr(trg_lines, predictions, opt.beam_size)

        key_tokens = decoder.tokenizer.tensor_to_tokens(reindexed_key_seqs,
                                                        dyn_id2word)
        trg_tokens = decoder.tokenizer.encoded_lines_to_tokens(trg_lines)
        if opt.beam_size > 1:
            out_tokens = decoder.tokenizer.encoded_lines_to_tokens(out)
        else:
            out_tokens = decoder.tokenizer.tensor_to_tokens(out_seqs,
                                                            dyn_id2word)

        # Overwrite if pretty print
        if opt.pretty_print:
            space = True if opt.print_space_between_key_tokens else False
            key = decoder.tokenizer.tokens_to_lines(key_tokens, space)
            trg = decoder.tokenizer.tokens_to_lines(trg_tokens)
            out = decoder.tokenizer.tokens_to_lines(out_tokens)
        return (list(zip(key, trg, out, key_tokens, trg_tokens, out_tokens)),
                avg_recon_loss,
                mrr,
                predictions)


def eval_all_batch(opt,
                   device,
                   encoder,
                   decoder,
                   loader,
                   autoencoder):
    results_all = []
    recon_loss = []
    mrr_all = []
    predictions_all = []
    for batch_index, batch in enumerate(loader):
        if autoencoder:
            results, loss, mrr, predictions = eval_ae_batch(
                opt, device, encoder, decoder, batch)
        else:
            results, loss, mrr, predictions = eval_key_batch(
                opt, device, encoder, decoder, batch)
        results_all.extend(results)
        recon_loss.append(loss)
        mrr_all.append(mrr)
        if predictions:
            predictions_all.extend(predictions)
        if len(results_all) >= opt.max_eval_num:
            break

    avg_recon_loss = np.mean(recon_loss)
    avg_mrr = np.sum(mrr_all) / len(results_all)  # CHECK Normalize
    return results_all, avg_recon_loss, avg_mrr, predictions_all


def eval_one_key(opt,
                 device,
                 encoder,
                 decoder,
                 key_seq,
                 key_line,
                 verbose=True):
    preds_lines = []
    preds_tokens = []

    decoder.eval()
    with torch.no_grad():
        key_seq, key_line = decoder.postprocess(key_seqs=key_seq,
                                                key_mask=None,
                                                encoded_lines=key_line)
        if verbose:
            print(f'KEY: {key_line}')
        # Decode
        (prob_over_dyn_vocab, attn_weight_over_key,
         dyn_word2id, dyn_id2word,
         gen_prob_per_timestep, predictions) = decoder.generate(key_seq,
                                                                key_line)
        if opt.beam_size > 1:
            for i, (pred, score) in enumerate(predictions[0], 1):
                out_tokens = decoder.tokenizer.encoded_line_to_tokens(pred)
                out = decoder.tokenizer.tokens_to_line(out_tokens)  # Raw line
                prob = round(math.exp(score) * 100, 2)
                preds_lines.append(out.replace("<eos>", ""))
                preds_tokens.append(out_tokens)
                if verbose:
                    print(f'OUT{i}: {out.replace("<eos>", ""):50} {prob}%')
        else:
            output = prob_over_dyn_vocab
            _, top = output.topk(1)
            out_seq = top.squeeze(2)
            out = decoder.tokenizer.tensor_to_lines(out_seq, dyn_id2word)[0]
            if verbose:
                print(f'OUT: {out.replace("<eos>", "")}')

    return preds_lines, preds_tokens


def eval_one_src(opt,
                 device,
                 encoder,
                 decoder,
                 src_seq,
                 src_line,
                 verbose=True):
    encoder.eval()
    decoder.eval()
    preds = []
    with torch.no_grad():
        # Encode
        key_prob_over_tokens, key_mask = encoder(src_seq.to(device))
        src_tokens = decoder.tokenizer.encoded_line_to_tokens(src_line[0])
        if True:
            for i in range(src_seq.size(1)):
                token = src_tokens[i]
                prob = round(key_prob_over_tokens[0][i].item() * 100, 2)
                prob_bar = "▇▇" * int(prob / 10)
                print(f'     {token:10} {prob_bar:20}| {prob}%')

        key_seq, key_line = decoder.postprocess(key_seqs=None,
                                                key_mask=key_mask,
                                                encoded_lines=src_line)

        # Decode
        (prob_over_dyn_vocab, attn_weight_over_key,
         dyn_word2id, dyn_id2word,
         gen_prob_per_timestep, predictions) = decoder.generate(key_seq,
                                                                key_line)
        key_tokens = decoder.tokenizer.encoded_line_to_tokens(key_line[0])
        key_line = decoder.tokenizer.tokens_to_line(key_tokens, space=True)
        if verbose:
            print(Fore.YELLOW + f'KEY: {key_line.replace("<eos>", "")}'
                  + Style.RESET_ALL)

        if opt.beam_size > 1:
            for i, (pred, score) in enumerate(predictions[0], 1):
                out_tokens = decoder.tokenizer.encoded_line_to_tokens(pred)
                out = decoder.tokenizer.tokens_to_line(out_tokens)
                prob = round(math.exp(score) * 100, 2)
                if verbose:
                    print(f'OUT{i}: {out.replace("<eos>", ""):50} {prob}%')
                preds.append(out.replace("<eos>", ""))
        else:
            output = prob_over_dyn_vocab
            _, top = output.topk(1)
            out_seq = top.squeeze(2)
            out = decoder.tokenizer.tensor_to_line(out_seq, dyn_id2word)
            if verbose:
                print(f'OUT: {out.replace("<eos>", "")}')
        return key_seq, key_line, key_tokens, preds


def evaluate(opt,
             exp,
             device,
             encoder,
             decoder,
             data_type,
             ae_loader,
             key_loader,
             pbar,
             log=False,
             print_only=["train"]):
    if ae_loader:
        (ae_results, ae_recon_loss, ae_mrr,
         ae_predictions) = eval_all_batch(opt,
                                          device,
                                          encoder,
                                          decoder,
                                          ae_loader,
                                          autoencoder=True)
        # Exact match score
        autoencoder_acc = len([src for src, key, out, _, _, _ in ae_results
                               if src == out]) / float(len(ae_results))

        # BLEU score
        _, _, _, src_tokens, key_tokens, out_tokens = zip(*ae_results)
        autoencoder_bleu = get_bleu(src_tokens, out_tokens)

    if key_loader:
        (key_results, key_recon_loss, key_mrr,
         key_predictions) = eval_all_batch(opt,
                                           device,
                                           encoder,
                                           decoder,
                                           key_loader,
                                           autoencoder=False)
        keyword_acc = len([key for key, trg, out, _, _, _ in key_results
                           if trg == out]) / float(len(key_results))

        _, _, _, key_tokens, trg_tokens, out_tokens = zip(*key_results)
        keyword_bleu = get_bleu(trg_tokens, out_tokens)


def evaluate_dataset(opt,
                     device,
                     exp,
                     tokenizer,
                     full_lines,
                     word2id,
                     id2word,
                     decoder,
                     encoder,
                     lambdas):
    # Build data loader
    num_test_examples = opt.num_examples
    ae_loader = data.get_loader(src_lines=full_lines,
                                trg_lines=full_lines,
                                word2id=word2id,
                                num_examples=num_test_examples,
                                max_seq_len=opt.max_seq_len,
                                keep_rate=1,
                                processed_src=True,
                                tokenizer=tokenizer,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                drop_last=False)

    # Decode data
    (ae_results, ae_recon_loss, ae_mrr,
     ae_predictions) = eval_all_batch(opt,
                                      device,
                                      encoder,
                                      decoder,
                                      ae_loader,
                                      autoencoder=True)

    # Process results
    result = Result(opt, exp, tokenizer, ae_results, ae_predictions)
    result.nll = round(ae_recon_loss, 2)
    result.generate_text_output()  # With default options
    return result


def load_uniform_model_for_external_usage(opt, device):
    modified_opt = opt
    modified_opt.uniform_encoder = True
    modified_opt.uniform_keep_rate = 0  # Doesn't matter for decoder
    modified_opt.load_trained_encoder = False
    existing_exp = re.match("(.*)_\d+?.pt", opt.uniform_model_name).group(1)

    # Tokenizer
    tokenizer = data.build_tokenizer(opt)

    # Vocabulary
    word2id, id2word = data.build_vocab(opt,
                                        existing_exp,
                                        tokenizer,
                                        use_existing=True)

    # Model
    uniform_decoder, uniform_encoder, lambdas = model.build_model(modified_opt,
                                                                  tokenizer,
                                                                  word2id,
                                                                  id2word,
                                                                  device)
    utils.load_model(modified_opt, uniform_encoder, uniform_decoder, lambdas)
    model_info = {'tokenizer': tokenizer,
                  'word2id': word2id,
                  'encoder': uniform_encoder,
                  'decoder': uniform_decoder}

    return model_info


def load_default_model_for_external_usage(opt, device):
    existing_exp = re.match("(.*)_\d+?.pt", opt.model_name).group(1)

    # Tokenizer
    tokenizer = data.build_tokenizer(opt)

    # Vocabulary
    word2id, id2word = data.build_vocab(opt,
                                        existing_exp,
                                        tokenizer,
                                        use_existing=True)

    # Model
    trained_decoder, trained_encoder, lambdas = model.build_model(opt,
                                                                  tokenizer,
                                                                  word2id,
                                                                  id2word,
                                                                  device)
    utils.load_model(opt, trained_encoder, trained_decoder, lambdas)
    model_info = {'tokenizer': tokenizer,
                  'word2id': word2id,
                  'encoder': trained_encoder,
                  'decoder': trained_decoder}

    return model_info


def infer_default_model_for_external_usage(opt,
                                           device,
                                           model_info,
                                           key_tokens):
    tokenizer = model_info['tokenizer']
    word2id = model_info['word2id']
    encoder = model_info['encoder']
    decoder = model_info['decoder']

    if key_tokens == ['']:
        key_tokens = ["<eos>"]
    else:
        key_tokens += ["<eos>"]

    key_tokens = [token.lower() for token in key_tokens if token != "#"]
    key_seq = tokenizer.tokens_to_tensor(key_tokens, word2id).long().view(1, -1)  # noqa
    key_line = [tokenizer.tokens_to_encoded_line(key_tokens)]
    preds_lines, preds_tokens = eval_one_key(opt,
                                             device,
                                             encoder=encoder,
                                             decoder=decoder,
                                             key_seq=key_seq,
                                             key_line=key_line,
                                             verbose=True)
    return preds_lines


def main(opt, device):
    existing_exp = re.match("(.*)_\d+?.pt", opt.model_name).group(1)

    # Tokenizer
    tokenizer = data.build_tokenizer(opt)

    # Vocabulary
    word2id, id2word = data.build_vocab(opt,
                                        existing_exp,
                                        tokenizer,
                                        use_existing=True)

    # Model
    trained_decoder, trained_encoder, lambdas = model.build_model(opt,
                                                                  tokenizer,
                                                                  word2id,
                                                                  id2word,
                                                                  device)
    utils.load_model(opt, trained_encoder, trained_decoder, lambdas)

    # Evaluate
    if opt.cross_eval_models:  # multiple encoders x multiple decoders
        # Load data
        if opt.path_test_data:
            path_test_data = opt.path_test_data
        else:  # If not specified, use last n (unused) sentences from data
            filename = "train_lines.txt"
            path_test_data = os.path.join(opt.root, opt.data_dir, filename)
        full_lines = data.load_data(path_test_data,
                                    check_validity=True,
                                    reverse=opt.reverse)  # Lines not in train set

        def parse_model_info(opt):
            # NOTE uniform encoder must include "uni" or "Uni" and "Kr" in model_name
            """
            Examples
            {
                "Uni-Kr0.1": "UNI_CD_S3_word_uniform_D500000_E300_Ld0.001_uniformE_Kr0.1_100000.pt",
                "Uni-Kr0.3": "UNI_CD_S3_word_uniform_D500000_E300_Ld0.001_uniformE_Kr0.3_100000.pt",
                "Uni-Kr0.5": "UNI_CD_S3_word_uniform_D500000_E300_Ld0.001_uniformE_Kr0.5_100000.pt",
                "Uni-Kr0.7": "UNI_CD_S3_word_uniform_D500000_E300_Ld0.001_uniformE_Kr0.7_100000.pt",
                "Uni-Kr0.9": "UNI_CD_S3_word_uniform_D500000_E300_Ld0.001_uniformE_Kr0.9_100000.pt",

                "Stop-Dr0.5": "STOPWORD_CD_S3_word_stopwordE_Dr0.5_D500000_E300_Ld0.001_trainedE_Le0.001_50000.pt",
                "Stop-Dr1.0": "STOPWORD_CD_S3_word_stopwordE_Dr1.0_D500000_E300_Ld0.001_trainedE_Le0.001_50000.pt",

                "Trained-Eps0.05": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps0.05_250000.pt",
                "Trained-Eps0.1": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps0.1_250000.pt",
                "Trained-Eps0.15": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps0.15_250000.pt",
                "Trained-Eps0.2": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps0.2_250000.pt",
                "Trained-Eps0.4": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps0.4_250000.pt",
                "Trained-Eps0.6": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps0.6_250000.pt",
                "Trained-Eps0.8": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps0.8_250000.pt",
                "Trained-Eps1.0": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps1.0_250000.pt",
                "Trained-Eps1.2": "REMOVE_CD_S3_word_scratch_initL5.0_D500000_E300_Ld0.001_trainedE_Le0.001_lagr_Ll0.01_Eps1.2_250000.pt",
            }
            """
            with open(opt.path_cross_eval_models, "rb") as fb:
                cross_eval_models = pickle.load(fb)

            return cross_eval_models


        only_these_encoders = []
        only_these_decoders = []

        model_info = parse_model_info(opt)
        encoders, decoders = model.build_models(opt,
                                                tokenizer,
                                                word2id,
                                                id2word,
                                                device,
                                                model_info)

        # Create directory to save results
        path_results = os.path.join(opt.root, opt.exp_dir, "results")
        timestemp = datetime.datetime.now().strftime("%m%d")
        results_dir = os.path.join(path_results, timestemp, "cross")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Print all evaluations to run
        print("[Eval] Evaluations to run:")
        print(f"{'Encoder':20}{'Decoder':20}")
        for decoder_name, decoder in decoders.items():
            if only_these_decoders and (decoder_name not in only_these_decoders):  # noqa
                continue
            for encoder_name, encoder in encoders.items():
                if only_these_encoders and (encoder_name not in only_these_encoders):  # noqa
                    continue
                eval_name = utils.name_eval(opt, encoder_name, decoder_name)
                path_result_p = os.path.join(results_dir, f"{eval_name}.p")
                if not os.path.exists(path_result_p):
                    print(f"{encoder_name:20}{decoder_name:20}")

        print("\n[Eval] Start cross-evaluation")
        for decoder_name, decoder in decoders.items():
            if only_these_decoders and (decoder_name not in only_these_decoders):  # noqa
                continue
            for encoder_name, encoder in encoders.items():
                if only_these_encoders and (encoder_name not in only_these_encoders):  # noqa
                    continue
                eval_name = utils.name_eval(opt, encoder_name, decoder_name)
                path_result_txt = os.path.join(results_dir, f"{eval_name}.txt")
                path_result_p = os.path.join(results_dir, f"{eval_name}.p")
                if (os.path.exists(path_result_txt)
                        and os.path.exists(path_result_p)):
                    print(f"[Eval] Found results for {encoder_name}E and {decoder_name}D.")  # noqa
                    continue

                print(f"[Eval] Evaluating with {encoder_name}E and {decoder_name}D.")  # noqa
                result = evaluate_dataset(opt,
                                          device,
                                          existing_exp,
                                          tokenizer,
                                          full_lines,
                                          word2id,
                                          id2word,
                                          decoder,
                                          encoder,
                                          lambdas)

                # Save results
                with open(path_result_txt, "w") as f:
                    f.write(result.text_output)
                with open(path_result_p, "wb") as f:
                    pickle.dump(result, f)
                print(f"[Eval] Saved results at {path_result_txt}.")

        print(f"[Eval] Results are saved at {results_dir}")

    else:  # multiple encoders x single decoder
        # Load data
        if opt.path_test_data:
            path_test_data = opt.path_test_data
        else:
            filename = "train_lines.txt"
            path_test_data = os.path.join(opt.root, opt.data_dir, filename)
        full_lines = data.load_data(path_test_data,
                                    check_validity=True,
                                    reverse=opt.reverse)

        # Build uniform encoders
        encoders = dict()
        if opt.use_baseline_encoders:
            encoders.update(model.build_baseline_encoders(opt,
                                                          tokenizer,
                                                          word2id,
                                                          id2word,
                                                          device))
        encoders["trained"] = trained_encoder

        # Create directory to save results
        path_results = os.path.join(opt.root, opt.exp_dir, "results")
        timestemp = datetime.datetime.now().strftime("%m%d")
        results_dir = os.path.join(path_results, timestemp, opt.model_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for encoder_name, encoder in encoders.items():
            print(f"[Eval] Evaluating with {encoder_name} encoder.")
            eval_name = utils.name_eval(opt, encoder_name)
            result = evaluate_dataset(opt,
                                      device,
                                      existing_exp,
                                      tokenizer,
                                      full_lines,
                                      word2id,
                                      id2word,
                                      trained_decoder,
                                      encoder,
                                      lambdas)

            # Save results
            path_txt_results = os.path.join(results_dir, f"{eval_name}.txt")
            path_p_results = os.path.join(results_dir, f"{eval_name}.p")
            with open(path_txt_results, "w") as f:
                f.write(result.text_output)
            with open(path_p_results, "wb") as f:
                pickle.dump(result, f)
            print(f"[Eval] Saved results as {path_p_results}")

        print(f"[Eval] Results are saved at {results_dir}\n")


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="eval.py")

    opts.basic_opts(parser)  # Make sure options match with pretrained models
    opts.train_opts(parser)
    opts.model_opts(parser)
    opts.eval_opts(parser)

    opt = parser.parse_args()
    device = utils.init_device()

    if opt.path_models:  # Evaluate multiple models
        failed = []
        with open(opt.path_models, "r") as f:
            model_names = list(filter(None, f.read().split("\n")))
        for model_name in model_names:
            # try:
            if "uniformE" in model_name:
                opt.load_trained_encoder = False
                opt.uniform_encoder = True
                opt.uniform_keep_rate = float(re.findall(
                    "uniformE_Kr(.*?)_", model_name)[0])
            elif "stopwordE" in model_name:
                opt.load_trained_encoder = False
                opt.stopword_encoder = True
                opt.stopword_drop_rate = float(re.findall(
                    "stopwordE_Dr(.*?)_", model_name)[0])

            opt.model_name = model_name
            main(opt, device)
    else:
        main(opt, device)
