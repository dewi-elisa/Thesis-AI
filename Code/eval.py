# import re
import os
# import copy
# import math
# import pickle
# import datetime
# import itertools
# import collections
import configargparse
import numpy as np
# from tqdm import tqdm
# from colorama import Fore, Style

import torch

import opts
import utils
import data
import model
import train
# from utils import cprint
# from result import Result

import torch.optim as optim
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.interpolate import interp1d


def eval_batch(opt, device, encoder, decoder, word2id, id2word, batch):
    src_seqs, trg_seqs, src_lines, trg_lines = batch
    src_seqs = src_seqs.squeeze(0).to(device)
    trg_seqs = trg_seqs.squeeze(0).to(device)

    # Add <sos> to src_seqs
    src_seqs = torch.cat((torch.tensor([word2id['<sos>']]).to(device), src_seqs))

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Encode
        subsentence, log_prob_mask, _ = encoder(src_seqs)
        subsentence_lines = [id2word[x.item()] for x in subsentence]

        # Decode
        sentence, log_prob_sentence = decoder(subsentence, trg_seqs, decode_function='greedy')
        sentence_lines = [id2word[x] for x in sentence]

        # Remove <sos> and <eos>
        if subsentence.size()[0] == 0:
            subsentence = subsentence
        elif subsentence[-1] == word2id["<eos>"]:
            subsentence = subsentence[1:-1]
        else:
            subsentence = subsentence[1:]

        if sentence[-1] == word2id["<eos>"]:
            sentence = sentence[1:-1]
        else:
            sentence = sentence[1:]
        src_seqs = src_seqs[1:-1]

        # Calculate metrics
        # print(src_lines)
        # print(subsentence_lines)
        # print(sentence_lines)
        # print(src_seqs.tolist())
        # print(subsentence)
        # print(trg_seqs)
        # print()
        # print(sentence)
        efficiency = (len(subsentence) / len(src_seqs.tolist())) * 100
        recon_loss = - log_prob_sentence.item()
        accuracy = (src_seqs.tolist() == sentence)
        loss = len(subsentence) + opt.linear_weight * - log_prob_sentence.item()

        return (list((src_lines, subsentence_lines, sentence_lines,
                      src_seqs, subsentence, sentence)),
                efficiency, loss, accuracy, recon_loss)


def evaluate(opt, device, encoder, decoder, word2id, id2word, loader):
    results_all = []
    efficiency_all = []
    loss_all = []
    accuracy_all = []
    recon_loss_all = []

    for batch_index, batch in enumerate(loader):
        with torch.no_grad():
            results, efficiency, loss, accuracy, recon_loss = eval_batch(opt, device,
                                                                         encoder, decoder,
                                                                         word2id, id2word,
                                                                         batch)

        results_all.extend(results)
        efficiency_all.append(efficiency)
        loss_all.append(loss)
        accuracy_all.append(accuracy)
        recon_loss_all.append(recon_loss)

        if len(results_all) >= opt.max_eval_num:
            break

    avg_efficiency = np.mean(efficiency_all)
    avg_loss = np.mean(loss_all)
    accuracy_all = np.sum(accuracy) / len(loss_all) * 100
    avg_recon_loss = np.mean(recon_loss_all)

    return results_all, avg_efficiency, avg_loss, accuracy_all, avg_recon_loss


def get_figure_data(opt, device, encoder, decoder, word2id, id2word, optimizer, loaders):
    (train_ae_loader, train_key_loader,
     val_ae_loader, val_key_loader,
     test_ae_loader, test_key_loader) = loaders

    efficiencies = []
    accuracies = []

    parameters = [4.0, 4.2, 4.4, 4.6, 4.8, 5.0]

    structured = 'unstructured'
    for parameter in parameters:
        model_name = structured + '_' + str(parameter) + '_' + str(opt.epochs) + '.pth'
        opt.linear_weight = parameter

        # If model is available, get data
        if os.path.exists('models/' + model_name):
            print()
            print('Loading model ' + model_name + '...')
            utils.load_model('lr_0.01/' + model_name, encoder, decoder)

            print('Evaluating the model...')
            encoder.eval()
            decoder.eval()
            # There is no test set, so val for now...
            _, efficiency, _, accuracy, _ = evaluate(opt, device, encoder, decoder,
                                                     word2id, id2word,
                                                     train_ae_loader, parameter)
            efficiencies.append(efficiency)
            accuracies.append(accuracy)
        elif os.path.exists('models/lr_0.01/' + structured + '_' + str(parameter) + '_' + str(10) + '.pth'):
            print()
            print('Loading model ' + model_name + '...')
            encoder = model.Encoder(opt, word2id, id2word, device).to(device)
            decoder = model.Decoder(opt, word2id, id2word, device).to(device)
            utils.load_model('lr_0.01/' + structured + '_' + str(parameter) + '_' + str(10) + '.pth', encoder, decoder)
            optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                   lr=opt.learning_rate)

            print('Training it now...')
            train.train(opt, device, encoder, decoder, word2id, id2word, optimizer, loaders)

            print('Saving it...')
            utils.save_model(encoder, decoder, parameter, opt.epochs+10)

            print('Evaluating the model...')
            encoder.eval()
            decoder.eval()
            # There is no test set, so val for now...
            _, efficiency, _, accuracy, _ = evaluate(opt, device, encoder, decoder,
                                                     word2id, id2word,
                                                     val_ae_loader)
            efficiencies.append(efficiency)
            accuracies.append(accuracy)
        else:
            print()
            print(model_name + ' not available!')
            encoder = model.Encoder(opt, word2id, id2word, device).to(device)
            decoder = model.Decoder(opt, word2id, id2word, device).to(device)
            optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                   lr=opt.learning_rate)

            print('Training it now...')
            train.train(opt, device, encoder, decoder, word2id, id2word, optimizer, loaders)

            print('Saving it...')
            utils.save_model(encoder, decoder, opt.epochs)

            print('Evaluating the model...')
            encoder.eval()
            decoder.eval()
            # There is no test set, so val for now...
            _, efficiency, _, accuracy, _ = evaluate(opt, device, encoder, decoder,
                                                     word2id, id2word,
                                                     val_ae_loader)
            efficiencies.append(efficiency)
            accuracies.append(accuracy)

    return efficiencies, accuracies, parameters


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="train.py")

    opts.basic_opts(parser)
    opts.train_opts(parser)
    opts.model_opts(parser)
    opts.eval_opts(parser)

    opt = parser.parse_args()
    exp = utils.name_exp(opt)
    device = utils.init_device()

    utils.init_seed(opt.seed)

    # Tokenizer
    tokenizer = data.build_tokenizer(opt)

    # Vocabulary
    word2id, id2word = data.build_vocab(opt, exp, tokenizer)

    # Data Loaders
    loaders = data.build_loaders(opt, tokenizer, word2id)

    # Model
    encoder = model.Encoder(opt, word2id, id2word, device).to(device)
    decoder = model.Decoder(opt, word2id, id2word, device).to(device)

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=opt.learning_rate)

    efficiency, accuracy, parameters = get_figure_data(opt, device,
                                                       encoder, decoder,
                                                       word2id, id2word,
                                                       optimizer, loaders)

    # parameters = [4, 4.2, 4.4, 4.6, 4.8]
    # efficiency = [15, 20, 30, 40, 45]
    # accuracy = [0, 3, 10, 15, 18, 23]

    plt.figure()
    print(efficiency)
    print(accuracy)
    plt.plot(efficiency, accuracy, marker='o')

    for i, parameter in enumerate(parameters):
        text = '$\lambda$ = ' + str(parameter)
        plt.text(efficiency[i]+1, accuracy[i]-.5, text)

    plt.xlabel('Kept (%)')
    plt.ylabel('Greedy accuracy (%)')
    # plt.legend(title='Model')
    plt.savefig('parameters.png')
