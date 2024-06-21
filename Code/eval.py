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

learning_rate = 0.001


def eval_batch(device, encoder, decoder, batch, parameter):
    src_seqs, trg_seqs, src_lines, trg_lines = batch
    src_seqs = src_seqs.to(device)
    trg_seqs = trg_seqs.to(device)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Encode
        subsentence, log_prob_mask = encoder(src_seqs)
        subsentence_lines = [encoder.id2word[x.item()] for x in subsentence]

        # Decode
        sentence, log_prob_sentence = decoder(subsentence, trg_seqs, decode_function='greedy')
        sentence_lines = [decoder.id2word[x.item()] for x in subsentence]

        # Calculate metrics
        efficiency = (len(subsentence) / len(src_seqs.squeeze(0))) * 100
        loss = - log_prob_sentence
        accuracy = (src_seqs == sentence)
        recon_loss = len(subsentence) + parameter * - log_prob_sentence

        return (list(zip(src_lines, subsentence_lines, sentence_lines,
                         src_seqs, subsentence, sentence)),
                efficiency, loss, accuracy, recon_loss)


def evaluate(opt, device, encoder, decoder, loader, parameter):
    results_all = []
    efficiency_all = []
    loss_all = []
    accuracy_all = []
    recon_loss_all = []

    for batch_index, batch in enumerate(loader):
        with torch.no_grad():
            results, efficiency, loss, accuracy, recon_loss = eval_batch(device,
                                                                         encoder, decoder,
                                                                         batch, parameter)
        results_all.extend(results)
        efficiency_all.append(efficiency)
        loss_all.append(loss)
        accuracy_all.append(accuracy)
        recon_loss_all.append(recon_loss)

        if len(results_all) >= opt.max_eval_num:
            break

    avg_efficiency = np.mean(efficiency_all)
    avg_loss = np.mean(loss_all)
    accuracy_all = np.sum(accuracy) / len(loss_all)
    avg_recon_loss = np.mean(recon_loss_all)

    return results_all, avg_efficiency, avg_loss, accuracy_all, avg_recon_loss


def get_figure_data(opt, device, encoder, decoder, word2id, id2word, optimizer, loaders):
    (train_ae_loader, train_key_loader,
     val_ae_loader, val_key_loader,
     test_ae_loader, test_key_loader) = loaders

    efficiencies = []
    accuracies = []

    parameters = [4, 4.2, 4.4, 4.6, 4.8]
    epoch = 10
    structured = 'unstructured'
    for parameter in parameters:
        model_name = structured + '_' + str(parameter) + '_' + str(epoch) + '.pth'

        # If model is available, get data
        if os.path.exists('models/' + model_name):
            print()
            print('Loading model ' + model_name + '...')
            utils.load_model(model_name, encoder, decoder)

            print('Evaluating the model...')
            encoder.eval()
            decoder.eval()
            # There is no test set, so val for now...
            _, efficiency, _, accuracy, _ = evaluate(opt, device, encoder, decoder,
                                                     val_ae_loader, parameter)
            efficiencies.append(efficiency)
            accuracies.append(accuracy)
        else:
            print()
            print(model_name + ' not available!')
            encoder = model.Encoder(opt, word2id, id2word, device)
            decoder = model.Decoder(opt, word2id, id2word, device)
            optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                   lr=learning_rate)

            print('Training it now...')
            train.train(opt, device, encoder, decoder, optimizer, loaders)

            print('Saving it...')
            utils.save_model(encoder, decoder, parameter, epoch)

            print('Evaluating the model...')
            encoder.eval()
            decoder.eval()
            # There is no test set, so val for now...
            _, efficiency, _, accuracy, _ = evaluate(opt, device, encoder, decoder,
                                                     val_ae_loader, parameter)
            efficiencies.append(efficiency)
            accuracies.append(accuracy)

    return efficiencies, accuracies, parameters


def func(x):
    return 1 / (1 + np.exp(-x))


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)


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
    encoder = model.Encoder(opt, word2id, id2word, device)
    decoder = model.Decoder(opt, word2id, id2word, device)

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=learning_rate)

    efficiency, accuracy, parameters = get_figure_data(opt, device,
                                                       encoder, decoder,
                                                       word2id, id2word,
                                                       optimizer, loaders)

    # parameters = [4, 4.2, 4.4, 4.6, 4.8]
    # efficiency = [15, 20, 30, 40, 45]
    # accuracy = [0, 3, 10, 15, 18]

    plt.figure()
    print(efficiency)
    print(accuracy)
    plt.plot(efficiency, accuracy, marker='o')

    # # fit the data to a curve
    # optimizedParameters, pcov = curve_fit(func, efficiency, accuracy)
    # print(optimizedParameters)
    # # Use the optimized parameters to plot the best fit
    # plt.plot(efficiency, func(*optimizedParameters), label="fit")

    # f = interp1d(efficiency, accuracy)
    # plt.plot(efficiency, f(efficiency))

    # p0 = [max(accuracy), np.median(efficiency), 1, min(accuracy)]  # mandatory initial guess
    # popt, pcov = curve_fit(sigmoid, efficiency, accuracy, p0)  # , method='dogbox')
    # plt.plot(efficiency, sigmoid(*popt), label="fit")

    for i, parameter in enumerate(parameters):
        text = '$\lambda$ = ' + str(parameter)
        plt.text(efficiency[i]+1, accuracy[i]-.5, text)

    plt.xlabel('Kept (%)')
    plt.ylabel('Greedy accuracy (%)')
    # plt.legend(title='Model')
    plt.savefig('parameters.png')
