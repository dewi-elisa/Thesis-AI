import opts
import utils
import data
import model
import train

import torch.optim as optim
import matplotlib.pyplot as plt
import configargparse
import numpy as np


def recon_loss_plot(opt, recon_loss, n_epochs, parameters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    x = np.linspace(1, n_epochs, n_epochs)
    plt.figure()

    for i, parameter in enumerate(parameters):
        (recon_loss_training, recon_loss_val) = recon_loss[i]
        plt.plot(x, recon_loss_training,
                 label='training_' + str(parameter),
                 linestyle='-', color=colors[i])
        plt.plot(x, recon_loss_val,
                 label='validation_' + str(parameter),
                 linestyle=':', color=colors[i])

    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction loss')
    plt.legend()
    plt.savefig('figs/recon_loss_' + str(opt.linear_weight) + '.png')


def cost_plot(opt, cost, n_epochs, parameters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    x = np.linspace(1, n_epochs, n_epochs)
    plt.figure()

    for i, parameter in enumerate(parameters):
        (cost_training, cost_val) = cost[i]
        plt.plot(x, cost_training,
                 label='training_' + str(parameter),
                 linestyle='-', color=colors[i])
        plt.plot(x, cost_val,
                 label='validation_' + str(parameter),
                 linestyle=':', color=colors[i])

    plt.xlabel('Epoch')
    plt.ylabel('Cost (%)')
    plt.legend()
    plt.savefig('figs/cost_' + str(opt.linear_weight) + '.png')


def training_obj_plot(opt, obj, n_epochs, parameters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    x = np.linspace(1, n_epochs, n_epochs)
    plt.figure()

    for i, parameter in enumerate(parameters):
        (obj_training, obj_val) = obj[i]
        plt.plot(x, obj_training,
                 label='training_' + str(parameter),
                 linestyle='-', color=colors[i])
        plt.plot(x, obj_val,
                 label='validation_' + str(parameter),
                 linestyle=':', color=colors[i])

    plt.xlabel('Epoch')
    plt.ylabel('Training objective')
    plt.legend()
    plt.savefig('figs/obj_' + str(opt.linear_weight) + '.png')


def acc_vs_cost_plot(accuracy, cost, parameters):
    accuracies_train, accuracies_val = [], []
    costs_train, costs_val = [], []

    for i, parameter in enumerate(parameters):
        (acc_train, acc_val) = accuracy[i]
        accuracies_train.append(acc_train[-1])
        accuracies_val.append(acc_val[-1])

        (cost_train, cost_val) = cost[i]
        costs_train.append(cost_train[-1])
        costs_val.append(cost_val[-1])

    plt.figure()
    plt.plot(costs_train, accuracies_train, marker='o', linestyle='-', label='training')
    plt.plot(costs_val, accuracies_val, marker='o', linestyle=':', label='validation')

    for i, parameter in enumerate(parameters):
        text = '$\lambda$ = ' + str(parameter)
        plt.text(costs_train[i]+1, accuracies_train[i]-.5, text)
        plt.text(costs_val[i]+1, accuracies_val[i]-.5, text)

    plt.xlabel('Kept (%)')
    plt.ylabel('Greedy accuracy (%)')
    plt.legend()
    plt.savefig('figs/acc_cost.png')


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
    opt.linear_weight = 4.3

    # Tokenizer
    tokenizer = data.build_tokenizer(opt)

    # Vocabulary
    word2id, id2word = data.build_vocab(opt, exp, tokenizer)

    # Data Loaders
    loaders = data.build_loaders(opt, tokenizer, word2id)

    # parameters = np.linspace(4, 5, 6).tolist()
    parameters = [10000]
    print(parameters)
    efficiencies, losses, accuracies, recon_losses = [], [], [], []

    for parameter in parameters:
        opt.linear_weight = parameter

        model_name = 'unstructured' + '_' + str(parameter) + '_' + str(opt.epochs) + '.pth'

        # Model
        encoder = model.Encoder(opt, word2id, id2word, device)
        decoder = model.Decoder(opt, word2id, id2word, device)

        # Optimizer
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=opt.learning_rate)

        print()
        print('Gathering data...')
        efficiency, loss, accuracy, recon_loss = train.train(opt, device, encoder, decoder,
                                                             word2id, id2word, optimizer, loaders)

        efficiencies.append(efficiency)
        losses.append(loss)
        accuracies.append(accuracy)
        recon_losses.append(recon_loss)

        print()
        print('Saving model...')
        utils.save_model(opt, encoder, decoder, opt.epochs)

    print()
    # print('Making the plots...')
    recon_loss_plot(opt, recon_losses, opt.epochs, parameters)
    cost_plot(opt, efficiencies, opt.epochs, parameters)
    training_obj_plot(opt, losses, opt.epochs, parameters)
    acc_vs_cost_plot(accuracies, efficiencies, parameters)
