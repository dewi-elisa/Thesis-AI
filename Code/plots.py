import opts
import utils
import data
import model
import train

import torch.optim as optim
import matplotlib.pyplot as plt
import configargparse
import numpy as np
import os


def recon_loss_plot(opt, recon_loss, n_epochs, parameters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:pink']
    x = np.linspace(1, n_epochs, n_epochs)
    plt.figure()

    for i, parameter in enumerate(parameters):
        (recon_loss_training, recon_loss_val) = recon_loss[i]
        plt.plot(x, recon_loss_training,
                 label='train_' + str(parameter),
                 linestyle='-', color=colors[i])
        plt.plot(x, recon_loss_val,
                 label='val_' + str(parameter),
                 linestyle=':', color=colors[i])

    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction loss')
    plt.legend()
    plt.savefig('figs/recon_loss_' + str(opt.linear_weight) + '.png')


def cost_plot(opt, cost, n_epochs, parameters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:pink']
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
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:pink']
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
    for i, parameter in enumerate(parameters):
        text = '$\lambda$ = ' + str(parameter)
        plt.text(costs_train[i], accuracies_train[i], text)
    plt.xlabel('Kept (%)')
    plt.ylabel('Greedy accuracy (%)')
    plt.legend()
    plt.savefig('figs/acc_cost_train.png')

    # plt.figure()
    plt.plot(costs_val, accuracies_val, marker='o', linestyle='-', label='validation')
    for i, parameter in enumerate(parameters):
        text = '$\lambda$ = ' + str(parameter)
        plt.text(costs_val[i], accuracies_val[i], text)
    plt.xlabel('Kept (%)')
    plt.ylabel('Greedy accuracy (%)')
    plt.legend()
    plt.savefig('figs/acc_cost_val.png')


def train_models(opt, parameters, device):
    # Tokenizer
    tokenizer = data.build_tokenizer(opt)

    # Vocabulary
    word2id, id2word = data.build_vocab(opt, exp, tokenizer)

    # Data Loaders
    loaders = data.build_loaders(opt, tokenizer, word2id)

    efficiencies, losses, accuracies, recon_losses = [], [], [], []

    for parameter in parameters:
        opt.linear_weight = parameter

        model_name = 'unstructured' + '_' + str(parameter) + '_' + str(opt.epochs) + '.pth'

        # Model
        encoder = model.Encoder(opt, word2id, id2word, device).to(device)
        decoder = model.Decoder(opt, word2id, id2word, device).to(device)

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

        return efficiencies, losses, accuracies, recon_losses


def load_data(file):
    results = np.load(file)
    parameters = [0.5]
    n_epochs = 10

    efficiencies = [(results['efficiencies_train'], results['efficiencies_train'])]
    losses = [(results['losses_train'], results['losses_val'])]
    accuracies = [(results['accuracies_train'], results['accuracies_val'])]
    recon_losses = [(results['recon_losses_train'], results['recon_losses_val'])]

    return efficiencies, losses, accuracies, recon_losses, parameters, n_epochs


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

    parameters = np.linspace(4, 5, 6).tolist()
    print(parameters)

    file = 'models/results_0.5_10.npz'
    if file is None:
        efficiencies, losses, accuracies, recon_losses = train_models(opt, parameters, device)
        n_epochs = opt.epochs
    else:
        efficiencies, losses, accuracies, recon_losses, parameters, n_epochs = load_data(file)

    print()
    print('Making the plots...')

    if not os.path.exists("figs/"):
        os.mkdir("figs")

    recon_loss_plot(opt, recon_losses, n_epochs, parameters)
    cost_plot(opt, efficiencies, n_epochs, parameters)
    training_obj_plot(opt, losses, n_epochs, parameters)
    acc_vs_cost_plot(accuracies, efficiencies, parameters)
