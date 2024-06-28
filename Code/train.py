import os
import sys
import configargparse
from tqdm import tqdm

import torch
import torch.utils.data
import torch.optim as optim

import numpy as np

import opts
import utils
import data
import model
import eval


def calculate_loss(opt, subsentence, log_q_alpha, log_p_beta):
    f = len(subsentence) + opt.linear_weight * - log_p_beta
    return log_q_alpha * f.detach() + f, f.detach()


def train_batch(opt, device, encoder, decoder, word2id, id2word, optimizer, batch):
    encoder.train()
    decoder.train()

    src_seqs, trg_seqs, src_lines, trg_lines = batch
    src_seqs = src_seqs.squeeze(0).to(device)
    trg_seqs = trg_seqs.squeeze(0).to(device)

    # Add <sos> to src_seqs
    src_seqs = torch.cat((torch.tensor([word2id['<sos>']]), src_seqs))

    # max_src_len = src_seqs.size()

    """
    Encode and decode
    """
    # Encode
    subsentence, log_prob_mask = encoder(src_seqs)

    # Process keywords
    # key_seqs, key_lines = decoder.postprocess(key_seqs=None,
    #                                           key_mask=mask,
    #                                           encoded_lines=src_lines)

    # Decode
    sentence, log_prob_sentence = decoder(subsentence, trg_seqs)

    """
    Stats
    """
    # Use unprocessed key_seqs for rewards and stats
    # num_key_tokens = torch.sum(key_seqs.gt(0), dim=1).float()

    # NOTE we count whitespace here for accurate rewards here
    num_src_tokens = torch.sum(src_seqs.gt(0), dim=0).float()
    num_key_tokens_from_encoder = float(subsentence.size(dim=0))

    kept_perc_per_sample = num_key_tokens_from_encoder / num_src_tokens

    # update optimizer
    optimizer.zero_grad()
    loss, actual_loss = calculate_loss(opt, subsentence, log_prob_mask, log_prob_sentence)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.clip)
    optimizer.step()

    loss_terms = (loss.item(), actual_loss.item(), torch.mean(kept_perc_per_sample).item())

    results = (src_seqs, subsentence, src_lines, trg_lines)

    return loss_terms, results


def print_examples(encoder, decoder, word2id, id2word, f, src_seqs, trg_seqs):
    # Add <sos> to src_seqs
    src_seqs = torch.cat((torch.tensor([word2id['<sos>']]), src_seqs))

    f.write('src_seqs:\n')
    for token in src_seqs.tolist():
        f.write(id2word[token] + ' ')
    f.write('\n')

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Encode
        subsentence, log_prob_mask = encoder(src_seqs)

        f.write('subsentence:\n')
        for token in subsentence.tolist():
            f.write(id2word[token] + ' ')
        f.write('\nlog probability: ' + str(log_prob_mask.item()) + '\n')

        # Decode
        sentence, log_prob_sentence = decoder(subsentence, trg_seqs, decode_function='greedy')

        f.write('sentence:\n')
        for token in sentence:
            f.write(id2word[token] + ' ')
        f.write('\nlog probability: ' + str(log_prob_sentence.item()) + '\n\n')


def train(opt, device, encoder, decoder, word2id, id2word, optimizer, loaders):
    print("\n[Train] Training for {} epochs.".format(opt.epochs))

    (train_ae_loader, train_key_loader,
     val_ae_loader, val_key_loader,
     test_ae_loader, test_key_loader) = loaders

    num_batches = len(train_ae_loader)

    efficiencies_train, losses_train, accuracies_train, recon_losses_train = [], [], [], []
    efficiencies_val, losses_val, accuracies_val, recon_losses_val = [], [], [], []

    with tqdm(total=opt.epochs * num_batches) as pbar:
        for epoch in range(opt.epochs):
            efficiency_train, loss_train, accuracy_train, recon_loss_train = [], [], [], []
            efficiency_val, loss_val, accuracy_val, recon_loss_val = [], [], [], []
            for batch in train_ae_loader:

                (loss, actual_loss, key_perc), results = train_batch(opt, device,
                                                                     encoder, decoder,
                                                                     word2id, id2word,
                                                                     optimizer, batch)

                pbar.set_description("Epoch {}".format(epoch + 1))
                pbar.set_postfix(loss=actual_loss)
                pbar.update()

                # Evaluation
                if epoch % opt.save_every == 0:
                    utils.save_model(opt, encoder, decoder, epoch, False)
                    eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                                  train_ae_loader)
                    eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                                  val_ae_loader)
                    # eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                    #               test_ae_loader)
                else:
                    if epoch % opt.train_every == 0:
                        eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                                      train_ae_loader)

                    if epoch % opt.val_every == 0:
                        eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                                      val_ae_loader)

                    # if epoch % opt.test_every == 0:
                        # eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                        #               test_ae_loader)
                if opt.plot:
                    _, efficiency, loss, accuracy, recon_loss = eval.evaluate(opt, device,
                                                                              encoder, decoder,
                                                                              word2id, id2word,
                                                                              train_ae_loader)
                    efficiency_train.append(efficiency)
                    loss_train.append(loss)
                    accuracy_train.append(accuracy)
                    recon_loss_train.append(recon_loss)

                    _, efficiency, loss, accuracy, recon_loss = eval.evaluate(opt, device,
                                                                              encoder, decoder,
                                                                              word2id, id2word,
                                                                              val_ae_loader)
                    efficiency_val.append(efficiency)
                    loss_val.append(loss)
                    accuracy_val.append(accuracy)
                    recon_loss_val.append(recon_loss)

            if opt.plot:
                efficiencies_train.append(np.mean(efficiency_train))
                losses_train.append(np.mean(loss_train))
                accuracies_train.append(np.mean(accuracy_train))
                recon_losses_train.append(np.mean(recon_loss_train))

                efficiencies_val.append(np.mean(efficiency_val))
                losses_val.append(np.mean(loss_val))
                accuracies_val.append(np.mean(accuracy_val))
                recon_losses_val.append(np.mean(recon_loss_val))

            if opt.examples:
                # Write examples to file
                with open('examples.txt', 'a') as f:
                    f.write('Epoch: ' + str(epoch+1) + '\n')
                    for batch_index, batch in enumerate([next(iter(train_ae_loader))]):
                        src_seqs, trg_seqs, src_lines, trg_lines = batch
                        src_seqs = src_seqs.squeeze(0).to(device)
                        trg_seqs = trg_seqs.squeeze(0).to(device)
                        f.write('Train example:\n')
                        print_examples(encoder, decoder, word2id, id2word, f, src_seqs, trg_seqs)
                    for batch_index, batch in enumerate(val_ae_loader):
                        src_seqs, trg_seqs, src_lines, trg_lines = batch
                        src_seqs = src_seqs.squeeze(0).to(device)
                        trg_seqs = trg_seqs.squeeze(0).to(device)
                        f.write('Validation example:\n')
                        print_examples(encoder, decoder, word2id, id2word, f, src_seqs, trg_seqs)
                    f.write('\n')

    if opt.plot:
        return (efficiencies_train,
                efficiencies_val), (losses_train,
                                    losses_val), (accuracies_train,
                                                  accuracies_val), (recon_losses_train,
                                                                    recon_losses_val)


def main(opt, exp, device):
    # Check model availability first
    if opt.model_name:
        path_model = os.path.join(opt.root, opt.exp_dir, "model", opt.model_name)
        if not os.path.exists(path_model):
            print("Could not find the model at {}.".format(path_model))
            sys.exit()

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
                           lr=opt.learning_rate)

    # Train
    train(opt, device, encoder, decoder, word2id, id2word, optimizer, loaders)

    # Save model
    utils.save_model(encoder, decoder, opt.epochs)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="train.py")

    opts.basic_opts(parser)
    opts.train_opts(parser)
    opts.model_opts(parser)
    opts.eval_opts(parser)

    opt = parser.parse_args()
    exp = utils.name_exp(opt)
    device = utils.init_device()
    parser = configargparse.ArgumentParser(description="train.py")

    utils.init_seed(opt.seed)
    main(opt, exp, device)
