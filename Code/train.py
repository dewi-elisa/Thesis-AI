import os
import sys
import configargparse
from tqdm import tqdm

import torch
import torch.utils.data
import torch.optim as optim

import opts
import utils
import data
import model
import eval

learning_rate = 0.001
parameter = 4.3
epochs = 10


def calculate_loss(subsentence, log_q_alpha, log_p_beta):
    f = len(subsentence) + parameter * - log_p_beta
    return log_q_alpha * f.detach() + f, f.detach()


def train_batch(device, encoder, decoder, word2id, id2word, optimizer, batch):
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
    loss, actual_loss = calculate_loss(subsentence, log_prob_mask, log_prob_sentence)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.clip)
    optimizer.step()

    loss_terms = (loss.item(), actual_loss.item(), torch.mean(kept_perc_per_sample).item())

    results = (src_seqs, subsentence, src_lines, trg_lines)

    return loss_terms, results


def train(opt, device, encoder, decoder, word2id, id2word, optimizer, loaders):
    print("\n[Train] Training for {} epochs.".format(opt.epochs))

    (train_ae_loader, train_key_loader,
     val_ae_loader, val_key_loader,
     test_ae_loader, test_key_loader) = loaders

    num_batches = len(train_ae_loader)
    with tqdm(total=epochs * num_batches) as pbar:
        for epoch in range(epochs):
            for batch in train_ae_loader:

                (loss, actual_loss, key_perc), results = train_batch(device, encoder, decoder,
                                                                     word2id, id2word,
                                                                     optimizer, batch)

                pbar.set_description("Epoch {}".format(epoch + 1))
                pbar.set_postfix(loss=actual_loss)
                pbar.update()

                # Evaluation
                if epoch % opt.save_every == 0:
                    utils.save_model(encoder, decoder, parameter, epoch, False)
                    eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                                  train_ae_loader, parameter)
                    eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                                  val_ae_loader, parameter)
                    # eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                    #               test_ae_loader, parameter)
                else:
                    if epoch % opt.train_every == 0:
                        eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                                      train_ae_loader, parameter)

                    if epoch % opt.val_every == 0:
                        eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                                      val_ae_loader, parameter)

                    # if epoch % opt.test_every == 0:
                        # eval.evaluate(opt, device, encoder, decoder, word2id, id2word,
                        #               test_ae_loader, parameter)


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
                           lr=learning_rate)

    # Train
    train(opt, device, encoder, decoder, word2id, id2word, optimizer, loaders)

    # Save model
    utils.save_model(encoder, decoder, parameter, epochs)


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
