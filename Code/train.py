import os
import sys
import configargparse
from tqdm import tqdm

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import opts
import utils
import data
import model
import eval


def binary_cross_entropy_with_sigmoid_out(input, target):
    mask = target != -1
    loss_over_masked = F.binary_cross_entropy(input[mask], target[mask], reduction="none")
    result = torch.zeros_like(input)
    result[mask] = loss_over_masked
    # print(target)
    # loss = F.binary_cross_entropy(input[mask], target[mask], reduction="none")
    # loss.masked_fill_(target.ge(0).bitwise_not(), 0)  # ignore_index=-1
    return result


def ce_loss_per_token(prob_over_dyn_vocab,
                      reindexed_trg_seqs,
                      len_dyn_word2id,
                      device,
                      eps=1e-30):
    # Pad trg_seqs for greedy/beam search decoded outputs
    if prob_over_dyn_vocab.size(1) > reindexed_trg_seqs.size(1):
        padded_trg_seqs = torch.zeros(prob_over_dyn_vocab.size(0),
                                      prob_over_dyn_vocab.size(1))
        padded_trg_seqs[:, :reindexed_trg_seqs.size(
            1)] = reindexed_trg_seqs[:, :reindexed_trg_seqs.size(1)]
        reindexed_trg_seqs = padded_trg_seqs.long()

    softmax_out = prob_over_dyn_vocab.view(-1, len_dyn_word2id)
    target = reindexed_trg_seqs.view(-1).to(device)
    return F.nll_loss(torch.log(softmax_out + eps),
                      target,
                      ignore_index=0,
                      reduction="none")


bce_loss_per_token = binary_cross_entropy_with_sigmoid_out


def train_batch(opt, encoder, decoder, lambdas,
                optimizerD, optimizerE, optimizerL,
                batch, batch_index, global_step, baseline):

    # Dummy values for optional stats
    encoder_loss = torch.tensor([0.])
    key_likelihood_per_sample = torch.tensor([0.])
    reward_per_sample = torch.tensor([0.])
    recon_term_per_sample = torch.tensor([0.])

    encoder.train()
    decoder.train()

    src_seqs, trg_seqs, src_lines, trg_lines = batch  # Encoded form
    src_seqs = src_seqs.to(device)
    trg_seqs = trg_seqs.to(device)

    batch_size, max_src_len = src_seqs.size()

    """
    Encode and decode
    """
    # Encode
    key_prob_over_tokens, key_mask = encoder(src_seqs=src_seqs,
                                             uniform=uniform)

    # Process keywords
    # Need to have encoded form for decoder to properly tokenize to
    # build a dynamic vocabulary (e.g. [batch, norm] -> "batchnorm" (x))
    key_seqs, key_lines = decoder.postprocess(key_seqs=None,
                                              key_mask=key_mask,
                                              encoded_lines=src_lines)

    # Decode
    (prob_over_dyn_vocab, attn_weight_over_key,
     dyn_word2id, dyn_id2word,
     gen_prob_per_timestep) = decoder(key_seqs,
                                      key_lines,
                                      trg_seqs)
    reindexed_trg_seqs = decoder.reindex(trg_seqs,
                                         trg_lines,
                                         dyn_word2id).to(device)

    """
    Stats
    """
    # Use unprocessed key_seqs for rewards and stats
    # num_key_tokens = torch.sum(key_seqs.gt(0), dim=1).float()

    # NOTE we count whitespace here for accurate rewards here
    num_src_tokens = torch.sum(src_seqs.gt(0), dim=1).float()
    num_key_tokens_from_encoder = torch.sum(key_mask.gt(0), dim=1).float()

    kept_perc_per_sample = num_key_tokens_from_encoder / num_src_tokens

    recon_term_per_token = - ce_loss_per_token(prob_over_dyn_vocab,
                                               reindexed_trg_seqs,
                                               len(dyn_word2id),
                                               device)
    recon_term_per_sample = torch.sum(
        recon_term_per_token.view(batch_size, -1),
        dim=1).div(num_src_tokens)
    recon_loss = - torch.mean(recon_term_per_sample)
    violation = Variable(recon_loss.detach() - opt.epsilon, requires_grad=True)

    """
    Decoder
    """
    optimizerD.zero_grad()
    if train_decoder_only:
        decoder_loss = - (recon_term_per_sample).sum().div(batch_size)
    else:
        decoder_loss = - (opt.linear_weight *
                          recon_term_per_sample).sum().div(batch_size)
    decoder_loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.clip)
    optimizerD.step()

    """
    Encoder
    """
    if not train_decoder_only:
        optimizerE.zero_grad()

        key_likelihood_per_token = - bce_loss_per_token(
            key_prob_over_tokens.view(-1),
            key_mask.float().view(-1))
        key_likelihood_per_sample = torch.sum(
            key_likelihood_per_token.view(batch_size, max_src_len),
            dim=1).div(num_src_tokens)

        recon_term_per_sample.detach_()
        if opt.lagrangian:
            reward_per_sample = (- (num_key_tokens_from_encoder)
                                 + (lambdas.detach() *
                                    (recon_term_per_sample + opt.epsilon)))
            if opt.use_baseline:
                reward_per_sample -= baseline
        else:  # Linear reward
            reward_per_sample = (- (num_key_tokens_from_encoder)
                                 + (opt.linear_weight *
                                    recon_term_per_sample))

        encoder_loss = - (key_likelihood_per_sample *
                          reward_per_sample).sum().div(batch_size)

        encoder_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.clip)
        optimizerE.step()

    obj_value = torch.mean(num_key_tokens_from_encoder) + lambdas * violation
    loss_terms = (decoder_loss.item(), encoder_loss.item(),
                  torch.mean(key_likelihood_per_sample).item(),
                  torch.mean(reward_per_sample).item(),
                  torch.mean(recon_term_per_sample).item(),
                  torch.mean(kept_perc_per_sample).item(),
                  torch.mean(lambdas).item(),
                  violation.item(),
                  obj_value.item())

    results = (src_seqs, key_seqs, reindexed_trg_seqs,
               src_lines, key_lines, trg_lines)

    return loss_terms, results


def train(opt,
          exp,
          encoder,
          decoder,
          lambdas,
          optimizerD,
          optimizerE,
          optimizerL,
          schedularD,
          schedularE,
          schedularL,
          loaders,
          global_step):
    print("\n[Train] Training for {} epochs.".format(opt.epochs))
    baseline, num = 0, 1.  # Stats for baseline

    (train_ae_loader, train_key_loader,
     val_ae_loader, val_key_loader,
     test_ae_loader, test_key_loader) = loaders

    num_batches = len(train_ae_loader)
    with tqdm(total=opt.epochs * num_batches) as pbar:
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            for batch_index, batch in enumerate(train_ae_loader):
                global_step += 1
                num += 1

                (decoder_loss, encoder_loss,
                 key_likelihood, reward,
                 recon_term, key_perc, lambdas_mean, violation, obj_value
                 ), results = train_batch(opt, encoder, decoder, lambdas,
                                          optimizerD, optimizerE, optimizerL,
                                          batch, batch_index, global_step,
                                          baseline)

                pbar.set_description("Epoch {}".format(epoch))
                pbar.set_postfix(decoder=decoder_loss)
                pbar.update()

                # Calculate running average for baseline
                baseline -= baseline / num
                baseline += reward / num

                # Statistics
                if global_step % 100 == 0:
                    writer.add_scalar(tag="loss/kept_perc_incl_whitespace",
                                      scalar_value=key_perc,
                                      global_step=global_step)
                    writer.add_scalar(tag="loss/decoder_loss",
                                      scalar_value=decoder_loss,
                                      global_step=global_step)
                    writer.add_scalar(tag="lr/decoder_lr",
                                      scalar_value=utils.get_lr(optimizerD),
                                      global_step=global_step)
                    writer.add_scalar(tag="lr/encoder_lr",
                                      scalar_value=utils.get_lr(optimizerE),
                                      global_step=global_step)
                    writer.add_scalar(tag="lr/lambda_lr",
                                      scalar_value=utils.get_lr(optimizerL),
                                      global_step=global_step)
                # Evaluation
                if global_step % opt.save_every == 0:
                    utils.save_model(opt, exp, global_step,
                                     encoder, decoder, lambdas)
                    eval.evaluate(opt, exp, device, encoder, decoder,
                                  "train", train_ae_loader, train_key_loader,
                                  global_step, writer, pbar, log=True)
                    eval.evaluate(opt, exp, device, encoder, decoder,
                                  "val", val_ae_loader, val_key_loader,
                                  global_step, writer, pbar, log=True)
                    eval.evaluate(opt, exp, device, encoder, decoder,
                                  "test", test_ae_loader, test_key_loader,
                                  global_step, writer, pbar, log=True)
                else:
                    if global_step % opt.train_every == 0:
                        eval.evaluate(opt, exp, device, encoder, decoder,
                                      "train", train_ae_loader, train_key_loader,  # noqa
                                      global_step, writer, pbar, log=True)

                    if global_step % opt.val_every == 0:
                        eval.evaluate(opt, exp, device, encoder, decoder,
                                      "val", val_ae_loader, val_key_loader,
                                      global_step, writer, pbar, log=True)

                    if global_step % opt.test_every == 0:
                        eval.evaluate(opt, exp, device, encoder, decoder,
                                      "test", test_ae_loader, test_key_loader,
                                      global_step, writer, pbar, log=True)


def main(opt, exp, device, writer):
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
    decoder, encoder, lambdas = model.build_model(opt,
                                                  tokenizer,
                                                  word2id,
                                                  id2word,
                                                  device)

    # Optimizers
    if opt.decoder_optimizer == "sgd":
        optimizerD = optim.SGD(decoder.parameters(),
                               lr=opt.decoder_learning_rate)
    elif opt.decoder_optimizer == "adam":
        optimizerD = optim.Adam(decoder.parameters(),
                                lr=opt.decoder_learning_rate)

    if opt.encoder_optimizer == "sgd":
        optimizerE = optim.SGD(encoder.parameters(),
                               lr=opt.encoder_learning_rate)
    elif opt.encoder_optimizer == "adam":
        optimizerE = optim.Adam(encoder.parameters(),
                                lr=opt.encoder_learning_rate)

    optimizerL = optim.SGD([lambdas],
                           lr=opt.lambdas_learning_rate,
                           momentum=opt.lambdas_momentum)

    # Scheduler for decaying learning rates
    schedularD = optim.lr_scheduler.StepLR(optimizerD,
                                           step_size=opt.step_sizeD,
                                           gamma=opt.gammaD)
    schedularE = optim.lr_scheduler.StepLR(optimizerE,
                                           step_size=opt.step_sizeE,
                                           gamma=opt.gammaE)
    schedularL = optim.lr_scheduler.StepLR(optimizerL,
                                           step_size=opt.step_sizeL,
                                           gamma=opt.gammaL)

    global_step = 0

    # Pretrained model
    if opt.load_pretrained_decoder:
        global_step = utils.load_model(opt, encoder, decoder, lambdas)

    # Train
    train(opt,
          exp,
          encoder,
          decoder,
          lambdas,
          optimizerD,
          optimizerE,
          optimizerL,
          schedularD,
          schedularE,
          schedularL,
          loaders,
          global_step)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="train.py")

    opts.basic_opts(parser)
    opts.train_opts(parser)
    opts.model_opts(parser)
    opts.eval_opts(parser)

    opt = parser.parse_args()
    exp = utils.name_exp(opt)
    device = utils.init_device()
    writer = utils.init_writer(opt, exp)

    utils.init_seed(opt.seed)
    main(opt, exp, device, writer)

    writer.close()
