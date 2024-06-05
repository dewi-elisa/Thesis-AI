import os
import sys
import random
import numpy as np
from colorama import Fore, Style

import torch
import torch.utils.data
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt


def name_exp(opt):
    if opt.uniform_encoder:
        encoder = "uniform_kr{}".format(opt.uniform_keep_rate)
    elif opt.stopword_encoder:
        encoder = "stopword_dr{}".format(opt.stopword_drop_rate)
    elif opt.lagrangian:
        encoder = "constrained_eps{}".format(opt.epsilon)
    else:
        encoder = "linear_w{}".format(opt.linear_weight)

    exp_name = encoder
    if opt.prefix:
        exp_name = opt.prefix + "_" + exp_name

    print(exp_name, "\n")
    print("Using pretrained decoder:", opt.load_pretrained_decoder, "\n")

    return exp_name


def name_eval(opt, encoder_type, decoder_type="trained"):
    eval_name = "{}E_{}D".format(encoder_type, decoder_type)
    if opt.whitespace != "remove":
        eval_name += "_w_{}".format(opt.whitespace)
    if opt.capitalization != "default":
        eval_name += "_c_{}".format(opt.capitalization)
    eval_name += "_beam{}".format(opt.beam_size)
    eval_name += "_{}".format(opt.num_examples)
    return eval_name


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("[Init] Initialized random seed with {}.".format(seed))


def init_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Init] Using {}.".format(device))
    return device


def init_writer(opt, exp):
    try:
        path_log = os.path.join(opt.root, opt.exp_dir, "log", opt.log_dir, exp)
        writer = SummaryWriter(path_log)
    except Exception as e:
        print(e)
        print("[Init] Check permission on directory for logging.")
        sys.exit()
    print("[Init] Logging at {}.".format(path_log))
    return writer


def save_model(opt, exp, encoder, decoder):
    path_model = os.path.join(opt.root, opt.exp_dir, "model")
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    save_path = os.path.join(path_model, "{}.pt".format(exp))
    save_dict = {"decoder": decoder.state_dict()}
    if (not opt.uniform_encoder) and (not opt.stopword_encoder):
        save_dict.update({"encoder": encoder.state_dict()})
    torch.save(save_dict, save_path)
    print("[Util] Saved model to {}.".format(save_path))


def load_model(opt, encoder, decoder, lambdas=0, global_step=0):
    path_model = os.path.join(opt.root, opt.exp_dir, "model")
    path_load = os.path.join(path_model, "{}".format(opt.model_name))
    if torch.cuda.is_available():
        loaded = torch.load(path_load)
    else:
        loaded = torch.load(path_load, map_location="cpu")

    decoder.load_state_dict(loaded.get("decoder"))
    if opt.load_trained_encoder:
        encoder.load_state_dict(loaded.get("encoder"))
    if opt.load_trained_lambdas:
        lambdas = loaded.get("lambdas")
    if opt.start_global_step > 0:
        global_step = opt.start_global_step
        print("[Util] Training model from {} step.".format(global_step))
    print("\n[Util] Loaded model from {}.".format(path_load))
    return global_step


def cprint(text, color, highlight=None):
    """ Print text in color while highlighting specified vocab. """
    if highlight:
        for vocab in highlight:
            text = text.replace(vocab, Fore.RED + vocab + color)
    print(color + text + Style.RESET_ALL)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
