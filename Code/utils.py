import os
import sys
import random
import numpy as np
from colorama import Fore, Style

import torch
import torch.utils.data
from tensorboardX import SummaryWriter


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


def save_model(encoder, decoder, parameter, epoch, structured=False):
    if structured is True:
        structured = 'structured'
    else:
        structured = 'unstructured'
    name = structured + '_' + str(parameter) + '_' + str(epoch) + '.pth'

    save_dict = {"decoder": decoder.state_dict()}
    save_dict.update({"encoder": encoder.state_dict()})
    torch.save(save_dict, "models/" + name)


def load_model(name, encoder, decoder):
    path = 'models/' + name
    model = torch.load(path)

    decoder.load_state_dict(model.get("decoder"))
    encoder.load_state_dict(model.get("encoder"))


def cprint(text, color, highlight=None):
    """ Print text in color while highlighting specified vocab. """
    if highlight:
        for vocab in highlight:
            text = text.replace(vocab, Fore.RED + vocab + color)
    print(color + text + Style.RESET_ALL)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
