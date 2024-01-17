#!/bin/bash

echo ================ Create output folder
mkdir experiments

echo ================ Start training constrained models
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 0.05 --decay_decoder_lr --epochs 65 || exit 1
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 0.1 --decay_decoder_lr --epochs 65 || exit 1
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 0.15 --decay_decoder_lr --epochs 65 || exit 1
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 0.2 --decay_decoder_lr --epochs 65 || exit 1
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 0.4 --decay_decoder_lr --epochs 65 || exit 1
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 0.6 --decay_decoder_lr --epochs 65 || exit 1
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 0.8 --decay_decoder_lr --epochs 65 || exit 1
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 1.0 --decay_decoder_lr --epochs 65 || exit 1
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 1.2 --decay_decoder_lr --epochs 65 || exit 1
