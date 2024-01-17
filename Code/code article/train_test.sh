#!/bin/bash

echo ================ Create output folder
mkdir experiments

echo ================ Start training one of linear models
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.25 --epochs 1 --num_examples 500 || exit 1

echo ================ Start training one of constrained models
time python3 autocomplete/train.py --root "./" --prefix CONSTRAINED --log_dir CONSTRAINED --lagrangian --epsilon 0.4 --decay_decoder_lr --epochs 1 --num_examples 500 || exit 1
