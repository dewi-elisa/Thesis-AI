#!/bin/bash

echo ================ Create output folder
mkdir experiments

echo ================ Start training uniform models
time python3 autocomplete/train.py --root "./" --prefix UNIFORM --log_dir UNIFORM --uniform_encoder --uniform_keep_rate 0.1 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix UNIFORM --log_dir UNIFORM --uniform_encoder --uniform_keep_rate 0.3 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix UNIFORM --log_dir UNIFORM --uniform_encoder --uniform_keep_rate 0.5 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix UNIFORM --log_dir UNIFORM --uniform_encoder --uniform_keep_rate 0.7 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix UNIFORM --log_dir UNIFORM --uniform_encoder --uniform_keep_rate 0.9 --epochs 30 || exit 1
