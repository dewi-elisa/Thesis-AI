#!/bin/bash

echo ================ Create output folder
mkdir experiments

echo ================ Start training stopword models
time python3 autocomplete/train.py --root "./" --prefix STOPWORD --log_dir STOPWORD --stopword_encoder --stopword_drop_rate 0.5 --epochs 15 || exit 1
time python3 autocomplete/train.py --root "./" --prefix STOPWORD --log_dir STOPWORD --stopword_encoder --stopword_drop_rate 1.0 --epochs 15 || exit 1
