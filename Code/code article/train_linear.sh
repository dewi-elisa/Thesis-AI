#!/bin/bash

echo ================ Create output folder
mkdir experiments

echo ================ Start training linear models
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 3.0 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.0 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.2 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.25 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.27 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.29 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.33 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.4 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.6 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 4.8 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 5.0 --epochs 30 || exit 1
time python3 autocomplete/train.py --root "./" --prefix LINEAR --log_dir LINEAR --linear_weight 6.0 --epochs 30 || exit 1
