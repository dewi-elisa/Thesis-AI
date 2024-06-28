import datetime

default_path = "/Users/dewi-elisa/Documents/Uni/scriptie KI/Thesis-AI/Code/"


def basic_opts(parser):
    group = parser.add_argument_group("Path")
    group.add("--root",
              type=str,
              default=default_path)
    group.add("--data_dir",
              type=str,
              default="data")
    group.add("--exp_dir",
              type=str,
              default="experiments")
    group.add("--log_dir",
              type=str,
              default=datetime.datetime.now().strftime("%m%d"))
    group.add("--prefix",
              type=str,
              default="")

    group = parser.add_argument_group("Initialization")
    group.add("--seed",
              type=int,
              default=3)

    group = parser.add_argument_group("Data")
    group.add("--num_examples",
              type=int,
              default=10)  # default=500000)
    group.add("--max_seq_len",
              type=int,
              default=20)
    group.add("--whitespace",
              default="remove",
              choices=["remove", "replace", "keep"])
    group.add("--capitalization",
              default="default",
              choices=["remove", "default"])

    group = parser.add_argument_group("Vocabulary")
    group.add("--build_new_vocab",
              action="store_true")
    group.add("--max_vocab_size",
              type=int,
              default=50000)


def train_opts(parser):
    group = parser.add_argument_group("Setup")

    # Lagrangian dual
    group.add("--lagrangian",
              action="store_true")
    group.add("--epsilon",
              type=float,
              default=0.6)
    group.add("--fix_lambda",
              action="store_true")
    group.add("--init_lambda",
              type=float,
              default=5.)

    group.add("--set_lambda_to_0",
              action="store_true")
    group.add("--relu_vio",
              action="store_true")

    # For linear objective
    group.add("--linear_weight",
              type=float,
              default=1.)

    # Options for encoder
    group.add("--fix_encoder",
              action="store_true")

    group = parser.add_argument_group("Optimization")
    group.add("--batch_size",
              type=int,
              default=1)  # default=128)
    group.add("--learning_rate",
              type=float,
              default=0.001)
    group.add("--clip",
              type=float,
              default=5.)
    group.add("--use_baseline",
              type=bool,
              default=True)

    group = parser.add_argument_group("Training")
    group.add("--start_epoch",
              type=int,
              default=1)
    group.add("--epochs",
              type=int,
              default=10)
    group.add("--start_global_step",
              type=int,
              default=0)
    group.add("--debug",
              action="store_true")

    group = parser.add_argument_group("Logging")
    group.add("--train_every",
              type=int,
              default=5000)
    group.add("--val_every",
              type=int,
              default=5000)
    group.add("--test_every",
              type=int,
              default=5000)
    group.add("--save_every",
              type=int,
              default=10)  # default=50000)
    group.add("--plot",
              type=bool,
              default=True)
    group.add("--examples",
              type=bool,
              default=True)
    group.add("--n",
              type=int,
              default=4)
    group.add("--verbose_autoencoder",
              type=bool,
              default=True)
    group.add("--verbose_keyword",
              action="store_true")
    group.add("--log_autoencoder_text",
              action="store_true")
    group.add("--log_keyword_text",
              action="store_true")


def model_opts(parser):
    group = parser.add_argument_group("Architecture")
    group.add("--embedding_dim",
              type=int,
              default=300)

    # Uniform encoder
    group.add("--uniform_encoder",
              action="store_true")
    group.add("--uniform_keep_rate",
              type=float,
              default=0.)

    # Stopword encoder
    group.add("--stopword_encoder",
              action="store_true")
    group.add("--stopword_drop_rate",
              type=float,
              default=1.)

    group = parser.add_argument_group("Load pretrained models")
    group.add("--load_pretrained_decoder",
              action="store_true")
    group.add("--load_trained_encoder",
              action="store_true")
    group.add("--load_trained_lambdas",
              action="store_true")
    group.add("--model_name",
              type=str)


def eval_opts(parser):
    group = parser.add_argument_group("Path")
    group.add("--path_test_data",
              type=str,
              default="",
              help="""Specify path to test_lines.txt. If not specified,
                      use last n (unused) sentences from train_lines.txt""")
    group.add("--path_models",
              type=str,
              default="",
              help="""Specify path to text file containing model names.
                      If empty, use --model_name""")
    group.add("--path_cross_eval_models",
              type=str,
              default="")

    group = parser.add_argument_group("Decoding")
    group.add("--max_eval_num",
              type=int,
              default=1000)
    group.add("--reverse",
              action="store_false")
    group.add("--beam_size",
              type=int,
              default=1,
              help="For decoding in evaluation")
    group.add("--use_baseline_encoders",
              action="store_true",
              help="For evaluating a single trained decoder")
    group.add("--cross_eval_models",
              action="store_true",
              help="For cross evaluating multiple encoders and decoders")

    group = parser.add_argument_group("Printing")
    group.add("--pretty_print",
              type=bool,
              default=True)
    group.add("--print_space_between_key_tokens",
              action="store_true")
    group.add("--print_src_first",
              type=bool,
              default=True)
