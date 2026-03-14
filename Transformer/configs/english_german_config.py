import os


class english_german_config:
    # TODO Add Docstring
    d_model = 512  # All sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512

    special_tokens = {  # The integer representations
        "pad_token": 0,  # '<PAD>'
        "unk_token": 1,  # '<UNK>'
        "sos_token": 2,  # '<SOS>'
        "eos_token": 3,  # '<EOS>'
    }

    # ===Encoder & Decoder
    h = 8  # How many H heads in the Multi-Head Attention
    dropout = 0.1
    d_ff = 2048
    N = 6
    enc_dec_N = 6  # Num of stacks of encoders and decoders

    # ====== Dataset ======
    max_seq_len = 5000 #TODO what is this for again?
    seq_max_len = 10  #TODO same here not sure this is needed! # Example if set to 4 "The brown rabbit" sentence is tuned into [21, 33, 15, 0, 0], the 0 is for the <padding>

    batch_size = 64

    vocab_size_constraint = (
        37_000  # Limit to let tokenizer trainer know when to stop merging sub-words.
    )
    vocab_size_dim = None  # The dimension of the embedding matrix. It Depends on size of loaded database, it is set later in code.
    tokenizer = "BPE"
    dataset_name = "WMT 2014 English-German"
    perc_to_download: int = 1  # percentage of database to download

    # ====== Training ======
    warmup_steps = 4_000
    step_num = 100_000

    # ====== Folder Structure ======
    CFG_PATH = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CFG_PATH, ".."))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
