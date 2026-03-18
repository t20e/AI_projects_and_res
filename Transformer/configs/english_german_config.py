import os

# TODO add  default english_german_config the one that was used to train the base model, I will only train on a small percentage of te dataset
class english_german_config:
    # TODO Add Docstring
    device = "mps" # if using NVIDIA GPU set to "cuda" or if no GPU available set to "cpu".
    d_model = 512  # All sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512

    special_tokens = {  # The integer representations
        "pad_token": 0,  # '<PAD>'
        "unk_token": 1,  # '<UNK>'
        "sos_token": 2,  # '<SOS>'
        "eos_token": 3,  # '<EOS>'
    }

    # ================== Encoder & Decoder ==================
    h = 8  # How many H heads in the Multi-Head Attention
    dropout = 0.1
    d_ff = 2048
    N = 6
    enc_dec_N = 6  # Num of stacks of encoders and decoders

    # ================== Dataset ==================
    pos_seq_len = 5000 # Positional Encoding sequence length
    batch_size = 64
    vocab_size = ( # TODO since were training on smaller ds, what is the best value here?
        16_000  # Paper: 37000. Limit to let tokenizer trainer know when to stop merging sub-words.
    )
    vocab_size_dim = None  # The dimension of the embedding matrix. It Depends on size of loaded database, it is set later in code.
    tokenizer = "BPE"
    dataset_name = "WMT 2014 English-German"
    perc_to_download: int = 1  # percentage of database to download
    total_sentence_pairs = None # The total number of English-German sentence pairs, will be set later in code.

    # ================== Training ==================
    warmup_steps = None # It Depends on size of loaded database, it is set later in code.  Paper: 4_000
    step_num_limit = 100_000 # Total number of steps to train the model.
    num_epochs = 10


    # ================== Folder Structure ==================
    CFG_PATH = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CFG_PATH, ".."))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
