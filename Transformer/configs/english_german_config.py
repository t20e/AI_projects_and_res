import os


class English_german_config:
    """
    🚨 Smaller config than the paper to train a decent model, that will not take days to train on GPU. Check Paper_english_german_config() class for the default paper config for its base model.
    """

    # TODO set config to get the device depending automatically
    device = (
        "mps"  # if using NVIDIA GPU set to "cuda" or if no GPU available set to "cpu".
    )

    d_model = 256  # All sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512



    # ================== Encoder & Decoder ==================
    h = 8  # How many H heads in the Multi-Head Attention
    dropout = 0.1
    d_ff = 1024
    N = 4
    enc_dec_N = 4  # Num of stacks of encoders and decoders



    # ================== Dataset ==================
    pos_seq_len = 5000  # Positional Encoding sequence length
    batch_size = 64
    vocab_size = (
        16_000  # Limit to let tokenizer trainer know when to stop merging sub-words.
    )
    tokenizer = "BPE"
    dataset_name = "WMT 2014 English-German"

    # Percentage of database to download, 10% is ~450,000 sentences
    perc_to_download: int = 10

    total_sentence_pairs = None  # The total number of English-German sentence pairs, will be set later in code.
    special_tokens = {  # The integer representations
        "pad_token": 0,  # '<PAD>'
        "unk_token": 1,  # '<UNK>'
        "sos_token": 2,  # '<SOS>'
        "eos_token": 3,  # '<EOS>'
    }



    # ================== Training ==================
    warmup_steps = None  # It Depends on size of loaded database, it is set later in code.  Paper: 4_000
    step_num_limit = 100_000  # Total number of steps to train the model.
    num_epochs = 15



    # ================== Folder Structure ==================
    CFG_PATH = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CFG_PATH, ".."))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
