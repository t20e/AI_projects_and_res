import os


class English_german_config:
    """
    🚨 Smaller config than the paper to train a decent model, that will not take days to train on GPU. Check Paper_english_german_config() class for the default paper config for its base model.
    """
    is_paper_config = False

    d_model = 512  # All sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512



    # ================== Encoder & Decoder ==================
    H = 8  # How many H heads in the Multi-Head Attention
    dropout = 0.1
    d_ff = 2048
    N = 6 # Num of stacks of encoders and decoders



    # ================== Dataset ==================
    max_seq_len = 128 # When training smaller model, not one from paper, to train faster we limit the size of each sequence.
    pos_seq_len = 5000  # Positional Encoding sequence length
    batch_size = 128
    vocab_size = (
        37000  # Limit to let tokenizer trainer know when to stop merging sub-words.
    )
    tokenizer = "BPE"
    dataset_name = "WMT 2014 English-German"

    # Percentage of database to download
    perc_to_download: int = 50

    total_sentence_pairs = None  # The total number of English-German sentence pairs, will be set later in code.
    special_tokens = {  # The integer representations
        "pad_token": 0,  # '<PAD>'
        "unk_token": 1,  # '<UNK>'
        "sos_token": 2,  # '<SOS>'
        "eos_token": 3,  # '<EOS>'
    }



    # ================== Training ==================
    num_workers = 4
    warmup_steps = None  # It Depends on size of loaded database, it is set later in code.  Paper: 4_000
    step_num_limit = 100_000  # Total number of steps to train the model.
    num_epochs = 15
    continue_from_chpt:bool = False # Continue training the model from a check point

    # Checkpoint filename to load! 🚨 NOTE: The same config that was used to train the checkpoint must be used to load it!
    checkpoint_name:str = "" 



    # ================== Folder Structure ==================
    CFG_PATH = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CFG_PATH, ".."))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
    folders_to_make = [
        "./data",
        "./model/saved_models",
        "./model/checkpoints",
    ]
