import os


class Paper_english_german_config:
    """
    🚨 This config is what the paper used to train its base model. It will likely take days to train on GPU, maybe more deepening on your hardware.
    ‼️ Note: I did not train on this config, so I can't verify if it will work or break!
    """

    is_paper_config = True


    device = (
        "mps"  # if using NVIDIA GPU set to "cuda" or if no GPU available set to "cpu".
    )

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
    pos_seq_len = 5000  # Positional Encoding sequence length
    batch_size = 64  # ‼️ Not sure what the paper used. Paper: "Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens."
    vocab_size = (
        37000  # Limit to let tokenizer trainer know when to stop merging sub-words.
    )

    tokenizer = "BPE"
    dataset_name = "WMT 2014 English-German"
    perc_to_download: int = 100  # percentage of database to download
    total_sentence_pairs = None  # The total number of English-German sentence pairs, will be set later in code.



    # ================== Training ==================
    warmup_steps = 4_000
    step_num_limit = 100_000  # Total number of steps to train the model.
    num_epochs = 16
    continue_from_chpt = False # Continue training the model from a check point
    checkpoint_name = "" # Checkpoint filename


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