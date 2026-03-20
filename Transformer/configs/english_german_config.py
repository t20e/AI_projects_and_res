import os


class English_german_config:
    """
    🚨🚨🚨 Smaller config than the paper to train a decent model, that will not take days to train on GPU. Check Paper_english_german_config() class for the default paper config for its base model.
    """
    is_paper_config = False

    d_model = 512  # All sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512



    # ================== Encoder & Decoder ==================
    H = 8  # How many H heads in the Multi-Head Attention
    dropout = 0.1
    d_ff = 2048
    N = 6 # Num of stacks of encoders and decoders



    # ================== Dataset ==================
    pos_seq_len = 5000  # Positional Encoding sequence length
    max_indiv_seq_len = 128 # Applies to individual sentences
    max_batch_seq_tokens = 6_000 # #TODO Paper: 25_000. Applies sequence limit to an entire batch of sequences.
    vocab_size = 37_000 # Limit to let tokenizer trainer know when to stop merging sub-words.
    tokenizer = "BPE"
    dataset_name = "WMT_2014_English_German"

    # Percentage of database to download
    perc_to_download: int = 1 #TODO test with 50

    total_sentence_pairs = None  # The total number of English-German sentence pairs, will be set later in code.
    special_tokens = {  # The integer representations
        "pad_token": 0,  # '<PAD>'
        "unk_token": 1,  # '<UNK>'
        "sos_token": 2,  # '<SOS>'
        "eos_token": 3,  # '<EOS>'
    }



    # ================== Training ==================
    num_workers = 0 # 🚨 NOTE: For NVIDIA GPU Golden Rule: num_worker = 4 * num_GPU | On Mac Silicone even though I have a 32 core GPU, but it is still only one GPU, best to num_workers = 0.
    warmup_steps = None  # It Depends on size of loaded database, it is set later in code.  Paper: 4_000

    # Either num_epochs or step_num_limit is reached first, and training stops.
    step_num_limit = 100_000  # Total number of steps to train the model.
    num_epochs = 1 # Paper trained til step_num_limit: "We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days)."


    continue_from_chpt:bool = False # Continue training the model from a check point

    # Checkpoint filename to load! 🚨 NOTE: The same config that was used to train the checkpoint must be used to load it!
    checkpoint_name:str = "" 

    # Train a overfitted model on the dataset, and see how it performs on one single sentence. Also change dataset percent to 1!
    train_overfitted_model:bool = True



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


    @classmethod
    def print(cfg):
        print(f"{'='*20} Config {'='*20}")

        print(f"\n[Model Architecture]")
        print(f"  - d_model = {cfg.d_model}")
        print(f"  - N Stacks = {cfg.N}")
        print(f"  - Heads (H) = {cfg.H}")
        print(f"  - d_ff = {cfg.d_ff}")
        print(f"  - Dropout = {cfg.dropout}")

        print(f"\n[Dataset & Tokenizer]")
        print(f"  - dataset_name = {cfg.dataset_name}%")
        print(f"  - perc_to_download = {cfg.perc_to_download}")
        print(f"  - Tokenizer = {cfg.tokenizer}")
        print(f"  - pos_seq_len = {cfg.pos_seq_len}")
        print(f"  - total_sentence_pairs = {cfg.total_sentence_pairs}")
        print(f"  - vocab_size = {cfg.vocab_size}")
        print(f"  - Max Seq Length (Individual) = {cfg.max_indiv_seq_len}")
        print(f"  - Max Seq Length (Batch) = {cfg.max_batch_seq_tokens}")

        print(f"\n[Training]")
        print(f"  - num_workers = {cfg.num_workers}")
        print(f"  - warmup_steps = {cfg.warmup_steps}")
        print(f"  - step_num_limit = {cfg.step_num_limit}")
        print(f"  - num_epochs = {cfg.num_epochs}")
        print(f"  - continue_from_chpt = {cfg.continue_from_chpt}")
        print(f"  - checkpoint_name = {cfg.checkpoint_name}")
        print(f"  - train_overfitted_model = {cfg.train_overfitted_model}")

        print("\n\n")