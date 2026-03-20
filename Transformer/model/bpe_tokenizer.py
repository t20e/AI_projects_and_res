# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: AI_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Implement The BPE Tokenizer
#
# The Target and Source are tokenized representations of the sequences.
#
# **Performs:**
#
# 1. The tokenizer maps every unique word or sub-word to a unique integer ID.
# 2. Adds special tokens.
#     - &lt;SOS&gt; (Start of sentence): Tells the Decoder to start generating.
#     - &lt;EOS&gt; (End of sentence): Tells the Decoder to stop generating.
#     - &lt;UNK&gt; (Unknown)
#     - &lt;PAD&gt; (Padding): Fills the remaining space in a batch so all sequences have the same length.
#         - Example: seq_len = 4 "John ate &lt;PAD&gt; &lt;PAD&gt;" or "I am leaving &lt;PAD&gt;"
# 3. Converts texts to numerical representation.
#     1. Example: "The rabbit" → [23, 14]
#
# - "We trained on the **standard WMT 2014 English-German dataset** consisting of about 4.5 million sentence pairs. Sentences were encoded using **byte-pair encoding** [3], which has a shared source target vocabulary of about 37000 tokens. For **English-French**, we used the significantly **larger WMT 2014 English-French dataset** consisting of 36M sentences and split tokens into a 32000 **word-piece** vocabulary [38]. Sentence pairs were batched together by approximate **sequence length**. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens."
#     - For English-German they used the byte-pair encoding (**BPE**) tokenizer.
#     - For English-French they used **WordPiece** tokenizer.
#

# %% [markdown]
# **Byte-Pair Encoding Sub-Word Tokenizer**
#
# Sub-word tokenizer breaks down words, e.g., "transformer" → ["trans", "former"]
#
# Instead of building the tokenizer myself, I used the `huggingface tokenizers` library.
#

# %%
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

# %%
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict
    from ..configs import English_german_config


def build_and_train_BPE_tokenizer(
    cfg: English_german_config,
    dataset_iterator: DatasetDict = None,
):
    """
    Build and train a Universal BPE tokenizer that the paper used for the standard WMT 2014 English-German dataset.
     - It is trained on the full dataset, so it can be used with any Transformer() model with different dataset percentages.

    Args:
        cfg: Adds the correct vocab_size to it.
        dataset_iterator: None if loading tokenizer for inference, else dataset iterator.
    """

    save_path = os.path.join(cfg.MODEL_DIR, "saved_models", "tokenizer")
    file_name = f"wmt_14_shared_bpe_tokenizer_universal.json"
    full_file_path = os.path.join(save_path, file_name)

    # If a tokenizer has already been trained load it
    if os.path.exists(full_file_path):
        tokenizer = Tokenizer.from_file(full_file_path)
        print(f"\nLoading existing Universal BPE tokenizer from: ({full_file_path})...")
        return tokenizer

    print(
        f"\nNo existing tokenizer found. Training new BPE tokenizer on the full dataset...\n"
    )

    if dataset_iterator is None:
        raise ValueError("A dataset iterator must be provided to train the tokenizer for the first time!") 

    # Init BPE model
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # Set the Pre-Tokenizer. Turns "The rabbit" → ["_The", "_rabbit"]
    tokenizer.pre_tokenizer = (
        pre_tokenizers.Metaspace()  # Paper used used whitespace -> pre_tokenizers.Whitespace()
    )

    # Tell the tokenizer how to merge sub-words back into words, e.g., ["rab", "bit"] → "rabbit"
    #   and ["_The", "_rabbit"] → "The rabbit"
    tokenizer.decoder = decoders.Metaspace()  # Paper used -> decoders.BPEDecoder()

    # Note: If you don't have enough memory to store large parts of the dataset, for example a massive paragraph than use a truncation -> tokenizer.enable_truncation(max_length=...)

    # Configure the Trainer
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"],
        # Integer representations: <PAD> = 0, "<UNK>" = 1, "<SOS>" = 2,  "<EOS>" = 3
        show_progress=True,
    )

    # Train the tokenizer on the shared dataset
    tokenizer.train_from_iterator(dataset_iterator, trainer=trainer)

    # Apply the post-processor whether we loaded or trained the tokenizer to be safe.
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<SOS> $A <EOS>",  # $A is the sentence sequence tokens.
        special_tokens=[
            ("<SOS>", tokenizer.token_to_id("<SOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    # Save the tokenizer
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(full_file_path)
    print(f"\nUniversal Tokenizer saved to {full_file_path}")
    return tokenizer


# %%
def test():
    import sys
    import os

    # We need to grab load_wmt14_en_de
    project_root = os.path.abspath("..")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project root added to sys.path: {project_root}")
    print(os.getcwd())

    from configs import English_german_config
    from utils.data_loader import load_wmt14_en_de, get_training_corpus

    cfg = English_german_config()

    raw_ds = load_wmt14_en_de(
        save_path=cfg.DATA_DIR, perc_to_download=cfg.perc_to_download
    )
    tokenizer = build_and_train_BPE_tokenizer(
        cfg=cfg,
        dataset_iterator=get_training_corpus(raw_ds),
        perc_to_download=cfg.perc_to_download,
    )

    test_sentence = "The brown rabbit ate the apple."
    print(f"\n\n\nTest sentence: {test_sentence}")

    # Tokenize
    tokenized = tokenizer.encode(test_sentence)
    print(f"Tokens: {tokenized.tokens}")
    print(f"IDs: {tokenized.ids}")

    # De-Tokenize
    detokenize = tokenizer.decode(tokenized.ids)
    print(f"Detokenized: {detokenize}")

# test()

# %%
