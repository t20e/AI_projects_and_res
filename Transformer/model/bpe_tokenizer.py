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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Implement The BPE Tokenizer
#
# The Target and Source are tokenized representations of the sequences. 
#
# **Performs:**
# 1. The tokenizer maps every unique word or sub-word to a unique integer ID.
# 2. Adds special tokens.
#    - &lt;SOS&gt; (Stat of sentence): Tells the Decoder to start generating.
#    - &lt;EOS&gt; (End of sentence): Tells the Decoder to stop generating.
#    - &lt;unk&gt; (Unknown)
#    - &lt;Pad&gt; (Padding): Fills the remaining space in a batch so all sequences have the same length.
#        - Example: seq_len = 4 "John ate &lt;Pad&gt; &lt;Pad&gt;" or "I am leaving &lt;Pad&gt;"
# 3. Converts texts to to numerical representation.
#    1. Example: "The rabbit" → [23, 14]
#
# - "We trained on the **standard WMT 2014 English-German dataset** consisting of about 4.5 million sentence pairs. Sentences were encoded using **byte-pair encoding** [3], which has a shared source target vocabulary of about 37000 tokens. For **English-French**, we used the significantly **larger WMT 2014 English-French dataset** consisting of 36M sentences and split tokens into a 32000 **word-piece** vocabulary [38]. Sentence pairs were batched together by approximate **sequence length**. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens."
#     - For English-German they used the byte-pair encoding (**BPE**) tokenizer.
#     - For English-French they used **WordPiece** tokenizer.

# %% [markdown]
# **Byte-Pair Encoding Sub-Word Tokenizer**
#
# Sub-word tokenizer breaks down words, e.g., "transformer" → [""trans", "former"]
#
# Instead of building the tokenizer my self I use a `tokenizers` library.

# %%
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os


# %%
def build_and_train_BPE_tokenizer(dataset_iterator, vocab_size=37_000):
    """
    Build and train the BPE tokenizer that the paper used for the standard WMT 2014 English-German dataset.

    Args:
        file_path:
        vocab_size: Vocabulary token size.
    """
    print("Initializing BPE tokenizer...")

    # Init BPE model
    print("Initializing BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Set the Pre-Tokenizer. Note: there are better modern tokenizers but we're sticking with the paper.
    tokenizer.pre_tokenizer = Whitespace()

    # Configure the Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "sos", "eos"],
        show_progress=True,
    )

    # Train the tokenizer on the shared dataset
    tokenizer.train_from_iterator(dataset_iterator, trainer=trainer)

    # Save the tokenizer
    os.makedirs("./model/saved_models", exist_ok=True)
    tokenizer.save("./model/saved_models/wmt_14_shared_bpe.json")
    print(f"Tokenizer saved to wmt_14_shared_bpe.json")

# %%
