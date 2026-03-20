import torch
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.training import TokenBatchSampler, Batch
from functools import partial
from datasets import load_from_disk, load_dataset, DatasetDict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs.english_german_config import English_german_config
    from tokenizers import Tokenizer

def load_wmt14_en_de(save_path: str, perc_to_download: int = 1) -> DatasetDict:
    """
    Load the WMT14 English-German dataset

    Args:
        perc_to_download: How much of the dataset to download.
        save_path: Full path to the ./data directory in project.
    """

    print(
        f"\nAttempting to get ({perc_to_download}%) of the WMT English-German dataset..."
    )
    ds_name = "wmt14"
    ds_config = "de-en"

    dataset = load_dataset(
        ds_name, ds_config, split=f"train[:{perc_to_download}%]", cache_dir=save_path
    )
    print("\nSuccessfully loaded raw dataset!\n")
    return dataset


def get_training_corpus(dataset):
    # Yield both English and German sentences for a shared Vocabulary.
    for example in dataset:
        yield example["translation"]["en"]
        yield example["translation"]["de"]


def get_pre_tokenized_ds(cfg, load_wmt14_en_de, tokenizer: Tokenizer):
    """Retrieves a Pre-Tokenized dataset

    Args:
        cfg: Configs
        load_wmt14_en_de: Function to load the dataset
        tokenizer: Tokenizer
    """

    # Create storage directory
    tokenized_path = os.path.join(
        cfg.DATA_DIR, f"tokenized_{cfg.dataset_name}_{cfg.perc_to_download}_percent_ds"
    )

    # Load pre-tokenized dataset if it exists
    if os.path.exists(tokenized_path):
        print(f"Loading existing pre-tokenized dataset from {tokenized_path}...")
        return load_from_disk(tokenized_path)


    print(f"Loading {cfg.perc_to_download}% of the dataset from disk...")
    raw_ds = load_wmt14_en_de(
        save_path=cfg.DATA_DIR, perc_to_download=cfg.perc_to_download
    )
    tokenized_ds = pre_tokenize_ds(cfg, raw_ds, tokenizer)
    del raw_ds # we only need to tokenized ds.
    return tokenized_ds


def pre_tokenize_ds(cfg: English_german_config, ds, tokenizer):
    """
    Pre-Tokenizes the raw sentences into lists of integers token IDs. This is done once before training.
    Saves the processed dataset to disk and to be loaded for later training runs with the same dataset percentage size.
    """
    print("\nPre-Tokenizing dataset...")

    # Create storage directory
    tokenized_path = os.path.join(
        cfg.DATA_DIR,
        f"tokenized_{cfg.dataset_name}_dataset_{cfg.perc_to_download}_percent_ds",
    )

    def _process_example(e):
        en_encoded = [tokenizer.encode(item["en"]).ids for item in e["translation"]]
        de_encoded = [tokenizer.encode(item["de"]).ids for item in e["translation"]]
        return {"src_ids": en_encoded, "tgt_ids": de_encoded}

    # batched=True uses Tokenizer's rust backend, which speeds up data prep
    tokenized_ds = ds.map(_process_example, batched=True)

    print(f"Saving pre-tokenized dataset to {tokenized_path}...")
    tokenized_ds.save_to_disk(tokenized_path)

    return tokenized_ds


def filter_ds(tokenized_ds, max_indiv_seq_len: int):
    """
    Applies a max sequence limit to **individual** sequences. I.e., Removes sequence outliers from the dataset. This prevents something like a single 500-word paragraph from causing a massive memory spike.

    Args:
        tokenized_ds: A dataset containing 'src_ids' and 'tgt_ids' keys.
        max_indiv_seq_len: The maximum number of tokens allowed in a sentence.
    """
    return tokenized_ds.filter(
        lambda x: len(x["src_ids"]) <= max_indiv_seq_len
        and len(x["tgt_ids"]) <= max_indiv_seq_len
    )


def collate_fn(batch, pad_token=0):
    """
    Transforms a list of raw dataset dictionaries into a single, padded Batch.

    Take raw list of integers and converts into torch Tensors.
        1. Finds the longest sequence in a specific batch.
        2. Pads all the shorter sequences in that batch to the length of the longest sequence in that batch with <PAD> tokens.
        3. Stacks them into a single, rectangular 2D tensor.

    Args:
        batch: A List of dictionaries, each containing 'src_ids' and 'tgt_ids' keys, fetched by the DatLoader.
        pad_token: The integer ID representation for the '<PAD>' token.
    """
    src_list, tgt_list = [], []

    for item in batch:
        # Grab the pre-tokenized ids
        src_list.append(torch.tensor(item["src_ids"]))
        tgt_list.append(torch.tensor(item["tgt_ids"]))

    # Pad sequences to the max length in this batch
    src_batch = pad_sequence(src_list, batch_first=True, padding_value=pad_token)
    tgt_batch = pad_sequence(tgt_list, batch_first=True, padding_value=pad_token)

    return Batch(src_batch, tgt_batch, pad_token)


def create_data_loaders(cfg: English_german_config, device, dataset, pad_token=0):
    """
    Assembles the sampler, collate_fn function and dataset into a PyTorch DataLoader.

    Args:
        pad_token: <PAD> integer ID representation.
    """
    bound_collate = partial(collate_fn, pad_token=pad_token)

    pin_memory = (
        True if device.type == "cuda" else False
    )  # Mac MPS (Metal) device uses Unified Memory, so pin_memory is False.

    batch_sampler = TokenBatchSampler(
        dataset=dataset, max_tokens=cfg.max_batch_seq_tokens, shuffle=True
    )

    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=bound_collate,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
