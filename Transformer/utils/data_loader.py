import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from model.training import Batch
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs.english_german_config import English_german_config


def pre_tokenize(ds, tokenizer):
    """
    Pre-Tokenize the dataset before training starts, and not during each batching.
    """

    def _process_example(e):
        # en_encoded = tokenizer.encode(e["translation"]["en"]).ids
        # de_encoded = tokenizer.encode(e["translation"]["de"]).ids

        en_encoded = [tokenizer.encode(item["en"]).ids for item in e["translation"]]
        de_encoded = [tokenizer.encode(item["de"]).ids for item in e["translation"]]

        return {"src_ids": en_encoded, "tgt_ids": de_encoded}

    return ds.map(_process_example, batched=True)


def filter_ds(tokenized_ds, max_seq_len: int):
    """
    Apply a sequence limit if needed.

    Args:
        tokenized_ds: A dataset that has called pre-tokenized().
        max_seq_len: Max length each sequence should be.
    """
    return tokenized_ds.filter(
        lambda x: len(x["src_ids"]) <= max_seq_len and len(x["tgt_ids"]) <= max_seq_len
    )


def collate_fn(batch, pad_token):
    """
    Sequence length batching: Ensure that for every batch, all sequences are padded to the length of the longest sequence in that specific batch, rather than a fixed global maximum.

    Args:
        batch: A List of samples, note its not the Batch() class
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


def create_data_loaders(
    cfg: English_german_config, device, dataset, batch_size, pad_token
):
    """
    Args:
        pad_token: <PAD> integer ID representation.
    """
    bound_collate = partial(collate_fn, pad_token=pad_token)

    if device.type == "mps":
        pin_memory = False  # MPS (Metal) device uses Unified Memory,
    elif device.type == "cuda":
        pin_memory = True
    else:
        pin_memory = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=bound_collate,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
