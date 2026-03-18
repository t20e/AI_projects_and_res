import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from model.training import Batch


def create_data_loaders(dataset, tokenizer: Tokenizer, batch_size, pad_token):
    """
    Args:
        pad_token: <PAD> integer ID representation.
    """

    def collate_fn(batch):
        """
        Sequence length batching: Ensure that for every batch, all sequences are padded to the length of the longest sequence in that specific batch, rather than a fixed global maximum.

        Args:
            batch: A List of samples, note its not the Batch() class
        """
        src_list, tgt_list = [], []

        for item in batch:
            # WMT14 format is { 'translation': {'en': '...', 'de': '...'}}
            en_text = item["translation"]["en"]  # english
            de_text = item["translation"]["de"]  # german

            src_list.append(torch.tensor(tokenizer.encode(en_text).ids))
            tgt_list.append(torch.tensor(tokenizer.encode(de_text).ids))

            # src_tokenized = tokenizer.encode(item["translation"]["en"]).ids
            # tgt_tokenized = tokenizer.encode(item["translation"]["de"]).ids

            # src_list.append(torch.tensor(src_tokenized))
            # tgt_list.append(torch.tensor(tgt_tokenized))

        # Pad sequences to the max length in this batch
        src_batch = pad_sequence(src_list, batch_first=True, padding_value=pad_token)
        tgt_batch = pad_sequence(tgt_list, batch_first=True, padding_value=pad_token)

        return Batch(src_batch, tgt_batch, pad_token)

    return DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
