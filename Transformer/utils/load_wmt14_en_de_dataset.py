from datasets import load_dataset
import os
from typing import TYPE_CHECKING

from datasets import DatasetDict


def load_wmt14_en_de(save_path: str, perc_to_download: int = 1)-> DatasetDict:
    """
    Load the WMT14 English-German dataset

    Args:
        perc_to_download: How much of the dataset to download.
        save_path: Full path to the ./data directory in project.
    """

    print(f"\n\nAttempting to get ({perc_to_download}%) of the WMT English-German dataset...")
    ds_name = "wmt14"
    ds_config = "de-en"


    dataset = load_dataset(
        ds_name, ds_config, split=f"train[:{perc_to_download}%]", cache_dir=save_path
    )
    print("\nSuccessfully loaded dataset!\n")
    return dataset


def get_training_corpus(dataset):
    # Yield both English and German sentences for a shared Vocabulary.
    for example in dataset:
        yield example["translation"]["en"]
        yield example["translation"]["de"]
