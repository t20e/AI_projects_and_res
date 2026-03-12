from datasets import load_dataset_builder, load_dataset
import os


def load_wmt14_en_de():
    print("\n\nLoading dataset...")
    save_path = "./data"
    ds_name = "wmt14"
    ds_config = "de-en"

    builder = load_dataset_builder(ds_name, ds_config, cache_dir=save_path)

    dataset_path = os.path.join(save_path, ds_name, ds_config)

    # Check if already downloaded
    if os.path.exists(dataset_path):
        print(f"Loading database from: {dataset_path}...")
    else:
        print(
            f"Downloading dataset to path: {dataset_path}, downloading only 1% of dataset..."
        )

    dataset = load_dataset(ds_name, ds_config, split="train[:1%]", cache_dir=save_path)
    print("Loaded dataset!\n")
    return dataset


def get_training_corpus(dataset):
    # Yield both English and German sentences for a shared Vocabulary.
    for example in dataset:
        yield example["translation"]["en"]
        yield example["translation"]["de"]
