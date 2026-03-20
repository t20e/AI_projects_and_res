import os
from datasets import load_dataset
import torch.nn as nn
import torch


from configs.english_german_config import English_german_config
from model.bpe_tokenizer import build_and_train_BPE_tokenizer
from model.Transformer import Transformer
from model.generator import Generator
from model.embedding import Embeddings
from model.pos_encoding import PositionalEncoding
from model.training import TrainModel
from model.utils import load_checkpoint
from utils.data_loader import (
    create_data_loaders,
    filter_ds,
    pre_tokenize,
    load_wmt14_en_de,
    get_training_corpus,
    get_pre_tokenized_ds,
)


if __name__ == "__main__":

    cfg = English_german_config()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ====== Setup ======
    for folder in cfg.folders_to_make:
        os.makedirs(folder, exist_ok=True)

    # ====== Dataset ======
    # Check if we already downloaded the dataset and pre-tokenized it
    tokenized_ds = get_pre_tokenized_ds(cfg)

    if tokenized_ds is None:  # Need to download the dataset and pre-tokenize it
        raw_ds = load_wmt14_en_de(
            save_path=cfg.DATA_DIR, perc_to_download=cfg.perc_to_download
        )
        tokenizer = build_and_train_BPE_tokenizer(
            cfg=cfg,
            dataset_iterator=get_training_corpus(raw_ds),
            perc_to_download=cfg.perc_to_download,
        )

        print("\n\nPre-Tokenizing dataset...")
        tokenized_ds = pre_tokenize(cfg, raw_ds, tokenizer)
    else:
        tokenizer = build_and_train_BPE_tokenizer(
            cfg=cfg,
            dataset_iterator=None,
            perc_to_download=cfg.perc_to_download,
        )

    print(
        f"\n\nFiltering out individual sentences with a max sequence len = {cfg.max_indiv_seq_len}"
    )
    tokenized_ds = filter_ds(tokenized_ds, cfg.max_indiv_seq_len)

    cfg.total_sentence_pairs = len(tokenized_ds)

    train_dataloader = create_data_loaders(
        cfg=cfg,
        device=device,
        dataset=tokenized_ds,
        pad_token=cfg.special_tokens["pad_token"],
    )

    # Set warmup to end after the first epoch
    cfg.warmup_steps = len(train_dataloader)

    # ====== Init model ======
    model = Transformer(cfg=cfg)
    model.to(device)

    trainer = TrainModel(cfg, model, device=device)

    start_epoch = 0

    if cfg.continue_from_chpt:
        start_epoch = load_checkpoint(trainer, cfg, device)
    else:
        # Xavier Init
        model.initialize_weights()


    trainer.train(train_dataloader, start_epoch=start_epoch)

 