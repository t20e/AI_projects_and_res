import os
from datasets import load_dataset
import torch.nn as nn
import torch

from configs.english_german_config import English_german_config
from utils.load_wmt14_en_de_dataset import load_wmt14_en_de, get_training_corpus
from model.bpe_tokenizer import build_and_train_BPE_tokenizer
from model.Transformer import Transformer
from model.generator import Generator
from model.embedding import Embeddings
from model.pos_encoding import PositionalEncoding
from model.training import TrainModel
from model.utils import load_checkpoint

from utils.data_loader import create_data_loaders, filter_ds, pre_tokenize


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
    raw_ds = load_wmt14_en_de(
        save_path=cfg.DATA_DIR, perc_to_download=cfg.perc_to_download
    )

    tokenizer = build_and_train_BPE_tokenizer(
        cfg=cfg,
        dataset_iterator=get_training_corpus(raw_ds),
        perc_to_download=cfg.perc_to_download,
    )

    print("\n\nPre-tokenizing dataset...")
    tokenized_ds = pre_tokenize(raw_ds, tokenizer)

    if not cfg.is_paper_config:
        # Filter out long sentences
        print(
            f"\n\nFiltering out long sentences with a max sequence len = {cfg.max_seq_len}"
        )
        tokenized_ds = filter_ds(tokenized_ds, cfg.max_seq_len)

    cfg.total_sentence_pairs = len(tokenized_ds)

    train_dataloader = create_data_loaders(
        cfg=cfg,
        device=device,
        dataset=tokenized_ds,
        batch_size=cfg.batch_size,
        pad_token=cfg.special_tokens["pad_token"],
    )

    # Set warmup to end after the first epoch
    cfg.warmup_steps = cfg.total_sentence_pairs // cfg.batch_size

    # ====== Init model ======
    model = Transformer(cfg=cfg)

    trainer = TrainModel(cfg, model, device=device)

    start_epoch = 0

    if cfg.continue_from_chpt:
        start_epoch = load_checkpoint(trainer, cfg, device)
    else:
        # Xavier Init
        model.initialize_weight()

    # Tie Weights:💡 From paper: "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]."
    #   - The pre-softmax linear transformation is the Generator, which is the last softmax + linear in the model.
    #   - So, we need to share weights between the scr_embed (Source Embedding), tgt_embed (Target Embedding), and the generator!
    shared_weights = model.src_embed[0].look_up_table.weight
    model.tgt_embed[0].look_up_table.weight = shared_weights
    model.generator.proj.weight = shared_weights

    model.to(device)

    trainer.train(train_dataloader, start_epoch=start_epoch)
