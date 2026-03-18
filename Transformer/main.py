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
from model.utils import initialize_weight
from model.training import TrainModel

from utils.data_loader import create_data_loaders

if __name__ == "__main__":

    # ====== Setup ======
    folders_to_make = [
        "./data",
        "./model/saved_models",
        "./model/checkpoints",
        "./model/pre_trained_models",
    ]
    for folder in folders_to_make:
        os.makedirs(folder, exist_ok=True)

    cfg = English_german_config()

    device = torch.device(
        "mps"
    )  # Make it so that it runs on any OS with different GPUs

    # ====== Dataset ======
    raw_ds = load_wmt14_en_de(
        save_path=cfg.DATA_DIR, perc_to_download=cfg.perc_to_download
    )

    cfg.total_sentence_pairs = len(raw_ds)

    tokenizer = build_and_train_BPE_tokenizer(
        cfg=cfg,
        dataset_iterator=get_training_corpus(raw_ds),
        perc_to_download=cfg.perc_to_download,
    )


    train_dataloader = create_data_loaders(
        raw_ds,
        tokenizer,
        batch_size=cfg.batch_size,
        pad_token=cfg.special_tokens["pad_token"],
    )

    # Set warmup to end after the first epoch
    cfg.warmup_steps = cfg.total_sentence_pairs // cfg.batch_size


    # ====== Init model ======
    model = Transformer(cfg=cfg)

    # Xavier Init
    initialize_weight(model)
    model.to(device)

    # 💡 From paper: "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]."
    #   - The pre-softmax linear transformation is the Generator, which is the last softmax + linear in the model.
    #   - So, we need to share weights between the scr_embed (Source Embedding), tgt_embed (Target Embedding), and the generator!
    shared_weights = model.src_embed[0].look_up_table.weight
    model.tgt_embed[0].look_up_table.weight = shared_weights
    model.generator.proj.weight = shared_weights

    trainer = TrainModel(cfg, model, device=device)
    trainer.train(train_dataloader)
