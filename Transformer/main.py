import os
import torch


from configs.english_german_config import English_german_config
from model.bpe_tokenizer import build_and_train_BPE_tokenizer
from model.Transformer import Transformer
from model.training import TrainModel, overfit_test
from model.utils import load_checkpoint
from utils.data_loader import (
    create_data_loaders,
    filter_ds,
    load_wmt14_en_de,
    get_training_corpus,
    get_pre_tokenized_ds,
)


if __name__ == "__main__":

    cfg = English_german_config()
    English_german_config.print()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    # ====== Setup ======
    for folder in cfg.folders_to_make:
        os.makedirs(folder, exist_ok=True)


    # ====== Dataset & Tokenizer ======
    tokenizer = build_and_train_BPE_tokenizer(cfg=cfg, load_wmt14_en_de=load_wmt14_en_de, get_training_corpus=get_training_corpus)

    tokenized_ds = get_pre_tokenized_ds(cfg, load_wmt14_en_de, tokenizer)

    print(
        f"\n\nFiltering out individual sentences with a max sequence len = {cfg.max_indiv_seq_len}"
    )
    tokenized_ds = filter_ds(tokenized_ds, cfg.max_indiv_seq_len)

    cfg.total_sentence_pairs = len(tokenized_ds)
    print(f"\nTotal sentence pairs = {cfg.total_sentence_pairs}")

    train_dataloader = create_data_loaders(
        cfg=cfg,
        device=device,
        dataset=tokenized_ds,
        pad_token=cfg.special_tokens["pad_token"],
    )

    # Set warmup to end after the first epoch
    cfg.warmup_steps = len(train_dataloader)
    print(f"\nWarmup steps = {cfg.warmup_steps}")

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

    if not cfg.train_overfitted_model:
        trainer.train(train_dataloader, start_epoch=start_epoch)
    else:
        # Overfit the model on a single sentence
        overfit_test(cfg=cfg, train_dataloader=train_dataloader, tokenizer=tokenizer, trainer=trainer)