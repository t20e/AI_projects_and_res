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
    pre_tokenize_ds,
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
    tokenizer_path = os.path.join(
        cfg.MODEL_DIR,
        "saved_models",
        "tokenizer",
        "wmt_14_shared_bpe_tokenizer_universal.json",
    )

    if not os.path.exists(tokenizer_path):
        print(
            f"Universal BPE Tokenizer not yet trained. Training it now on 100% of the dataset..."
        )
        full_raw_ds = load_wmt14_en_de(save_path=cfg.DATA_DIR, perc_to_download=100)
        tokenizer = build_and_train_BPE_tokenizer(
            cfg=cfg,
            dataset_iterator=get_training_corpus(full_raw_ds),
        )
        del full_raw_ds  # delete the dataset to save memory, even if we are still using 100% of it for training, it is just loaded from disk, very fast!
    else:
        print(f"Loading existing Universal BPE Tokenizer from {tokenizer_path}...")
        tokenizer = build_and_train_BPE_tokenizer(cfg, dataset_iterator=None)

    # Check if we already downloaded the dataset and pre-tokenized it
    tokenized_ds = get_pre_tokenized_ds(cfg)

    if tokenized_ds is None:  # Need to download the dataset and pre-tokenize it
        print(
            f"Loading {cfg.perc_to_download}% of the dataset from disk..."
        )
        raw_ds = load_wmt14_en_de(
            save_path=cfg.DATA_DIR, perc_to_download=cfg.perc_to_download
        )
        print("\n\nPre-Tokenizing dataset...")
        tokenized_ds = pre_tokenize(cfg, raw_ds, tokenizer)
        del raw_ds

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
        print("\n\n\nTraining an Overfitted Model on a single sentence...")
        single_batch = next(iter(train_dataloader))

        # Keep only one sentence
        single_batch.src = single_batch.src[0:1]
        single_batch.tgt = single_batch.tgt[0:1]
        single_batch.tgt_y = single_batch.tgt_y[0:1]
        single_batch.src_padding_mask = single_batch.src_padding_mask[0:1]
        single_batch.tgt_no_peek_mask = single_batch.tgt_no_peek_mask[0:1]
        single_batch.non_tokens = (
            single_batch.tgt_y != cfg.special_tokens["pad_token"]
        ).data.sum()

        fake_dataloader = [
            single_batch
        ] * 2000  # Multiple by 2000 to make a massive 1-epoch dataset, so it only runs for 1 epoch.

        # Decode the first sentence in the batch
        src_ids = single_batch.src[0].cpu().tolist()
        tgt_ids = single_batch.tgt[0].cpu().tolist()

        # Sentences
        en_sentence = tokenizer.decode(src_ids, skip_special_tokens=True)
        de_sentence = tokenizer.decode(tgt_ids, skip_special_tokens=True)

        print(f"\n[Source English Sentence]: {en_sentence}")
        print(f"\n[Target German Sentence]: {de_sentence}")
        print(
            "\n ⚠️ Copy the English sentence above to use in inference.py later. The German sentence should be the one generated for it!\n"
        )

        # Override the optimizer and scheduler
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=5e-4)
        trainer.compute_loss.opt = trainer.optimizer

        # Turn off Dropout
        for module in trainer.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

        # Turn off label smoothing
        trainer.criterion = nn.CrossEntropyLoss(
            ignore_index=cfg.special_tokens["pad_token"],
            reduction="sum",
        )
        trainer.compute_loss.criterion = trainer.criterion

        # Dummy scheduler, so when its called doesn't nothing and code continues
        class DummyScheduler:
            def step(self):
                pass

            def state_dict(self):
                return {}

        trainer.scheduler = DummyScheduler()

        trainer.cfg.num_epochs = 1
        trainer.cfg.step_num_limit = 2000
        trainer.cfg.perc_to_download = "OVERFIT_TEST"

        trainer.train(train_dataloader=fake_dataloader, start_epoch=0)

        print("\n\nOverfitted model saved! Test it in inference!\n")
