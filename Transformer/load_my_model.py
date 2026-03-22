from huggingface_hub import snapshot_download
import os
from configs.english_german_config import English_german_config

def download_my_pretrained_model(cfg: English_german_config):
    """Model is hosted on huggingface.co"""

    print(f"\n\n⭐️ Downloading my pretrained model from huggingface.co (~700MB)...\n")
    repo_id = "t20e/Transformer"

    # Download model weights
    checkpoint_path = os.path.join(cfg.MODEL_DIR, "checkpoints")
    print(f"Downloading the model weights to {checkpoint_path}/...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=checkpoint_path,
        repo_type="model",
        allow_patterns=["transformer_epoch_15_20_percent_ds.pt"],
    )

    # Download the tokenizer
    tokenizer_path = os.path.join(cfg.MODEL_DIR, "saved_models", "tokenizer")
    print(f"Downloading the model's tokenizer to {tokenizer_path}/...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=tokenizer_path,
        allow_patterns=["wmt_14_shared_bpe_tokenizer_universal.json"],
    )

if __name__ == "__main__":
    cfg = English_german_config()
    for folder in cfg.folders_to_make:
        os.makedirs(folder, exist_ok=True)
    download_my_pretrained_model(cfg=cfg)