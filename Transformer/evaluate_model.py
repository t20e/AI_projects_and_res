"""

# Evaluate BLEU Score

"""

import os
import torch
import evaluate
from datasets import load_dataset
from tqdm import tqdm

from configs.english_german_config import English_german_config
from model.Transformer import Transformer
from model.beam_search import BeamSearch
from model.bpe_tokenizer import build_and_train_BPE_tokenizer
from model.utils import load_checkpoint
from inference import translate_sentence
from utils.data_loader import load_wmt14_en_de, get_training_corpus
from utils.safetensors_utils import load_trained_model


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# def load_chpt(cfg: English_german_config, device) -> Transformer:
#     model = Transformer(cfg)
#     chpt_path = os.path.join(cfg.MODEL_DIR, "checkpoints", cfg.checkpoint_name)

#     if not os.path.exists(chpt_path):
#         raise FileNotFoundError(f"Checkpoint not found at {chpt_path}")

#     print(f"Loading checkpoint from {chpt_path}...")
#     checkpoint = torch.load(chpt_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)
#     model.eval()
#     return model


cfg = English_german_config()

# NOTE: Config must be the same as the one used to train the this checkpoint!
cfg.checkpoint_name = "transformer_epoch_15_20_percent_ds.pt"

tokenizer = build_and_train_BPE_tokenizer(
    cfg=cfg, load_wmt14_en_de=load_wmt14_en_de, get_training_corpus=get_training_corpus
)

model = load_trained_model(cfg, device)

# Load the BLEU metric
sacreblue = evaluate.load("sacrebleu")

print("\nLoading validation split of the dataset...")
val_ds = load_dataset("wmt14", "de-en", split="validation")

# Test on a subset to save time
num_samples = 100
subset = val_ds.select(range(num_samples))

predictions = []
references = []

print(f"\nTranslating {num_samples} sentences for evaluation...")
for i in tqdm(subset):
    eng_text = i["translation"]["en"]
    target_de_text = i["translation"]["de"]

    pred_text = translate_sentence(eng_text, model, tokenizer, cfg, device)

    predictions.append(pred_text)
    references.append([target_de_text])

print("\nCalculating BLEU score...")
results = sacreblue.compute(predictions=predictions, references=references)
print(f"\n Final BLEU score: {results['score']:.2f}")
