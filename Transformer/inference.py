"""
Implement English to German Translation Inference

Make sure the model being loaded has the same config as the one used to train it!
Run with `python inference.py`

"""

import os
import torch

from configs.english_german_config import English_german_config
from model.bpe_tokenizer import build_and_train_BPE_tokenizer
from model.Transformer import Transformer
from model.beam_search import BeamSearch
from utils.data_loader import load_wmt14_en_de, get_training_corpus
from utils.safetensors_utils import load_trained_model




def translate_sentence(english_input, model, tokenizer, cfg, device):
    src_encoded = tokenizer.encode(english_input)
    src_tensor = torch.tensor([src_encoded.ids], dtype=torch.long, device=device)

    pad_symbol = cfg.special_tokens["pad_token"]
    src_padding_mask = (src_tensor != pad_symbol).unsqueeze(-2).unsqueeze(-2).to(device)

    # Failsafe so model does not get stuck in an infinite loop generating gibberish. If you feed model a 10-word English sentence, the max_len is set to $10+50=60$. From paper: "We set the maximum output length during inference to input length + 50, but terminate early when possible"
    max_len = src_tensor.size(1) + 50

    with torch.no_grad():
        pred_seq = BeamSearch(
            model=model,
            src=src_tensor,
            src_padding_mask=src_padding_mask,
            max_len=max_len,
            pad_token_id=0,
            start_token=cfg.special_tokens["sos_token"],
            eos_token=cfg.special_tokens["eos_token"],
            beam_size=4,
            device=device,
        )

    pred_ids = pred_seq.cpu().numpy().tolist()
    translated_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
    return translated_text


if __name__ == "__main__":
    cfg = English_german_config()
    English_german_config.print()

    # NOTE: Config must be the same as the one used to train the this checkpoint!
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = build_and_train_BPE_tokenizer(
        cfg=cfg,
        load_wmt14_en_de=load_wmt14_en_de,
        get_training_corpus=get_training_corpus,
    )

    model = load_trained_model(cfg, device)

    print("\n\nUsing device:", device)
    print("\nType 'quit', 'exit', 'q' to quit!")
    while True:
        english_input = input("\nEnglish you want to translate to German: ")
        if english_input.lower() in ["quit", "exit", "q"]:
            break
        german_output = translate_sentence(english_input, model, tokenizer, cfg, device)
        print(f"\nGerman Translation: {german_output}")
