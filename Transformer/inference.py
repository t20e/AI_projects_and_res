"""
Implement English to German Inference

Run with python3

"""

import os
import torch

from configs.english_german_config import English_german_config
from model.bpe_tokenizer import build_and_train_BPE_tokenizer
from model.Transformer import Transformer
from model.beam_search import BeamSearch



if __name__ == "__main__":

    cfg = English_german_config()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    def load_trained_model(cfg: English_german_config, device) -> Transformer:
        """Load a checkpoint, change config for which checkpoint"""

        model = Transformer(cfg=cfg)

        # Tie Weights:💡 From paper: "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]."
        #   - The pre-softmax linear transformation is the Generator, which is the last softmax + linear in the model.
        #   - So, we need to share weights between the scr_embed (Source Embedding), tgt_embed (Target Embedding), and the generator!
        shared_weights = model.src_embed[0].look_up_table.weight
        model.tgt_embed[0].look_up_table.weight = shared_weights
        model.generator.proj.weight = shared_weights

        chpt_path = os.path.join(cfg.MODEL_DIR, "checkpoints", cfg.checkpoint_name)

        if not os.path.exists(chpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {chpt_path}")

        print(f"\nLoading weights from ({chpt_path})...")
        checkpoint = torch.load(chpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    tokenizer = build_and_train_BPE_tokenizer(
        cfg=cfg, perc_to_download=cfg.perc_to_download, dataset_iterator=None
    )

    model = load_trained_model(cfg, device)

    def translate_sentence(english_input, model, tokenizer, cfg, device):
        src_encoded = tokenizer.encode(english_input)
        src_tensor = torch.tensor([src_encoded.ids], dtype=torch.long, device=device)

        pad_symbol = cfg.special_tokens["pad_token"]
        src_padding_mask = (
            (src_tensor != pad_symbol).unsqueeze(-2).unsqueeze(-2).to(device)
        )

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

    print("\n\n\nType 'quit', 'exit', 'q' to quit!")
    while True:
        english_input = input("\nEnglish you want to translate to german: ")
        if english_input.lower() in ["quit", "exit", "q"]:
            break
        german_output = translate_sentence(english_input, model, tokenizer, cfg, device)
        print(f"German: {german_output}")
