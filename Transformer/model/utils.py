import copy
import os
import torch.nn as nn
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs.english_german_config import English_german_config
    from training import TrainModel


def clones(module, N):
    """Produce N identical layers of the EncoderLayer or DecoderLayer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def make_target_mask(tgt_tokens, pad_token: int):
    """
    Hides padding and future tokens (no-peeking) of the Target sequence, so the model doesn't cheat. This is specifically for the Masked Multi-Head Attention in the DecoderLayer.

    Args:
        tgt_tokens: The target tokens. (batch_size, seq_len)
        pad_token: The integer representation for the '<PAD>' token

    Return:
        Final target mask: (batch_size, 1, seq_len, seq_len)
    """

    # Find the padding in the target sequence.
    tgt_padding_mask = (tgt_tokens != pad_token).unsqueeze(1).unsqueeze(2)

    # Create no-peek triangle mask
    seq_len = tgt_tokens.size(1)
    no_peek_triangle = torch.tril(
        torch.ones(seq_len, seq_len, device=tgt_tokens.device)
    ).bool()

    # Combine to form the final target mask
    return tgt_padding_mask & no_peek_triangle


def load_checkpoint(trainer: TrainModel, cfg: English_german_config, device) -> int:
    """
    Load a checkpoint.

    Returns the last epoch the checkpoint was trained on.

    """

    chpt_path = os.path.join(cfg.MODEL_DIR, "checkpoints", cfg.checkpoint_name)

    if os.path.exists(chpt_path):
        print(f"Loading checkpoint: {cfg.checkpoint_name}...")
        chpt = torch.load(chpt_path, map_location=device)

        # Load all states into the trainer
        trainer.model.load_state_dict(chpt["model_state_dict"])
        trainer.optimizer.load_state_dict(chpt["optimizer_state_dict"])

        # MPS device bug issue, fix by moving optimizer state tensors to mps device
        for state in trainer.optimizer.state.values():
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if "scheduler_state_dict" in chpt:
            trainer.scheduler.load_state_dict(chpt["scheduler_state_dict"])
        if "step_counter" in chpt:
            trainer.step_counter = chpt["step_counter"]

        last_epoch = chpt["epoch"] + 1
        print(
            f"\nResuming training from Epoch {last_epoch} at Step {trainer.step_counter}..."
        )
        return last_epoch

    else:
        print(f"\n\nCheckpoint not found at {chpt_path}! Check configurations!")
        import sys
        sys.exit(1)
