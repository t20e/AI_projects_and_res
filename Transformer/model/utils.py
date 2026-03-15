import copy
import torch.nn as nn
import torch

def clones(module, N):
    """Produce N identical layers of the EncoderLayer or DecoderLayer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def initialize_weight(model):
    """Initialize parameters with Xavier uniform"""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print(f"\n\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters!\n\n")


def make_target_mask(tgt_tokens, pad_token:int):
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