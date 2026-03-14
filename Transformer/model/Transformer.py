# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Implement The Full Transformer Model

# %%
import torch.nn as nn

try: # works when ran via main.py (package mode)
    from .decoder import Decoder, DecoderLayer
    from .encoder import Encoder, EncoderLayer
    from .generator import Generator
    from .pos_encoding import PositionalEncoding
    from .embedding import Embeddings
except ImportError:
    # Works when running from inside Jupyter Notebook
    from decoder import Decoder, DecoderLayer
    from encoder import Encoder, EncoderLayer
    from generator import Generator
    from pos_encoding import PositionalEncoding
    from embedding import Embeddings

# %%
from typing import TYPE_CHECKING

if TYPE_CHECKING: # for type checks example cfg: english_german_config
    from configs.english_german_config import english_german_config

class Transformer(nn.Module):
    def __init__(
        self,
        cfg: english_german_config
    ):
        super().__init__()
        """
        The Transformer Model

        Args:
            cfg: Configurations
        """
        self.cfg = cfg
        self.decoder = Decoder(
            DecoderLayer(cfg.d_model, cfg.h, cfg.d_ff, cfg.dropout), cfg.N
        )
        self.encoder = Encoder(
            EncoderLayer(cfg.d_model, cfg.h, cfg.d_ff, cfg.dropout), cfg.N
        )

        # The embedded Source sequence
        self.src_embed = nn.Sequential(
            Embeddings(d_model=cfg.d_model, vocab_size_dim=cfg.vocab_size_dim),
            PositionalEncoding(
                d_model=cfg.d_model, max_seq_len=cfg.max_seq_len, dropout=cfg.dropout
            ),
        )

        # The embedded Target sequence
        self.tgt_embed = nn.Sequential(
            Embeddings(d_model=cfg.d_model, vocab_size_dim=cfg.vocab_size_dim),
            PositionalEncoding(
                d_model=cfg.d_model, max_seq_len=cfg.max_seq_len, dropout=cfg.dropout
            ),
        )
        self.generator = Generator(self.cfg.d_model, self.cfg.vocab_size_dim)

    def encode(self, src, src_padding_mask):
        """
        Args:
            src: The source vocab.
            src_padding_mask: The <padding> masking for the source.
        """
        return self.encoder(self.src_embed(src), src_padding_mask)

    def decode(self, x, encoder_output, src_padding_mask, tgt_no_peek_mask):
        """
        Args:
            x: Target sequence.
            src: The source vocab.
            src_padding_mask: The <padding> masking for the source.
        """
        return self.decoder(
            self.tgt_embed(x), encoder_output, src_padding_mask, tgt_no_peek_mask
        )

    def forward(self, src, tgt, src_padding_mask, tgt_no_peek_mask):
        # Run the encoder
        encoder_out = self.encode(src, src_padding_mask)
        # Run Decoder
        decoder_out = self.decode(tgt, encoder_out, src_padding_mask, tgt_no_peek_mask)

        # Project to vocabulary probabilities
        return self.generator(decoder_out)

# %%
