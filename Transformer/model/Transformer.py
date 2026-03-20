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
#     display_name: AI_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Implement The Full Transformer Model
#
# ![Transformer model](../showcase_images/from_paper/main.png)

# %% [markdown]
# - Tie Weights:💡 From paper: "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]."

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

if TYPE_CHECKING: # for type checks example cfg: English_german_config
    from configs.english_german_config import English_german_config

class Transformer(nn.Module):
    def __init__(
        self,
        cfg: English_german_config
    ):
        super().__init__()
        """
        The Transformer Model

        Args:
            cfg: Configurations
        """
        self.cfg = cfg
        self.decoder = Decoder(
            DecoderLayer(cfg.d_model, cfg.H, cfg.d_ff, cfg.dropout), cfg.N
        )
        self.encoder = Encoder(
            EncoderLayer(cfg.d_model, cfg.H, cfg.d_ff, cfg.dropout), cfg.N
        )

        # The embedded Source sequence
        self.src_embed = nn.Sequential(
            Embeddings(d_model=cfg.d_model, vocab_size=cfg.vocab_size),
            PositionalEncoding(
                d_model=cfg.d_model, pos_seq_len=cfg.pos_seq_len, dropout=cfg.dropout
            ),
        )

        # The embedded Target sequence
        self.tgt_embed = nn.Sequential(
            Embeddings(d_model=cfg.d_model, vocab_size=cfg.vocab_size),
            PositionalEncoding(
                d_model=cfg.d_model, pos_seq_len=cfg.pos_seq_len, dropout=cfg.dropout
            ),
        )
        self.generator = Generator(self.cfg.d_model, self.cfg.vocab_size)

        # Tie weights after all other layers have been initialized
        self.tie_weights()

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

        # Run last Linear + Softmax layers
        return self.generator(decoder_out)

    def initialize_weights(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        print(f"\n\nModel initialized with {sum(p.numel() for p in self.parameters()):,} parameters!\n\n")

    def tie_weights(self):
        """
        Tie Weights:💡 From paper: "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]."
           - The pre-softmax linear transformation is the Generator, which is the last softmax + linear in the model.
           - So, we need to share weights between the src_embed (Source Embedding), tgt_embed (Target Embedding), and the generator!
        """
        shared_weights = self.src_embed[0].look_up_table.weight
        self.tgt_embed[0].look_up_table.weight = shared_weights
        self.generator.proj.weight = shared_weights


# %%
