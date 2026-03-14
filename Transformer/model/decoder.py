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
# # Implement The Decoder Block
#
# ![Main Encoder Block](../showcase_images/from_paper/decoder.png)

# %% [markdown]
# - **Decoder Stack**:
#   - Composed of a stack of $Nx=6$ identical decoding layers.
#   - The Decoder first sees its own output shifted right, then sees the output from the Encoder.
#   - Has both the layers the Encoder has, but also has a **Masked Multi-Head Attention** that prevents cheating by helping the Decoder understand the sequence it has generated so far, allowing it to only see past tokens in the output sentence.
#   - "This masking [Masked Multi-Head Attention], combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$."
#   - The Decoder also uses the **padding_mask** similar to how the [Encoder](./encoder.ipynb) uses it. 
#     - Except for its **Masked Multi-Head Attention**:
#       - It receives the Decoder's output (shift right), i.e., the **Target** as input.
#         - **tgt_no_peek_mask**: This mask hides both the &lt;padding&gt; tokens like the src_padding_mask, but it also hides the future generated tokens as well, so that model can't cheat.

# %%
import torch.nn as nn
import torch

import torch.nn as nn

try: # works when ran via main.py (package mode)
    from .residual_con_layer_norm import ResidualConnection, LayerNorm
    from .multi_head_attention import Multi_Head_Attention
    from .FeedForwardNetwork import FeedForwardNetwork
    from .model_utils import clones, make_target_mask
except ImportError:
    # Works when running from inside Jupyter Notebook
    from residual_con_layer_norm import ResidualConnection, LayerNorm
    from multi_head_attention import Multi_Head_Attention
    from FeedForwardNetwork import FeedForwardNetwork
    from model_utils import clones, make_target_mask


# %%
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        """
        A single layer of the Decoder Stack.
        """
        super().__init__()
        self.d_model = d_model

        # Masked Attention
        self.masked_mha = Multi_Head_Attention(d_model, h, dropout)
        # (Cross attention connection between the Encoder and Decoder)
        self.cross_mha = Multi_Head_Attention(d_model, h, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        # Three residual connections: One for the Masked Multi-Head Attention, the other for the normal Multi-Head Attention, and the last for the Feed Forward
        self.residual_1_masked_mha = ResidualConnection(d_model, dropout)
        self.residual_2_mha = ResidualConnection(d_model, dropout)
        self.residual_3_ffn = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_padding_mask, tgt_no_peek_mask):
        """
        Args:
            x: Target sequence (from Decoder)
            encoder_output: The Encoder's final output
            src_padding_mask: The padding for source sequence.
            tgt_no_peek_mask: The mask for the Masked Multi-Head Attention target sequence.
        """

        # Masked Attention Sublayer
        x = self.residual_1_masked_mha(
            x, lambda x: self.masked_mha(x, x, x, tgt_no_peek_mask)
        )

        # Normal Attention sublayer connection between Encoder-Decoder
        x = self.residual_2_mha(
            x, lambda x: self.cross_mha(x, encoder_output, encoder_output, src_padding_mask)
        )

        # Feed-Forward Sublayer
        x = self.residual_3_ffn(x, self.ffn)
        return x


# %%
class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N):
        """
        The full Decoder that is a Nx stack of DecoderLayer()

        Args:
            layer: A single DecoderLayer().
            N: The stack size of DecoderLayer()
        """
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, encoder_output, src_padding_mask, tgt_no_peek_mask):
        """
        Arg:
            encoder_output: The Encoder's final output
            src_padding_mask: The padding for the Source sequence. Example with 3 paddings: "The brown rabbit ate the apple. <Padding> <Padding> <Padding>". This mask hides the paddings from the Encoder, so it only sees non-padding tokens.
            tgt_no_peek_mask: The mask for the Target sequence, hides both the <padding> and the future tokens.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_padding_mask, tgt_no_peek_mask)
        return self.norm(x)


# %%
def test():
    print("\n\nTesting the Decoder...")
    d_model, h, d_ff, N, dropout = 512, 8, 2048, 6, 0.1
    batch_size, seq_len = 2, 10

    base_dec_layer = DecoderLayer(d_model, h, d_ff, dropout)
    decoder = Decoder(base_dec_layer, N)

    # Inputs
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)

    # Masks

    # Target sequence Masked multi-head attention input
    tgt_tokens = torch.ones(batch_size, seq_len)
    tgt_tokens[:, -3:] = 0  # last 3 are padding
    tgt_no_peek_mask = make_target_mask(tgt_tokens, pad_token=0)

    # Source tokens for the cross-attention MHA
    src_tokens = torch.ones(batch_size, seq_len)
    src_tokens[:, -2:] = 0  # has 2 padding tokens
    src_padding_mask = (src_tokens != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)

    output = decoder(dummy_input, encoder_output, src_padding_mask, tgt_no_peek_mask)
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape mismatch!"

    # Verify Gradient flow
    output.mean().backward()
    # Pick a parameter at the very beginning of the network (e.g., the first masked attention sublayer weights)
    start_param = decoder.layers[0].masked_mha.w_q.weight
    assert start_param.grad is not None, "Gradient did not reach the first layer!"
    assert torch.any(start_param.grad != 0), "Gradients are zero (Vanishing Gradient)!"
    decoder.zero_grad()
    print("Test Passed!")


test()

# %%

# %%
