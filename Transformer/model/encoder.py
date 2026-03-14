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
# # Implement The Encoder Block
#
# ![Main Encoder Block](../showcase_images/from_paper/encoder.png)

# %% [markdown]
# **Encoder Stack**:
# - Composed of a stack of $Nx=6$ identical encoding layers.
# - The stacked encoding layers are identical in structure but do not share weights.
# - The Encoder's **input** would look, e.g., "The brown rabbit ate the apple. &lt;padding&gt; &lt;padding&gt; &lt;padding&gt;". We want to ensure that the &lt;padding&gt; tokens have zero influence on the other tokens, to that order we use a **src_padding_mask** filter to hide the padding.
#
# 1. The Encoder starts at the top of the Nx stack of EncoderLayers. 
#    1. It passes its input into its [Multi-Head Attention](./multi_head_attention.ipynb) sublayer. 
#    2. Then, residual + LayerNorm (Add & Norm) sublayer.
#    3. Then, feed through a Feed Forward (Feed-Forward Neural Network, also called an MLP)
#    4. Then, another residual + LayerNorm (Add & Norm) sublayer.
# 2. Once a single EncoderLayer has been passed through, it feeds the next EncoderLayer of the Nx stack, and once it finally reaches the $6$'th EncoderLayer it passes its output to the Decoder. As shown in the below image.
#    - ![enc_to_dec](../showcase_images/from_paper/enc_to_dec.png)
#    - So, path is: Input → EncoderLayer $1$ → EncoderLayer $2$ → ... → EncoderLayer $6$ → Decoder
#    - The output of the $6'th$ EncoderLayer (specifically its Keys & Values) is sent to every single DecoderLayer in the Decoder stack simultaneously (except to the *Mask Multi-Head Attention*).
#
#

# %%
import torch.nn as nn
import torch
import copy

try: # works when ran via main.py (package mode)
    from .residual_con_layer_norm import ResidualConnection, LayerNorm
    from .multi_head_attention import Multi_Head_Attention
    from .FeedForwardNetwork import FeedForwardNetwork
    from .model_utils import clones
except ImportError:
    # Works when running from inside Jupyter Notebook
    from model_utils import clones
    from residual_con_layer_norm import ResidualConnection, LayerNorm
    from multi_head_attention import Multi_Head_Attention
    from FeedForwardNetwork import FeedForwardNetwork


# %%
class EncoderLayer(nn.Module):
    """A single layer of the Encoder stack"""

    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.mha = Multi_Head_Attention(d_model, h, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        # Two residual connections: one for the MHA and the other for the FFN.
        self.residual_1_mha = ResidualConnection(d_model, dropout)
        self.residual_2_ffn = ResidualConnection(d_model, dropout)

    def forward(self, x, src_padding_mask):
        # Attention sublayer
        x = self.residual_1_mha(x, lambda x: self.mha(x, x, x, src_padding_mask))

        # Feed-Forward Sublayer
        x = self.residual_2_ffn(x, self.ffn)
        return x


# %%
class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N):
        """The full Encoder that is a Nx stack of EncoderLayer()

        Args:
            layer: A single EncoderLayer().
            N: The stack size of EncoderLayer()
        """
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, src_padding_mask):
        """
        Arg:
            src_padding_mask: The padding for sequences. Example with 3 paddings sequence: "The brown rabbit ate the apple. <Padding> <Padding> <Padding>". This mask hides the paddings from the Encoder, so it only sees non-padding tokens.
        """
        # Pass the input through each layer in the stack
        for layer in self.layers:
            x = layer(x, src_padding_mask)
        return self.norm(x)


# %%
def test():
    print("\n\nTesting the Encoder...")
    d_model = 512
    h = 8
    d_ff = 2048
    N = 6
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    base_enc_layer = EncoderLayer(d_model, h, d_ff, dropout)
    encoder = Encoder(base_enc_layer, N)

    dummy_input = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {dummy_input.shape}")

    # Create a simple padding Mask, the last 3 tokens are padding
    src_padding_mask = torch.ones(batch_size, 1, 1, seq_len)
    src_padding_mask[:, :, :, -3:] = 0  # A 0 means is for "<padding>"

    output = encoder(dummy_input, src_padding_mask)
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape mismatch!"

    # Verify Gradient flow
    output.mean().backward() 
    # Pick a parameter at the very beginning of the network (e.g., the first EncoderLayer's weights)
    start_param = encoder.layers[0].mha.w_q.weight
    assert start_param.grad is not None, "Gradient did not reach the first layer!"
    assert torch.any(start_param.grad != 0), "Gradients are zero (Vanishing Gradient)!"

    encoder.zero_grad()
    
    print("Test Passed!")

test()

# %%
