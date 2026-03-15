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
# # Implement Positional Encoding
#
# ![Alt text](../showcase_images/from_paper/pos_encoding.png)
#
# * From Figure 1 in the paper.
#

# %% [markdown]
# $$PE_{(pos, 2i)} = sin(pos/10000^{2 i /d_{model}})$$
#
# $$PE_{(pos, 2i+1)} = cos(pos/10000^{2 i /d_{model}})$$
#
# **Note**:
# - In order for the model to make use of the order of the sequence, we inject some information about the relative or absolute position of the tokens in the sequence.
# - The position encodings have the same dimension $d_{model}$ as the embeddings
# - $pos$ is the position.
# - $i$ is the dimension. Each dimension corresponds to a sinusoid.
#
# - **Regularization**:
#   - Dropout from paper: "In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of $p_{drop} = 0.1$.

# %%
import torch.nn as nn
import torch
import math


# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, pos_seq_len = 5000, dropout=0.1):
        """
        Implements Positional Encoding, which adds positional data to sequence vectors.
        Args:
            d_model: Dimensionality of the vectors.
            pos_seq_len: The size of the static `pos_enc`, needs to be large enough to cover any sentence it might ever encounter.
            dropout: Dropout regularization
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # === Compute the positional encodings once in log space.
        pos_enc = torch.zeros(pos_seq_len, d_model)

        # Create a vector of positions [0, 1, ..., pos_seq_len-1]
        position = torch.arange(0, pos_seq_len, dtype=torch.float).unsqueeze(-1)

        # Calculate division denominator, 2 correlates to 2i in formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10_000.0) / d_model)
        )

        # Fill the pos_enc matrix
        pos_enc[:, 0::2] = torch.sin(position * div_term) # Even indices get sine
        pos_enc[:, 1::2] = torch.cos(position * div_term) # Odd indices get cosine


        pos_enc = pos_enc.unsqueeze(0) # Add a batch -> (1, pos_seq_len, d_model)

        # Register as buffer so it is saved with the model, but is not a learned parameter.
        self.register_buffer("pos_enc", pos_enc)
    
    def forward(self, x):
        """x.shape: (batch, seq_len, d_model)"""
        # x = x + self.pos_enc[:, : x.size(1)].requires_grad_(False)
        x = x + self.pos_enc[:,  :x.size(1), :]
        return self.dropout(x)


# %%
def test():
    print(f"\n\nTesting Positional Encoding...")
    pos_enc_layer = PositionalEncoding(d_model=512, pos_seq_len=100)
    sample_input = torch.zeros(1, 100, 512)
    output = pos_enc_layer(sample_input)

    print(output.shape)

# test()

# %%
