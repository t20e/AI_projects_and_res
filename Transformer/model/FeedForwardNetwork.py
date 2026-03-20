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
# # Implement The Feed Forward Network
#
# ![FFN](../showcase_images/from_paper/ffn.png)
#
# $$FFN(x) = max(0, x W_1 + b_1) W_2 + b_2$$

# %% [markdown]
# **Notes**:
# - The FFN is applied to each position separately and identically.
# - It consists of two linear transformations with a **ReLU** activation in between. The $max(...)$ is the ReLU.
#   - The first linear expands the dimension from $d_{model} = 512$ to $d_{ff} = 2048$
#     - By expanding we give the model more parameters to learn non-linear functions. This is also where the heavy pattern recognition happens.
#   - The second linear brings it back down to $512$
# - While the linear transformations are the same across different positions, they use different parameters from layer to layer.
# - **Regularization**:
#   - Dropout: "We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized." The base model has a dropout $= 0.1$

# %%
import torch.nn as nn
import torch
import torch.nn.functional as F


# %%
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        """
        Implement The Feed Forward Network layer

        Args:
            d_model: model size
            d_ff: How much to expand the first layer in the FFN
            dropout: The dropout rate
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        return self.w_2(x)


# %%
def test():
    print("\n\nTesting FFN...")

    FFN = FeedForwardNetwork()
    batch_size = 2
    seq_len = 10
    d_model = 512

    sample_input = torch.zeros(batch_size, seq_len, d_model)
    output = FFN(sample_input)
    print(f"Output size: {output.shape}")

# test()

# %%
