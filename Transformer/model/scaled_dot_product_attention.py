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
# # Implement the Scaled Dot-Product Attention
#
# ![Alt text](../showcase_images/from_paper/scaled_dot_product_attention.png)
#
# * *From Figure 2 in the paper.*
#

# %% [markdown]
# Inputs: **Queries**, **Keys**, and **Values**.
# - Queries & Keys dimension = $d_k$ 
# - Values dimension = $d_v$
#
# **Formula:**
# $$Attention(Q, K, V) = softmax(\frac{Q K^T}{\sqrt{d_k}})V$$
#

# %% [markdown]
# 1. **MatMul**: Compute the dot products of the query with all keys of dimension $d_k$. As shown above: $Q K^T$. Determines which tokens are relevant, we take the dot-product between a token's $Q$ and the $K$ of all other tokens in the sequence,
# 2. **Scale**: Divide the each result by $\sqrt{d_k}$, this prevents gradients from vanishing during softmax.  
# 3. **Mask (optional)**: Apply a mask $-∞$, this is only used for the **Masked Multi-Head Attention**, which ensures that future generated tokens from the Decoder's output, do not have any influence on the current token.
# 4. **Softmax**: Then we normalize with a softmax to turn them into attention weights, these weights act as focus levels.
# 5. **Weighted Sum/ last MatMul**: Multiply the weights by the Values. As shown above: $softmax(...)V$

# %%
import torch
import torch.nn.functional as F
import math


# %%
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None, dropout=None
):
    # TODO add Docstring
    d_k = q.size(-1)  # dim of the Keys vectors

    # 1. MatMul
    scores = torch.matmul(q, k.transpose(-2, -1))

    # 2. Scale
    scores /= math.sqrt(d_k)

    # 3. Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 4. Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply Dropout from paper: "We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized (softmax)."
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # 5. Weighted sum/Matmul
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


# %%
def test():
    print("\n\nRunning Scaled Dot-Product Attention Test...")
    batch_size = 2
    heads = 8
    seq_len = 10
    d_k = 64

    q = torch.randn(batch_size, heads, seq_len, d_k)
    k = torch.randn(batch_size, heads, seq_len, d_k)
    v = torch.randn(batch_size, heads, seq_len, d_k)

    output, weights = scaled_dot_product_attention(q, k, v)

    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")

# test()

# %%
