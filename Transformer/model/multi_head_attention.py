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
# # Implement Multi-Head Attention (MHA)
#
# ![Alt text](../showcase_images/from_paper/multi_head_attention.png)
#
# * From Figure 2 in the paper.

# %% [markdown]
# Notes:
# 1. The **Masked Multi-Head Attention** also uses this, its just that it does it with a different set of Keys, Values, Queries!
# 2.  **Projection**: The MHA layer first creates 3 different representations of each token in the sequence: $V$ (Values), $K$ (Keys), $Q$ (Queries).
#    - The $Q$ asks "What am I looking for?".
#    - The $K$ answers "What information do I contain?".
#    - The $V$ answers "If I am relevant, what information should I actually pass forward?".
#    - Example: In a sentence the tokens would be words:
#     - "The brown rabbit ate the apple."
#       - The token for the word "rabbit":
#           1. would have the 3 vectors create for it.
#           2. its $Q$ would ask the question and it would use all the other tokens $K$ and $V$ in the sequence to answer this. This also occurs for all other tokens in the sequence.
#    1. Input: A single token matrix representing the sequence, shape (batch_size, seq_len, $d_{model}$), this matrix contains the word embeddings plus positional encodings. Every row is a vector for one token. 
#    2. Operation: matrix multiplication $(X * W)$.
#    3.  Result: Three new matrices of the same shape: $Q$, $K$, and $V$.
# 3. [**Scaled Dot-Product Attention**](./scaled_dot_product_attention.ipynb) part.
#    1. We take the three ($Q$, $K$, $V$) matrices and split them along the $d_{model}$ dimension into $h$ "heads".
#       1. If $d_{model} = 512$ and $h$ = $8$, it splits each $512$-length vector into eight $64$-length vectors ($8 * 64 = 512$).
#       2. Each head will look for different things. One head might focus on grammar, another on punctuation, etc...
# 4. **ConCat**: Concatenate the heads back together.
# 5. Perform the final Linear.
#

# %% [markdown]
# **Inputs**: **Queries**, **Keys**, and **Values**.
# - Queries 
# - Keys & Values dimensions = $d_k = d_V = d_{model} / h = 64$
# - $h$ = $8$ parallel attention heads.
#
# **Regularization**:
# - Base model used a dropout rate of $0.1$
#
# **Formula:**
#
# $$MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O$$

# %% [markdown]
# Where:
#
# - $head_i = Attention(Q W^{Q}_i, K W^{K}_i, V W^{V}_i)$
# - We need to project the Queries, Keys, and Values
# - The projections are parameter matrices: $W^Q_i \in \mathbb{R}^{d_{model} * d_k}$, $W^K_i \in \mathbb{R}^{d_{model} * d_k}$, $W^V_i \in \mathbb{R}^{d_{model} * d_v}$, and $W^O \in \mathbb{R}^{h d_v * d_{model}}$
#   - $W^Q_i \in \mathbb{R}^{d_{model} * d_k}$ means:
#     - Take an input vector of size $d_{model}$ (e.g., $512$)
#     - Multiply it by a matrix to shrink it down to $d_k$ (e.g., $64$)
#     - Because you use $h$ heads (e.g., $8$), you do this 8 times in parallel $(8 * 64 = 512)$
#   - $W^O$ is the final weight output projection.

# %%
import torch.nn as nn
import math

# %%
try: # works when ran via main.py (package mode)
    from .scaled_dot_product_attention import scaled_dot_product_attention
except ImportError:
    # Works when running from inside Jupyter Notebook
    from scaled_dot_product_attention import scaled_dot_product_attention

import torch


# %%
class Multi_Head_Attention(nn.Module):
    """
    Implement Multi-Head Attention as shown in the right side from Figure 2 in the paper.
    """

    def __init__(self, d_model=512, H=8, dropout=0.1):
        super().__init__()
        assert d_model % H == 0, "d_model must be divisible by h"

        self.d_model = d_model
        self.H = H  # Num heads
        self.d_k = d_model // H  # Defaults then d_k=64

        # Linear for the Queries, Keys, and Values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: The Queries sequence.
            k: The Keys sequence.
            v: The values sequence.
            mask: Whether this is a Mask Multi-Head Attention (will contain a mask) or the regular Multi-Head Attention (will be None)
        """
        batch_size = q.size(0)

        query_seq_len = q.size(1)
        keys_value_seq_len = k.size(1) # keys and value have the same length

        # 1. Projection and reshape the Queries, keys, and values and Reshape
        q = self.w_q(q).view(batch_size, query_seq_len, self.H, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, keys_value_seq_len, self.H, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, keys_value_seq_len, self.H, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        out, self.attn_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 3. Concat back
        out = out.transpose(1, 2).contiguous().view(batch_size, query_seq_len, self.d_model)

        # 4. Final Linear
        return self.w_o(out)


# %%
def test():
    print("\n\nRunning MHA Test...")

    d_model = 512
    H = 8
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    MHA = Multi_Head_Attention(d_model, H, dropout)
    # print(MHA.dropout)

    # Inputs as they would come from the Embedding layer
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    # Test without mask
    output = MHA(q, k, v)
    print(f"Output shape (no mask): {output.shape}")

    # Test with mask that is used in the Mask Multi-Head Attention
    mask = torch.tril(torch.ones(seq_len, seq_len))
    output_mask = MHA(q,k,v, mask=mask.unsqueeze(0).unsqueeze(0))
    print(f"Output shape (with mask): {output_mask.shape}")
    
# test()

# %%


