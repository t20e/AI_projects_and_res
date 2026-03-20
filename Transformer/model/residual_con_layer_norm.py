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
# # Implement The Residual Connection & Layer Normalization
#
# ![residual_layer_normalization](../showcase_images/from_paper/residual_layer_norm.png)

# %% [markdown]
# Paper's **Post-Layer Normalization** Formula:
#
# $$\text{LayerNorm}(x + \text{Sublayer}(x))$$
#
# - Where:
#   - $\text{Sublayer}(x)$ "is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce an output of dimension $d_{model} = 512$"
#
# ⭐️ Note: Modern best practice is to apply **Pre-Layer Normalization**, so instead of using above formula, we use:
#
# $$x + \text{Sublayer}(\text{LayerNorm}(x))$$
#

# %% [markdown]
# ## Layer Normalization
#
# From [Layer Normalization](https://arxiv.org/pdf/1607.06450) paper & [pytorch doc](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html):
#
# $$y = \frac{x - E[x]}{\sqrt{\text{Var} [x] + \epsilon}} * \gamma + \beta$$

# %% [markdown]
# Stability:
# - Utilizing layer normalization helps keep training stable, e,g., keeping gradients in check to prevent exploding gradients or vanishing gradients.
#
# LayerNorm looks at a vector for a single token and ensures its mean is $0$ and its standard deviation is $1$. Then, it applies two learned parameters: $\gamma$ (gamma) to scale it, and $\beta$ (beta) to shift it.

# %%
import torch.nn as nn
import torch


# %%
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        # Learned parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
    
        # std = x.std(-1, keepdim=True)
        # return self.gamma * (x - mean) / (std + self.eps) + self.beta

        


# %% [markdown]
# ## Residual Connection
#
# "We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]."
#
# - The residual connections is a way to preserve information.
#   - Example without residual connections: Pass the sentence "The brown rabbit ate the apple." into a filter. The filter might get distracted by the "brown" and lose the "rabbit". By the time the data gets through 6 layers, the original meaning might be totally lost.
#     -  With a residual connection: You pass the sentence into a filter, but also carry a copy of the original "The brown rabbit ate the apple." around the outside of the filter, and add it back after the filter.
#
# Regularization:
# - Dropout: "We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized."
#

# %%
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # Note: Paper used (Post_LN), e.g., self.norm(x). But modern best practice is to apply LayerNorm before the sublayer (Pre-LN). I am using the latter.  
        out = sublayer(self.norm(x))
        out = self.dropout(out)
        return x + out

# %%

# %%
