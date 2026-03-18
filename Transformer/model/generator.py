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
# # Implement Generator
#
# ![Generator image](../showcase_images/from_paper/generator.png)

# %% [markdown]
# The generator is the final `Linear` + `Softmax` layer to convert the Decoder's vector output back into vocabulary probabilities.

# %%
import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

# %%
