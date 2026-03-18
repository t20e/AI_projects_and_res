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
# <!-- Reviewed: ✅ -->

# %% [markdown]
# # Implement Embedding
# ![Embedding section image](../showcase_images/from_paper/embedding.png)

# %% [markdown]
# **Word Embeddings** are numerical representations of words in a **higher-dimensional space**, that capture semantic, syntactic, and contextual information. Words with similar meanings are positioned close to each other. The distance between vectors encodes the **degree of similarity between words**.
#
# A good [Embedding Space Visualization](https://projector.tensorflow.org/) by tensorflow.org |  another by [google](https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space)

# %% [markdown]
# ---
#
# **How It Works In The Transformer:**
#
# - The Transformer model uses **learned, task-specific embedding layers**, rather than a pre-existing algorithm like Word2Vec, or GloVe.
#
# - The embedding layer itself is a **static lookup table**. The **Self-Attention** mechanism in the encoder and decoder will mix these static vectors with the rest of the sequence to create **contextualized** representations.
#
#
#
# **Notes From Paper:**
#
# - "To facilitate these residual connections, all sub-layers in the model, as well as the **embedding layers**, produce outputs of dimension $d_{model}$ = 512."
# - "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$"
#
#
#
# **Regularization**:
# - **Dropout**: "In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of $P_{drop} = 0.1$".

# %%
import torch.nn as nn
import torch
import math


# %%
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.look_up_table = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.look_up_table(x) * math.sqrt(self.d_model)


# %%
try: # works when ran via main.py (package mode)
    from .pos_encoding import PositionalEncoding
except ImportError:
    # Works when running from inside Jupyter Notebook
    from pos_encoding import PositionalEncoding

def test():
    print("Testing Embedding...")
    d_model = 512  # Dimensionality of the vectors
    vocab_size = 1000  # Size of the dictionary
    batch_size = 2
    seq_len = 5  # num of words in each sentence

    em = Embeddings(d_model, vocab_size)

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {dummy_input.shape}")

    output = em(dummy_input)
    print(f"Output shape: {output.shape}")

    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, "Shape mismatch"
    
    # test with the positional Encoding
    pe = PositionalEncoding(d_model, dropout=0.1)
    x_em = em(dummy_input)
    x_final = pe(x_em)

    print(f"Embedding shape: {x_em.shape}")
    print(f"After Positional Encoding shape: {x_final.shape}")
    
    assert not torch.equal(x_em, x_final)
    print("Success\n")

# test()

# %%

# %%
