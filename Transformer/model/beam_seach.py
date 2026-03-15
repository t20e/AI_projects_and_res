# ---
# jupyter:
#   jupytext:
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
# # Implement Beam Search
#

# %% [markdown]
# - Beam Search is what turns 
# - Steps:
#   1. The model is given a sequence of token IDS representing sub-words.
#   2. It outputs a vector the size of the vocabulary (total number of unique sub-words from the tokenizer)
#   3. Every position in that vector is a logit (score)
#   4. Beam search selects the best scores.
#   5. The tokenizer looks up which string (like "car") belongs to those indices and joins them into a human readable string.
#
# - During training we run Beam Search on a few sentence pairs from the validation set every epoch or so, to validate how well the model is perform while training. Then after training they used Beam Search on a test set to get the BLEU score. 

# %%
