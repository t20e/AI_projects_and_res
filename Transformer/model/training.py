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
# # Implement Model Training

# %% [markdown]
# - **Loss function**
#   - **Regularization**:
#     - **Label Smoothing**: "During training, we employed label smoothing of value $\in_{ls} = 0.1$. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score" to the loss function during training loop.
#       - Its added to the loss
#
# -  **Adam optimizer**: "We used the Adam optimizer [20] with $\beta_1 = 0.9$, $\beta_2 = 0.98$, and $\epsilon = 10^{−9}$"
#    -  $$l \text{rate} = d^{-0.5}_{\text{model}} * \text{min} (\text{step\_num}^{-0.5}, \text{step\_num} * \text{warmup\_steps}^{-1.5}) $$
#       -  $d_{model} = 512,\quad \text{step\_num} = 100000,\quad \text{warmup\_steps} = 4000$
#          -  $l \text{rate} = 512^{-0.5} * \text{min}(100000 * 4000^{-1.5}) = 0.000139...$
#    -  "This corresponds to increasing the learning rate [$l \text{rate}$] linearly for the first $\text{warmup\_steps}$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used $\text{warmup\_steps} = 4000$."

# %%
# <!-- #TODO make sure formulas are displayed correct on the repo -->

# %%
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR


try:  # works when ran via main.py (package mode)
    from .model_utils import make_target_mask
except ImportError:
    # Works when running from inside Jupyter Notebook
    from model_utils import make_target_mask

# %% [markdown]
# # TODO: 
# - Validate via greedy search

# %%
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checks example cfg: english_german_config below
    from .Transformer import Transformer
    from ..configs import english_german_config
    from tokenizers import Tokenizer


# %% [markdown]
# ## Batch

# %%
class Batch:
    def __init__(self, src, tgt=None, pad_token: int = 0):
        """
        Handles token masking logic.

        Args:
            src = A batch of tokenized Source Sequence.
            tgt = A batch of tokenized Target Sequence.
            pad_token: The integer representation for the '<PAD>' token
        """

        # TODO understand this?
        """Object to hold a batch of data with mask during training."""
        self.src = src
        self.src_padding_mask = (src != pad_token).unsqueeze(-2).unsqueeze(-2)

        self.tgt = tgt

        if tgt is not None:
            # Previous context: Take all tokens excerpt for the last one, this includes the <SOS> at the start.
            self.tgt = tgt[:, :-1]

            # 🌟 OUTPUTS (SHIFTED RIGHT): We take all tokens except for the first one (<SOS>), this start with the first actual word and ends with <EOS>
            self.tgt_y = tgt[:, 1:] # this is what the model is trying to predict at each time step

            self.tgt_no_peek_mask = make_target_mask(self.tgt, pad_token)

            # non_tokens: Used to divide the total loss by the number of non-padding tokens.
            self.non_tokens = (self.tgt_y != pad_token).data.sum()


# %% [markdown]
# ## Learning Rate Schedule

# %%
def get_std_opt(model: Transformer, d_model=512, warmup_steps=4_000, step_num=100_000):
    # TODO understand this?
    # lr=1 so the scheduler controls the absolute value.
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: lrate(step, d_model, factor=1, warmup=warmup_steps),
    )
    return optimizer, lr_scheduler


def lrate(step_num, model_size, factor, warmup_steps):
    """The lrate formula as shown in the above image."""
    if step_num == 0:
        step_num = 1
    return factor * (
        model_size ** (-0.5)
        * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    )


# %% [markdown]
# ## Losss

# %%
class LabelSmoothing(nn.Module):
    # TODO understand this?
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


# %%
if TYPE_CHECKING:  # for type checks example cfg: english_german_config below
    from .generator import Generator


class SimpleLossCompute:
    def __init__(self, generator: Generator, criterion, opt=None):
        """
        Compute a simple loss function

        Args:
            generator: The last Linear → softmax layers
            opt: Adam optimizer
        """

        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data * norm


# %% [markdown]
# ## Train Model

# %%
class TrainModel(nn.Module):
    def __init__(self, cfg: english_german_config, model: Transformer, device):
        super().__init__()
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1, ignore_index=cfg.special_tokens["pad_token"]
        )
        self.model = model
        self.device = device

    def forward():
        print("\n" + "#" * 64)
        print(f"\nTraining Model")
        print("\n" + "#" * 64)

    def run_epoch(self, dataloader, compute_loss, optimizer, scheduler, device):
        """Run a single epoch"""
        total_tokens = 0
        total_loss = 0

        self.model.train()
        for i, batch in enumerate(dataloader):
            # Move to device
            src = batch.src.to(device)
            tgt = batch.tgt.to(device)
            tgt_y = batch.tgt_y.to(device)
            src_padding_mask = batch.src_padding_mask.to(device)
            tgt_no_peek_mask = batch.tgt_no_peek_mask.to(device)

            # Forward Pass
            output = self.model(src, tgt, src_padding_mask, tgt_no_peek_mask)

            # Loss compute
            loss = compute_loss(output, tgt_y, batch.non_tokens)
            total_loss += loss
            total_tokens += batch.non_tokens

            # Update Learning Rate
            scheduler.step()

# %%
