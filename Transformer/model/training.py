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
# **How the data flow through the model during training:**
# - During training, we use **Teacher Forcing** method.
#   - **Input:** The Encoder receives the full Source sequence, and creates a context vector representation of it.
#   - **Shifted Target:** The Decoder receives the shifted Target sequence, e.g., if the goal is to predict "The brown rabbit", the Decoder receives `[<SOS>, The, brown]`.
#   - **Masking:** The padding mask ensures the model ignores the `<PAD>` tokens, and specifically the **Masked Multi-Head Attention** will ignore both the `<PAD>`, and the future tokens, e.g., (same example as shifted target) when the model is predicting "brown", it can't peek at the answer "rabbit".
#   - **Decoder's output:** Produces a vector if size $d_{model} = 512$ for every token position.
#   - **Generator:** *(The last two layers: linear→softmax)* take those $512$-dim vectors and projects them on the size of the vocabulary (tokenizer=$37000$).

# %% [markdown]
#
# -  **Adam optimizer**: "We used the Adam optimizer [20] with $\beta_1 = 0.9$, $\beta_2 = 0.98$, and $\epsilon = 10^{−9}$"
#    -  $$ \text{lrate} = d^{-0.5}_{\text{model}} * \text{min} (\text{stepNum}^{-0.5}, \quad \text{stepNum} * \text{warmupSteps}^{-1.5}) $$
#       -  Example: $\quad d_{model} = 512,\quad \text{stepNum} = 100000,\quad \text{warmupSteps} = 4000$
#          - $ \text{lrate} = 512^{-0.5} * \text{min}(100000 * 4000^{-1.5}) = 0.000139...$
#            - The $\text{stepNum} = 100000$ is the **current step** (e.g., step 1, step 2, ...) it is a counter that ticks every time a single batch is processed.
#            - Paper: "We trained the base models for a total of 100,000 steps or 12 hours." This means that they stopped training after step $100000$.
#              - So $ 0.000139...$ is the learning rate at step $100000$
#    -  "This corresponds to increasing the learning rate [$\text{lrate}$] linearly for the first $\text{warmupSteps}$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used $\text{warmupSteps} = 4000$."
#       -  **Warmup**: For the first $4000$ steps, the $\text{stepNum} * \text{warmupSteps}^{-1.5}$ term is smaller, this causes the learning rate to increase linearly, which "warms-up" the model, preventing the gradients from exploding early on when gradients are random.
#       -  **Decay** $\text{min}( \text{stepNum}^{-0.5} > ...)$: After $4000$ steps, the $\text{stepNum}^{-0.5}$ is smaller inside the $\text{min}()$. This causes the learning rate to decrease following the inverse square root.  
#          -  Paper: "This corresponds to increasing the learning rate linearly for the first $\text{warmupSteps}$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used $\text{warmupSteps} = 4000$."
#       -  The dim ($d^{-0.5}_{model}$): This scales the entire learning rate based on the model size.
#   -  Note: that these $\quad d_{model} = 512,\quad \text{stepNum} = 100000,\quad \text{warmupSteps} = 4000$ is what the paper used to train on the full 4.5 million sentence from the **WMT14 dataset**, I will not use the entire dataset, the lrate will be updated differently depending on the size of the dataset.
#      - **Dataset Size Adjustment:** 
#        - **Example:** If we were to train on only $45088$ sentence pairs ($1$% of the WMT14 en-de dataset) with a batch_size $= 64$, than $45088/64 = 704$ **steps per epoch**.
#           - The $= 704$ means the optimizer will update the model's weights $= 704$ times per epoch.
#           - **Formulas:**
#             - **1 Step (or iteration) $=$** One forward pass + one backward pass on **one batch**.
#             - **Steps Per Epoch** $=$ Total_sentence_pairs $\div$ batch_size
#             - **Total Training Steps $=$** Steps per epoch $\times$ Total Epochs
#             - So, in this example set $\text{warmupSteps} = 704$ so that the learning rate starts to decrease at about the end of the first epoch, which coincides with the end of the warmup and the model can start training. 
#               - Result: 
#                 - Batch $1$ & $\text{stepNum} = 1$, learning rate is small.
#                 - Batch $704$ & $\text{stepNum} = 704$, learning rate is at its **peak** (end of epoch $1$)
#                 - Batch $705$ & $\text{stepNum} = 705$, learning rate begins to **decay** (start of epoch $2$)
#           - This example is shown in `lrate_growth_example()` below
#

# %% [markdown]
# **Notes:**
# - The paper used a fixed number of **tokens per batch**, not a fixed number of sentence pairs → "Each training batch contained a set of sentence pairs containing approximately $25000$ source tokens and $25000$ target tokens."

# %% [markdown]
# - **Validation**: check [beam_search.ipynb](./beam_search.ipynb)

# %% [markdown]
# - **Loss function**
#   - **Regularization**:
#     - **Label Smoothing**: "During training, we employed label smoothing of value $\in_{ls} = 0.1$. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score" to the loss function during training loop.
#       - Its added to the loss

# %%
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime, timedelta
import time
import os

try:  # works when ran via main.py (package mode)
    from .utils import make_target_mask
except ImportError:
    # Works when running from inside Jupyter Notebook
    from utils import make_target_mask

# %%
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checks example cfg: English_german_config below
    from .Transformer import Transformer
    from ..configs import English_german_config
    from tokenizers import Tokenizer

# %% [markdown]
# ## Batch

# %%
import torch
from  torch.utils.data import Sampler
import random

class TokenBatchSampler:
    def __init__(self, dataset, max_tokens, shuffle=True):
        """
        Groups sentences into batches based on their total token count.

        Provide a list of indices (e.g., [5, 102, 43]), which the dataloader uses to identify which sentences belong to current batch it is processing.

        Args:
            dataset: A dataset.
            max_tokens: The maximum number of tokens per sequence.
            shuffle: Whether to shuffle the dataset before sampling.
        """
        self.max_tokens = max_tokens
        self.shuffle = shuffle

        # Find teh length of the longest sentence, either source or target for each pair.
        lengths = [max(len(item["src_ids"]), len(item["tgt_ids"])) for item in dataset]

        # Get the indices that would sort the dataset from shortest to longest sentences.
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        self.batches = []
        curr_batch = []
        max_batch_len = 0

        for idx in sorted_indices:
            seq_len = lengths[idx]

            max_batch_len = max(max_batch_len, seq_len)

            # Update the max length for the current batch we are building.
            if max_batch_len * (len(curr_batch) + 1) > self.max_tokens:
                self.batches.append(curr_batch)
                curr_batch = [idx]
                max_batch_len = seq_len
            else:
                curr_batch.append(idx)
        
        # Add the last partially filled batch to the list of batches.
        if curr_batch:
            self.batches.append(curr_batch)
    
    def __iter__(self):
        # Shuffle the order of batches, not the contents inside them.
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# %%
class Batch:
    def __init__(self, src, tgt=None, pad_token: int = 0):
        """
        A single batch.

        Args:
            src = A batch of tokenized Source Sequence sentences.
            tgt = A batch of tokenized Target Sequence.
            pad_token: The integer representation for the '<PAD>' token
        """
        self.src = src

        # Make source mask
        self.src_padding_mask = (src != pad_token).unsqueeze(-2).unsqueeze(-2)

        self.tgt = tgt

        if tgt is not None:
            # Previous context: Take all tokens excerpt for the last one, this includes the <SOS> at the start.
            self.tgt = tgt[:, :-1]

            # 🌟 OUTPUTS (SHIFTED RIGHT): We take all tokens except for the first one (<SOS>), this start with the first actual word and ends with <EOS>
            self.tgt_y = tgt[:, 1:] # this is what the model is trying to predict at each time step

            # High future tokens so model doesn't cheat
            self.tgt_no_peek_mask = make_target_mask(self.tgt, pad_token)

            # non_tokens: Used to divide the total loss by the number of non-special tokens.
            self.non_tokens = (self.tgt_y != pad_token).data.sum()


# %% [markdown]
# ## Learning Rate Schedule

# %%
def get_std_opt(model: Transformer, d_model=512, warmup_steps=4_000):
    """
    Get the scheduler and optimizer.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1,  # lr=1 so the scheduler controls the absolute value.
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: lrate(
            step, d_model, factor=1, warmup_steps=warmup_steps
        ),
    )
    return optimizer, lr_scheduler


def lrate(step_num, d_model, warmup_steps, factor=1.0):
    """The lrate formula as shown in the top formula.
    
    Args:
        step_num: The current step.
        d_model: Size of the model.
        factor: Default to 1.0. If the model is learning to slowly, increase to (factor=2.0 or higher), if the loss is exploding becoming NaN, lower factor (e.g., 0.5).
        warmup_steps: How many steps to "warmup" the model.
    """
    if step_num == 0:
        step_num = 1
    return factor * (
        d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    )


# %%
def lrate_growth_example():
    batch_size = 64
    total_sentences = 45088
    d_model = 512
    # Calculate how many steps (batches) make up one epoch
    steps_per_epoch = total_sentences // batch_size
    warmup = steps_per_epoch

    print(f"Lrate example on 1% of WMT 14 dataset")
    print(f"Batch size: {batch_size}")
    print(f"Total sentences in dataset: {total_sentences:,}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Warmup ends at step: {warmup} (End of epoch 1)")
    print(
        f"\n\n{'Step':>8} | {'Approx Epoch':>12} | {'Sentences Seen':>16} | {'Learning-Rate':>12} | {'Phase'}"
    )

    milestones = [
        1,
        350,
        704,
        705,
        1500,
        3000,
        5000,
    ]  # Where the learning rate is changed.

    for step in milestones:
        lr = lrate(step_num=step, d_model=d_model, factor=1.0, warmup_steps=warmup)

        # Calculate progress
        curr_epoch = step / steps_per_epoch
        sentences_seen = step * batch_size

        if step <= warmup:
            phase = "Warmup (Linear up ↑)"
        else:
            phase = "Decay (Inverse Square Root ↓)"

        print(
            f"{step:8d} | {curr_epoch:12.2f} | {sentences_seen:16,d} | {lr:12.8f} | {phase}"
        )


# lrate_growth_example()

# %% [markdown]
# ## Loss

# %%
if TYPE_CHECKING:
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
        loss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()  # Clear gradients before the next batch.
        return loss.item() * norm


# %% [markdown]
# ## plot

# %%
import matplotlib.pyplot as plt


def plot_loss_history(loss_history, cfg: English_german_config):
    plt.figure(figsize=(12, 8))
    plt.plot(loss_history, label="Raw Loss", alpha=0.3, color="blue")

    if len(loss_history) > 100:
        window = 100
        import numpy as np

        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(loss_history)),
            smoothed,
            label="Smoothed Loss",
            color="red",
        )

    plt.title(f"Training Loss - {cfg.perc_to_download}% of Dataset")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    config_info = (
        f"Model: d_model={cfg.d_model}, N={cfg.N}, h ={cfg.H}, d_ff={cfg.d_ff}\n"
        f"Training: batch_size={cfg.batch_size}, vocab_size={cfg.vocab_size}, num_epochs={cfg.num_epochs}, step_limit={cfg.step_num_limit}, num_workers={cfg.num_workers}, max_batch_seq_tokens={cfg.max_batch_seq_tokens}, "
        f"max_indiv_seq_len={cfg.max_indiv_seq_len}, warmup_steps={cfg.warmup_steps},\n"
        f"{cfg.perc_to_download}% Percent of {cfg.dataset_name} dataset, "
        f"Total Sentence Pairs: {cfg.total_sentence_pairs},\n"
        f"Final Step: {len(loss_history):,}"
    )

    # Place config info at bottom of plot
    plt.figtext(
        0.5,
        0.01,
        config_info,
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
        bbox={"facecolor": "orange", "alpha": 0.1, "pad": 10},
    )
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    plot_path = os.path.join(
        cfg.MODEL_DIR,
        "checkpoints",
        f"training_loss_plot_{cfg.perc_to_download}_percent_ds.png",
    )
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()


# %% [markdown]
# ## Train Model

# %%
class TrainModel(nn.Module):
    def __init__(self, cfg: English_german_config, model: Transformer, device):

        super().__init__()
        self.cfg = cfg
        self.model = model
        self.device = device

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1,  # Used Torch's built-in Label Smoothing.
            ignore_index=cfg.special_tokens["pad_token"],
            reduction="sum",
        )

        self.optimizer, self.scheduler = get_std_opt(
            model=model,
            d_model=cfg.d_model,
            warmup_steps=cfg.warmup_steps,
        )
        self.compute_loss = SimpleLossCompute(
            model.generator, self.criterion, self.optimizer
        )

        # Track the steps. Once it reaches `step_num_limit` we end training
        self.step_counter = 0

        self.loss_history = []

        self.start_time = None

    def save_checkpoint(self, epoch, avg_loss):
        checkpoint_name = (
            f"transformer_epoch_{epoch+1}_{self.cfg.perc_to_download}_percent_ds.pt"
        )
        checkpoint_path = os.path.join(
            self.cfg.MODEL_DIR, "checkpoints", checkpoint_name
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step_counter": self.step_counter,
                "loss_history": self.loss_history,
                "loss": avg_loss,
            },
            checkpoint_path,
        )
        print(f"Saved Checkpoint to -> {checkpoint_path}")

    def train(self, train_dataloader, start_epoch):
        """
        Train a model
        Args:
            start_epoch: Will depend on if we are training from a checkpoint or training a new model.
        """
        num_epochs = self.cfg.num_epochs
        self.start_time = time.time()
        total_steps_to_go = self.cfg.step_num_limit - self.step_counter

        print("\n" + "#" * 64)
        print(f"\nTraining Model")
        print(
            f"Num epochs: {num_epochs} | device: {self.device} | Max steps {self.cfg.step_num_limit:,}"
        )
        print("\n" + "#" * 64)

        for epoch in range(start_epoch, num_epochs):
            avg_loss = self.run_epoch(train_dataloader)
            print(
                f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}"
            )

            if self.step_counter >= self.cfg.step_num_limit:
                print(
                    f"Reached step limit: {self.cfg.step_num_limit}. Final Checkpoint..."
                )
                self.save_checkpoint(epoch=epoch, avg_loss=avg_loss)
                break

            self.save_checkpoint(epoch=epoch, avg_loss=avg_loss)

        plot_loss_history(loss_history=self.loss_history, cfg=self.cfg)
        print("Training complete!\n\n")

    def run_epoch(self, dataloader):
        """Run a single epoch"""
        total_tokens = 0
        total_loss = 0
        device = self.device
        self.model.train()

        for i, batch in enumerate(dataloader):

            if self.step_counter >= self.cfg.step_num_limit:
                return total_loss / (total_tokens if total_tokens > 0 else 1)

            # Move batch to device
            src = batch.src.to(device)
            tgt = batch.tgt.to(device)
            tgt_y = batch.tgt_y.to(device)
            src_padding_mask = batch.src_padding_mask.to(device)
            tgt_no_peek_mask = batch.tgt_no_peek_mask.to(device)
            non_tokens = batch.non_tokens.to(device)

            # Forward Pass (The model returns decoder output before the generator (last linear + softmax layers))
            output = self.model(src, tgt, src_padding_mask, tgt_no_peek_mask)

            # Loss compute, performs backward prop
            loss = self.compute_loss(output, tgt_y, non_tokens)

            curr_step_loss = loss / non_tokens.item()
            self.loss_history.append(curr_step_loss)

            total_loss += loss.item()
            total_tokens += non_tokens.item()

            # Update Learning Rate per step
            self.scheduler.step()
            self.step_counter += 1

            if i % 10 == 0:
                self.print_step_info(
                    total_tokens, loss, non_tokens, num_batches=len(dataloader)
                )

        # Average the loss over the exact number of real words, excluding all the special tokens!
        return total_loss / total_tokens

    def print_step_info(self, total_tokens, loss, non_tokens, num_batches):
        elapsed = time.time() - self.start_time
        steps_completed = self.step_counter

        if steps_completed > 0:  # So it doesn't divide by zero
            avg_time_per_step = elapsed / steps_completed

            # Calculate the actual total steps expected to go for this run
            total_epochs = self.cfg.num_epochs
            steps_per_epoch = num_batches

            # There is a step_num_limit=100_000, but the training can also end before that depending on config.
            actual_total_steps = min(
                self.cfg.step_num_limit, total_epochs * steps_per_epoch
            )

            remaining_steps = actual_total_steps - steps_completed
            remaining_steps = max(0, remaining_steps)  # Prevent negative ETA
            eta_seconds = remaining_steps * avg_time_per_step

            # Convert seconds to readable
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(elapsed)))

            print(
                f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] "
                f"Step: {self.step_counter}/{actual_total_steps} | "
                f"Loss: {loss/non_tokens.item():.4f} | "
                f"Tokens: {total_tokens} | "
                f"Time Elapsed: {elapsed_str} | "
                f"ETA: {eta_str}"
            )

# %%

# %%
