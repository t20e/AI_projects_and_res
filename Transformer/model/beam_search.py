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
# [Beam Search explained](https://d2l.ai/chapter_recurrent-modern/beam-search.html#id1)
#

# %% [markdown]
# **During Inference:**
#
# - Beam Search acts as the controller, it wraps around the Transformer model and drives the generation process.
# - Steps:
#     1. Beam Search begins the process, and feeds the model an initial sequence containing only the starts-of-sequence token (<SOS>).
#     2. The model performs a forward pass and outputs a vocabulary-sized vector of logits (probabilities) the size of the vocabulary (total number of unique sub-words from the tokenizer) for the very next position.
#     3. Beam Search takes these logits, applies a **log-softmax** to get log probabilities, and selects the top $\boldsymbol{k}$ candidates, where $\boldsymbol{k}$ is the set to **beam width**.
#     4. **Loop**: Beam Search creates a new separate $\boldsymbol{k}$ sequences by appending each chosen token to the original start token. It then feeds all $\boldsymbol{k}$ sequences back into the model for another forward pass to predict the next token for each branch.
#         1. The model outputs logits for these new positions, and Beam Search calculates the cumulative score for all possible next steps across all branches.
#         2. It the prunes the possibilities, keeping only the top $\boldsymbol{k}$ sequences overall.
#         3. This loop continues until a sequence generates an end-of-sequences token (<EOS>), or hits a predefined maximum length limit.
#         4. Finally (after the loop), the **tokenizer** takes over to decode the winning sequence of integer IDs back into human-readable string.
#

# %% [markdown]
# **During Training:**
#
# - Periodically during training, to calculate the **BLEU** score, the model switches to inference mode, and Beam Search takes the wheel, this is solely for the purpose of **validation** during training.
#

# %% [markdown]
# **From Paper:**
#
# - "For the base models, we used a single model obtained by averaging the last $5$ checkpoints, which were written at $10$-minute intervals. For the big models, we averaged the last $20$ checkpoints. We used **beam search** with a **beam size** of $4$ and **length penalty** $\alpha = 0.6$ [38]. These hyperparameters were chosen after experimentation on the development set. We set the **maximum output length** during **inference** to input length $+ 50$, **but terminate early when possible** [38]."
#

# %%
import torch

try:  # works when ran via main.py (package mode)
    from .utils import make_target_mask
except ImportError:
    # Works when running from inside Jupyter Notebook
    from utils import make_target_mask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Transformer import Transformer


# %%
def BeamSearch(
    model: Transformer,
    src,
    src_padding_mask,
    max_len,
    pad_token_id,
    start_token,
    eos_token,
    device="mps",
    beam_size=4,
):
    """
    Implement Beam Search for a single sequence (batch_size=1).

    Args:
        model: Initialized Transformer()
        src: Source Tokens.
        src_padding_mask: The padding for source sequence.
        max_len: Failsafe so model does not get stuck in an infinite loop generating gibberish. If you feed model a 10-word English sentence, the max_len is set to $10+50=60$. From paper: "We set the maximum output length during inference to input length + 50, but terminate early when possible"
        pad_token_id: The ID of the <PAD> token, should be set to 0.
        start_token: <SOS> token
        eos_token: <EOS>  token.
        device: CPU or GPU Pytorch device
        beam_size=4: Also called **Beam Width** or **k**, how many top candidates sequences to keep. At every step of generation, the model hold onto the 4 most probable partial sentences.
    """
    model.eval()

    with torch.no_grad():
        # Encode once
        encoder_out = model.encode(src, src_padding_mask)

        # Create K sequences and scores to keep track of surviving sequences
        alive_seq = torch.full((1, 1), start_token, dtype=torch.long, device=device)
        alive_scores = torch.zeros(1, 1, device=device)

        finished_beams = []

        for step in range(max_len):
            num_active = alive_seq.size(0)

            # Expand encoder's output & source mask to match the number of active beams, this is so we can append next generated tokens
            exp_encoder_out = encoder_out.expand(num_active, -1, -1)
            exp_src_mask = src_padding_mask.expand(num_active, -1, -1, -1)

            tgt_no_peek_mask = make_target_mask(alive_seq, pad_token=pad_token_id).to(
                device
            )

            # Decode and get log_softmax for the next token
            decoder_out = model.decode(
                alive_seq, exp_encoder_out, exp_src_mask, tgt_no_peek_mask
            )
            log_probs = model.generator(decoder_out[:, -1, :])

            # Add new log_probs to cumulative scores
            scores = alive_scores + log_probs

            # Flatten to find the top k candidates across all branches
            flat_scores = scores.view(-1)
            k = min(beam_size - len(finished_beams), flat_scores.size(0))
            top_scores, top_indices = torch.topk(flat_scores, k)

            # Map 1D indices back to beam and vocabulary tokens
            vocab_size = log_probs.size(-1)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Prepare for the next timestep
            next_seqs = []
            next_scores = []

            # Update sequences and check if <EOS> token was generated
            for score, beam_idx, token_idx in zip(
                top_scores, beam_indices, token_indices
            ):
                seq = torch.cat([alive_seq[beam_idx], token_idx.unsqueeze(0)])

                if token_idx.item() == eos_token:  # Found <EOS>
                    # Apply length penalty
                    length_penalty = seq.size(0) ** 0.6
                    finished_beams.append((seq, score.item() / length_penalty))
                else:
                    next_seqs.append(seq)
                    next_scores.append(score)

            # Stop if all beams hit <EOS> or we have enough finished beams
            if not next_seqs or len(finished_beams) >= beam_size:
                break

            alive_seq = torch.stack(next_seqs)
            alive_scores = torch.tensor(next_scores, device=device).unsqueeze(1)

        # If max_len reached before <EOS> is generated, add remaining active beams to finished
        if not finished_beams:
            for i in range(alive_seq.size(0)):
                length_penalty = alive_seq[i].size(0) ** 0.6
                finished_beams.append(
                    (alive_seq[i], alive_scores[i].item() / length_penalty)
                )

        # Sort by best penalized score
        finished_beams.sort(key=lambda x: x[1], reverse=True)

        # Return the raw tensor sequence of the best beam
        return finished_beams[0][0]

# %%
