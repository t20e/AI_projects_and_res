# Implement The Transformer Model From The OG Paper

- Paper → [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

Useful links:

- [Transformers Step-by-Step Explained by ByteByteGo](https://www.youtube.com/watch?v=avjX3QrYkls)


**TODO:**

- Regularization:
  - Label Smoothing: "During training, we employed label smoothing of value $\in_{ls} = 0.1$. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score" to the loss function during training loop.

## How The Transformer Architecture Works

1. Scaled Dot-Product Attention:1 Implement the formula:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
2. Multi-Head Attention (MHA): This is where you handle the "splitting" of heads. Focus on the reshape and transpose operations required to process multiple heads in parallel.
3. Positional Encoding: Implement the sine and cosine functions. This is a standalone module and easy to verify visually or via unit tests.
4. The Feed-Forward Network: The simple two-layer linear transformation used in every encoder/decoder block.
5. The Encoder Layer: Combine MHA, LayerNorm, and Feed-Forward with residual connections.

Tips for the Build

- Watch your dimensions: Keep a comment block at the top of each function noting the expected input/output shapes (e.g., (batch_size, seq_len, d_model)).
- Masking is the hardest part: Spend extra time on the "Look-Ahead Mask" for the decoder. It's the most common source of bugs.
- Use dummy data: Don't try to train it on a real dataset immediately. Pass a random tensor of shape (batch_size, seq_len, d_model) through each component to ensure it doesn't crash.


