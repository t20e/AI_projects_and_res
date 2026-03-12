# Implement The Transformer Model From The OG Paper

- Paper → [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

Useful Resources:

- [Transformers Step-by-Step Explained by ByteByteGo](https://www.youtube.com/watch?v=avjX3QrYkls)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

**TODO:**

- Embeddings

- Training:
  - Regularization:
    - Label Smoothing: "During training, we employed label smoothing of value $\in_{ls} = 0.1$. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score" to the loss function during training loop.

---

- 💡 The transformer architecture is used in many SOTA models that need to process sequential data. It can be reconfigured to perform a lot of different tasks other than just Naturel Language Processing.

![Figure one from paper](./showcase_images/from_paper/main.png)

- 💡 All the components are implemented in there own notebooks in [./model](./model/).

- **Terminology**:
  - The **Encoder** contains a **Nx** stack of **EncoderLayers**
    - Inside a single **EncoderLayer** there are two main components (or **sublayers**.
      - The first is the **Multi-Head Attention** and its (Add & norm).
      - The second is the **Feed Forward** (FFN) and its (Add & norm).
    - This goes for the decoder as well, except it has an additional **sublayer**, the **Masked Multi-Head Attention**.
