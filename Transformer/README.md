# Implement The Transformer Model From The OG Paper

- Paper → [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

Useful Resources:

- [Transformers Step-by-Step Explained by ByteByteGo](https://www.youtube.com/watch?v=avjX3QrYkls)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

**TODO:**

- Calculate **BLEU** scores
- Once I have a good model store it on huggingface with its tokenizer, and let user know to download it.

- Do one more final review of all code!
- If I create the EasyJupyter library implement it here.

---

💡 The transformer architecture is used in many SOTA models that process sequential data. However, it can be reconfigured to perform other tasks than just Naturel Language Processing (NLP).

![Figure one from paper](./showcase_images/from_paper/main.png)

- ✨ All the model's layers are implemented in there own notebooks in [./model](./model/).

- **Terminology**:
  - The **Encoder** contains a **Nx** stack of **EncoderLayers**
    - Inside a single **EncoderLayer** there are two main components (or **sublayers**.
      - The first is the **Multi-Head Attention** and its (Add & norm).
      - The second is the **Feed Forward** (FFN) and its (Add & norm).
    - This is the same for the **Decoder**, except it has an additional **sublayer**, the **Masked Multi-Head Attention**.
    - **Source (X)** the Encoder's input, if we were training the model to translate english to german, the source would be the english tokens.
    - **Target (Y)** the Decoder's input. Note, the connection between the Encoder and Decoder the Source is being passed! Target is being feed to the Decoder where the Outputs are (shifted right). This would be the german tokens.
      - You can think of the Decoder as: It consumes its own previous output (target) while simultaneously cross-referencing the source.
    - **Model size** = d_model = $d_{model}$
    - Sentences and sequences are used interchangeably.

## How To Train Model

```bash
python 
```

## How To Run Inference

```bash
python 
```
