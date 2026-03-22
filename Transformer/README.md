# Implement The Transformer Model From The OG Paper

- Paper → [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- Model will translate English to German.
- Model weights are hosted on [huggingface.co](https://huggingface.co/t20e/Transformer/blob/main/README.md).

Useful Resources:

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformers Step-by-Step Explained by ByteByteGo](https://www.youtube.com/watch?v=avjX3QrYkls)

---

💡 The transformer architecture is used in many SOTA models that process sequential data. However, it can be reconfigured to perform other tasks than just Natural Language Processing (NLP).

![Figure one from paper](./showcase_images/from_paper/main.png)

- ✨ All the model's layers are implemented in their own notebooks in [./model](./model/).

- **Terminology**:
  - The **Encoder** contains a **Nx** stack of **EncoderLayers**
    - Inside a single **EncoderLayer** there are two main components (or **sublayers**).
      - The first is the **Multi-Head Attention** and its (Add & norm).
      - The second is the **Feed Forward** (FFN) and its (Add & norm).
    - This is the same for the **Decoder**, except it has an additional **sublayer**, the **Masked Multi-Head Attention**.
    - **Source (X)** the Encoder's input, if we were training the model to translate english to german, the source would be the english tokens.
    - **Target (Y)** the Decoder's input. Note, the connection between the Encoder and Decoder the Source is being passed! Target is being fed to the Decoder where the Outputs are (shifted right). This would be the german tokens.
      - You can think of the Decoder as: It consumes its own previous output (target) while simultaneously cross-referencing the source.
    - **Model size** = d_model = $d_{model}$

## Installation & Setup

1. **Prerequisites**:
   - Must have Conda installed.

```bash
git clone --filter=blob:none --sparse https://github.com/t20e/AI_projects_and_res.git;
cd AI_projects_and_res;
git sparse-checkout set Transformer;
cd Transformer;
conda env create -f environment.yml;
conda activate transformer_env;
```

### Run Inference Translation With My Pre-Trained Model

```bash
python inference.py
```

### How To Evaluate A Model

- Run [evaluate.ipynb](./evaluate.ipynb)

### How To Train A Model

1. Configure hyperparameters in the `configs/english_german_config.py`.
   - Note: you could use `configs/paper_english_german_config.py`
     - Just change all the `None` values to be something similar to the paper, and edit the other values to fit your hardware.
     - And change all the imports from `from configs.english_german_config import English_german_config` → to → `from configs.paper_english_german_config import Paper_english_german_config`

```bash
python main.py
```
