# Trained Model Info

- Trained on only **20%** of the WMT 2014 English-German dataset for 15 epochs.

- Took **~17 hours** to train on a M1 Mac with 32 core GPU and 64GB RAM.
- It's not a SOTA model, but it's a good baseline.

- **My Model's BLEU** score: **26.01**
  - The paper's base model achieved a BLEU score of **27.3 BLEU** on the same dataset.

Last step print:

```shell
[03-21 11:05:59] Step: 54463/54495 | Loss: 2.6400 | Tokens: 26339688 | Time Elapsed: 16:56:27 | ETA: 0:00:35

Epoch [15/15] completed. Average Loss: 2.6815
```

![Training Loss](./showcase_images/training_loss_plot_20_percent_ds.png)

- **Loss Image**: training_loss_plot_20_percent_ds.png

**Model's Config:**

```python
d_model = 512
H = 8
dropout = 0.1
d_ff = 2048
N = 6

pos_seq_len = 5000  
max_indiv_seq_len = 128 
max_batch_seq_tokens = 8_000  
vocab_size = 37_000  
num_epochs = 15
```

**Translation example:**

```bash
English you want to translate to German: The cat is fat
# Because of not adding a period at the end of the sentence, the model got the sentence wrong.
German Translation: Die Katze ist tödlich.

English you want to translate to German: The cat is fat.
# Adding a period at the end of the sentence, the model got the sentence right.

German Translation: Die Katze ist Fett.
```
