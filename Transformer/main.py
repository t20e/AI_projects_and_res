from datasets import load_dataset


from configs.english_german_config import english_german_config
from utils.load_wmt14_en_de_dataset import load_wmt14_en_de, get_training_corpus
from model.bpe_tokenizer import build_and_train_BPE_tokenizer

if __name__ == "__main__":
    # Get Dataset
    dataset = load_wmt14_en_de()
    tokenizer = build_and_train_BPE_tokenizer(get_training_corpus(dataset))
    print(tokenizer)