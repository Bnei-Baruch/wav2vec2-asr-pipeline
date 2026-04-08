import json
import os
from .utils import load_dataset_from_dir
from .constants import VOCAB_PATH

def prepare_vocab():
    dataset = load_dataset_from_dir()

    if os.path.exists(VOCAB_PATH):
        os.remove(VOCAB_PATH)

    print("Create Vocabulary")

    def extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab]}

    print("Extract All Chars")
    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=1000,
        keep_in_memory=False,
        remove_columns=dataset.column_names,
    )

    print("Create Vocab Dict")
    vocab_set = set()
    for v in vocabs["vocab"]:
        vocab_set.update(v)
    vocab_list = sorted(vocab_set)

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(VOCAB_PATH, "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
    print(f"Vocab saved to {VOCAB_PATH}")


if __name__ == "__main__":
    prepare_vocab()
