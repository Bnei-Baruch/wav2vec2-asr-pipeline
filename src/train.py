import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, Audio
import evaluate
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

# MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
# MODEL_ID = "facebook/mms-1b "
MODEL_ID = "facebook/wav2vec2-base"
USE_LOCAL_DATA = True
LOCAL_DATA_DIR = "./dataset"
OUTPUT_DIR = "./models/wav2vec2-large-xlsr-custom"

def train():
    print("Load Dataset")
    print(f"Loading local dataset from {LOCAL_DATA_DIR}...")
    dirs = [os.path.join(LOCAL_DATA_DIR, uid) for uid in os.listdir(LOCAL_DATA_DIR)]
    print(f"Dirs: {dirs}")
    all_datasets = [load_dataset("audiofolder", data_dir=d)["train"] for d in dirs]
    dataset = concatenate_datasets(all_datasets)
    print(f"Dataset: {dataset}")

    print("Text Preprocessing")
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\]]'

    def remove_special_characters(batch):
        print(f"Batch dataset: {batch}")
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
        return batch

    dataset = dataset.map(remove_special_characters)

    print("Create Vocabulary")
    def extract_all_chars(batch):
        print(f"Batch vocab: {batch}")
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    print("Extract All Chars")
    # vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names)
    vocabs = dataset.map(extract_all_chars, batched=True, batch_size=500, keep_in_memory=False, remove_columns=dataset.column_names)
   
    print("Create Vocab Dict")
    vocab_list = list(set(vocabs["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    print("Create Processor")
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    print("Prepare Audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1)

    print("Create Data Collator")
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.tokenizer.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    print("Create Metric")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    print("Create Model")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()

    print("Create Trainer")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        #per_device_train_batch_size=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        #num_train_epochs=30,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        #save_steps=500,
        save_steps=50,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset,
        eval_dataset=dataset, 
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    train()
