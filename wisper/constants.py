BASE_MODEL_ID = "ivrit-ai/whisper-large-v3"
LANGUAGE = "he"
TASK = "transcribe"

DATASET_DIR = "./dataset_whisper"
ROW_DATA_DIR = "./row_data"
MODEL_DIR = "./models/whisper-large-v3-he"

TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-5,
    "warmup_steps": 100,
    "num_train_epochs": 3,
    "bf16": True,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
    "logging_steps": 25,
    "predict_with_generate": True,
    "per_device_eval_batch_size": 2,
    "eval_accumulation_steps": 4,
    "generation_max_length": 225,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "remove_unused_columns": False,
}
