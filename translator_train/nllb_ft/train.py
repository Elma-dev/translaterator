from configs import DARIJA_COLUMN
from configs import MAX_LENGTH
from configs import ENGLISH_COLUMN
from configs import SAVE_TOTAL_LIMIT
from configs import SAVE_STEPS
from configs import RUN_NAME
from configs import W_DECAY
from configs import LEARNING_RATE
from configs import EVAL_STEPS
from configs import LOG_STEPS
from configs import TEST_BATCH_SIZE
from configs import TRAIN_BATCH_SIZE
from configs import OUTPUT_DIR
from configs import TEST_SIZE
from configs import DATASET_SPLIT
from configs import DATASET_ID
from configs import FREEZE_ENCODER
from configs import SRC_LANG
from configs import TGT_LANG
from configs import MODEL_ID
from configs import DEVICE
from transformers import (
    AutoModelForSeq2Seq,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset
from configs import *


def tokenize_dataset(dataset, tokenizer):
    english_token_ids = tokenizer(
        dataset[ENGLISH_COLUMN], padding=True, truncation=True, max_length=MAX_LENGTH
    )

    darija_token_ids = tokenizer(
        dataset[DARIJA_COLUMN], padding=True, truncation=True, max_length=MAX_LENGTH
    )

    final_dataset = Dataset.from_list(
        {
            "input_ids": english_token_ids["input_ids"],
            "attention_mask": english_token_ids["attention_mask"],
            "labels": darija_token_ids["input_ids"],
        }
    )
    return final_dataset


if __name__ == "__main__":
    model = AutoModelForSeq2Seq.from_pretrained(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, src_lang=SRC_LANG, tgt_lang=TGT_LANG
    )

    # freeze encoder
    if FREEZE_ENCODER:
        for params in model.model.encode.parameters():
            params.requires_grad = False

    # load dataset
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    dataset_split = dataset.train_test_split(test_size=TEST_SIZE, seed=42)
    test, train = dataset_split["test"], dataset_split["train"]
    test = tokenize_dataset(test, tokenizer)
    train = tokenize_dataset(train, tokenizer)

    # prepare Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch=TRAIN_BATCH_SIZE,
        per_device_eval_batch=TEST_BATCH_SIZE,
        logging_steps=LOG_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        num_train_epochs=3,
        learning_rate=LEARNING_RATE,
        weight_decay=W_DECAY,
        report_to="trackio",
        predict_with_generate=True,
        run_name=RUN_NAME,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
