import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset
from configs import *

# Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler()],
# )
logger = logging.getLogger(__name__)


def tokenize_dataset(dataset_obj, tokenizer):
    logger.info(f"Tokenizing dataset with {len(dataset_obj)} examples...")

    def preprocess_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples[ENGLISH_COLUMN],
            max_length=MAX_LENGTH or 300,  # Default to 128 if MAX_LENGTH is None
            truncation=True,
        )

        # Tokenize targets
        labels = tokenizer(
            text_target=examples[DARIJA_COLUMN],
            max_length=MAX_LENGTH or 128,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset_obj.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_obj.column_names,
        desc="Running tokenizer on dataset",
    )
    return tokenized_dataset


if __name__ == "__main__":
    logger.info(f"Loading model and tokenizer: {MODEL_ID}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, src_lang=SRC_LANG, tgt_lang=TGT_LANG
    )

    # freeze encoder
    if FREEZE_ENCODER:
        logger.info("Freezing encoder parameters...")
        # NLLB/M2M100 models have 'model.encoder'
        for params in model.get_encoder().parameters():
            params.requires_grad = False

    # load dataset
    logger.info(f"Loading dataset: {DATASET_ID} (split: {DATASET_SPLIT})")
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)

    logger.info(f"Splitting dataset (test_size={TEST_SIZE})...")
    dataset_split = dataset.train_test_split(test_size=TEST_SIZE, seed=42)
    test_raw, train_raw = dataset_split["test"], dataset_split["train"]

    logger.info("Preparing tokenized datasets...")
    test = tokenize_dataset(test_raw, tokenizer)
    train = tokenize_dataset(train_raw, tokenizer)

    # prepare Trainer
    logger.info("Setting up Seq2SeqTrainingArguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TEST_BATCH_SIZE,
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

    logger.info("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete.")
