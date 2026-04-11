import logging

import torch
from configs import *
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format
from utils import format_to_chat
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if torch.cuda.is_available():
        logger.info(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Load tokenizer/model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

    logger.info("Tokenizer setup chat template...")
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    logger.info("Load dataset...")
    dataset = load_dataset(DATASET_REPO, split=DATASET_CONFIG)
    print(f"Dataset: {dataset}")

    logger.info("Format dataset...")
    train_dataset = dataset.map(format_to_chat)

    logger.info("Prepare configs...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_BATCH_SIZE,
        gradient_checkpointing=True,  # Changed to False
        learning_rate=LEARNING_RATE,  # Curriculum Learning
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        optim=OPTIM,
        # T4: FP16
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=OUTPUT_MODEL_ID,
        # hub_strategy="every_save",
        dataset_text_field=DATASET_TEXT_FIELD,
        packing=PACKING,
        report_to=REPORT_TO,  # noqa: F405
    )

    logger.info("Prepare Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    logger.info("Start training...")
    trainer.train()

    logger.info("Training Done...")

    logger.info("💾 PUSH MODEL To Hub...")

    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

    trainer.push_to_hub(OUTPUT_MODEL_ID)
