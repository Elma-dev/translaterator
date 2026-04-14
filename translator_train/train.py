import logging

import numpy as np
import torch
from configs import *
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer  # , setup_chat_format
from utils import format_to_chat
from dotenv import load_dotenv

try:
    import sacrebleu
except ImportError:
    raise ImportError("Run: pip install sacrebleu")

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if torch.cuda.is_available():
        logger.info(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Load tokenizer/model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

    logger.info("Tokenizer setup chat template...")
    # if tokenizer.chat_template is None:
    #    model, tokenizer = setup_chat_format(model, tokenizer)

    logger.info("Load dataset...")
    dataset = load_dataset(DATASET_REPO, split=DATASET_CONFIG)
    print(f"Dataset: {dataset}")

    logger.info("Split dataset into train/eval...")
    split = dataset.train_test_split(test_size=EVAL_SPLIT_SIZE, seed=42)
    train_raw, eval_raw = split["train"], split["test"]

    # Cap eval size to avoid slow/OOM eval passes on large datasets
    if len(eval_raw) > MAX_EVAL_SAMPLES:
        eval_raw = eval_raw.select(range(MAX_EVAL_SAMPLES))
        logger.info(f"Eval set capped at {MAX_EVAL_SAMPLES} samples")
    else:
        logger.info(f"Eval set size: {len(eval_raw)} samples")

    logger.info("Format dataset...")
    train_dataset = train_raw.map(format_to_chat)
    eval_dataset = eval_raw.map(format_to_chat)

    logger.info("Build compute_metrics...")

    def compute_metrics(eval_pred):
        """Compute chrF and BLEU from teacher-forced predictions."""
        logits, labels = eval_pred
        # Argmax over vocab dim to get predicted token ids
        predictions = np.argmax(logits, axis=-1)

        decoded_preds, decoded_labels = [], []
        for pred_ids, label_ids in zip(predictions, labels):
            # Mask out -100 (ignored) positions
            valid_mask = label_ids != -100
            pred_tokens = pred_ids[valid_mask]
            label_tokens = label_ids[valid_mask]
            decoded_preds.append(
                tokenizer.decode(pred_tokens, skip_special_tokens=True)
            )
            decoded_labels.append(
                tokenizer.decode(label_tokens, skip_special_tokens=True)
            )

        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        return {
            "chrF": round(chrf.score, 4),
            "BLEU": round(bleu.score, 4),
        }

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
        max_grad_norm=MAX_GRAD_NORM,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=OUTPUT_MODEL_ID,
        # hub_strategy="every_save",
        dataset_text_field=DATASET_TEXT_FIELD,
        packing=PACKING,
        report_to=REPORT_TO,  # noqa: F405
        run_name=RUN_NAME,
    )

    logger.info("Prepare Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Start training...")
    trainer.train()

    logger.info("Training Done...")

    logger.info("💾 PUSH MODEL To Hub...")

    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

    trainer.push_to_hub(OUTPUT_MODEL_ID)
