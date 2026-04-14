from datetime import datetime

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
DATASET_REPO = "abdeljalilELmajjodi/darija_english_translation_new"
DATASET_CONFIG = "train"
OUTPUT_DIR = "LFM_2.5_1.2b_Translator"
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.05
OPTIM = "paged_adamw_32bit"
BF16 = True
LOGGING_STEPS = 10
# evaluation
EVAL_STRATEGY = "steps"
EVAL_STEPS = 200
EVAL_SPLIT_SIZE = 0.05  # 5% held out for eval
MAX_EVAL_SAMPLES = 1500
SAVE_STRATEGY = "steps"
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 2


OUTPUT_MODEL_ID = (
    f"abdeljalilELmajjodi/translator_{LEARNING_RATE}_{PER_DEVICE_TRAIN_BATCH_SIZE}"
)
MAX_GRAD_NORM = 1.0
PUSH_TO_HUB = False
# hub_model_id=OUTPUT_MODEL_ID,
# hub_strategy="every_save",
DATASET_TEXT_FIELD = "messages"
PACKING = False
REPORT_TO = "trackio"
RUN_NAME = f"{MODEL_ID.split('/')[-1]}-lr{LEARNING_RATE}-bs{PER_DEVICE_TRAIN_BATCH_SIZE}-{datetime.now().strftime('%m%d-%H%M')}"
