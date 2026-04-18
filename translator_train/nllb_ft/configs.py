from datetime import datetime

MODEL_ID = "facebook/nllb-200-distilled-600M"
DATASET_ID = "abdeljalilELmajjodi/darija_english_translation_new"
DATASET_SPLIT = "train"
SRC_LANG = "eng_Latn"
TGT_LANG = "ary_Arab"
FREEZE_ENCODER = True
ENGLISH_COLUMN = "english"
DARIJA_COLUMN = "darija"
MAX_LENGTH = None
TEST_SIZE = 0.1
OUTPUT_DIR = ""
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
LOG_STEPS = 100
EVAL_STEPS = 1000
LEARNING_RATE = 2e-5
W_DECAY = 0.02
RUN_NAME = f"{MODEL_ID.split('/')[-1]}-lr{LEARNING_RATE}-bs{TRAIN_BATCH_SIZE}-{datetime.now().strftime('%m%d-%H%M')}"
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
DEVICE = "cuda"
