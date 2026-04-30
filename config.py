#config.py
MODEL_NAME = "roberta-base"

MAX_LEN = 64            
BATCH_SIZE = 4            
EPOCHS = 3                 
LEARNING_RATE = 2e-5
DROPOUT = 0.3
RANDOM_STATE = 42

THRESHOLD = 0.7           
MODEL_SAVE_PATH = "models/best_model.pt"
DATA_PATH = "data/tram_with_all_labels.csv"

EARLY_STOPPING_PATIENCE = 2

GRADIENT_ACCUMULATION_STEPS = 4