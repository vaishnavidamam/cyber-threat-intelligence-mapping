#train.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from collections import Counter

import config
from model import RobertaMultiLabel
from utils import calculate_metrics
from dataset import CTIDataset

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv(config.DATA_PATH)
text_column = df.columns[0]
label_columns = df.columns[1:]

X = df[text_column].values
y = df[label_columns].values
num_labels = len(label_columns)
print(f"Samples: {len(X)}, Labels: {num_labels}")

mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=config.RANDOM_STATE)
for train_idx, temp_idx in mskf.split(X, y):
    break

X_train, X_temp = X[train_idx], X[temp_idx]
y_train, y_temp = y[train_idx], y[temp_idx]

mskf2 = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=config.RANDOM_STATE)
for val_idx, test_idx in mskf2.split(X_temp, y_temp):
    break

X_val, X_test = X_temp[val_idx], X_temp[test_idx]
y_val, y_test = y_temp[val_idx], y_temp[test_idx]

label_combos = [tuple(row) for row in y_train]
combo_counts = Counter(label_combos)
rare_combos = [combo for combo, cnt in combo_counts.items() if cnt < 5]

oversample_texts = []
oversample_labels = []

for idx, combo in enumerate(label_combos):
    if combo in rare_combos:
        oversample_texts.append(X_train[idx])
        oversample_labels.append(y_train[idx])
        oversample_texts.append(X_train[idx])
        oversample_labels.append(y_train[idx])

X_train = np.concatenate([X_train, np.array(oversample_texts)])
y_train = np.concatenate([y_train, np.array(oversample_labels)])
print("Training set after oversampling:", len(X_train))

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)

train_dataset = CTIDataset(X_train, y_train, tokenizer, config.MAX_LEN)
val_dataset = CTIDataset(X_val, y_val, tokenizer, config.MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

model = RobertaMultiLabel(num_labels).to(device)

for name, param in model.roberta.named_parameters():
    if "encoder.layer" in name:
        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
        if layer_num < 6:
            param.requires_grad = False

label_tensor = torch.tensor(y_train)
pos_counts = label_tensor.sum(dim=0)
neg_counts = label_tensor.shape[0] - pos_counts
pos_weight = (neg_counts / (pos_counts + 1e-5)).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

best_f05 = 0
early_counter = 0

for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
    start_time = time.time()
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Avg Train Loss: {avg_train_loss:.4f}")

    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    all_preds = (all_probs > config.THRESHOLD).astype(int)
    metrics = calculate_metrics(all_labels, all_preds)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if metrics["F0.5_score"] > best_f05:
        best_f05 = metrics["F0.5_score"]
        early_counter = 0
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print("Best model saved!")
    else:
        early_counter += 1
        print(f"No improvement. Early stop counter: {early_counter}")
        if early_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Epoch time: {time.time() - start_time:.2f} sec")

print("\nTraining complete.")