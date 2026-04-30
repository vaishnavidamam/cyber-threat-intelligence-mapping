#evaluate.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score

import config
from dataset import CTIDataset
from model import RobertaMultiLabel
from utils import calculate_metrics



TEMPERATURE = 1.5


def find_best_thresholds(labels, probs):

    n_labels = labels.shape[1]
    thresholds = np.zeros(n_labels)

    for i in range(n_labels):

        best_t = 0.5
        best_f1 = 0

        for t in np.arange(0.1, 0.9, 0.05):

            preds = (probs[:, i] > t).astype(int)

            f1 = f1_score(labels[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[i] = best_t

    return thresholds


def main():

    print("Loading dataset...")

    df = pd.read_csv(config.DATA_PATH)

    text_column = df.columns[0]
    label_cols = df.columns[1:]

    texts = df[text_column].values
    labels = df[label_cols].values

    print("Total samples:", len(df))

    tokenizer = RobertaTokenizer.from_pretrained(
        config.MODEL_NAME,
        local_files_only=True
    )

    dataset = CTIDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Loading trained model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobertaMultiLabel(num_labels=len(label_cols))

    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    print("\nStarting evaluation...\n")

    with torch.no_grad():

        progress_bar = tqdm(loader, desc="Evaluating", ncols=100)

        for batch in progress_bar:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids, attention_mask)

           
            logits = outputs / TEMPERATURE

            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.extend(labels)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\nFinding best threshold per label...")

    thresholds = find_best_thresholds(all_labels, all_probs)

    print("Best thresholds:", thresholds)

    all_preds = (all_probs > thresholds).astype(int)

    metrics = calculate_metrics(all_labels, all_preds)

    print("\n==============================")
    print("Final Evaluation Metrics")
    print("==============================\n")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()