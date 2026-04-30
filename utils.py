#utils.py
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, fbeta_score

def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics["Precision"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["F1_score"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["F0.5_score"] = fbeta_score(y_true, y_pred, beta=0.5, average="micro", zero_division=0)
    metrics["Hamming_Loss"] = hamming_loss(y_true, y_pred)
    return metrics