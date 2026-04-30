import pandas as pd
from sklearn.metrics import precision_score, recall_score

# Replace these with your actual outputs
# y_true = actual labels from dataset
# y_pred = predicted labels from your model

# Example structure (replace this!)
# y_true = [...]
# y_pred = [...]

categories = sorted(list(set(y_true)))  # keeps order clean

results = []

for cat in categories:
    y_true_binary = [1 if y == cat else 0 for y in y_true]
    y_pred_binary = [1 if y == cat else 0 for y in y_pred]

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

    results.append([
        cat,
        round(precision * 100),
        round(recall * 100)
    ])

# Create final table
df = pd.DataFrame(results, columns=["Tactic Category", "Precision (%)", "Recall (%)"])

print(df)