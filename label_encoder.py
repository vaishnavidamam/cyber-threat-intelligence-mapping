#label_encoder.py
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("data/tram_with_all_labels.csv")

label_columns = df.columns.drop("text")

mlb = MultiLabelBinarizer()
mlb.fit([label_columns])

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("label_encoder.pkl created successfully")
print("Total labels:", len(label_columns))