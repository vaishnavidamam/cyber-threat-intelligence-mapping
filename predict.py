#predict.py
import torch
import pandas as pd
from transformers import RobertaTokenizer
from model import RobertaMultiLabel

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

data = pd.read_csv("data/tram_with_all_labels.csv")

label_columns = data.columns[1:]
id2label = list(label_columns)

NUM_LABELS = len(id2label)

MAX_LEN = 64
THRESHOLD = 0.75
TOP_K = 5

model = RobertaMultiLabel(NUM_LABELS)
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
model.eval()


def predict(text):

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )

    probs = torch.sigmoid(outputs)[0]

    results = []

    for i, p in enumerate(probs):
        if p.item() >= THRESHOLD:
            results.append((id2label[i], p.item()))

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:TOP_K]

    return [(label, prob) for label, prob in results]

if __name__ == "__main__":

    while True:

        sentence = input("\nEnter CTI sentence (type 'exit' to quit): ")

        if sentence.lower() == "exit":
            break

        predictions = predict(sentence)

        print("\nTop Predicted MITRE Techniques:\n")

        if len(predictions) == 0:
            print("No techniques above threshold")
        else:
            for tech in predictions:
                print(tech)