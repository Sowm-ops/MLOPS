import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import json

df = pd.read_csv("data/dataset.csv")
X = df[["feature1", "feature2"]]
y = df["label"]

_, X_test, _, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

with open("data/model.pkl", "rb") as f:
    model = pickle.load(f)

accuracy = model.score(X_test, y_test)

with open("data/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print("Model accuracy:", accuracy)
