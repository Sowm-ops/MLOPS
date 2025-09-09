import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]
df = pd.read_csv("data/dataset.csv")
X = df[["feature1", "feature2"]]
y = df["label"]

X_train, _, y_train, _ = train_test_split(X, y, train_size=params["split"], random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

with open("data/model.pkl", "wb") as f:
    pickle.dump(model, f)
