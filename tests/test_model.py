import joblib
import pandas as pd
import numpy as np
import os

def test_imdb_model_exists():
    assert os.path.exists("models/imdb_best.pkl"), "IMDB model missing"
    assert os.path.exists("models/imdb_vectorizer.pkl"), "IMDB vectorizer missing"

def test_heart_model_exists():
    assert os.path.exists("models/heart_best.pkl"), "Heart model missing"
    assert os.path.exists("models/heart_scaler.pkl"), "Heart scaler missing"

def test_heart_model_prediction():
    model = joblib.load("models/heart_best.pkl")
    df = pd.read_csv("data/heart_test.csv")
    X = df.drop(columns=["target"])
    pred = model.predict(X)

    assert len(pred) == len(X), "Prediction length mismatch"
    assert not np.isnan(pred).any(), "Heart model produced NaN predictions"

def test_imdb_model_prediction():
    model = joblib.load("models/imdb_best.pkl")
    df = pd.read_csv("data/imdb_test.csv")
    X = df.drop(columns=["sentiment"])

    pred = model.predict(X)
    assert len(pred) == len(X), "IMDB prediction length mismatch"
