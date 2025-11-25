import pandas as pd
import os

def test_imdb_processed_exists():
    assert os.path.exists("data/imdb_train.csv"), "Missing imdb_train.csv"
    assert os.path.exists("data/imdb_test.csv"), "Missing imdb_test.csv"

def test_heart_processed_exists():
    assert os.path.exists("data/heart_train.csv"), "Missing heart_train.csv"
    assert os.path.exists("data/heart_test.csv"), "Missing heart_test.csv"

def test_imdb_schema():
    df = pd.read_csv("data/imdb_train.csv")
    # Ensure TF-IDF columns exist
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    assert len(tfidf_cols) == 1500, f"Expected 1500 TFIDF features, found {len(tfidf_cols)}"

    # Label column
    assert "sentiment" in df.columns
    assert df["sentiment"].isin([0, 1]).all()

def test_heart_schema():
    df = pd.read_csv("data/heart_train.csv")
    assert "target" in df.columns, "Heart label column missing"
    assert df["target"].isin([0, 1]).all(), "Heart labels must be 0 or 1"
    assert df.drop(columns=["target"]).isna().sum().sum() == 0, "Heart features contain missing values"
