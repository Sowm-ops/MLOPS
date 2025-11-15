#!/usr/bin/env python
"""
Trains best model for IMDB and Heart separately.
Saves: models/imdb_best.pkl  and  models/heart_best.pkl
"""
import os, yaml, joblib, mlflow, mlflow.sklearn
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
with open("params.yaml") as f:
    cfg = yaml.safe_load(f) or {}

LABEL_IMDB  = cfg["data"]["label_column"]
LABEL_HEART = cfg["data"]["heart_label_column"]
SAMPLE_SZ   = cfg["train"]["sample_size"]
RS          = cfg["train"]["random_state"]
CV          = cfg["train"]["cv_folds"]
MODELS_CFG  = cfg["models"]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_sample(df, n, label):
    if n >= len(df): return df.copy()
    pos = df[df[label] == 1]
    neg = df[df[label] == 0]
    n_pos = min(len(pos), n // 2)
    n_neg = min(len(neg), n // 2)
    if n_pos == 0 or n_neg == 0: return df.copy()
    return pd.concat([pos.sample(n_pos, random_state=RS),
                      neg.sample(n_neg, random_state=RS)]).sample(frac=1, random_state=RS)

def get_model(name):
    if name == "lr":       return LogisticRegression(max_iter=300, n_jobs=-1, random_state=RS)
    if name == "linearsvc":return LinearSVC(max_iter=2000, random_state=RS)
    if name == "xgb":      return xgb.XGBClassifier(random_state=RS, n_jobs=-1)
    if name == "gbm":      return GradientBoostingClassifier(random_state=RS)
    raise ValueError(name)

# -------------------------------------------------
# TRAIN ONE DATASET
# -------------------------------------------------
def train_one(prefix, train_path, test_path, label_col):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Map labels
    if label_col == LABEL_IMDB:
        train_df[label_col] = train_df[label_col].str.lower().map({"positive":1, "negative":0})
        test_df[label_col]  = test_df[label_col].str.lower().map({"positive":1, "negative":0})
    train_df = train_df.dropna(subset=[label_col])
    test_df  = test_df.dropna(subset=[label_col])

    # Sampling
    train_s = safe_sample(train_df, SAMPLE_SZ, label_col)
    X_train = train_s.drop(columns=[label_col])
    y_train = train_s[label_col]
    X_test  = test_df.drop(columns=[label_col])
    y_test  = test_df[label_col]

    # Drop raw text / clean columns
    drop = [c for c in X_train.columns if X_train[c].dtype == "object" and not c.startswith("tfidf_")]
    drop += [c for c in X_train.columns if c.endswith("_clean")]
    X_train = X_train.drop(columns=drop, errors="ignore")
    X_test  = X_test.drop(columns=drop, errors="ignore")

    # Encode categoricals
    encoders = {}
    for col in X_train.select_dtypes("object"):
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col]  = X_test[col].astype(str).map(lambda x: x if x in le.classes_ else "<UNK>")
        le.classes_ = np.append(le.classes_, "<UNK>")
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le
        joblib.dump(le, f"models/encoder_{col}.pkl")

    # Grid search
    mlflow.set_experiment(f"{prefix}_experiment")
    best_acc = 0
    best_mod = None
    best_name = ""
    with mlflow.start_run():
        for name, mc in MODELS_CFG.items():
            if not mc.get("enabled"): continue
            print(f"\n{prefix.upper()} – {name.upper()}")
            mod = get_model(name)
            grid = GridSearchCV(mod, mc["params"], cv=CV, scoring="accuracy", n_jobs=-1)
            grid.fit(X_train, y_train)
            acc = accuracy_score(y_test, grid.predict(X_test))
            mlflow.log_metric(f"{name}_acc", acc)
            if acc > best_acc:
                best_acc, best_mod, best_name = acc, grid.best_estimator_, name
        # Save
        path = Path("models") / f"{prefix}_best.pkl"
        joblib.dump(best_mod, path)
        mlflow.sklearn.log_model(best_mod, f"{prefix}_model")
        print(f"{prefix.upper()} best → {best_name} ({best_acc:.4f})")

# -------------------------------------------------
# RUN BOTH
# -------------------------------------------------
train_one("imdb", "data/imdb_train.csv", "data/imdb_test.csv", LABEL_IMDB)
train_one("heart","data/heart_train.csv","data/heart_test.csv", LABEL_HEART)