#!/usr/bin/env python
"""
Streamlit UI – IMDB + Heart Disease + Ensemble
FULLY FIXED & WORKING
"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import yaml
import re
import nltk
import plotly.express as px
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# NLTK
# -------------------------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
with open("params.yaml") as f:
    cfg = yaml.safe_load(f) or {}

LABEL_IMDB  = cfg["data"]["label_column"]
LABEL_HEART = cfg["data"]["heart_label_column"]
MAX_WORDS   = cfg["preprocess"]["text"]["max_words"]

# -------------------------------------------------
# LOAD ARTIFACTS (cached)
# -------------------------------------------------
@st.cache_resource
def load_artifact(prefix, name):
    path = Path("models") / f"{prefix}_{name}.pkl"
    if not path.exists():
        st.error(f"{prefix.upper()} {name} not found – run `train.py` first.")
        return None
    return joblib.load(path)

model_imdb   = load_artifact("imdb",  "best")
model_heart  = load_artifact("heart", "best")
vec_imdb     = load_artifact("imdb",  "vectorizer")
scaler_heart = load_artifact("heart", "scaler")

# -------------------------------------------------
# LOAD TEST DATA
# -------------------------------------------------
@st.cache_data
def load_test(prefix):
    df = pd.read_csv(f"data/{prefix}_test.csv")
    label = LABEL_IMDB if prefix == "imdb" else LABEL_HEART
    if prefix == "imdb":
        df[label] = df[label].map({"positive": 1, "negative": 0})
    df = df.dropna(subset=[label])
    return df

test_imdb  = load_test("imdb")  if model_imdb else None
test_heart = load_test("heart") if model_heart else None

# -------------------------------------------------
# TEXT PREPROCESS (same as data_prep)
# -------------------------------------------------
def preprocess_text(txt):
    txt = re.sub(r"<[^>]+>", " ", str(txt))
    txt = re.sub(r"[^a-zA-Z]", " ", txt).lower()
    txt = re.sub(r"\s+", " ", txt).strip()
    tokens = [lemmatizer.lemmatize(t) for t in word_tokenize(txt) if t not in stop_words]
    return " ".join(tokens)

# -------------------------------------------------
# CLEAN TEST DATA (drop non-feature columns)
# -------------------------------------------------
def clean_features(df, label_col, is_imdb=False):
    X = df.drop(columns=[label_col])
    drop_cols = [c for c in X.columns if X[c].dtype == "object" and not c.startswith("tfidf_")]
    drop_cols += [c for c in X.columns if c.endswith("_clean")]
    return X.drop(columns=drop_cols, errors="ignore")

# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(page_title="IMDB & Heart Predictor", layout="wide")
st.title("Hybrid MLOps: IMDB + Heart Disease + Ensemble")

dataset = st.radio(
    "Select Mode",
    ["IMDB (Text)", "Heart Disease (Numeric)", "Both (Ensemble)"],
    horizontal=True
)

# -------------------------------------------------------------------------
# 1. IMDB ONLY
# -------------------------------------------------------------------------
if dataset == "IMDB (Text)":
    if not (model_imdb and vec_imdb):
        st.stop()

    review = st.text_area("Enter movie review", height=150, value="Amazing film!")
    if st.button("Predict Sentiment", type="primary"):
        clean = preprocess_text(review)
        vec = vec_imdb.transform([clean]).toarray()[0]
        row = {f"tfidf_{i}": vec[i] for i in range(MAX_WORDS)}
        df_in = pd.DataFrame([row]).reindex(columns=clean_features(test_imdb, LABEL_IMDB).columns, fill_value=0)
        pred = int(model_imdb.predict(df_in)[0])
        prob = model_imdb.predict_proba(df_in)[0]
        st.success(f"**Sentiment: {'Positive' if pred else 'Negative'}**")
        c1, c2 = st.columns(2)
        c1.metric("Negative", f"{prob[0]:.1%}")
        c2.metric("Positive", f"{prob[1]:.1%}")

# -------------------------------------------------------------------------
# 2. HEART ONLY
# -------------------------------------------------------------------------
elif dataset == "Heart Disease (Numeric)":
    if not (model_heart and scaler_heart):
        st.stop()

    st.subheader("Patient Features")
    cols = [c for c in test_heart.columns if c != LABEL_HEART]
    inputs = {}
    for c in cols:
        mean_val = test_heart[c].mean()
        inputs[c] = st.number_input(c, value=float(mean_val), step=0.1, key=f"heart_{c}")

    if st.button("Predict Heart Disease", type="primary"):
        df_in = pd.DataFrame([inputs])
        df_in = pd.DataFrame(scaler_heart.transform(df_in), columns=cols)
        pred = int(model_heart.predict(df_in)[0])
        prob = model_heart.predict_proba(df_in)[0]
        st.success(f"**Heart Disease: {'Present' if pred else 'Absent'}**")
        c1, c2 = st.columns(2)
        c1.metric("Absent", f"{prob[0]:.1%}")
        c2.metric("Present", f"{prob[1]:.1%}")

# -------------------------------------------------------------------------
# 3. ENSEMBLE
# -------------------------------------------------------------------------
else:
    if not (model_imdb and vec_imdb and model_heart and scaler_heart):
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("IMDB Review")
        review = st.text_area("Movie review", height=120, value="Great acting!", key="review_ens")
    with col2:
        st.subheader("Heart Patient")
        heart_inputs = {}
        for c in test_heart.columns:
            if c == LABEL_HEART: continue
            mean_val = test_heart[c].mean()
            heart_inputs[c] = st.number_input(c, value=float(mean_val), step=0.1, key=f"ens_h_{c}")

    if st.button("Predict Ensemble", type="primary"):
        # IMDB
        clean = preprocess_text(review)
        vec = vec_imdb.transform([clean]).toarray()[0]
        imdb_row = {f"tfidf_{i}": vec[i] for i in range(MAX_WORDS)}
        imdb_df = pd.DataFrame([imdb_row]).reindex(columns=clean_features(test_imdb, LABEL_IMDB).columns, fill_value=0)
        imdb_prob = model_imdb.predict_proba(imdb_df)[0]

        # Heart
        heart_df = pd.DataFrame([heart_inputs])
        heart_df = pd.DataFrame(scaler_heart.transform(heart_df), columns=heart_inputs.keys())
        heart_prob = model_heart.predict_proba(heart_df)[0]

        # Ensemble (average)
        ens_prob = (imdb_prob + heart_prob) / 2
        ens_pred = int(ens_prob.argmax())

        st.success(f"**Ensemble: {'Positive / Present' if ens_pred else 'Negative / Absent'}**")
        c1, c2 = st.columns(2)
        c1.metric("Negative / Absent", f"{ens_prob[0]:.1%}")
        c2.metric("Positive / Present", f"{ens_prob[1]:.1%}")

# -------------------------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("Model Performance")

if model_imdb and test_imdb is not None:
    X_i_clean = clean_features(test_imdb, LABEL_IMDB)
    y_i = test_imdb[LABEL_IMDB].astype(int)
    acc_i = accuracy_score(y_i, model_imdb.predict(X_i_clean))
    st.metric("IMDB Test Accuracy", f"{acc_i:.1%}")

if model_heart and test_heart is not None:
    X_h = test_heart.drop(columns=[LABEL_HEART])
    y_h = test_heart[LABEL_HEART].astype(int)
    acc_h = accuracy_score(y_h, model_heart.predict(X_h))
    st.metric("Heart Test Accuracy", f"{acc_h:.1%}")

colA, colB = st.columns(2)
with colA:
    if model_imdb and test_imdb is not None:
        y_pred_i = model_imdb.predict(X_i_clean)
        rep = classification_report(y_i, y_pred_i, output_dict=True, target_names=["Negative","Positive"])
        st.dataframe(pd.DataFrame(rep).transpose())
with colB:
    if model_heart and test_heart is not None:
        y_pred_h = model_heart.predict(X_h)
        rep = classification_report(y_h, y_pred_h, output_dict=True, target_names=["Absent","Present"])
        st.dataframe(pd.DataFrame(rep).transpose())