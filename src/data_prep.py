#!/usr/bin/env python
"""
Dual Dataset Prep – FINAL FIXED VERSION
"""
import os, re, yaml, joblib, nltk
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -------------------------------------------------
# NLTK
# -------------------------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
with open("params.yaml") as f:
    cfg = yaml.safe_load(f) or {}

IMDB_PATH   = "IMDB Dataset.csv"
HEART_PATH  = "heart.csv"
LABEL_IMDB  = cfg["data"]["label_column"]
LABEL_HEART = cfg["data"]["heart_label_column"]
MAX_WORDS   = cfg["preprocess"]["text"]["max_words"]
TEST_SIZE   = cfg["train"]["test_size"]
RS          = cfg["train"]["random_state"]

# -------------------------------------------------
# TEXT PREPROCESS
# -------------------------------------------------
def preprocess_text(text):
    txt = re.sub(r"<[^>]+>", " ", str(text))
    txt = re.sub(r"[^a-zA-Z]", " ", txt).lower()
    txt = re.sub(r"\s+", " ", txt).strip()
    tokens = [lemmatizer.lemmatize(t) for t in word_tokenize(txt) if t not in stop_words]
    return " ".join(tokens)

# -------------------------------------------------
# 1. IMDB DATASET
# -------------------------------------------------
print("Processing IMDB Dataset...")
df_imdb = pd.read_csv(IMDB_PATH)

# Auto-detect text column
text_cols = [c for c in df_imdb.columns if df_imdb[c].dtype == "object" and c != LABEL_IMDB]
if not text_cols:
    raise ValueError("No text column found in IMDB dataset.")
TEXT_COL = text_cols[0]
print(f"  → Text column: '{TEXT_COL}'")

X_imdb = df_imdb.drop(columns=[LABEL_IMDB])
y_imdb = df_imdb[LABEL_IMDB]

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imdb, y_imdb, test_size=TEST_SIZE, random_state=RS, stratify=y_imdb
)

# === SAVE TEXT BEFORE DROPPING ===
train_text_raw = X_train_i[TEXT_COL].copy()
test_text_raw  = X_test_i[TEXT_COL].copy()

# === PREPROCESS TEXT ===
train_text_clean = train_text_raw.apply(preprocess_text)
test_text_clean  = test_text_raw.apply(preprocess_text)

# === TF-IDF ON CLEANED TEXT ===
vec = TfidfVectorizer(max_features=MAX_WORDS)
train_tfidf = vec.fit_transform(train_text_clean).toarray()
test_tfidf  = vec.transform(test_text_clean).toarray()

tfidf_train = pd.DataFrame(train_tfidf, index=X_train_i.index, columns=[f"tfidf_{i}" for i in range(MAX_WORDS)])
tfidf_test  = pd.DataFrame(test_tfidf,  index=X_test_i.index,  columns=[f"tfidf_{i}" for i in range(MAX_WORDS)])

# === DROP ORIGINAL TEXT COLUMN ===
X_train_i = X_train_i.drop(columns=[TEXT_COL])
X_test_i  = X_test_i.drop(columns=[TEXT_COL])

# === CONCAT TF-IDF ===
X_train_i = pd.concat([X_train_i, tfidf_train], axis=1)
X_test_i  = pd.concat([X_test_i,  tfidf_test],  axis=1)

# === ADD CLEANED TEXT (NOW SAFE) ===
X_train_i["text_clean"] = train_text_clean
X_test_i["text_clean"]   = test_text_clean

# -------------------------------------------------
# 2. HEART DATASET
# -------------------------------------------------
print("Processing Heart Disease Dataset...")
df_heart = pd.read_csv(HEART_PATH)
X_heart = df_heart.drop(columns=[LABEL_HEART])
y_heart = df_heart[LABEL_HEART]

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart, y_heart, test_size=TEST_SIZE, random_state=RS, stratify=y_heart
)

scaler = StandardScaler()
X_train_h_scaled = pd.DataFrame(scaler.fit_transform(X_train_h), columns=X_train_h.columns, index=X_train_h.index)
X_test_h_scaled  = pd.DataFrame(scaler.transform(X_test_h),      columns=X_test_h.columns,  index=X_test_h.index)

# -------------------------------------------------
# 3. SAVE ALL
# -------------------------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# IMDB
pd.concat([X_train_i, y_train_i], axis=1).to_csv("data/imdb_train.csv", index=False)
pd.concat([X_test_i,  y_test_i],  axis=1).to_csv("data/imdb_test.csv",  index=False)
joblib.dump(vec, "models/imdb_vectorizer.pkl")

# Heart
pd.concat([X_train_h_scaled, y_train_h], axis=1).to_csv("data/heart_train.csv", index=False)
pd.concat([X_test_h_scaled,  y_test_h],  axis=1).to_csv("data/heart_test.csv",  index=False)
joblib.dump(scaler, "models/heart_scaler.pkl")

# Legacy
pd.concat([X_train_i, y_train_i], axis=1).to_csv("data/train.csv", index=False)
pd.concat([X_test_i,  y_test_i],  axis=1).to_csv("data/test.csv",  index=False)

print("\nDATA PREP SUCCESSFUL!")
print("   data/imdb_train.csv")
print("   data/imdb_test.csv")
print("   data/heart_train.csv")
print("   data/heart_test.csv")
print("   models/imdb_vectorizer.pkl")
print("   models/heart_scaler.pkl")