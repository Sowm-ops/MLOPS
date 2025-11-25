from src.data_prep import preprocess_text
import re

def test_preprocess_removes_html():
    text = "<div>Hello World</div>"
    out = preprocess_text(text)
    assert "<" not in out and ">" not in out, "HTML not removed"

def test_preprocess_lowercase():
    out = preprocess_text("HELLO WORLD")
    assert out == out.lower(), "Text not converted to lowercase"

def test_preprocess_lemmatization():
    out = preprocess_text("cars")
    # Lemmatizer should convert "cars" → "car"
    assert "car" in out, "Lemmatization failed"

def test_preprocess_removes_stopwords():
    out = preprocess_text("this is a sample")
    assert "this" not in out and "is" not in out, "Stopwords not removed"
