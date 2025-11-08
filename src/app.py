import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px

st.set_page_config(page_title="MLOps Prediction UI", layout="wide")
st.title("MLOps: SVM Prediction Analysis")

# Load local data & model
@st.cache_data
def load_all():
    test_df = pd.read_csv('data\\test.csv')
    model = joblib.load('models\\best_model.pkl')
    return test_df, model

test_df, model = load_all()
X_test = test_df[['feature1', 'feature2']]
y_test = test_df['label']
y_pred = model.predict(X_test)

col1, col2 = st.columns(2)
with col1:
    st.metric("Test Accuracy", f"{model.score(X_test, y_test):.3f}")
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
with col2:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

# Real-time prediction
st.subheader("New Prediction")
col3, col4 = st.columns(2)
with col3:
    feat1 = st.number_input("Feature 1", value=2.0, step=0.1)
with col4:
    feat2 = st.number_input("Feature 2", value=3.0, step=0.1)
if st.button("Predict Label"):
    pred = model.predict([[feat1, feat2]])[0]
    prob = model.predict_proba([[feat1, feat2]])[0]
    st.success(f"Predicted Label: {pred}")
    st.info(f"Probabilities: Class 0 = {prob[0]:.3f}, Class 1 = {prob[1]:.3f}")

st.caption("MLflow UI: http://localhost:5000 | Model from DVC pipeline")