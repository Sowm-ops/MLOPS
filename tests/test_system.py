import os

def test_dvc_outputs_exist():
    assert os.path.exists("metrics/train_metrics.json"), "train_metrics.json missing"
    assert os.path.exists("models"), "models/ folder missing"

def test_model_artifacts():
    files = os.listdir("models")
    assert any("imdb" in f for f in files), "IMDB model artifacts missing"
    assert any("heart" in f for f in files), "Heart model artifacts missing"

def test_mlflow_folder_created():
    # Models logged via mlflow go inside ./mlruns/
    assert os.path.exists("mlruns"), "MLflow tracking folder missing"
