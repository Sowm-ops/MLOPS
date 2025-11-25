import os

def test_dvc_outputs_exist():
    assert os.path.exists("metrics/train_metrics.json"), "train_metrics.json missing"
    assert os.path.exists("models"), "models/ folder missing"

def test_model_artifacts():
    files = os.listdir("models")

    # Required core artifacts
    expected = {
        "imdb_best.pkl",
        "heart_best.pkl",
        "imdb_vectorizer.pkl",
        "heart_scaler.pkl"
    }

    missing = [f for f in expected if f not in files]
    assert not missing, f"Missing model artifacts: {missing}"


def test_mlflow_folder_created():
    # In local runs mlruns should exist. In CI/CD, it may not.
    if os.getenv("GITHUB_ACTIONS") == "true":
        assert True  # Skip test in CI
    else:
        assert os.path.exists("mlruns"), "MLflow tracking folder missing"

