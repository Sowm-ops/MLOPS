import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Local MLflow (default ./mlruns)
mlflow.set_experiment("local_mlops_demo")

# Load params
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Load local data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
X_train, y_train = train_df[['feature1', 'feature2']], train_df['label']
X_test, y_test = test_df[['feature1', 'feature2']], test_df['label']

with mlflow.start_run(run_name="local_svm_tuning"):
    # Example: SVM with expanded params
    model = SVC(random_state=42, probability=True)
    param_grid = {
        'C': params['models']['svm']['C'],
        'kernel': params['models']['svm']['kernel']
    }
    grid = GridSearchCV(model, param_grid, cv=params['train']['cv_folds'], scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Local logging
    mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
    mlflow.log_metric("cv_score", grid.best_score_)
    y_pred = grid.predict(X_test)
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    mlflow.sklearn.log_model(grid.best_estimator_, "local_model")

    # Save local model file
    joblib.dump(grid.best_estimator_, 'models/best_model.pkl')
    print(f"Local model saved: accuracy {accuracy_score(y_test, y_pred):.3f}")