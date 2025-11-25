import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def test_imdb_drift():
    train = pd.read_csv("data/imdb_train.csv")
    test = pd.read_csv("data/imdb_test.csv")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train, current_data=test)
    result = report.as_dict()

    assert "metrics" in result, "Drift report malformed"
