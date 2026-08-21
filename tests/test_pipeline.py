"""Tests de entrenamiento y artefactos lógicos."""
from src.config import DATA_RAW
from src.data_loader import load_raw_data, prepare_dataframe
from src.train_pipeline import train_model


def test_train_returns_expected_metrics():
    df = prepare_dataframe(load_raw_data(DATA_RAW)).sample(n=3000, random_state=7)
    _, result = train_model(df, n_iter=1, cv_splits=2, test_size=0.25)
    for metric in ("recall", "precision", "f1", "roc_auc", "pr_auc"):
        assert metric in result["metrics"]
        assert 0.0 <= result["metrics"][metric] <= 1.0
    assert isinstance(result["best_params"], dict)
