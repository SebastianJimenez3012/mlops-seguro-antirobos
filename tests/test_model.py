"""Tests del pipeline de ML en modo rápido."""
import numpy as np
import pytest

from src.config import DATA_RAW, FEATURES
from src.data_loader import load_raw_data, prepare_dataframe
from src.train_pipeline import train_model


@pytest.fixture(scope="module")
def trained_model():
    df = prepare_dataframe(load_raw_data(DATA_RAW))
    sample = df.sample(n=4000, random_state=42)
    model, _ = train_model(sample, n_iter=1, cv_splits=2, test_size=0.25)
    return model, sample


def test_model_exposes_predict_and_predict_proba(trained_model):
    model, _ = trained_model
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_probabilities_are_valid(trained_model):
    model, sample = trained_model
    proba = model.predict_proba(sample[FEATURES].head(20))
    assert proba.shape == (20, 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
