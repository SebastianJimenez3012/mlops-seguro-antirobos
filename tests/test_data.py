"""Tests del contrato y preparación de datos."""
import pandas as pd

from src.config import DATA_RAW, TARGET
from src.data_loader import load_raw_data, prepare_dataframe


def test_raw_dataset_has_expected_size_order_of_magnitude():
    df = load_raw_data(DATA_RAW)
    assert len(df) > 20_000


def test_prepare_target_is_binary_and_not_null():
    clean = prepare_dataframe(load_raw_data(DATA_RAW))
    assert set(clean[TARGET].unique()).issubset({0, 1})
    assert clean[TARGET].isna().sum() == 0


def test_prepare_removes_duplicates():
    raw = load_raw_data(DATA_RAW)
    clean = prepare_dataframe(raw)
    assert len(clean) == len(raw.drop_duplicates())


def test_positive_rate_is_imbalanced_but_nonzero():
    clean = prepare_dataframe(load_raw_data(DATA_RAW))
    rate = clean[TARGET].mean()
    assert 0.01 < rate < 0.05


def test_loader_returns_dataframe():
    assert isinstance(load_raw_data(DATA_RAW), pd.DataFrame)
