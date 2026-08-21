"""Preprocesamiento sin leakage y pipeline de entrenamiento."""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Aprende topes por percentil únicamente con los datos recibidos en fit()."""

    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.caps_ = np.nanquantile(values, self.quantile, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float).copy()
        return np.minimum(values, self.caps_)


def build_preprocessor() -> ColumnTransformer:
    """Crea transformaciones para numéricas, categóricas y flags."""
    numeric = Pipeline(
        steps=[
            ("cap_p99", QuantileCapper(0.99)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    lima_flag = SimpleImputer(strategy="most_frequent")
    unicef_flag = SimpleImputer(strategy="constant", fill_value=0)

    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("lima", lima_flag, [BINARY_FEATURES[0]]),
            ("unicef", unicef_flag, [BINARY_FEATURES[1]]),
        ],
        remainder="drop",
    )


def build_training_pipeline() -> ImbPipeline:
    """Pipeline CV-safe: preprocess -> selección -> SMOTE -> Random Forest."""
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("selector", SelectKBest(score_func=f_classif, k="all")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
