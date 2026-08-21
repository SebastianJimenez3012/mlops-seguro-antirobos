"""Tests de transformaciones reproducibles."""
import numpy as np
import pandas as pd

from src.config import FEATURES
from src.preprocessing import QuantileCapper, build_preprocessor


def test_quantile_capper_caps_extreme_values():
    values = np.array([[1.0], [2.0], [3.0], [1000.0]])
    capper = QuantileCapper(quantile=0.75).fit(values)
    transformed = capper.transform(values)
    assert transformed.max() <= capper.caps_[0]


def test_preprocessor_handles_missing_and_unknown_category():
    rows = [
        {
            "Mto_TC": 1000,
            "SUELDO_ESTIMADO": np.nan,
            "EDAD": 30,
            "ANTIGUEDAD_MES": 12,
            "MARCA": "Visa",
            "Nombre_territorio": "T.CENTRO",
            "REGION": "LIMA CENTRO",
            "SEXO": "F",
            "SEGMENTO": "CLASICO",
            "FLAG_LIMA_PROVINCIA": 1,
            "FLAG_UNICEF": 0,
        },
        {
            "Mto_TC": 1500,
            "SUELDO_ESTIMADO": 2500,
            "EDAD": 40,
            "ANTIGUEDAD_MES": 60,
            "MARCA": "MARCA_NUEVA",
            "Nombre_territorio": "T.CENTRO",
            "REGION": "REGION_NUEVA",
            "SEXO": "M",
            "SEGMENTO": "VIP",
            "FLAG_LIMA_PROVINCIA": np.nan,
            "FLAG_UNICEF": np.nan,
        },
    ]
    frame = pd.DataFrame(rows).reindex(columns=FEATURES)
    transformer = build_preprocessor()
    transformed = transformer.fit_transform(frame)
    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()
