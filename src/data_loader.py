"""Carga, contrato y preparación básica de datos."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EXPECTED_RAW_COLUMNS, FEATURES, TARGET


class DataContractError(ValueError):
    """Error de contrato de datos de entrada."""


def load_raw_data(path: Path) -> pd.DataFrame:
    """Carga el CSV original separado por punto y coma."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset: {path}")
    return pd.read_csv(path, sep=";")


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Valida columnas y condiciones mínimas antes de modelar."""
    missing = sorted(set(EXPECTED_RAW_COLUMNS) - set(df.columns))
    if missing:
        raise DataContractError(f"Columnas obligatorias faltantes: {missing}")
    if df.empty:
        raise DataContractError("El dataset está vacío")

    # El diccionario del caso solo documenta FLAG_SS=venta. El notebook interpreta
    # sus NaN como no compra; esta regla se conserva explícitamente como supuesto.
    non_null_target = set(df[TARGET].dropna().unique().tolist())
    if not non_null_target.issubset({0, 1, 0.0, 1.0}):
        raise DataContractError(
            f"{TARGET} contiene valores no binarios: {sorted(non_null_target)}"
        )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia duplicados y formaliza variables binarias sin usar información de test."""
    validate_raw_schema(df)
    clean = df.copy().drop_duplicates().reset_index(drop=True)

    # Supuesto heredado del notebook del caso: NaN en FLAG_SS = no compra.
    clean[TARGET] = clean[TARGET].fillna(0).astype(int)

    # FLAG_UNICEF contiene casi exclusivamente 1 cuando viene informado; NaN se
    # interpreta como ausencia del flag, por lo que se transforma en 0.
    clean["FLAG_UNICEF"] = clean["FLAG_UNICEF"].fillna(0).astype(int)

    # Mantener únicamente variables del modelo + target para evitar columnas ocultas.
    clean = clean[FEATURES + [TARGET]]
    validate_prepared_data(clean)
    return clean


def validate_prepared_data(df: pd.DataFrame) -> None:
    """Contrato posterior a preparación básica."""
    if df[TARGET].isna().any():
        raise DataContractError("El target no puede contener nulos")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise DataContractError("El target debe ser binario 0/1")
    positive_rate = float(df[TARGET].mean())
    if not 0.005 <= positive_rate <= 0.15:
        raise DataContractError(
            f"Tasa positiva fuera del rango esperado: {positive_rate:.2%}"
        )
