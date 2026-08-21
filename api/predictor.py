"""Carga perezosa del artefacto y servicio de inferencia."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH


class ModelPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self._model = None

    @property
    def available(self) -> bool:
        return self.model_path.exists() or self._model is not None

    def load(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modelo no disponible: {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, payload: dict) -> dict:
        model = self.load()
        frame = pd.DataFrame([payload]).reindex(columns=FEATURES)
        probability = float(model.predict_proba(frame)[0, 1])
        prediction = int(probability >= 0.50)
        if probability >= 0.70:
            priority = "alta"
        elif probability >= 0.50:
            priority = "media"
        else:
            priority = "baja"
        return {
            "probability": probability,
            "prediction": prediction,
            "priority": priority,
        }
