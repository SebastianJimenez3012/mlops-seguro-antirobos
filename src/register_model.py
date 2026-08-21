"""Registro del mejor artefacto en MLflow Model Registry."""
from __future__ import annotations

import json
import os

from src.config import METRICS_PATH

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "SeguroAntirobosClassifier")


def register_best_model() -> None:
    """Registra el modelo del run ganador y lo deja en Staging para QA."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("Instala mlflow para registrar el modelo") from exc

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    with open(METRICS_PATH, encoding="utf-8") as handle:
        metrics = json.load(handle)
    run_id = metrics.get("mlflow_run_id")
    if not run_id:
        raise RuntimeError("metrics.json no contiene mlflow_run_id")

    result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        stage_message = "Stage: Staging"
    except Exception:
        # Compatibilidad defensiva si la versión de MLflow usa aliases en lugar de stages.
        client.set_registered_model_alias(MODEL_NAME, "staging", result.version)
        stage_message = "Alias: staging"

    print(f"Modelo registrado: {MODEL_NAME} v{result.version} | {stage_message}")
    print("La promoción a Production debe ocurrir después de validación QA.")


if __name__ == "__main__":
    register_best_model()
