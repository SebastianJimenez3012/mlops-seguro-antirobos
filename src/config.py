"""Configuración central del proyecto de propensión de compra de seguro."""
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"


def _load_params() -> dict:
    if not PARAMS_PATH.exists():
        return {}
    with open(PARAMS_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


PARAMS = _load_params()
TRAINING = PARAMS.get("training", {})
QUALITY = PARAMS.get("quality_gate", {})

DATA_RAW = ROOT_DIR / "data" / "raw" / "DS_Seguro_Antirobos.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed" / "seguro_antirobos_clean.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = ARTIFACTS_DIR / "roc_curve.png"

TARGET = "FLAG_SS"
RANDOM_STATE = int(TRAINING.get("random_state", 42))
TEST_SIZE = float(TRAINING.get("test_size", 0.30))
DEFAULT_N_ITER = int(TRAINING.get("n_iter", 6))
DEFAULT_CV_SPLITS = int(TRAINING.get("cv_splits", 3))

NUMERIC_FEATURES = [
    "Mto_TC",
    "SUELDO_ESTIMADO",
    "EDAD",
    "ANTIGUEDAD_MES",
]
CATEGORICAL_FEATURES = [
    "MARCA",
    "Nombre_territorio",
    "REGION",
    "SEXO",
    "SEGMENTO",
]
BINARY_FEATURES = ["FLAG_LIMA_PROVINCIA", "FLAG_UNICEF"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
EXPECTED_RAW_COLUMNS = FEATURES + [TARGET]

# Quality gate: calibrado a partir de la línea base del caso, no copiado de otro caso.
MIN_RECALL = float(QUALITY.get("min_recall", 0.25))
MIN_ROC_AUC = float(QUALITY.get("min_roc_auc", 0.73))
MIN_F1 = float(QUALITY.get("min_f1", 0.30))
