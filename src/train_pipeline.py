"""Entrenamiento con tuning, CV, trazabilidad y artefactos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.config import (
    ARTIFACTS_DIR,
    BEST_PARAMS_PATH,
    CONFUSION_MATRIX_PATH,
    DATA_PROCESSED,
    FEATURES,
    METRICS_PATH,
    MIN_F1,
    MIN_RECALL,
    MIN_ROC_AUC,
    MODEL_PATH,
    RANDOM_STATE,
    DEFAULT_N_ITER,
    DEFAULT_CV_SPLITS,
    ROC_CURVE_PATH,
    TARGET,
    TEST_SIZE,
)
from src.evaluate import calculate_metrics, save_evaluation_plots, save_json
from src.prepare_data import prepare_and_save
from src.preprocessing import build_training_pipeline


def search_space() -> dict:
    """Espacio de búsqueda acotado para mantener costo computacional razonable."""
    return {
        "selector__k": [15, 25, "all"],
        "model__n_estimators": [80, 120, 180],
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": [1, 3, 5],
        "model__class_weight": [None, "balanced_subsample"],
    }


def _start_mlflow_run(params: dict, metrics: dict, model, data_version: str | None):
    """Registra el run cuando MLflow está instalado; no bloquea el pipeline si no lo está."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "seguro-antirobos-tuning"))
    with mlflow.start_run(run_name="random_forest_best_run") as run:
        mlflow.log_params({str(k): v for k, v in params.items()})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "algorithm": "RandomForestClassifier",
                "data_version": data_version or "local",
                "use_case": "propension_seguro_antirobos",
            }
        )
        mlflow.sklearn.log_model(model, artifact_path="model")
        for artifact in (METRICS_PATH, BEST_PARAMS_PATH, CONFUSION_MATRIX_PATH, ROC_CURVE_PATH):
            if Path(artifact).exists():
                mlflow.log_artifact(str(artifact), artifact_path="evaluation")
        return run.info.run_id


def train_model(
    df: pd.DataFrame,
    n_iter: int = 6,
    cv_splits: int = 3,
    test_size: float = TEST_SIZE,
) -> tuple[object, dict]:
    """Entrena el pipeline completo sin fuga de información entre folds."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = RandomizedSearchCV(
        estimator=build_training_pipeline(),
        param_distributions=search_space(),
        n_iter=n_iter,
        scoring={
            "recall": "recall",
            "roc_auc": "roc_auc",
            "f1": "f1",
            "average_precision": "average_precision",
        },
        refit="recall",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    metrics = calculate_metrics(model, X_test, y_test)
    metrics.update(
        {
            "cv_best_recall": float(search.best_score_),
            "test_rows": int(len(X_test)),
            "test_positive_rows": int(y_test.sum()),
            "quality_gate": {
                "min_recall": MIN_RECALL,
                "min_roc_auc": MIN_ROC_AUC,
                "min_f1": MIN_F1,
            },
        }
    )
    return model, {"metrics": metrics, "best_params": search.best_params_}


def run_training(n_iter: int = DEFAULT_N_ITER, cv_splits: int = DEFAULT_CV_SPLITS) -> dict:
    """Ejecuta preparación, entrenamiento, evaluación y logging."""
    if not DATA_PROCESSED.exists():
        prepare_and_save()
    df = pd.read_csv(DATA_PROCESSED)

    model, result = train_model(df, n_iter=n_iter, cv_splits=cv_splits)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_json(result["best_params"], BEST_PARAMS_PATH)

    # Se recalcula el split determinista para generar los mismos plots del test final.
    X = df[FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    save_evaluation_plots(
        model,
        X_test,
        y_test,
        CONFUSION_MATRIX_PATH,
        ROC_CURVE_PATH,
    )
    save_json(result["metrics"], METRICS_PATH)

    data_version = os.getenv("DATA_VERSION")
    run_id = _start_mlflow_run(
        result["best_params"],
        {k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))},
        model,
        data_version,
    )
    if run_id:
        result["metrics"]["mlflow_run_id"] = run_id
        save_json(result["metrics"], METRICS_PATH)

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(n_iter=args.n_iter, cv_splits=args.cv_splits)
