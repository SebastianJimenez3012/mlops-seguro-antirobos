"""Evaluación y persistencia de artefactos de desempeño."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(model, X_test, y_test) -> dict:
    """Calcula métricas apropiadas para clasificación desbalanceada."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }


def save_evaluation_plots(model, X_test, y_test, cm_path: Path, roc_path: Path) -> None:
    """Guarda matriz de confusión y curva ROC."""
    cm_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Matriz de confusión - Seguro antirrobos")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=140)
    plt.close(fig)

    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="Modelo")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Aleatorio")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC - Seguro antirrobos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=140)
    plt.close(fig)


def save_json(payload: dict, path: Path) -> None:
    """Persiste un diccionario JSON de forma legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
