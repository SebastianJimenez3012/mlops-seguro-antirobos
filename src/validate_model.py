"""Quality gate de métricas para impedir promoción de modelos débiles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import METRICS_PATH, MIN_F1, MIN_RECALL, MIN_ROC_AUC


def validate(metrics_path: Path = METRICS_PATH) -> None:
    """Finaliza con código 1 si alguna métrica mínima no se cumple."""
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} no existe. Entrena primero el modelo.")
        raise SystemExit(1)

    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)

    checks = {
        "recall": (float(metrics.get("recall", 0.0)), MIN_RECALL),
        "roc_auc": (float(metrics.get("roc_auc", 0.0)), MIN_ROC_AUC),
        "f1": (float(metrics.get("f1", 0.0)), MIN_F1),
    }
    print("=" * 60)
    print("QUALITY GATE — MODELO DE SEGURO ANTIRROBOS")
    print("=" * 60)
    failed = []
    for name, (value, threshold) in checks.items():
        status = "OK" if value >= threshold else "FAIL"
        print(f"{name:10s}: {value:.4f} | mínimo {threshold:.4f} | {status}")
        if value < threshold:
            failed.append(name)

    if failed:
        print(f"\nRECHAZADO. No continúa a Docker: {', '.join(failed)}")
        raise SystemExit(1)
    print("\nAPROBADO. El pipeline puede continuar a build Docker.")


if __name__ == "__main__":
    try:
        validate()
    except SystemExit as exc:
        sys.exit(exc.code)
