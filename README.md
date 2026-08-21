# Proyecto Final MLOps — Propensión de Compra de Seguro Antirrobos

Proyecto **end-to-end de MLOps** construido a partir del caso de Venta de Seguros Antirrobos y de la progresión de las Unidades 1–7 del curso *MLOps: Del Modelo al Entorno Productivo*.

## 1. Caso de negocio

El objetivo es estimar la probabilidad de que un cliente acepte un seguro antirrobos y utilizar ese score para priorizar campañas comerciales.

**Target:** `FLAG_SS`

- `1`: venta/aceptación del seguro.
- `NaN -> 0`: supuesto utilizado por el notebook original del caso para representar no compra. El diccionario solo documenta `FLAG_SS` como venta del seguro, por lo que esta interpretación se registra explícitamente como supuesto.

El dataset original contiene 28,313 registros y una clase positiva cercana al 2.6%, por lo que **accuracy no se utiliza como criterio principal**. Las métricas prioritarias son Recall, ROC-AUC, F1 y PR-AUC.

## 2. Arquitectura

```text
Raw Data
   |
   v
Data Contract / Cleaning
   |
   v
Train/Test Split
   |
   v
Preprocess (fit SOLO en train/fold)
   |-- P99 capping
   |-- median/mode imputation
   |-- One-Hot Encoding
   v
SelectKBest
   |
   v
SMOTE dentro de CV
   |
   v
Random Forest + RandomizedSearchCV
   |
   +--> MLflow Tracking
   |
   v
Evaluation + Quality Gate
   |
   v
Model Artifact
   |
   v
FastAPI -> Docker -> GitHub Actions
```

## 3. Evolución de las Unidades 1–7

| Unidad | Implementación en este repositorio |
|---|---|
| 1 | `data/`, `notebooks/`, `src/`, `requirements.txt`, README |
| 2 | Código modular, PEP8, `pytest`, tests por responsabilidad |
| 3 | Git/GitHub, `dvc.yaml`, `params.yaml`, descriptor `.dvc` del raw |
| 4 | Pipeline reproducible, data contract, fail-fast, artefactos |
| 5 | FastAPI, `/health`, `/predict`, Docker y docker-compose |
| 6 | RandomizedSearchCV, Stratified CV, MLflow Tracking y script de Registry |
| 7 | Makefile, flake8, coverage, quality gate y GitHub Actions CI/CD |

## 4. Correcciones respecto al notebook base

El notebook entregado con el caso se conserva en `notebooks/` como línea base, pero el pipeline productivo corrige problemas importantes:

1. El capping P99 y las imputaciones se **aprenden dentro del pipeline**, no antes del split.
2. SMOTE se ejecuta **dentro de cada fold de validación cruzada**, evitando contaminar validación.
3. Se guarda el **pipeline completo** (preprocesamiento + selección + balanceo/modelo) para evitar train-serving skew.
4. `FLAG_UNICEF` usa `NaN -> 0`; imputarlo con moda sería incorrecto porque los valores no nulos son casi todos 1.
5. La evaluación regional del notebook anterior a convertir `FLAG_SS` en 0/1 no se usa como evidencia de conversión.

## 5. Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## 6. Ejecución local

```bash
make lint
make test
make validate-data
make prepare
make train
make validate
```

O todo el pipeline:

```bash
make all
```

Con DVC:

```bash
dvc repro
```

## 7. MLflow

El entrenamiento registra automáticamente el mejor run si `mlflow` está instalado.

```bash
python -m src.train_pipeline
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir: `http://localhost:5000`

Para registrar el mejor artefacto en Model Registry:

```bash
python -m src.register_model
```

## 8. API

Después de entrenar:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`
- Predicción: `POST http://localhost:8000/predict`

Ejemplo de entrada:

```json
{
  "Mto_TC": 12000,
  "MARCA": "Visa",
  "Nombre_territorio": "T.CENTRO",
  "FLAG_LIMA_PROVINCIA": 1,
  "REGION": "LIMA MODERNA",
  "SUELDO_ESTIMADO": 4000,
  "EDAD": 35,
  "SEXO": "F",
  "ANTIGUEDAD_MES": 80,
  "SEGMENTO": "CLASICO",
  "FLAG_UNICEF": 0
}
```

## 9. Docker

```bash
docker build -t seguro-antirobos-api:local .
docker run --rm -p 8000:8000 seguro-antirobos-api:local
```

O:

```bash
docker compose up --build
```

## 10. Resultado verificado localmente

Con el pipeline corregido y `RandomizedSearchCV` (6 configuraciones, 3 folds), la ejecución local generó:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9795 |
| Precision | 0.8095 |
| Recall | 0.3036 |
| F1 | 0.4416 |
| ROC-AUC | 0.7654 |
| PR-AUC | 0.3722 |

Mejores hiperparámetros de esta corrida: `selector__k=25`, `n_estimators=80`, `max_depth=None`, `min_samples_leaf=1`, `class_weight=None`. Los resultados están persistidos en `artifacts/`.

**Verificación local:** 13 tests aprobados con cobertura total aproximada de 52% sobre `src/` + `api/`.

## 11. Quality Gate

Los umbrales iniciales están definidos en `src/config.py` y documentados en `params.yaml`:

- Recall >= 0.25
- ROC-AUC >= 0.73
- F1 >= 0.30

No son valores copiados de otro caso; parten de la línea base del dataset de seguro antirrobos y deben recalibrarse con evidencia de negocio futura.

## 12. CI/CD

`.github/workflows/ml_pipeline.yml` implementa:

```text
Push / Pull Request
  -> flake8
  -> pytest + coverage
  -> data quality gate
  -> prepare
  -> training + MLflow
  -> model quality gate
  -> upload artifacts
  -> Docker build
  -> API smoke test
```

La versión entregada implementa **Continuous Integration + Continuous Delivery**: la imagen queda validada y lista para ser promovida. No se afirma Continuous Deployment porque el push/deploy automático a infraestructura productiva no está configurado.

## 13. Estructura principal

```text
.github/workflows/ml_pipeline.yml
api/
artifacts/
data/raw/
data/processed/
docs/
notebooks/
src/
tests/
Dockerfile
docker-compose.yml
dvc.yaml
params.yaml
MLproject
Makefile
requirements.txt
```




## 14. Fuente del caso

Los documentos originales del caso se preservan en `docs/` y el notebook de análisis en `notebooks/` para trazabilidad académica.
